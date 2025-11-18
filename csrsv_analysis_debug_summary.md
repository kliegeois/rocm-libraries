# CSRSv Analysis Memory Corruption Debug Summary

**Date:** November 19, 2025  
**Issue:** `csrsv` test crashes on iteration 4/50 with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`  
**Matrix:** bmwcra_1.csr (m=148770, nnz=10,644,002)  
**Test Command:** `./rocsparse-test --gtest_filter=nightly/csrsv.level2/f32_r_1_0_0_T_ND_U_force_auto_0b_csr_0_bmwcra_1 --gtest_repeat=50`

---

## Problem Analysis

### Root Cause Identified
The test calls `rocsparse_csrsv_analysis()` **50 times with the same `mat_info` object**. Each call invokes `trm_analysis()` which:
1. Computes transpose of the matrix (for transposed operations)
2. Stores transposed arrays in `trm_info` structure
3. On subsequent calls, the arrays already exist but contain **stale data** from previous iterations

### Key Discovery: Memory Reuse Bug
- **`temp_buffer`** is allocated once and reused across all 50 iterations
- Inside `temp_buffer`, we have `tmp_work1` and `tmp_work2` arrays
- `radix_sort_pairs()` uses a **double_buffer pattern** that swaps between:
  - `keys`: {tmp_work1, transposed_col_ind}
  - `vals`: {transposed_perm, tmp_work2}
- After radix_sort, `vals.current()` may point to either `transposed_perm` OR `tmp_work2`
- On iteration 2+, `tmp_work2` contains **garbage from iteration 1**, leading to corruption

### Crash Location
The crash occurs in `gthr_kernel` at:
```cpp
x_val[idx] = y[x_ind[idx] - idx_base];
```
Where `x_ind` is `transposed_perm` containing invalid indices (negative values, values > nnz), causing out-of-bounds memory access.

---

## Investigation Timeline

### Phase 1: Initial Debugging (Added Extensive Logging)
**File Modified:** `/app/rocm-libraries/projects/rocsparse/library/src/level2/rocsparse_csrsv_analysis_impl.cpp`

Added debug output throughout `trm_analysis()`:
- Print statements at every major step
- `hipDeviceSynchronize()` after each operation to ensure sequential execution
- Validation code to check `transposed_perm` indices are in valid range [0, nnz)

**Finding:** Validation showed `transposed_perm[0] = -1897936846` on iteration 2, confirming corruption.

### Phase 2: Root Cause - Double Buffer Initialization
Discovered that `radix_sort_pairs()` may swap buffers, so we need to initialize **BOTH** arrays:
- `transposed_perm` ✓ (initialized with identity permutation)
- `tmp_work2` ✗ (NOT initialized - contains stale data!)

**Attempted Fix 1:** Initialize both `transposed_perm` AND `tmp_work2` with identity permutation
```cpp
// Create identity permutation in BOTH buffers
rocsparse_create_identity_permutation(handle, nnz, transposed_perm);
rocsparse_create_identity_permutation(handle, nnz, tmp_work2);
```

**Result:** Still got garbage values - this didn't fully solve the problem.

### Phase 3: Buffer Reuse Logic
**Original Code Issue:** Had an error check that would fail if transposed arrays already existed:
```cpp
if(*ref_transposed_mat != nullptr)
{
    return rocsparse_status_internal_error;  // This prevented reuse!
}
```

**Attempted Fix 2:** Free and reallocate buffers on every call
- Used `rocsparse_hipFreeAsync()` to free existing buffers
- Reallocate with `rocsparse_hipMallocAsync()`

**Result:** Caused page faults and memory corruption - async free/realloc is unreliable!

### Phase 4: Smart Reuse Strategy
**Current Implementation (lines 66-119):**
```cpp
// Check if analysis was already done: buffers exist and dimensions already stored
bool analysis_already_done = (trm_info->get_transposed_perm() != nullptr
                              && trm_info->get_transposed_row_ptr() != nullptr
                              && trm_info->get_transposed_col_ind() != nullptr
                              && trm_info->get_m() == m && trm_info->get_nnz() == nnz);

if(analysis_already_done)
{
    // Skip entire transpose computation and return early
    return rocsparse_status_success;
}

// Check if dimensions changed and we need to free old buffers
bool buffers_exist = (trm_info->get_transposed_perm() != nullptr
                     || trm_info->get_transposed_row_ptr() != nullptr
                     || trm_info->get_transposed_col_ind() != nullptr);

if(buffers_exist && (trm_info->get_m() != m || trm_info->get_nnz() != nnz))
{
    // Free old buffers only if dimensions changed
    // Then reallocate with new sizes
}
```

**Logic:**
1. **Iteration 1:** Buffers are nullptr → allocate and compute transpose → store dimensions
2. **Iteration 2-50:** Buffers exist AND dimensions match → early return (reuse existing data)
3. **If dimensions change:** Free old buffers, reallocate new ones

---

## Current Status: Test Hanging

### Symptom
Test hangs on **iteration 1** - doesn't complete even a single run. Timeout after 60+ seconds.

### Likely Cause: Excessive Debug Code
The validation code (lines 270-315) runs on every call:
```cpp
// Copy 10M+ elements to host for validation
std::vector<I> h_transposed_perm(nnz);  // nnz = 10,644,002
hipMemcpyAsync(h_transposed_perm.data(), transposed_perm, sizeof(I) * nnz, ...);
hipStreamSynchronize(stream);

// Loop through ALL elements checking bounds
for(I i = 0; i < nnz; ++i)  // 10M+ iterations!
{
    I idx = h_transposed_perm[i] - rocsparse_index_base_zero;
    if(idx < 0 || idx >= nnz) { /* error */ }
}
```

This validation:
- Copies **40 MB** of data from device to host
- Loops through **10,644,002** elements
- Runs on **every single call** to `trm_analysis()`

Additionally, there are **~50+ `hipDeviceSynchronize()` calls** throughout the function, each forcing full GPU synchronization.

---

## Files Modified

### Primary File
`/app/rocm-libraries/projects/rocsparse/library/src/level2/rocsparse_csrsv_analysis_impl.cpp`

**Key Changes:**
1. Removed original error check for existing buffers
2. Added buffer reuse logic with early return
3. Added dimension checking (get_m(), get_nnz())
4. Initialize both double_buffer arrays with identity permutation
5. Added extensive debug logging (lines 63, 75, 88, 142, 145, 157, 164, etc.)
6. Added validation code (lines 270-315)
7. Multiple `hipDeviceSynchronize()` calls throughout

### Secondary Files (Minor Changes)
`/app/rocm-libraries/projects/rocsparse/library/src/level1/rocsparse_gthr_impl.cpp`
- Added debug output for gthr operation
- Added validation of nnz and pointers

---

## Data Structures

### `trm_info_t` Structure
Located in: `/app/rocm-libraries/projects/rocsparse/library/src/include/rocsparse_trm_info.hpp`

Key members:
```cpp
struct trm_info_t {
    int64_t m;                        // Matrix rows - get_m(), set_m()
    int64_t nnz;                      // Non-zeros - get_nnz(), set_nnz()
    void* transposed_perm;            // Permutation array
    void* transposed_row_ptr;         // Transposed row pointers
    void* transposed_col_ind;         // Transposed column indices
    const void* row_ptr;              // Original row pointers
    const void* col_ind;              // Original column indices
    // ...
};
```

### Buffer Layout in `temp_buffer`
```
+-------------------+
| tmp_work1 (J*)    |  Size: nnz * sizeof(J)  - for column indices
+-------------------+
| tmp_work2 (I*)    |  Size: nnz * sizeof(I)  - for permutation
+-------------------+
| rocprim_buffer    |  Size: rocprim_size     - for radix_sort
+-------------------+
```

**Problem:** `tmp_work2` is reused across iterations but never cleared!

---

## Radix Sort Double Buffer Pattern

```cpp
rocsparse::primitives::double_buffer<J> keys(tmp_work1, transposed_col_ind);
rocsparse::primitives::double_buffer<I> vals(transposed_perm, tmp_work2);

radix_sort_pairs(keys, vals, ...);

// After sort, vals.current() may point to EITHER:
//   - transposed_perm (if even number of swaps)
//   - tmp_work2 (if odd number of swaps)

if(vals.current() != transposed_perm)
{
    // Copy result back to transposed_perm
    hipMemcpyAsync(transposed_perm, vals.current(), sizeof(I) * nnz, ...);
}
```

**Critical Insight:** We must ensure BOTH `transposed_perm` and `tmp_work2` contain valid data before calling `radix_sort_pairs()`.

---

## Next Steps / TODO

### Immediate Action: Remove Debug Code
1. **Remove ALL `std::cout` debug statements** (50+ lines)
2. **Remove ALL unnecessary `hipDeviceSynchronize()` calls** (keep only essential ones)
3. **Remove validation code** (lines 270-315) - this is causing the hang
4. **Test with clean code** to see if basic logic works

### Verify Fix Works
1. Test with `--gtest_repeat=50`
2. Check all iterations pass without crashes
3. Verify no memory corruption
4. Check performance (should be fast without debug overhead)

### Potential Alternative Solutions (if simple fix doesn't work)

**Option A: Clear temp_buffer on each call**
```cpp
// At function start for transpose case
if(nnz > 0)
{
    hipMemsetAsync(tmp_work2, 0, sizeof(I) * nnz, stream);
}
```

**Option B: Force result into transposed_perm**
```cpp
// Construct double_buffer to ensure result ends in transposed_perm
rocsparse::primitives::double_buffer<I> vals(tmp_work2, transposed_perm);
// This way vals.current() starts at tmp_work2, and after sort it ends in transposed_perm
```

**Option C: Always copy result (safest but slower)**
```cpp
// Always copy regardless of vals.current()
hipMemcpyAsync(transposed_perm, vals.current(), sizeof(I) * nnz, ...);
```

**Option D: Don't reuse at all (original behavior)**
```cpp
// Return error if buffers already exist - requires test to create new info each time
if(trm_info->get_transposed_perm() != nullptr)
{
    return rocsparse_status_internal_error;
}
```

---

## Key Insights

1. **async operations are tricky:** `rocsparse_hipFreeAsync()` + `rocsparse_hipMallocAsync()` in quick succession is unreliable even with synchronization

2. **Double buffer pattern requires both buffers initialized:** If radix_sort swaps to alternate buffer, that buffer MUST have valid data

3. **temp_buffer is persistent:** It's allocated once for the entire test run (all 50 iterations), so any data left in it persists

4. **Early return is the right approach:** If analysis was already done, skip all work and return immediately. This is both correct and performant.

5. **Validation is expensive:** Validating 10M+ elements with host copies and synchronization makes the test unbearably slow

6. **Debug synchronization changes behavior:** All the `hipDeviceSynchronize()` calls change timing and may hide/expose race conditions

---

## Build & Test Commands

```bash
# Build debug version
cd /app/rocm-libraries/projects/rocsparse/build/debug
make -j6

# Run single iteration to test basic functionality
cd clients/staging
./rocsparse-test --gtest_filter=nightly/csrsv.level2/f32_r_1_0_0_T_ND_U_force_auto_0b_csr_0_bmwcra_1 --gtest_repeat=1

# Run 50 iterations to test stability
./rocsparse-test --gtest_filter=nightly/csrsv.level2/f32_r_1_0_0_T_ND_U_force_auto_0b_csr_0_bmwcra_1 --gtest_repeat=50

# Timeout after 3 minutes if hanging
timeout 180 ./rocsparse-test --gtest_filter=... --gtest_repeat=50
```

---

## Questions to Answer Tomorrow

1. **Does removing all debug code fix the hang?**
   - If yes → test with 50 iterations
   - If no → there's a deeper issue (infinite loop, deadlock?)

2. **Does the early return logic actually work correctly?**
   - Check that iteration 2-50 actually skip computation
   - Verify dimensions are set correctly after iteration 1

3. **Is the double buffer initialization sufficient?**
   - Or do we need to clear `tmp_work2` explicitly on each call?

4. **Should we test with the release build?**
   - Debug builds have different optimization levels
   - Release build at: `/app/rocm-libraries/projects/rocsparse/build/release`

5. **Are there other functions with similar issues?**
   - `csrsm_analysis`, `bsrsv_analysis`, etc. may have same problem
   - Check if they also need buffer reuse logic

---

## References

### Code Locations
- Main function: `rocsparse::trm_analysis()` - line 774 in csrsv_analysis_impl.cpp
- Transpose logic: lines 60-340
- Buffer allocation: lines 146-176
- Radix sort: lines 180-250
- Validation (problematic): lines 270-315
- Dimension storage: lines 776-777

### Related Files
- `/app/rocm-libraries/projects/rocsparse/library/src/include/rocsparse_trm_info.hpp`
- `/app/rocm-libraries/projects/rocsparse/library/src/common/rocsparse_trm_info.cpp`
- `/app/rocm-libraries/projects/rocsparse/library/src/level1/gthr_device.h`
- `/app/rocm-libraries/projects/rocsparse/library/src/level1/rocsparse_gthr_impl.cpp`

### Git Branch
- Repository: rocm-libraries
- Branch: csrsv_d
- Base: develop

---

## Final Recommendation

**Priority 1:** Strip out all debug code and test basic functionality  
**Priority 2:** If tests pass, verify performance is acceptable  
**Priority 3:** If tests still fail, investigate alternative buffer management strategies  

The core logic (early return when buffers exist + dimensions match) is sound. The problem is likely the overhead and side effects of the extensive debugging code added during investigation.
