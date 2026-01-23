# Code Coverage Gap Analysis: develop vs reduced_ci

**Generated:** January 23, 2026  
**Comparison:** `coverage_develop` → `coverage_reduced_ci`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total files affected** | 73 |
| **Total lines lost coverage** | 1,682 |
| **Line coverage drop** | 77.9% → 76.2% (-1.7%) |
| **Function coverage drop** | 51.4% → 49.8% (-1.6%) |

## Coverage Loss by Capability

| Capability | Lines Lost | Files Affected |
|------------|-----------|----------------|
| level2 (SpMV, SpSV) | 811 | 23 |
| extra (SpGEMM, SpGEAM) | 416 | 15 |
| level3 (SpMM, SpSM) | 302 | 19 |
| conversion | 99 | 11 |
| precond (GTSV) | 42 | 3 |
| reordering | 8 | 1 |
| common | 4 | 1 |

---

## Detailed Analysis by Operation

### 1. SELLMV (Sliced ELLPACK Matrix-Vector Multiply)

**File:** `rocsparse_sellmv_impl.cpp`  
**Lines Lost:** 95

#### Missing Test Scenarios

| Scenario | Description | Parameters |
|----------|-------------|------------|
| Transpose | Transpose operation not tested | `trans = rocsparse_operation_transpose` |
| Conjugate Transpose | Conjugate transpose not tested | `trans = rocsparse_operation_conjugate_transpose` |
| Large Slice Size | Large slice kernel path | `sell_slice_size > 128` |

#### Specific Kernels Not Covered
- `sellmvt_kernel<8>` - Transpose with small slice size
- `sellmvt_large_slice_kernel<256>` - Transpose with large slice size

#### Recommended Tests
```yaml
- function: rocsparse_sellmv
  trans: [T, H]
  slice_size: [64, 128, 256, 512]
  datatypes: [f32, f64, c32, c64]
```

---

### 2. GEBSRMV (General Block Sparse Row Matrix-Vector)

**Files:**
- `rocsparse_gebsrmv_template_row_block_dim_1.cpp` (90 lines)
- `rocsparse_gebsrmv_template_row_block_dim_2.cpp` (84 lines)
- `rocsparse_gebsrmv_template_row_block_dim_3.cpp` (45 lines)
- `rocsparse_gebsrmv_template_row_block_dim_4.cpp` (48 lines)

**Total Lines Lost:** 267

#### Missing Test Scenarios

| Scenario | Description | Parameters |
|----------|-------------|------------|
| Row Block Dim 1 | Various column block dimensions | `row_block_dim = 1`, `col_block_dim = 2-16` |
| Row Block Dim 2 | Various column block dimensions | `row_block_dim = 2`, `col_block_dim = 1-16` |
| Wavefront 32 | Code path for wavefront_size == 32 | Requires MI100/MI200 or similar GPU |
| High blocks_per_row | Dense blocks per row | `blocks_per_row >= 64` |

#### Root Cause
The lost coverage is specifically in the `wavefront_size == 32` branch paths. These are only executed on certain AMD GPU architectures.

#### Recommended Tests
```yaml
- function: rocsparse_gebsrmv
  row_block_dim: [1, 2, 3, 4]
  col_block_dim: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  blocks_per_row: [4, 8, 16, 32, 64, 128]
  # Note: Test on wavefront_size=32 GPU (e.g., gfx1030)
```

---

### 3. COOMV (COO Matrix-Vector Multiply)

**Files:**
- `rocsparse_coomv_impl.cpp` (71 lines)
- `rocsparse_coomv_aos_impl.cpp` (49 lines)

**Total Lines Lost:** 120

#### Missing Test Scenarios

| Scenario | Description | Parameters |
|----------|-------------|------------|
| Transpose | Transpose operation | `trans = rocsparse_operation_transpose` |
| Conjugate Transpose | Conjugate transpose operation | `trans = rocsparse_operation_conjugate_transpose` |
| Atomic Algorithm | Atomic kernel path | `alg = rocsparse_coomv_alg_atomic` |

#### Specific Kernels Not Covered
- `coomvt_kernel<1024>` - Transpose kernel
- Atomic loop kernels with transpose

#### Recommended Tests
```yaml
- function: rocsparse_coomv
  trans: [T, H]
  algorithm: [atomic, segmented]
  datatypes: [f32, f64, c32, c64]
```

---

### 4. CSRMV (CSR Matrix-Vector Multiply)

**Files:**
- `rocsparse_csrmv_template_lrb.cpp` (66 lines)
- `rocsparse_csrmv_template_nnzsplit.cpp` (40 lines)
- `rocsparse_csrmv_template_rowsplit.cpp` (27 lines)
- `rocsparse_csrmv_impl.cpp` (27 lines)

**Total Lines Lost:** 160

#### Missing Test Scenarios

| Algorithm | Scenario | Parameters |
|-----------|----------|------------|
| LRB | Medium row kernel | Matrices with specific row length distributions |
| LRB | Conjugate operation | `conj = true` |
| NNZ-Split | Transpose | `trans = rocsparse_operation_transpose` |
| Row-Split | Transpose | `trans = rocsparse_operation_transpose` |

#### Recommended Tests
```yaml
- function: rocsparse_csrmv
  trans: [N, T, H]
  algorithm: [lrb, nnzsplit, rowsplit, adaptive]
  matrix_types:
    - uniform_short_rows   # < 32 nnz per row
    - uniform_medium_rows  # 32-256 nnz per row
    - uniform_long_rows    # > 256 nnz per row
    - mixed_row_lengths    # variable row lengths
```

---

### 5. HYBMV (Hybrid ELL+COO Matrix-Vector)

**File:** `rocsparse_hybmv.cpp`  
**Lines Lost:** 29

#### Missing Test Scenarios

| Scenario | Description | Parameters |
|----------|-------------|------------|
| Device Pointer Mode | Alpha/beta on device | `pointer_mode = rocsparse_pointer_mode_device` |
| Transpose | Transpose operation | `trans = rocsparse_operation_transpose` |
| COO portion with device pointers | COO part with device scalars | `ell_nnz > 0 && coo_nnz > 0`, device mode |

#### Recommended Tests
```yaml
- function: rocsparse_hybmv
  trans: [N, T, H]
  pointer_mode: [host, device]
  partition: [auto, user, max]
```

---

### 6. BSRGEMM (Block Sparse GEMM)

**File:** `rocsparse_bsrgemm_calc.cpp`  
**Lines Lost:** 123

#### Missing Test Scenarios

| Group | NNZ per Row Range | Description |
|-------|-------------------|-------------|
| 5 | 129 - 256 | Medium density output rows |
| 6 | 257 - 512 | High density output rows |
| 7 | 513+ | Very high density output rows |

#### Root Cause
The reduced CI doesn't include test matrices that produce output rows with high non-zero counts. These are typically generated when multiplying semi-dense or dense block matrices.

#### Recommended Tests
```yaml
- function: rocsparse_bsrgemm
  test_matrices:
    - name: medium_fill_result
      description: "A*B produces 129-256 nnz per row"
    - name: high_fill_result
      description: "A*B produces 257-512 nnz per row"
    - name: very_high_fill_result
      description: "A*B produces 513+ nnz per row"
  block_dim: [2, 3, 4, 5, 6, 7, 8]
```

---

### 7. CSRGEMM NNZ Calculation

**File:** `rocsparse_csrgemm_nnz_calc.cpp`  
**Lines Lost:** 106

#### Missing Test Scenarios

| Group | Intermediate Products | Hash Size |
|-------|----------------------|-----------|
| 4 | 1025 - 2048 | 2048 |
| 5 | 2049 - 4096 | 4096 |
| 6 | 4097 - 8192 | 8192 |
| 10 | Very large | Multipass |

#### Root Cause
SpGEMM symbolic phase requires matrices that produce large numbers of intermediate products per row. This typically requires multiplying matrices with overlapping sparsity patterns.

#### Recommended Tests
```yaml
- function: rocsparse_csrgemm
  phase: symbolic (nnz calculation)
  test_matrices:
    - name: high_intermediate_products
      A_avg_nnz_per_row: 50
      B_avg_nnz_per_row: 50
      overlap_factor: high
    - name: very_high_intermediate_products
      A_avg_nnz_per_row: 100
      B_avg_nnz_per_row: 100
```

---

### 8. GEBSRMM (General Block Sparse Row Matrix-Matrix)

**File:** `rocsparse_gebsrmm.cpp`  
**Lines Lost:** 38

#### Missing Test Scenarios

| Scenario | Description | Parameters |
|----------|-------------|------------|
| TransB | Transpose on dense matrix B | `transB = rocsparse_operation_transpose` |
| Conjugate TransB | Conjugate transpose on B | `transB = rocsparse_operation_conjugate_transpose` |

#### Recommended Tests
```yaml
- function: rocsparse_gebsrmm
  transA: [N]
  transB: [N, T, H]
  row_block_dim: [1, 2, 3, 4, 5, 8, 16]
  col_block_dim: [1, 2, 3, 4, 5, 8, 16]
```

---

### 9. CSRITSV Analysis

**File:** `rocsparse_csritsv_analysis.cpp`  
**Lines Lost:** 33

#### Missing Test Scenarios

| Scenario | Description |
|----------|-------------|
| Lower triangular, unit diagonal check | Missing diagonal detection for lower matrices |
| Upper triangular, unit diagonal check | Missing diagonal detection for upper matrices |
| Non-triangular matrix detection | Diagonal counting for non-triangular types |

#### Recommended Tests
```yaml
- function: rocsparse_csritsv
  fill_mode: [lower, upper]
  diag_type: [non_unit, unit]
  matrix_types:
    - with_missing_diagonal
    - with_zero_diagonal
    - fully_populated_diagonal
```

---

### 10. SPTRSM (Sparse Triangular Solve - Multiple RHS)

**File:** `rocsparse_sptrsm.cpp`  
**Lines Lost:** 27

#### Missing Test Scenarios

| Scenario | Description | Parameters |
|----------|-------------|------------|
| COO format | SpTRSM with COO input | `format = rocsparse_format_coo` |
| CSC format | SpTRSM with CSC input | `format = rocsparse_format_csc` |

#### Recommended Tests
```yaml
- function: rocsparse_sptrsm
  format: [csr, coo, csc]
  operation: [N, T, H]
  fill_mode: [lower, upper]
```

---

### 11. GTSV Preconditioner

**Files:**
- `rocsparse_gtsv.cpp`
- `rocsparse_gtsv_no_pivot.cpp`
- `rocsparse_gtsv_no_pivot_strided_batch.cpp`

**Total Lines Lost:** 42

#### Missing Test Scenarios
- Specific matrix size edge cases
- Batch size variations

---

### 12. Conversion Operations

**Files affected:** 11 files  
**Total Lines Lost:** 99

Key files:
- `rocsparse_gebsr2gebsr.cpp`
- `rocsparse_gebsr2gebsc.cpp`
- `rocsparse_csr2ell.cpp`

#### Missing Test Scenarios
- Edge cases in block format conversions
- Specific block dimension combinations

---

## Test Priority Matrix

| Priority | Operation | Impact (Lines) | Complexity |
|----------|-----------|----------------|------------|
| **High** | GEBSRMV (wavefront32) | 267 | Medium - requires specific GPU |
| **High** | COOMV transpose | 120 | Low - just add trans parameter |
| **High** | BSRGEMM high-fill | 123 | Medium - need dense test matrices |
| **High** | CSRGEMM symbolic | 106 | Medium - need overlapping matrices |
| **Medium** | SELLMV transpose | 95 | Low - just add trans parameter |
| **Medium** | CSRMV transpose | 160 | Low - add trans to existing tests |
| **Medium** | GEBSRMM transB | 38 | Low - just add transB parameter |
| **Low** | HYBMV device mode | 29 | Low |
| **Low** | Conversion edge cases | 99 | Medium |

---

## Recommended Test Additions

### Quick Wins (Low effort, high coverage recovery)

1. **Add transpose tests for all SpMV operations:**
   - SELLMV, COOMV, CSRMV, HYBMV
   - Estimated recovery: ~400 lines

2. **Add transB tests for SpMM operations:**
   - GEBSRMM, CSRMM
   - Estimated recovery: ~70 lines

### Medium Effort

3. **Create dense/semi-dense SpGEMM test matrices:**
   - For BSRGEMM and CSRGEMM symbolic phase
   - Estimated recovery: ~230 lines

### Hardware-Dependent

4. **Run tests on wavefront_size=32 GPU:**
   - GEBSRMV tests on MI100, MI200, or gfx1030
   - Estimated recovery: ~270 lines

---

## Appendix: Top 20 Files by Coverage Loss

| Lines Lost | File |
|------------|------|
| 123 | `extra/rocsparse_bsrgemm_calc.cpp` |
| 106 | `extra/rocsparse_csrgemm_nnz_calc.cpp` |
| 95 | `level2/rocsparse_sellmv_impl.cpp` |
| 90 | `level2/rocsparse_gebsrmv_template_row_block_dim_1.cpp` |
| 84 | `level2/rocsparse_gebsrmv_template_row_block_dim_2.cpp` |
| 71 | `level2/rocsparse_coomv_impl.cpp` |
| 66 | `level2/rocsparse_csrmv_template_lrb.cpp` |
| 49 | `level2/rocsparse_coomv_aos_impl.cpp` |
| 48 | `level2/rocsparse_gebsrmv_template_row_block_dim_4.cpp` |
| 45 | `extra/rocsparse_csrgemm.cpp` |
| 45 | `level2/rocsparse_gebsrmv_template_row_block_dim_3.cpp` |
| 40 | `level2/rocsparse_csrmv_template_nnzsplit.cpp` |
| 38 | `level3/rocsparse_gebsrmm.cpp` |
| 33 | `level2/rocsparse_bsrmv_impl.cpp` |
| 33 | `level2/rocsparse_csritsv_analysis.cpp` |
| 29 | `level2/rocsparse_hybmv.cpp` |
| 28 | `level3/rocsparse_csrmm_template_row_split.cpp` |
| 27 | `level2/rocsparse_csrmv_impl.cpp` |
| 27 | `level2/rocsparse_csrmv_template_rowsplit.cpp` |
| 27 | `level3/rocsparse_sptrsm.cpp` |
