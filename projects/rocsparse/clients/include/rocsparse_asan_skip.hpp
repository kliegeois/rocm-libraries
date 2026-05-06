/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#ifdef ROCSPARSE_WITH_ASAN

#include "rocsparse_arguments.hpp"
#include <cstring>

// Returns non-null skip reason if this test should be skipped under ASAN builds.
// Returns nullptr if the test should run normally.
//
// All remaining failures are ASAN infrastructure artifacts — not rocsparse bugs.
// Real OOB/UAF bugs found by ASAN have been fixed separately (see PR #7010).
//
// Three categories:
//
//  A. Functions that call rocprim (sort, scan, reduce) internally.
//     rocprim emits ASAN callbacks for every memory access, causing 1.5–10×
//     slowdowns and test timeouts. SpGEMM additionally shows numerical divergence
//     because rocprim sort order changes under ASAN, causing NNZ mismatch between
//     the symbolic (counting) and fill phases.
//     Lift this entire block once rocprim is ASAN-clean.
//
//  B. False-positive crashes from HIP memory pool page recycling.
//     HIP's GPU allocator recycles freed pages; ASAN reports use-after-free
//     on the recycled addresses. Not a rocsparse memory safety issue.
//     Confirmed: tests that crash in-suite pass when run in isolation.
//
//  C. Spin-loop family: HIP memory pool false positives.
//     These kernels use inter-workgroup atomic synchronization (spin-loops).
//     The kernels are annotated with KERNEL_NO_ASAN (no_sanitize) to prevent
//     hangs, and the spin-loop logic itself is correct. However, when run
//     in-suite, HIP's memory pool recycles freed pages from earlier tests,
//     and ASAN reports false-positive overflows on the recycled addresses.
//     Confirmed: failing tests pass when run in isolation.
//     Skip all parameterizations — the crash depends on test ordering, not
//     on (M, N) values.
//
inline const char* rocsparse_asan_skip_reason(const Arguments& arg)
{
    const char* func = arg.function;

    // ── Category A: functions that call rocprim ────────────────────────────
    // Skip all parameterizations of these functions. The root cause is in
    // rocprim, not rocsparse. Remove this block when rocprim is ASAN-clean.
    // clang-format off
    static const char* rocprim_users[] = {
        // SpGEMM / SpGEAM — rocprim sort (COO tuples) + prefix scan
        "csrgemm",              "csrgemm_reuse",        "bsrgemm",
        "spgemm_csr",           "spgemm_bsr",           "spgemm_reuse_csr",
        "csrgeam",              "bsrgeam",
        "spgeam_csr",           "spgeam_reuse_csr",
        // SpMM — rocprim reductions
        "csrmm",                "bsrmm",                "gebsrmm",
        "spmm_csr",             "spmm_csc",             "spmm_coo",
        "spmm_bsr",             "spmm_bell",
        "spmm_batched_csr",     "spmm_batched_csc",     "spmm_batched_coo",
        // SpMV — rocprim-based backends
        "spmv_csr",             "spmv_csc",             "spmv_coo",
        "spmv_ell",             "spmv_coo_aos",
        "v2_spmv_csr",          "v2_spmv_csc",          "v2_spmv_coo",
        "v2_spmv_ell",          "v2_spmv_sell",         "v2_spmv_coo_aos",
        "v2_spmv_csr_res",
        // Sort — rocprim segmented_radix_sort
        "csrsort",              "cscsort",              "coosort",
        // Format conversion — rocprim prefix scan / sort
        "csr2gebsr",            "csr2bsr",              "bsr2csr",
        "csr2csc",              "csr2hyb",              "hyb2csr",
        "ell2csr",              "gebsr2gebsc",
        // Other rocprim users
        "gemvi",                "nnz",
        "gtsv",                 "gtsv_no_pivot",        "gtsv_no_pivot_strided_batch",
        "sparse_to_sparse",
        "bsrxmv",               "copy_info",
        "prune_csr2csr_by_percentage",
        "prune_dense2csr",      "prune_dense2csr_by_percentage",
        nullptr
    };
    // clang-format on
    for(int i = 0; rocprim_users[i]; ++i)
    {
        if(std::strcmp(func, rocprim_users[i]) == 0)
            return "ASAN skip: uses rocprim (rocprim ASAN issues tracked separately)";
    }

    // ── Category B: HIP memory pool false-positive crashes ─────────────────
    if(arg.a_type == rocsparse_datatype_f32_c || arg.a_type == rocsparse_datatype_f64_c
       || arg.compute_type == rocsparse_datatype_f32_c
       || arg.compute_type == rocsparse_datatype_f64_c)
    {
        if(std::strcmp(func, "coomv") == 0 || std::strcmp(func, "hybmv") == 0)
            return "ASAN false positive: HIP memory pool page recycling (complex type)";
    }
    if(std::strcmp(func, "bsrpad_value") == 0)
        return "ASAN false positive: device crash (HIP pool recycling)";
    if(std::strcmp(func, "csric0") == 0)
        return "ASAN false positive: HIP pool recycling (all csric0 tests crash in-suite)";
    // Large external matrix: too slow under ASAN regardless of rocprim
    if(arg.filename[0] != '\0' && std::strstr(arg.filename, "amazon0312") != nullptr)
        return "ASAN overhead: large external matrix (amazon0312) exceeds timeout";

    // ── Category C: spin-loop family — HIP memory pool false positives ──────
    // These functions use inter-workgroup atomic spin-loops (KERNEL_NO_ASAN).
    // When run in-suite, HIP's memory pool recycles freed pages from earlier
    // tests, causing false-positive ASAN reports. Tests pass in isolation.
    // Skip all parameterizations — crash depends on test ordering, not (M, N).
    // clang-format off
    static const char* spin_loop_funcs[] = {
        "bsric0",       "bsrilu0",      "bsrsm",        "bsrsv",
        "csrilu0",      "csritilu0",    "csritilu0_ex",
        "csrsm",        "csrsv",        "csritsv",      "csricsv",      "csrilusv",
        "spic0",        "spilu0",
        "spsm_csr",     "spsm_coo",
        "spsv_csr",     "spsv_coo",
        "sptrsm_csr",   "sptrsm_coo",
        "sptrsv",       "sptrsv_csr",   "sptrsv_coo",
        "spitsv_csr",
        nullptr
    };
    // clang-format on
    for(int i = 0; spin_loop_funcs[i]; ++i)
    {
        if(std::strcmp(func, spin_loop_funcs[i]) == 0)
            return "ASAN false positive: spin-loop family, HIP pool recycling (passes in isolation)";
    }

    return nullptr;
}

#define ROCSPARSE_ASAN_CHECK_SKIP(arg)                                   \
    do                                                                   \
    {                                                                    \
        const char* _asan_skip_reason = rocsparse_asan_skip_reason(arg); \
        if(_asan_skip_reason)                                            \
        {                                                                \
            GTEST_SKIP() << _asan_skip_reason;                           \
            return;                                                      \
        }                                                                \
    } while(0)

#else // !ROCSPARSE_WITH_ASAN

#define ROCSPARSE_ASAN_CHECK_SKIP(arg) \
    do                                 \
    {                                  \
    } while(0)

#endif // ROCSPARSE_WITH_ASAN
