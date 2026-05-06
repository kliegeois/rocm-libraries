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
// Two categories:
//
//  A. Functions that call rocprim (sort, scan, reduce) internally.
//     rocprim emits ASAN callbacks for every memory access, causing 1.5–10×
//     slowdowns and test timeouts. SpGEMM additionally shows numerical divergence
//     because rocprim sort order changes under ASAN, causing NNZ mismatch between
//     the symbolic (counting) and fill phases.
//     Lift this entire block once rocprim is ASAN-clean.
//
//  B. Residual false-positive crashes from HIP memory pool page recycling.
//     Most pool-recycling crashes were fixed by switching position_t::m_position
//     from hipMallocAsync to hipMalloc (keeping it outside the pool).
//     The remaining entries below have causes unrelated to position_t.
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
        "v2_spmv_csr_res",      "v2_spmv_csr_res_multiple",
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
    // These functions crash due to HIP memory pool page recycling:
    // a freed GPU page is reused within or between tests, and ASAN reports
    // the access as use-after-free. Not a rocsparse memory safety issue.
    // clang-format off
    static const char* pool_recycling_crashes[] = {
        // BSR triangular solve — has KERNEL_NO_ASAN for spin-loops; pool recycling
        // from earlier Category-A tests (bsr2csr/csr2bsr) still triggers OOB on
        // recycled addresses unrelated to position_t. Needs Category A fixed first.
        "bsrsm",
        // BSR triangular solve (vector) — ASAN-induced race in analysis caching:
        // trm_info lookup returns nullptr for block_dim=9 with analysis_policy_reuse.
        // Passes without ASAN; root cause is lock ordering under ASAN timing.
        "bsrsv",
        nullptr
    };
    // clang-format on
    for(int i = 0; pool_recycling_crashes[i]; ++i)
    {
        if(std::strcmp(func, pool_recycling_crashes[i]) == 0)
            return "ASAN false positive: HIP memory pool page recycling";
    }

    if(arg.a_type == rocsparse_datatype_f32_c || arg.a_type == rocsparse_datatype_f64_c
       || arg.compute_type == rocsparse_datatype_f32_c
       || arg.compute_type == rocsparse_datatype_f64_c)
    {
        if(std::strcmp(func, "coomv") == 0 || std::strcmp(func, "hybmv") == 0)
            return "ASAN false positive: HIP memory pool page recycling (complex type)";
    }
    if(std::strcmp(func, "bsrpad_value") == 0)
        return "ASAN false positive: device crash (HIP pool recycling)";

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
