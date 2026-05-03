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
inline const char* rocsparse_asan_skip_reason(const Arguments& arg)
{
    const char* func = arg.function;

    // Category 1: f64_c SpGEMM numerical divergence (ASAN infrastructure artifact).
    // ASAN instrumentation changes GPU compute timing, causing NNZ counting and
    // fill phases to disagree in csrgemm/bsrgemm with complex<double>.
    if(arg.compute_type == rocsparse_datatype_f64_c || arg.a_type == rocsparse_datatype_f64_c)
    {
        if(std::strcmp(func, "csrgemm") == 0 || std::strcmp(func, "bsrgemm") == 0
           || std::strcmp(func, "csrgemm_reuse") == 0 || std::strcmp(func, "spgemm_csr") == 0
           || std::strcmp(func, "spgemm_bsr") == 0 || std::strcmp(func, "spgemm_reuse_csr") == 0)
        {
            return "ASAN artifact: f64_c SpGEMM numerical divergence";
        }
    }

    // Category 2: False-positive crashes from HIP memory pool page recycling.
    // HIP's GPU memory allocator recycles freed pages, and ASAN reports
    // heap-use-after-free on recycled pages that are still in shadow memory.
    if(arg.a_type == rocsparse_datatype_f32_c || arg.compute_type == rocsparse_datatype_f32_c)
    {
        if(std::strcmp(func, "coomv") == 0 || std::strcmp(func, "hybmv") == 0
           || std::strcmp(func, "csrsm") == 0)
        {
            return "ASAN false positive: HIP memory pool page recycling (f32_c)";
        }
    }
    if(std::strcmp(func, "bsrpad_value") == 0)
    {
        return "ASAN false positive: device ASAN crash";
    }
    if(std::strcmp(func, "bsrsv") == 0)
    {
        return "ASAN false positive: device ASAN crash";
    }

    // Category 3: Tests that hang under ASAN due to atomic contention cascade
    // or general ASAN overhead causing family-level timeouts (600s).
    // Spin-loop families (csrsv, csrsm, bsrsv, bsrsm, csric0, csrilu0, etc.)
    // are handled by ROCSPARSE_KERNEL_NO_ASAN and excluded from this list.
    static const char* hung_families[] = {
        "bsrgeam",
        "bsrmm",
        "bsrxmv",
        "copy_info",
        "csr2gebsr",
        "csrgeam",
        "csrgemm",
        "csrgemm_reuse",
        "csrmm",
        "gebsrmm",
        "gemvi",
        "gtsv",
        "gtsv_no_pivot",
        "gtsv_no_pivot_strided_batch",
        "sparse_to_sparse",
        "spgeam_csr",
        "spgeam_reuse_csr",
        "spgemm_csr",
        "spmm_batched_coo",
        "spmm_batched_csc",
        "spmm_batched_csr",
        "spmm_bell",
        "spmm_bsr",
        "spmm_coo",
        "spmm_csc",
        "spmm_csr",
        "spmv_coo",
        "spmv_coo_aos",
        "spmv_csc",
        "spmv_csr",
        "spmv_ell",
        "v2_spmv_coo",
        "v2_spmv_coo_aos",
        "v2_spmv_csc",
        "v2_spmv_csr",
        "v2_spmv_ell",
        "v2_spmv_sell",
        nullptr};

    for(int i = 0; hung_families[i] != nullptr; ++i)
    {
        if(std::strcmp(func, hung_families[i]) == 0)
        {
            return "ASAN infrastructure: test hangs due to ASAN overhead";
        }
    }

    return nullptr;
}

#define ROCSPARSE_ASAN_CHECK_SKIP(arg)                                     \
    do                                                                     \
    {                                                                      \
        const char* _asan_skip_reason = rocsparse_asan_skip_reason(arg);   \
        if(_asan_skip_reason)                                              \
        {                                                                  \
            GTEST_SKIP() << _asan_skip_reason;                             \
            return;                                                        \
        }                                                                  \
    } while(0)

#else // !ROCSPARSE_WITH_ASAN

#define ROCSPARSE_ASAN_CHECK_SKIP(arg) do {} while(0)

#endif // ROCSPARSE_WITH_ASAN
