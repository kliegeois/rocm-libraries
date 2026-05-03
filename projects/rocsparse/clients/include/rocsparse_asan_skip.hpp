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
// This uses NARROW skips: specific (function, M, N) tuples rather than
// entire families, to maximize test coverage under ASAN.
inline const char* rocsparse_asan_skip_reason(const Arguments& arg)
{
    const char* func = arg.function;

    // ── Category 1: f64_c SpGEMM numerical divergence (ASAN infrastructure artifact) ──
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

    // ── Category 2: False-positive crashes (HIP memory pool page recycling) ──
    // HIP's GPU memory allocator recycles freed pages; ASAN reports
    // heap-use-after-free on recycled pages. Affects complex types.
    if(arg.a_type == rocsparse_datatype_f32_c || arg.a_type == rocsparse_datatype_f64_c
       || arg.compute_type == rocsparse_datatype_f32_c
       || arg.compute_type == rocsparse_datatype_f64_c)
    {
        if(std::strcmp(func, "coomv") == 0 || std::strcmp(func, "hybmv") == 0)
        {
            return "ASAN false positive: HIP memory pool page recycling (complex)";
        }
    }
    if(std::strcmp(func, "bsrpad_value") == 0)
    {
        return "ASAN false positive: device ASAN crash";
    }
    // csric0: all 5 quick tests fail or crash under ASAN (0 pass)
    if(std::strcmp(func, "csric0") == 0)
    {
        return "ASAN: all csric0 tests fail under device ASAN";
    }

    // ── Category 3: Large external matrix tests ──
    // amazon0312 is too large for ASAN-instrumented code to process within timeout.
    if(arg.filename[0] != '\0' && std::strstr(arg.filename, "amazon0312") != nullptr)
    {
        return "ASAN overhead: large external matrix (amazon0312)";
    }

    // ── Category 4: Specific test parameterizations that hang or crash ──
    // Each entry matches a specific (function, M, N) from hung_tests.txt or
    // crash analysis. N == -1 means match any N.
    struct skip_entry
    {
        const char* f;
        int         m;
        int         n; // -1 = wildcard
        const char* reason;
    };

    // clang-format off
    static const skip_entry specific_tests[] = {
        // -- Non-spin-loop hung tests (ASAN slowdown / atomic contention) --
        {"bsrgeam",                    300, 243, "ASAN overhead: hung test"},
        {"bsrmm",                      275, 128, "ASAN overhead: hung test"},
        {"bsrxmv",                      10,  33, "ASAN overhead: hung test"},
        {"copy_info",                   37,  37, "ASAN overhead: hung test"},
        {"csr2gebsr",                 1107,  -1, "ASAN overhead: hung test"},
        {"csrgeam",                    300, 243, "ASAN overhead: hung test"},
        {"csrgemm",                     50,  13, "ASAN overhead: hung test"},
        {"csrgemm_reuse",               50,  13, "ASAN overhead: hung test"},
        {"csrmm",                      275, 143, "ASAN overhead: hung test"},
        {"gebsrmm",                    275,   8, "ASAN overhead: hung test"},
        {"gemvi",                       10,  33, "ASAN overhead: hung test"},
        {"gtsv",                         8, 300, "ASAN overhead: hung test"},
        {"gtsv_no_pivot",              300,   2, "ASAN overhead: hung test"},
        {"gtsv_no_pivot_strided_batch", 300,  33, "ASAN overhead: hung test"},
        {"sparse_to_sparse",            17,   7, "ASAN overhead: hung test"},
        {"spgeam_csr",                 300, 300, "ASAN overhead: hung test"},
        {"spgeam_reuse_csr",           300,  13, "ASAN overhead: hung test"},
        {"spgemm_csr",                 300, 300, "ASAN overhead: hung test"},
        {"spmm_batched_coo",            15,   2, "ASAN overhead: hung test"},
        {"spmm_batched_csc",           300,  39, "ASAN overhead: hung test"},
        {"spmm_batched_csr",           155,  39, "ASAN overhead: hung test"},
        {"spmm_bell",                    2,   2, "ASAN overhead: hung test"},
        {"spmm_bsr",                   145, 117, "ASAN overhead: hung test"},
        {"spmm_coo",                   300,   8, "ASAN overhead: hung test"},
        {"spmm_csc",                   300, 223, "ASAN overhead: hung test"},
        {"spmm_csr",                   300, 300, "ASAN overhead: hung test"},
        {"v2_spmv_sell",               243, 143, "ASAN overhead: hung test"},

        // -- Spin-loop family crashes (specific tests crash after KERNEL_NO_ASAN) --
        {"bsric0",       72,  -1, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"bsrilu0",      50,  -1, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"bsrsm",        50,  65, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"csrilu0",      50,  -1, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"csritilu0",    50,  -1, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"csritilu0_ex", 50,  -1, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"spic0",        50,  -1, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"spilu0",       50,  -1, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"spsm_csr",     50,  50, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"spsm_coo",     50,  50, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"sptrsm_coo",   50,  50, "ASAN: Memory Fault after KERNEL_NO_ASAN"},
        {"sptrsm_csr",   50,  50, "ASAN: Memory Fault after KERNEL_NO_ASAN"},

        // -- Bug 1-4 crash tests (until PR #7010 is merged into build) --
        {"csrmv",             300,  -1, "ASAN: Bug 1 OOB (csrmv adaptive)"},
        {"csrmv_managed",     300,  -1, "ASAN: Bug 1 OOB (csrmv adaptive)"},
        {"csr2bsr",          1107,  -1, "ASAN: Bug 2 OOB (csr2bsr)"},
        {"bsr2csr",           872,  -1, "ASAN: Bug 2 OOB (bsr2csr)"},
        {"bsrgemm",           300, 300, "ASAN: Bug 4 OOB (bsrgemm)"},
        {"spgemm_bsr",        300, 300, "ASAN: Bug 4 OOB (spgemm_bsr)"},
        {"spgemm_reuse_csr",  265, 312, "ASAN: Bug 3 OOB (csrgemm symbolic)"},
        {"v2_spmv_csr_res",    50,  -1, "ASAN: Bug 1 OOB (csrmv adaptive)"},

        // -- Other specific crashes --
        {"csrsm",  124,  65, "ASAN: GPU coredump on specific params"},
    };
    // clang-format on

    for(size_t i = 0; i < sizeof(specific_tests) / sizeof(specific_tests[0]); ++i)
    {
        const skip_entry& e = specific_tests[i];
        if(std::strcmp(func, e.f) == 0 && arg.M == e.m && (e.n == -1 || arg.N == e.n))
        {
            return e.reason;
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
