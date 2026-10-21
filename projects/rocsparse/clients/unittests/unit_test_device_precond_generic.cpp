/*! \file */
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

// Generic incomplete-factorization API (rocsparse_spic0 / spildlt0): the kernel regimes
// selected by the row length (hash vs binary search), including longest rows with
// exactly as many entries as the hash table of the kernel on a wavefront of 32 (32, 64,
// 128, 256 or 512). Every factorization is compared with a host reference and with the
// residual of the factorization on the sparsity pattern.

#include "unit_test_device_precond_common.hpp"

#include <string>

using namespace precond_ut;

class PrecondGeneric : public HandleTest
{
};

// ---------------------------------------------------------------------------
// CSR
// ---------------------------------------------------------------------------

// IC(0) inner sums with several non-zero terms (rows that share columns below the
// diagonal) through the hash and the binary-search kernel, with real-valued entries.
TEST_F(PrecondGeneric, spic0_csr_nonzero_inner_sums_hash)
{
    run_options opt;
    // Longest rows of 3, 6, 21, 32 and 41 entries.
    for(const int32_t w : {2, 5, 20, 31, 40})
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            check_csr_matrix<Kind::ic, T>(
                handle, make_lower_band_matrix<T>(60, w, false), opt, "band");
        });
    }
    // A dense last row over narrow band rows: table sizes 32 .. 512 (lengths below and
    // equal to the sizes) and band 3 so that the sums have three terms.
    for(const int32_t fat : {8, 32, 40, 64, 100, 128, 256, 300, 511, 512})
    {
        check_csr_matrix<Kind::ic, double>(
            handle, make_fat_row_matrix<double>(fat, false, 3), opt, "fat " + std::to_string(fat));
        check_csr_matrix<Kind::ic, float>(
            handle, make_fat_row_matrix<float>(fat, false, 3), opt, "fat " + std::to_string(fat));
    }
    // 64-bit indices with a row that fills the table.
    check_csr_matrix<Kind::ic, double, int64_t, int64_t>(
        handle, make_fat_row_matrix<double>(64, false, 3), opt, "fat 64 i64");
    // Band and fat-row matrices (non-zero inner sums) in the other index base.
    const auto one = rocsparse_index_base_one;
    check_csr_matrix<Kind::ic, double>(
        handle, make_lower_band_matrix<double>(60, 5, false), opt, "band", one);
    check_csr_matrix<Kind::ic, double>(
        handle, make_fat_row_matrix<double>(520, false, 3), opt, "fat 520", one);
    check_csr_matrix<Kind::ilu, double>(
        handle, with_upper_triangle(make_lower_band_matrix<double>(60, 5)), opt, "band", one);
    check_csr_matrix<Kind::ilu, double>(
        handle,
        with_upper_triangle(make_fat_row_matrix<double>(520, true, 3)),
        opt,
        "fat 520",
        one);
    check_csr_matrix<Kind::ildlt, double>(
        handle, make_fat_row_matrix<double>(40, true, 3), opt, "fat 40", one);
}

// ---------------------------------------------------------------------------
// Row-length regimes of ILDLT(0): hash kernel up to 512 entries
// ---------------------------------------------------------------------------
TEST_F(PrecondGeneric, spildlt0_csr_hash_row_lengths)
{
    // Row lengths up to 512 select the hash kernel; the table size grows with the length.
    // The lengths equal to a table size (32, 64, ..., 512) fill the whole table.
    run_options opt;
    for(const int32_t fat : {8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512})
    {
        check_csr_matrix<Kind::ildlt, double>(
            handle, make_fat_row_matrix<double>(fat), opt, "fat=" + std::to_string(fat));
        check_csr_matrix<Kind::ildlt, double>(handle,
                                              make_fat_row_matrix<double>(fat, true, 3),
                                              opt,
                                              "band 3 fat=" + std::to_string(fat));
    }
    // Complex values and 64-bit indices with a row that fills the table.
    check_csr_matrix<Kind::ildlt, rocsparse_double_complex, int64_t, int64_t>(
        handle,
        make_fat_row_matrix<rocsparse_double_complex>(32, true, 3),
        opt,
        "fat 32 i64 complex");
}
