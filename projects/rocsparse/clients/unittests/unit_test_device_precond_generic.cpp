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

// Generic incomplete-factorization API (rocsparse_spilu0 / spic0 / spildlt0): CSR
// matrices through the hash and the binary-search kernels, with complex IC(0) inner
// sums that are not real. Every factorization is compared with a host reference and
// with the residual of the factorization on the sparsity pattern.

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
// diagonal) through the hash and the binary-search kernel. The complex types get
// imaginary off-diagonal parts, so the products of the inner sums are not real and
// the conjugation order of the update shows.
TEST_F(PrecondGeneric, spic0_csr_nonzero_inner_sums_hash)
{
    run_options opt;
    // Longest rows of 3, 6, 21 and 41 entries.
    for(const int32_t w : {2, 5, 20, 40})
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            check_csr_matrix<Kind::ic, T>(handle, make_lower_band_matrix<T>(60, w), opt, "band");
        });
    }
    // A dense last row over narrow band rows: table sizes 32 .. 512 (lengths below
    // the sizes) and band 3 so that the sums have three terms.
    for(const int32_t fat : {8, 40, 100, 300, 511})
    {
        check_csr_matrix<Kind::ic, double>(
            handle, make_fat_row_matrix<double>(fat, false, 3), opt, "fat " + std::to_string(fat));
        check_csr_matrix<Kind::ic, float>(
            handle, make_fat_row_matrix<float>(fat, false, 3), opt, "fat " + std::to_string(fat));
        check_csr_matrix<Kind::ic, rocsparse_float_complex>(
            handle,
            make_fat_row_matrix<rocsparse_float_complex>(fat, true, 3),
            opt,
            "complex fat " + std::to_string(fat));
        check_csr_matrix<Kind::ic, rocsparse_double_complex>(
            handle,
            make_fat_row_matrix<rocsparse_double_complex>(fat, true, 3),
            opt,
            "complex fat " + std::to_string(fat));
    }
    // Binary-search kernel (longest row above 512) with complex products.
    check_csr_matrix<Kind::ic, rocsparse_float_complex>(
        handle, make_fat_row_matrix<rocsparse_float_complex>(520, true, 3), opt, "complex fat 520");
    check_csr_matrix<Kind::ic, rocsparse_double_complex>(
        handle,
        make_fat_row_matrix<rocsparse_double_complex>(520, true, 3),
        opt,
        "complex fat 520");
    // Band and fat-row matrices (non-zero inner sums) in the other index base.
    const auto one = rocsparse_index_base_one;
    check_csr_matrix<Kind::ic, double>(
        handle, make_lower_band_matrix<double>(60, 5, false), opt, "band", one);
    check_csr_matrix<Kind::ic, double>(
        handle, make_fat_row_matrix<double>(520, false, 3), opt, "fat 520", one);
    check_csr_matrix<Kind::ic, rocsparse_double_complex>(
        handle, make_lower_band_matrix<rocsparse_double_complex>(60, 5), opt, "complex band", one);
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
