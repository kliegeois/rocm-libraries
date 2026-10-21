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

// Pruning: quick returns and argument checks of prune_csr2csr_by_percentage and
// prune_dense2csr_by_percentage. The checks pin the statuses documented in the public
// headers.

#include "unit_test_device_conv_common.hpp"

#include <vector>

using namespace convut;

namespace
{
    using ConvPrune = ConvTest;

    // Column major dense matrix with distinct magnitudes 1 .. 97 (and a zero where
    // (3 i + j) % 19 == 0, so that low percentages prune nonzero entries too), optionally
    // with alternating signs. The padding between the columns is -777.
    template <typename T>
    std::vector<T> make_dense(int m, int n, int ld, bool signs)
    {
        std::vector<T> a(size_t(ld) * n, T(-777));
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
            {
                const int k   = i * n + j;
                const T   mag = static_cast<T>(1 + (k * 17) % 97);
                a[size_t(j) * ld + i]
                    = ((3 * i + j) % 19 == 0) ? T(0) : (signs && k % 2 ? -mag : mag);
            }
        return a;
    }

    TEST_F(ConvPrune, PruneCsr2CsrByPercentageQuickReturnsAndArguments)
    {
        MatDescr     descr_a, descr_c;
        MatInfo      info;
        DevBuf<I>    ptr(std::vector<I>{0, 0, 0}), ptr_c(3);
        DevBuf<char> tmp(64);

        // No rows, or no columns and no entries.
        for(bool device_mode : {false, true})
            for(I m : {0, 2})
            {
                const I n = m ? 0 : 4;
                EXPECT_EQ(nnz_via(handle,
                                  device_mode,
                                  [&](I* p) {
                                      return rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                                                        m,
                                                                                        n,
                                                                                        0,
                                                                                        descr_a,
                                                                                        nullptr,
                                                                                        ptr,
                                                                                        nullptr,
                                                                                        10.f,
                                                                                        descr_c,
                                                                                        ptr_c,
                                                                                        p,
                                                                                        info.get(),
                                                                                        tmp);
                                  }),
                          0);
                EXPECT_EQ(rocsparse_sprune_csr2csr_by_percentage(handle,
                                                                 m,
                                                                 n,
                                                                 0,
                                                                 descr_a,
                                                                 nullptr,
                                                                 ptr,
                                                                 nullptr,
                                                                 0.f,
                                                                 descr_c,
                                                                 nullptr,
                                                                 ptr_c,
                                                                 nullptr,
                                                                 info.get(),
                                                                 tmp),
                          st_ok);
            }

        // A percentage outside of [0, 100] is an invalid value.
        DevBuf<double> val(std::vector<double>{1.0, 2.0});
        DevBuf<I>      rp(std::vector<I>{0, 1, 2}), ci(std::vector<I>{0, 1});
        I              nnz_c = 0;
        size_t         size  = 0;
        EXPECT_EQ(rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                             2,
                                                             2,
                                                             0,
                                                             descr_a,
                                                             nullptr,
                                                             ptr,
                                                             nullptr,
                                                             -1.f,
                                                             descr_c,
                                                             ptr_c,
                                                             &nnz_c,
                                                             info.get(),
                                                             tmp),
                  st_val);
        EXPECT_EQ(rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle,
                                                                     2,
                                                                     2,
                                                                     2,
                                                                     descr_a,
                                                                     val,
                                                                     rp,
                                                                     ci,
                                                                     150.0,
                                                                     descr_c,
                                                                     nullptr,
                                                                     rp,
                                                                     nullptr,
                                                                     info.get(),
                                                                     &size),
                  st_val);
        EXPECT_EQ(rocsparse_dprune_csr2csr_by_percentage(handle,
                                                         2,
                                                         2,
                                                         2,
                                                         descr_a,
                                                         val,
                                                         rp,
                                                         ci,
                                                         101.0,
                                                         descr_c,
                                                         nullptr,
                                                         ptr_c,
                                                         nullptr,
                                                         info.get(),
                                                         tmp),
                  st_val);
    }

    // Argument checks of the double precision percentage routines.
    TEST_F(ConvPrune, PruneDense2CsrByPercentageArguments)
    {
        const int      m = 9, n = 7;
        const auto     A = make_dense<double>(m, n, m, true);
        MatDescr       descr;
        MatInfo        info;
        DevBuf<double> d_A(A);
        DevBuf<I>      d_ptr(m + 1);

        size_t size = 0;
        ASSERT_EQ(
            rocsparse_dprune_dense2csr_by_percentage_buffer_size(
                handle, m, n, d_A, m, 40.0, descr, nullptr, d_ptr, nullptr, info.get(), &size),
            st_ok);
        DevBuf<char> temp(size);
        const I      nnz = nnz_via(handle, false, [&](I* p) {
            return rocsparse_dprune_dense2csr_nnz_by_percentage(
                handle, m, n, d_A, m, 40.0, descr, d_ptr, p, info.get(), temp);
        });
        ASSERT_GT(nnz, 0);
        DevBuf<I>      d_ind(nnz);
        DevBuf<double> d_val(nnz);

        const auto compute
            = [&](rocsparse_handle h, double pct, I ld, double* val, I* ind, void* buffer) {
                  return rocsparse_dprune_dense2csr_by_percentage(
                      h, m, n, d_A, ld, pct, descr, val, d_ptr, ind, info.get(), buffer);
              };
        // The output arrays are required when there are entries.
        EXPECT_EQ(compute(handle, 40.0, m, nullptr, d_ind, temp), st_ptr);
        EXPECT_EQ(compute(handle, 40.0, m, d_val, nullptr, temp), st_ptr);
        EXPECT_EQ(compute(handle, 40.0, m, nullptr, nullptr, temp), st_ptr);
        EXPECT_EQ(compute(nullptr, 40.0, m, d_val, d_ind, temp), rocsparse_status_invalid_handle);
        EXPECT_EQ(compute(handle, 40.0, m - 1, d_val, d_ind, temp), st_size);
        EXPECT_EQ(compute(handle, 40.0, m, d_val, d_ind, nullptr), st_ptr);
        // A percentage outside of [0, 100] is an invalid value.
        EXPECT_EQ(compute(handle, -1.0, m, d_val, d_ind, temp), st_val);
        I count = 0;
        EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                      handle, m, n, d_A, m, 101.0, descr, d_ptr, &count, info.get(), temp),
                  st_val);
        EXPECT_EQ(
            rocsparse_dprune_dense2csr_by_percentage_buffer_size(
                handle, m, n, d_A, m, 101.0, descr, nullptr, d_ptr, nullptr, info.get(), &size),
            st_val);
        EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                      handle, m, n, d_A, m, 40.0, descr, d_ptr, nullptr, info.get(), temp),
                  st_ptr);
        EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                      handle, m, n, d_A, m, 40.0, descr, d_ptr, &count, info.get(), nullptr),
                  st_ptr);
        EXPECT_EQ(
            rocsparse_dprune_dense2csr_by_percentage_buffer_size(
                handle, m, n, d_A, m, 40.0, descr, nullptr, d_ptr, nullptr, info.get(), nullptr),
            st_ptr);
    }
} // namespace
