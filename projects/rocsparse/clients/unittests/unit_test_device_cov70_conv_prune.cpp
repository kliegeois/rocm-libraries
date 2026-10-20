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

// rocsparse_dprune_dense2csr_by_percentage (buffer size, nnz and compute) and the
// pointer checks of rocsparse_dprune_csr2csr_by_percentage. The pruning is checked
// against a host reference: with n the number of candidate entries (all m * n
// entries for the dense routine, nnz for the CSR one), the threshold is the
// element at position ceil(n * percentage / 100) - 1 of the sorted magnitudes, and
// the entries whose magnitude is strictly larger than the threshold are kept.

#include "unit_test_device_cov70_conv_common.hpp"

#include "rocsparse_handle.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

using namespace rocsparse_ut;
using namespace cov70conv;

namespace
{
    // Shrinks the handle scratch buffer so the routine has to allocate its own
    // temporary storage, restores it on destruction.
    struct ShrinkHandleBuffer
    {
        rocsparse_handle handle;
        size_t           saved;
        explicit ShrinkHandleBuffer(rocsparse_handle h)
            : handle(h)
            , saved(h->buffer_size)
        {
            h->buffer_size = 0;
        }
        ~ShrinkHandleBuffer()
        {
            handle->buffer_size = saved;
        }
    };

    struct PointerModeScope
    {
        rocsparse_handle       handle;
        rocsparse_pointer_mode saved;
        PointerModeScope(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
            , saved(h->pointer_mode)
        {
            EXPECT_EQ(rocsparse_set_pointer_mode(h, mode), rocsparse_status_success);
        }
        ~PointerModeScope()
        {
            (void)rocsparse_set_pointer_mode(handle, saved);
        }
    };

    struct HostCsr
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<double>        val;
    };

    // Entries of |values| strictly above the percentile threshold.
    double percentile_threshold(std::vector<double> magnitudes, double percentage)
    {
        const int64_t count = static_cast<int64_t>(magnitudes.size());
        std::sort(magnitudes.begin(), magnitudes.end());
        int64_t pos
            = static_cast<int64_t>(std::ceil(static_cast<double>(count) * (percentage / 100.0)))
              - 1;
        pos = std::min(pos, count - 1);
        pos = std::max(pos, static_cast<int64_t>(0));
        return magnitudes[pos];
    }

    // Column-major dense matrix with distinct magnitudes (a permutation of 1..97 stepped
    // by 17), explicit zeros on a diagonal pattern and alternating signs.
    std::vector<double> make_dense(int m, int n, int lda)
    {
        std::vector<double> a(static_cast<size_t>(lda) * n, -777.0);
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < m; ++i)
            {
                const int    k   = i * n + j;
                const double mag = 1.0 + static_cast<double>((k * 17) % 97);
                double       v   = ((i + j) % 4 == 0) ? 0.0 : ((k % 2 == 0) ? mag : -mag);
                a[static_cast<size_t>(j) * lda + i] = v;
            }
        }
        return a;
    }

    HostCsr
        reference_dense(int m, int n, int lda, const std::vector<double>& a, double pct, int base)
    {
        std::vector<double> mags;
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < m; ++i)
            {
                mags.push_back(std::fabs(a[static_cast<size_t>(j) * lda + i]));
            }
        }
        const double thr = percentile_threshold(mags, pct);

        HostCsr r;
        r.row_ptr.push_back(base);
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                const double v = a[static_cast<size_t>(j) * lda + i];
                if(std::fabs(v) > thr)
                {
                    r.col_ind.push_back(j + base);
                    r.val.push_back(v);
                }
            }
            r.row_ptr.push_back(static_cast<rocsparse_int>(r.val.size()) + base);
        }
        return r;
    }

    class Cov70ConvPrune : public HandleTest
    {
    };
}

// The dense routine over a grid of percentages, index bases, pointer modes and
// scratch buffer situations (handle buffer large enough, or too small so that the
// routine allocates its own storage).
TEST_F(Cov70ConvPrune, Dense2CsrByPercentageDouble)
{
    const int                 m = 9, n = 7, lda = 11;
    const std::vector<double> hA = make_dense(m, n, lda);
    device_vector<double>     dA(hA);
    ASSERT_TRUE(dA.ptr);

    for(double pct : {0.0, 10.0, 40.0, 77.0, 100.0})
    {
        for(int base : {0, 1})
        {
            for(rocsparse_pointer_mode mode :
                {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device})
            {
                for(bool shrink : {false, true})
                {
                    SCOPED_TRACE("pct=" + std::to_string(pct) + " base=" + std::to_string(base)
                                 + " device_mode="
                                 + std::to_string(mode == rocsparse_pointer_mode_device)
                                 + " shrink=" + std::to_string(shrink));

                    const HostCsr ref = reference_dense(m, n, lda, hA, pct, base);

                    MatDescr descr(base == 0 ? rocsparse_index_base_zero
                                             : rocsparse_index_base_one);
                    MatInfo  info;
                    ASSERT_TRUE(descr.d && info.i);

                    PointerModeScope pm(handle, mode);

                    device_vector<rocsparse_int> d_row_ptr(static_cast<size_t>(m + 1));
                    ASSERT_TRUE(d_row_ptr.ptr);

                    size_t buffer_size = 0;
                    ASSERT_EQ(rocsparse_dprune_dense2csr_by_percentage_buffer_size(handle,
                                                                                   m,
                                                                                   n,
                                                                                   dA,
                                                                                   lda,
                                                                                   pct,
                                                                                   descr.d,
                                                                                   nullptr,
                                                                                   d_row_ptr,
                                                                                   nullptr,
                                                                                   info.i,
                                                                                   &buffer_size),
                              rocsparse_status_success);
                    EXPECT_EQ(buffer_size, sizeof(double) * 2 * m * n);

                    device_vector<char> temp(std::max(buffer_size, size_t(1)));
                    ASSERT_TRUE(temp.ptr);

                    std::unique_ptr<ShrinkHandleBuffer> shrinker;
                    if(shrink)
                    {
                        shrinker.reset(new ShrinkHandleBuffer(handle));
                    }

                    rocsparse_int                nnz_host = -1;
                    device_vector<rocsparse_int> nnz_dev(static_cast<size_t>(1));
                    ASSERT_TRUE(nnz_dev.ptr);
                    rocsparse_int* nnz_ptr
                        = (mode == rocsparse_pointer_mode_device) ? nnz_dev.ptr : &nnz_host;

                    ASSERT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(handle,
                                                                           m,
                                                                           n,
                                                                           dA,
                                                                           lda,
                                                                           pct,
                                                                           descr.d,
                                                                           d_row_ptr,
                                                                           nnz_ptr,
                                                                           info.i,
                                                                           temp.ptr),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                    const rocsparse_int nnz = (mode == rocsparse_pointer_mode_device)
                                                  ? to_host(nnz_dev.ptr, 1)[0]
                                                  : nnz_host;
                    ASSERT_EQ(nnz, static_cast<rocsparse_int>(ref.val.size()));
                    EXPECT_EQ(to_host(d_row_ptr.ptr, m + 1), ref.row_ptr);

                    device_vector<rocsparse_int> d_col_ind(std::max<size_t>(nnz, 1));
                    device_vector<double>        d_val(std::max<size_t>(nnz, 1));
                    ASSERT_TRUE(d_col_ind.ptr && d_val.ptr);

                    ASSERT_EQ(rocsparse_dprune_dense2csr_by_percentage(handle,
                                                                       m,
                                                                       n,
                                                                       dA,
                                                                       lda,
                                                                       pct,
                                                                       descr.d,
                                                                       d_val,
                                                                       d_row_ptr,
                                                                       d_col_ind,
                                                                       info.i,
                                                                       temp.ptr),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                    EXPECT_EQ(to_host(d_row_ptr.ptr, m + 1), ref.row_ptr);
                    EXPECT_EQ(to_host(d_col_ind.ptr, nnz), ref.col_ind);
                    EXPECT_EQ(to_host(d_val.ptr, nnz), ref.val);
                }
            }
        }
    }
}

// Empty matrices: the nnz routine fills the row pointer with the index base and
// reports zero entries, the compute routine returns immediately.
TEST_F(Cov70ConvPrune, Dense2CsrByPercentageEmpty)
{
    struct Size
    {
        int m, n;
    };

    for(Size sz : {Size{0, 5}, Size{4, 0}, Size{0, 0}})
    {
        for(int base : {0, 1})
        {
            for(rocsparse_pointer_mode mode :
                {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device})
            {
                SCOPED_TRACE("m=" + std::to_string(sz.m) + " n=" + std::to_string(sz.n)
                             + " base=" + std::to_string(base) + " device_mode="
                             + std::to_string(mode == rocsparse_pointer_mode_device));

                const int lda = std::max(sz.m, 1);
                MatDescr  descr(base == 0 ? rocsparse_index_base_zero : rocsparse_index_base_one);
                MatInfo   info;
                ASSERT_TRUE(descr.d && info.i);
                PointerModeScope pm(handle, mode);

                device_vector<double>        dA(static_cast<size_t>(1));
                device_vector<rocsparse_int> d_row_ptr(
                    std::vector<rocsparse_int>(static_cast<size_t>(sz.m + 1), -5));
                device_vector<rocsparse_int> d_nnz(std::vector<rocsparse_int>{-7});
                ASSERT_TRUE(dA.ptr && d_row_ptr.ptr && d_nnz.ptr);
                rocsparse_int  nnz_host = -9;
                rocsparse_int* nnz_ptr
                    = (mode == rocsparse_pointer_mode_device) ? d_nnz.ptr : &nnz_host;

                size_t buffer_size = 1;
                ASSERT_EQ(rocsparse_dprune_dense2csr_by_percentage_buffer_size(handle,
                                                                               sz.m,
                                                                               sz.n,
                                                                               dA,
                                                                               lda,
                                                                               50.0,
                                                                               descr.d,
                                                                               nullptr,
                                                                               d_row_ptr,
                                                                               nullptr,
                                                                               info.i,
                                                                               &buffer_size),
                          rocsparse_status_success);
                EXPECT_EQ(buffer_size, size_t(0));

                ASSERT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(handle,
                                                                       sz.m,
                                                                       sz.n,
                                                                       dA,
                                                                       lda,
                                                                       50.0,
                                                                       descr.d,
                                                                       d_row_ptr,
                                                                       nnz_ptr,
                                                                       info.i,
                                                                       nullptr),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                const rocsparse_int nnz
                    = (mode == rocsparse_pointer_mode_device) ? to_host(d_nnz.ptr, 1)[0] : nnz_host;
                EXPECT_EQ(nnz, 0);
                EXPECT_EQ(to_host(d_row_ptr.ptr, sz.m + 1),
                          std::vector<rocsparse_int>(static_cast<size_t>(sz.m + 1), base));

                EXPECT_EQ(rocsparse_dprune_dense2csr_by_percentage(handle,
                                                                   sz.m,
                                                                   sz.n,
                                                                   dA,
                                                                   lda,
                                                                   50.0,
                                                                   descr.d,
                                                                   nullptr,
                                                                   d_row_ptr,
                                                                   nullptr,
                                                                   info.i,
                                                                   nullptr),
                          rocsparse_status_success);
            }
        }
    }
}

// Argument checks of the double precision dense routines, including the
// csr_val / csr_col_ind arrays being checked against the number of entries found
// by the nnz routine.
TEST_F(Cov70ConvPrune, Dense2CsrByPercentageArguments)
{
    const int                    m = 9, n = 7, lda = 9;
    const std::vector<double>    hA = make_dense(m, n, lda);
    device_vector<double>        dA(hA);
    MatDescr                     descr;
    MatInfo                      info;
    device_vector<rocsparse_int> d_row_ptr(static_cast<size_t>(m + 1));
    ASSERT_TRUE(dA.ptr && descr.d && info.i && d_row_ptr.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_dprune_dense2csr_by_percentage_buffer_size(handle,
                                                                   m,
                                                                   n,
                                                                   dA,
                                                                   lda,
                                                                   40.0,
                                                                   descr.d,
                                                                   nullptr,
                                                                   d_row_ptr,
                                                                   nullptr,
                                                                   info.i,
                                                                   &buffer_size),
              rocsparse_status_success);
    device_vector<char> temp(buffer_size);
    ASSERT_TRUE(temp.ptr);

    rocsparse_int nnz = 0;
    ASSERT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                  handle, m, n, dA, lda, 40.0, descr.d, d_row_ptr, &nnz, info.i, temp.ptr),
              rocsparse_status_success);
    ASSERT_GT(nnz, 0);

    device_vector<rocsparse_int> d_col_ind(static_cast<size_t>(nnz));
    device_vector<double>        d_val(static_cast<size_t>(nnz));
    ASSERT_TRUE(d_col_ind.ptr && d_val.ptr);

    // Arrays that are too small to be checked: null while nnz > 0.
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage(
            handle, m, n, dA, lda, 40.0, descr.d, nullptr, d_row_ptr, d_col_ind, info.i, temp.ptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage(
            handle, m, n, dA, lda, 40.0, descr.d, d_val, d_row_ptr, nullptr, info.i, temp.ptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage(
            handle, m, n, dA, lda, 40.0, descr.d, nullptr, d_row_ptr, nullptr, info.i, temp.ptr),
        rocsparse_status_invalid_pointer);

    // Other invalid arguments.
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage(
            nullptr, m, n, dA, lda, 40.0, descr.d, d_val, d_row_ptr, d_col_ind, info.i, temp.ptr),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage(
            handle, m, n, dA, lda, -1.0, descr.d, d_val, d_row_ptr, d_col_ind, info.i, temp.ptr),
        rocsparse_status_invalid_value);
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage(
            handle, m, n, dA, m - 1, 40.0, descr.d, d_val, d_row_ptr, d_col_ind, info.i, temp.ptr),
        rocsparse_status_invalid_size);
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage(
            handle, m, n, dA, lda, 40.0, descr.d, d_val, d_row_ptr, d_col_ind, info.i, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                  handle, m, n, dA, lda, 101.0, descr.d, d_row_ptr, &nnz, info.i, temp.ptr),
              rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                  handle, m, n, dA, lda, 40.0, descr.d, d_row_ptr, nullptr, info.i, temp.ptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                  handle, m, n, dA, lda, 40.0, descr.d, d_row_ptr, &nnz, info.i, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_dprune_dense2csr_by_percentage_buffer_size(
            handle, m, n, dA, lda, 40.0, descr.d, nullptr, d_row_ptr, nullptr, info.i, nullptr),
        rocsparse_status_invalid_pointer);
}

// The CSR routine checks csr_val_C / csr_col_ind_C against the number of kept
// entries when they are not provided. Keeping nothing (percentage 100) makes null
// arrays legal.
TEST_F(Cov70ConvPrune, Csr2CsrByPercentageOutputArrays)
{
    const int m = 8, n = 6;

    // Row i holds the columns j with (i + j) % 2 == 0, magnitudes all distinct.
    std::vector<rocsparse_int> h_row_ptr{0};
    std::vector<rocsparse_int> h_col_ind;
    std::vector<double>        h_val;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            if((i + j) % 2 == 0)
            {
                h_col_ind.push_back(j);
                h_val.push_back((h_val.size() % 2 == 0 ? 1.0 : -1.0)
                                * (1.0 + static_cast<double>((h_val.size() * 7) % 31)));
            }
        }
        h_row_ptr.push_back(static_cast<rocsparse_int>(h_val.size()));
    }
    const int nnz_A = static_cast<int>(h_val.size());

    device_vector<rocsparse_int> d_row_ptr(h_row_ptr);
    device_vector<rocsparse_int> d_col_ind(h_col_ind);
    device_vector<double>        d_val(h_val);
    ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr);

    MatDescr descr_A, descr_C;
    MatInfo  info;
    ASSERT_TRUE(descr_A.d && descr_C.d && info.i);

    for(double pct : {100.0, 50.0})
    {
        SCOPED_TRACE("pct=" + std::to_string(pct));

        device_vector<rocsparse_int> d_row_ptr_C(static_cast<size_t>(m + 1));
        ASSERT_TRUE(d_row_ptr_C.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle,
                                                                     m,
                                                                     n,
                                                                     nnz_A,
                                                                     descr_A.d,
                                                                     d_val,
                                                                     d_row_ptr,
                                                                     d_col_ind,
                                                                     pct,
                                                                     descr_C.d,
                                                                     nullptr,
                                                                     d_row_ptr_C,
                                                                     nullptr,
                                                                     info.i,
                                                                     &buffer_size),
                  rocsparse_status_success);
        device_vector<char> temp(std::max(buffer_size, size_t(1)));
        ASSERT_TRUE(temp.ptr);

        rocsparse_int nnz_C = -1;
        ASSERT_EQ(rocsparse_dprune_csr2csr_nnz_by_percentage(handle,
                                                             m,
                                                             n,
                                                             nnz_A,
                                                             descr_A.d,
                                                             d_val,
                                                             d_row_ptr,
                                                             d_col_ind,
                                                             pct,
                                                             descr_C.d,
                                                             d_row_ptr_C,
                                                             &nnz_C,
                                                             info.i,
                                                             temp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Host reference of the number of kept entries.
        std::vector<double> mags;
        for(double v : h_val)
        {
            mags.push_back(std::fabs(v));
        }
        const double  thr          = percentile_threshold(mags, pct);
        rocsparse_int expected_nnz = 0;
        for(double mag : mags)
        {
            expected_nnz += (mag > thr) ? 1 : 0;
        }
        ASSERT_EQ(nnz_C, expected_nnz);
        EXPECT_EQ(to_host(d_row_ptr_C.ptr, m + 1).back(), expected_nnz);

        if(expected_nnz == 0)
        {
            // Nothing is written: null output arrays are accepted.
            EXPECT_EQ(rocsparse_dprune_csr2csr_by_percentage(handle,
                                                             m,
                                                             n,
                                                             nnz_A,
                                                             descr_A.d,
                                                             d_val,
                                                             d_row_ptr,
                                                             d_col_ind,
                                                             pct,
                                                             descr_C.d,
                                                             nullptr,
                                                             d_row_ptr_C,
                                                             nullptr,
                                                             info.i,
                                                             temp.ptr),
                      rocsparse_status_success);
        }
        else
        {
            device_vector<rocsparse_int> d_col_ind_C(static_cast<size_t>(expected_nnz));
            device_vector<double>        d_val_C(static_cast<size_t>(expected_nnz));
            ASSERT_TRUE(d_col_ind_C.ptr && d_val_C.ptr);

            EXPECT_EQ(rocsparse_dprune_csr2csr_by_percentage(handle,
                                                             m,
                                                             n,
                                                             nnz_A,
                                                             descr_A.d,
                                                             d_val,
                                                             d_row_ptr,
                                                             d_col_ind,
                                                             pct,
                                                             descr_C.d,
                                                             nullptr,
                                                             d_row_ptr_C,
                                                             d_col_ind_C,
                                                             info.i,
                                                             temp.ptr),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(rocsparse_dprune_csr2csr_by_percentage(handle,
                                                             m,
                                                             n,
                                                             nnz_A,
                                                             descr_A.d,
                                                             d_val,
                                                             d_row_ptr,
                                                             d_col_ind,
                                                             pct,
                                                             descr_C.d,
                                                             d_val_C,
                                                             d_row_ptr_C,
                                                             nullptr,
                                                             info.i,
                                                             temp.ptr),
                      rocsparse_status_invalid_pointer);

            // With the arrays the result is the reference one.
            ASSERT_EQ(rocsparse_dprune_csr2csr_by_percentage(handle,
                                                             m,
                                                             n,
                                                             nnz_A,
                                                             descr_A.d,
                                                             d_val,
                                                             d_row_ptr,
                                                             d_col_ind,
                                                             pct,
                                                             descr_C.d,
                                                             d_val_C,
                                                             d_row_ptr_C,
                                                             d_col_ind_C,
                                                             info.i,
                                                             temp.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            std::vector<rocsparse_int> ref_col;
            std::vector<double>        ref_val;
            std::vector<rocsparse_int> ref_ptr{0};
            for(int i = 0; i < m; ++i)
            {
                for(rocsparse_int k = h_row_ptr[i]; k < h_row_ptr[i + 1]; ++k)
                {
                    if(std::fabs(h_val[k]) > thr)
                    {
                        ref_col.push_back(h_col_ind[k]);
                        ref_val.push_back(h_val[k]);
                    }
                }
                ref_ptr.push_back(static_cast<rocsparse_int>(ref_val.size()));
            }
            EXPECT_EQ(to_host(d_row_ptr_C.ptr, m + 1), ref_ptr);
            EXPECT_EQ(to_host(d_col_ind_C.ptr, expected_nnz), ref_col);
            EXPECT_EQ(to_host(d_val_C.ptr, expected_nnz), ref_val);
        }
    }
}
