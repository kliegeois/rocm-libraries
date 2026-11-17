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

//
// csrmv inputs the quick and pre-checkin YAML suites never build.
//
// The harness fills y with init_exact, which is finite, so a beta = 0 case
// cannot catch a kernel that still multiplies y. Random matrices do not
// produce a 400-nonzero row, a 4096-nonzero row, or an empty prefix followed
// by a nonempty last row. The matrices below match the gap probe.
//
// Expected y follows host_csrmv_general: when beta is 0, y is replaced and a
// NaN already stored there is not multiplied into the reference.
//
// TARGET: rocsparse-unit-test-device. Public rocsparse_Xcsrmv and
// rocsparse_spmv only.
//
#include "unit_test_utils.hpp"

#include "rocsparse_mat_info.hpp"

#include "rocsparse.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Host CSR. Values stay in double until they are converted for upload.
    struct HostCsr
    {
        int64_t              m = 0;
        int64_t              n = 0;
        std::vector<int64_t> row_ptr;
        std::vector<int64_t> col_ind;
        std::vector<double>  val;
    };

    // m rows, each with per_row contiguous ones starting at column 0.
    HostCsr band(int m, int n, int per_row)
    {
        HostCsr a;
        a.m = m;
        a.n = n;
        a.row_ptr.assign(static_cast<size_t>(m) + 1, 0);
        const int count = std::min(per_row, n);
        for(int row = 0; row < m; ++row)
        {
            for(int k = 0; k < count; ++k)
            {
                a.col_ind.push_back(k);
                a.val.push_back(1.0);
            }
            a.row_ptr[static_cast<size_t>(row) + 1] = static_cast<int64_t>(a.col_ind.size());
        }
        return a;
    }

    // Every row empty except the last, which is dense with ones.
    HostCsr last_row_dense(int m, int n)
    {
        HostCsr a;
        a.m = m;
        a.n = n;
        a.row_ptr.assign(static_cast<size_t>(m) + 1, 0);
        for(int k = 0; k < n; ++k)
        {
            a.col_ind.push_back(k);
            a.val.push_back(1.0);
        }
        a.row_ptr[static_cast<size_t>(m)] = n;
        return a;
    }

    template <typename T>
    std::vector<T> filled(size_t n, T value)
    {
        return std::vector<T>(n, value);
    }

    // host_csrmv_general replaces y when beta is 0. 0 * NaN is NaN, so the
    // reference must not read y in that case.
    template <typename T>
    std::vector<T> host_reference(const HostCsr&             a,
                                  T                          alpha,
                                  T                          beta,
                                  const std::vector<T>&      x,
                                  const std::vector<double>& aval,
                                  std::vector<T>             y)
    {
        for(int64_t row = 0; row < a.m; ++row)
        {
            T sum = static_cast<T>(0);
            for(int64_t k = a.row_ptr[static_cast<size_t>(row)];
                k < a.row_ptr[static_cast<size_t>(row) + 1];
                ++k)
            {
                const T entry = static_cast<T>(aval[static_cast<size_t>(k)]);
                sum = static_cast<T>(sum + alpha * entry * x[a.col_ind[static_cast<size_t>(k)]]);
            }
            if(beta == static_cast<T>(0))
            {
                y[static_cast<size_t>(row)] = sum;
            }
            else
            {
                y[static_cast<size_t>(row)]
                    = static_cast<T>(sum + beta * y[static_cast<size_t>(row)]);
            }
        }
        return y;
    }

    template <typename T>
    bool finite_near(T got, T want)
    {
        if(std::isnan(got) || std::isnan(want) || std::isinf(got) || std::isinf(want))
        {
            return false;
        }
        const double g   = static_cast<double>(got);
        const double w   = static_cast<double>(want);
        const double tol = 4.0 * static_cast<double>(std::numeric_limits<T>::epsilon())
                           * std::max(1.0, std::abs(w));
        return std::abs(g - w) <= tol;
    }

    template <typename T>
    bool rows_match(const std::vector<T>& got, const std::vector<T>& want, const std::string& what)
    {
        if(got.size() != want.size())
        {
            ADD_FAILURE() << what << ": size " << got.size() << " want " << want.size();
            return false;
        }
        for(size_t i = 0; i < got.size(); ++i)
        {
            if(!finite_near(got[i], want[i]))
            {
                ADD_FAILURE() << what << ": row " << i << " got " << got[i] << " want " << want[i];
                return false;
            }
        }
        return true;
    }

    std::vector<rocsparse_int> to_index(const std::vector<int64_t>& v)
    {
        return std::vector<rocsparse_int>(v.begin(), v.end());
    }

    template <typename T>
    std::vector<T> to_values(const std::vector<double>& v)
    {
        std::vector<T> out(v.size());
        for(size_t i = 0; i < v.size(); ++i)
        {
            out[i] = static_cast<T>(v[i]);
        }
        return out;
    }

    template <typename T>
    struct OnDevice
    {
        device_vector<rocsparse_int> row_ptr;
        device_vector<rocsparse_int> col_ind;
        device_vector<T>             val;
        device_vector<T>             x;
        device_vector<T>             y;

        OnDevice(const HostCsr& a, const std::vector<T>& hx, const std::vector<T>& hy)
            : row_ptr(to_index(a.row_ptr))
            , col_ind(to_index(a.col_ind))
            , val(to_values<T>(a.val))
            , x(hx)
            , y(hy)
        {
        }

        bool ok() const
        {
            return row_ptr.ptr != nullptr && col_ind.ptr != nullptr && val.ptr != nullptr
                   && x.ptr != nullptr && y.ptr != nullptr;
        }
    };

    struct LegacyGuard
    {
        rocsparse_mat_descr descr = nullptr;
        rocsparse_mat_info  info  = nullptr;

        ~LegacyGuard()
        {
            if(info != nullptr)
            {
                (void)rocsparse_destroy_mat_info(info);
            }
            if(descr != nullptr)
            {
                (void)rocsparse_destroy_mat_descr(descr);
            }
        }
    };

    struct SpmvGuard
    {
        rocsparse_spmat_descr mat = nullptr;
        rocsparse_dnvec_descr x   = nullptr;
        rocsparse_dnvec_descr y   = nullptr;

        ~SpmvGuard()
        {
            if(mat != nullptr)
            {
                (void)rocsparse_destroy_spmat_descr(mat);
            }
            if(x != nullptr)
            {
                (void)rocsparse_destroy_dnvec_descr(x);
            }
            if(y != nullptr)
            {
                (void)rocsparse_destroy_dnvec_descr(y);
            }
        }
    };

    template <typename T>
    rocsparse_status call_analysis(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   const T*                  val,
                                   const rocsparse_int*      row_ptr,
                                   const rocsparse_int*      col_ind,
                                   rocsparse_mat_info        info);

    template <>
    rocsparse_status call_analysis<float>(rocsparse_handle          handle,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             nnz,
                                          const rocsparse_mat_descr descr,
                                          const float*              val,
                                          const rocsparse_int*      row_ptr,
                                          const rocsparse_int*      col_ind,
                                          rocsparse_mat_info        info)
    {
        return rocsparse_scsrmv_analysis(
            handle, rocsparse_operation_none, m, n, nnz, descr, val, row_ptr, col_ind, info);
    }

    template <>
    rocsparse_status call_analysis<double>(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const double*             val,
                                           const rocsparse_int*      row_ptr,
                                           const rocsparse_int*      col_ind,
                                           rocsparse_mat_info        info)
    {
        return rocsparse_dcsrmv_analysis(
            handle, rocsparse_operation_none, m, n, nnz, descr, val, row_ptr, col_ind, info);
    }

    template <typename T>
    rocsparse_status call_csrmv(rocsparse_handle          handle,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                rocsparse_int             nnz,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr,
                                const T*                  val,
                                const rocsparse_int*      row_ptr,
                                const rocsparse_int*      col_ind,
                                rocsparse_mat_info        info,
                                const T*                  x,
                                const T*                  beta,
                                T*                        y);

    template <>
    rocsparse_status call_csrmv<float>(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       rocsparse_int             nnz,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr,
                                       const float*              val,
                                       const rocsparse_int*      row_ptr,
                                       const rocsparse_int*      col_ind,
                                       rocsparse_mat_info        info,
                                       const float*              x,
                                       const float*              beta,
                                       float*                    y)
    {
        return rocsparse_scsrmv(handle,
                                rocsparse_operation_none,
                                m,
                                n,
                                nnz,
                                alpha,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                info,
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status call_csrmv<double>(rocsparse_handle          handle,
                                        rocsparse_int             m,
                                        rocsparse_int             n,
                                        rocsparse_int             nnz,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             val,
                                        const rocsparse_int*      row_ptr,
                                        const rocsparse_int*      col_ind,
                                        rocsparse_mat_info        info,
                                        const double*             x,
                                        const double*             beta,
                                        double*                   y)
    {
        return rocsparse_dcsrmv(handle,
                                rocsparse_operation_none,
                                m,
                                n,
                                nnz,
                                alpha,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                info,
                                x,
                                beta,
                                y);
    }

    // info == nullptr selects rowsplit. A real info, after analysis, selects
    // the adaptive kernel.
    template <typename T>
    void legacy_csrmv(rocsparse_handle      handle,
                      const HostCsr&        a,
                      T                     alpha,
                      T                     beta,
                      const std::vector<T>& x,
                      const std::vector<T>& y0,
                      bool                  adaptive,
                      bool                  check_stream_block,
                      const char*           what)
    {
        OnDevice<T> dev(a, x, y0);
        ASSERT_TRUE(dev.ok()) << what << ": device allocation failed";

        const rocsparse_int m   = static_cast<rocsparse_int>(a.m);
        const rocsparse_int n   = static_cast<rocsparse_int>(a.n);
        const rocsparse_int nnz = static_cast<rocsparse_int>(a.col_ind.size());

        LegacyGuard guard;
        ASSERT_EQ(rocsparse_create_mat_descr(&guard.descr), rocsparse_status_success) << what;
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success)
            << what;

        if(adaptive)
        {
            ASSERT_EQ(rocsparse_create_mat_info(&guard.info), rocsparse_status_success) << what;
            ASSERT_EQ(call_analysis<T>(handle,
                                       m,
                                       n,
                                       nnz,
                                       guard.descr,
                                       dev.val.ptr,
                                       dev.row_ptr.ptr,
                                       dev.col_ind.ptr,
                                       guard.info),
                      rocsparse_status_success)
                << what;

            // Two short rows must share one stream row block, with more than
            // one thread in the reduction (the tree path, not one thread per row).
            if(check_stream_block)
            {
                rocsparse_csrmv_info csrmv = guard.info->get_csrmv_info();
                ASSERT_NE(csrmv, nullptr) << what;
                ASSERT_EQ(csrmv->adaptive.size, static_cast<size_t>(2)) << what;
                std::vector<rocsparse_int> blocks(2);
                std::vector<rocsparse_int> wg(2);
                UT_CHECK_HIP(hipMemcpy(blocks.data(),
                                       csrmv->adaptive.row_blocks,
                                       blocks.size() * sizeof(rocsparse_int),
                                       hipMemcpyDeviceToHost));
                UT_CHECK_HIP(hipMemcpy(wg.data(),
                                       csrmv->adaptive.wg_ids,
                                       wg.size() * sizeof(rocsparse_int),
                                       hipMemcpyDeviceToHost));
                EXPECT_EQ(blocks[0], 0) << what;
                EXPECT_EQ(blocks[1], m) << what;
                EXPECT_GT(wg[0], 1) << what;
            }
        }

        // Plant y on the device. init_exact never does this, so a YAML case
        // cannot put a NaN here.
        UT_CHECK_HIP(hipMemcpy(dev.y.ptr, y0.data(), y0.size() * sizeof(T), hipMemcpyHostToDevice));

        ASSERT_EQ(call_csrmv<T>(handle,
                                m,
                                n,
                                nnz,
                                &alpha,
                                guard.descr,
                                dev.val.ptr,
                                dev.row_ptr.ptr,
                                dev.col_ind.ptr,
                                guard.info,
                                dev.x.ptr,
                                &beta,
                                dev.y.ptr),
                  rocsparse_status_success)
            << what;
        UT_CHECK_HIP(hipDeviceSynchronize());

        std::vector<T> got(y0.size());
        UT_CHECK_HIP(
            hipMemcpy(got.data(), dev.y.ptr, got.size() * sizeof(T), hipMemcpyDeviceToHost));
        const std::vector<T> want = host_reference<T>(a, alpha, beta, x, a.val, y0);
        (void)rows_match(got, want, what);
    }

    template <typename T>
    void generic_spmv(rocsparse_handle      handle,
                      const HostCsr&        a,
                      rocsparse_spmv_alg    alg,
                      T                     alpha,
                      T                     beta,
                      const std::vector<T>& x,
                      const std::vector<T>& y0,
                      int                   trials,
                      const char*           what)
    {
        OnDevice<T> dev(a, x, y0);
        ASSERT_TRUE(dev.ok()) << what << ": device allocation failed";

        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success)
            << what;

        SpmvGuard guard;
        ASSERT_EQ(rocsparse_create_csr_descr(&guard.mat,
                                             a.m,
                                             a.n,
                                             static_cast<int64_t>(a.col_ind.size()),
                                             dev.row_ptr.ptr,
                                             dev.col_ind.ptr,
                                             dev.val.ptr,
                                             it_of<rocsparse_int>(),
                                             it_of<rocsparse_int>(),
                                             rocsparse_index_base_zero,
                                             dt_of<T>()),
                  rocsparse_status_success)
            << what;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&guard.x, a.n, dev.x.ptr, dt_of<T>()),
                  rocsparse_status_success)
            << what;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&guard.y, a.m, dev.y.ptr, dt_of<T>()),
                  rocsparse_status_success)
            << what;

        const std::vector<T> want = host_reference<T>(a, alpha, beta, x, a.val, y0);

        for(int trial = 0; trial < trials; ++trial)
        {
            std::ostringstream label;
            label << what << " trial " << trial;

            UT_CHECK_HIP(
                hipMemcpy(dev.y.ptr, y0.data(), y0.size() * sizeof(T), hipMemcpyHostToDevice));

            size_t buffer_size = 0;
            ASSERT_EQ(rocsparse_spmv(handle,
                                     rocsparse_operation_none,
                                     &alpha,
                                     guard.mat,
                                     guard.x,
                                     &beta,
                                     guard.y,
                                     dt_of<T>(),
                                     alg,
                                     rocsparse_spmv_stage_buffer_size,
                                     &buffer_size,
                                     nullptr),
                      rocsparse_status_success)
                << label.str();

            device_vector<char> buffer(buffer_size > 0 ? buffer_size : static_cast<size_t>(1));
            ASSERT_NE(buffer.ptr, nullptr) << label.str();

            ASSERT_EQ(rocsparse_spmv(handle,
                                     rocsparse_operation_none,
                                     &alpha,
                                     guard.mat,
                                     guard.x,
                                     &beta,
                                     guard.y,
                                     dt_of<T>(),
                                     alg,
                                     rocsparse_spmv_stage_preprocess,
                                     &buffer_size,
                                     buffer.ptr),
                      rocsparse_status_success)
                << label.str();
            ASSERT_EQ(rocsparse_spmv(handle,
                                     rocsparse_operation_none,
                                     &alpha,
                                     guard.mat,
                                     guard.x,
                                     &beta,
                                     guard.y,
                                     dt_of<T>(),
                                     alg,
                                     rocsparse_spmv_stage_compute,
                                     &buffer_size,
                                     buffer.ptr),
                      rocsparse_status_success)
                << label.str();
            UT_CHECK_HIP(hipDeviceSynchronize());

            std::vector<T> got(y0.size());
            UT_CHECK_HIP(
                hipMemcpy(got.data(), dev.y.ptr, got.size() * sizeof(T), hipMemcpyDeviceToHost));
            if(!rows_match(got, want, label.str()))
            {
                return;
            }
        }
    }

    template <typename T>
    void check_rowsplit_beta0_nan(rocsparse_handle handle)
    {
        const HostCsr        a   = band(8, 16, 5);
        const T              nan = std::numeric_limits<T>::quiet_NaN();
        const std::vector<T> x   = filled<T>(static_cast<size_t>(a.n), static_cast<T>(1));
        const std::vector<T> y0  = filled<T>(static_cast<size_t>(a.m), nan);
        legacy_csrmv<T>(handle,
                        a,
                        static_cast<T>(1),
                        static_cast<T>(0),
                        x,
                        y0,
                        false,
                        false,
                        "rowsplit_beta0_nan");
    }

    template <typename T>
    void check_adaptive_beta0_nan(rocsparse_handle handle)
    {
        // 2 x 40 = 80 nonzeros, under the 1024-entry stream block, so both rows
        // land in one row block and take the tree reduction. Analysis first.
        const HostCsr        a   = band(2, 64, 40);
        const T              nan = std::numeric_limits<T>::quiet_NaN();
        const std::vector<T> x   = filled<T>(static_cast<size_t>(a.n), static_cast<T>(1));
        const std::vector<T> y0  = filled<T>(static_cast<size_t>(a.m), nan);
        legacy_csrmv<T>(handle,
                        a,
                        static_cast<T>(1),
                        static_cast<T>(0),
                        x,
                        y0,
                        true,
                        true,
                        "adaptive_beta0_nan");
    }

    template <typename T>
    void check_spmv_alpha0_beta2(rocsparse_handle handle)
    {
        // The quick return reads host scalars and fires only for alpha = 0 and
        // beta = 1. beta = 2 must scale y. A dropped check leaves y at 3.
        const HostCsr        a  = band(4, 8, 3);
        const std::vector<T> x  = filled<T>(static_cast<size_t>(a.n), static_cast<T>(1));
        const std::vector<T> y0 = filled<T>(static_cast<size_t>(a.m), static_cast<T>(3));
        generic_spmv<T>(handle,
                        a,
                        rocsparse_spmv_alg_csr_rowsplit,
                        static_cast<T>(0),
                        static_cast<T>(2),
                        x,
                        y0,
                        1,
                        "spmv_alpha0_beta2");
    }

    template <typename T>
    void check_nnzsplit_long_row(rocsparse_handle handle)
    {
        // 400 nonzeros is more than one nnzsplit block (128 threads on wave32,
        // 256 on wave64). alpha = 2 must reach every block: a dropped alpha
        // returns 400 instead of 800. The same row misses the block reduction
        // if the syncthreads before it is gone.
        const HostCsr        a  = band(1, 400, 400);
        const std::vector<T> x  = filled<T>(static_cast<size_t>(a.n), static_cast<T>(1));
        const std::vector<T> y0 = filled<T>(static_cast<size_t>(a.m), static_cast<T>(0));
        generic_spmv<T>(handle,
                        a,
                        rocsparse_spmv_alg_csr_nnzsplit,
                        static_cast<T>(2),
                        static_cast<T>(0),
                        x,
                        y0,
                        5,
                        "nnzsplit_long");
    }

    template <typename T>
    void check_nnzsplit_last_row(rocsparse_handle handle)
    {
        // Empty prefix, nonempty last row. An off-by-one in the nnzsplit block
        // starts faults on this shape.
        const HostCsr        a  = last_row_dense(4, 32);
        const std::vector<T> x  = filled<T>(static_cast<size_t>(a.n), static_cast<T>(1));
        const std::vector<T> y0 = filled<T>(static_cast<size_t>(a.m), static_cast<T>(0));
        generic_spmv<T>(handle,
                        a,
                        rocsparse_spmv_alg_csr_nnzsplit,
                        static_cast<T>(1),
                        static_cast<T>(0),
                        x,
                        y0,
                        1,
                        "nnzsplit_last_row");
    }

    template <typename T>
    void check_lrb_long_row(rocsparse_handle handle)
    {
        // 4096 is bin 12, above LR_THRESHOLD (bin 11). Several workgroups share
        // the row (BLOCK_MULTIPLIER * 256 entries each) and combine with an
        // atomic. A short row never reaches that kernel. Repeated because a
        // plain += drops contributions that race.
        const HostCsr        a  = band(1, 4096, 4096);
        const std::vector<T> x  = filled<T>(static_cast<size_t>(a.n), static_cast<T>(1));
        const std::vector<T> y0 = filled<T>(static_cast<size_t>(a.m), static_cast<T>(0));
        generic_spmv<T>(handle,
                        a,
                        rocsparse_spmv_alg_csr_lrb,
                        static_cast<T>(1),
                        static_cast<T>(0),
                        x,
                        y0,
                        8,
                        "lrb_long_row");
    }

    using CsrmvGap = HandleTest;
}

TEST_F(CsrmvGap, rowsplit_beta0_nan)
{
    check_rowsplit_beta0_nan<float>(handle);
    check_rowsplit_beta0_nan<double>(handle);
}

TEST_F(CsrmvGap, adaptive_beta0_nan)
{
    check_adaptive_beta0_nan<float>(handle);
    check_adaptive_beta0_nan<double>(handle);
}

TEST_F(CsrmvGap, spmv_alpha0_beta2)
{
    check_spmv_alpha0_beta2<float>(handle);
    check_spmv_alpha0_beta2<double>(handle);
}

TEST_F(CsrmvGap, nnzsplit_long_row)
{
    check_nnzsplit_long_row<float>(handle);
    check_nnzsplit_long_row<double>(handle);
}

TEST_F(CsrmvGap, nnzsplit_last_row)
{
    check_nnzsplit_last_row<float>(handle);
    check_nnzsplit_last_row<double>(handle);
}

TEST_F(CsrmvGap, lrb_long_row)
{
    check_lrb_long_row<float>(handle);
    check_lrb_long_row<double>(handle);
}
