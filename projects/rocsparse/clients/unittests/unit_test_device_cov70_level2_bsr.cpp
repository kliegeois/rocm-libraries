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
// Device unit tests of the block level2 products: gebsrmv (row_block_dim !=
// col_block_dim), bsrxmv and bsrmv.
//
// Every call is compared with a host reference of y = alpha * A * x + beta * y.
//
// rocsparse_bsrmv / rocsparse_bsrxmv route every block dimension to the
// general kernel when handle->wavefront_size == 32 (gfx1201). The kernels
// specialised for 2x2, 3x3, 4x4, 5x5, 8x8, 16x16 and 17..32 are selected only
// for another wavefront size, so the tests report wavefront 64 on the handle
// while the call runs. Those kernels are not built on the wave size, except:
//   * blocks_per_row >= 64 selects a 64-lane logical wavefront (2x2, 3x3, 4x4)
//     which does not exist on a wave32 GPU, so it is not exercised here;
//   * the float 5x5 and 8x8 kernels then use 64 lane warps, so those two
//     dimensions are checked with double and complex data only.
//
#include "unit_test_device_cov70_level2_common.hpp"

#include <sstream>
#include <string>

using namespace cov70l2;

namespace
{
#define L2_GEBSRMV_WRAPPER(NAME, TYPE)                                                            \
    inline rocsparse_status gebsrmv(rocsparse_handle          handle,                             \
                                    rocsparse_direction       dir,                                \
                                    rocsparse_operation       trans,                              \
                                    int32_t                   mb,                                 \
                                    int32_t                   nb,                                 \
                                    int32_t                   nnzb,                               \
                                    const TYPE*               alpha,                              \
                                    const rocsparse_mat_descr descr,                              \
                                    const TYPE*               val,                                \
                                    const int32_t*            ptr,                                \
                                    const int32_t*            ind,                                \
                                    int32_t                   rbd,                                \
                                    int32_t                   cbd,                                \
                                    const TYPE*               x,                                  \
                                    const TYPE*               beta,                               \
                                    TYPE*                     y)                                  \
    {                                                                                             \
        return NAME(                                                                              \
            handle, dir, trans, mb, nb, nnzb, alpha, descr, val, ptr, ind, rbd, cbd, x, beta, y); \
    }

    L2_GEBSRMV_WRAPPER(rocsparse_sgebsrmv, float)
    L2_GEBSRMV_WRAPPER(rocsparse_dgebsrmv, double)
    L2_GEBSRMV_WRAPPER(rocsparse_cgebsrmv, rocsparse_float_complex)
    L2_GEBSRMV_WRAPPER(rocsparse_zgebsrmv, rocsparse_double_complex)
#undef L2_GEBSRMV_WRAPPER

#define L2_BSRXMV_WRAPPER(NAME, TYPE)                                      \
    inline rocsparse_status bsrxmv(rocsparse_handle          handle,       \
                                   rocsparse_direction       dir,          \
                                   rocsparse_operation       trans,        \
                                   int32_t                   size_of_mask, \
                                   int32_t                   mb,           \
                                   int32_t                   nb,           \
                                   int32_t                   nnzb,         \
                                   const TYPE*               alpha,        \
                                   const rocsparse_mat_descr descr,        \
                                   const TYPE*               val,          \
                                   const int32_t*            mask,         \
                                   const int32_t*            ptr,          \
                                   const int32_t*            end,          \
                                   const int32_t*            ind,          \
                                   int32_t                   bd,           \
                                   const TYPE*               x,            \
                                   const TYPE*               beta,         \
                                   TYPE*                     y)            \
    {                                                                      \
        return NAME(handle,                                                \
                    dir,                                                   \
                    trans,                                                 \
                    size_of_mask,                                          \
                    mb,                                                    \
                    nb,                                                    \
                    nnzb,                                                  \
                    alpha,                                                 \
                    descr,                                                 \
                    val,                                                   \
                    mask,                                                  \
                    ptr,                                                   \
                    end,                                                   \
                    ind,                                                   \
                    bd,                                                    \
                    x,                                                     \
                    beta,                                                  \
                    y);                                                    \
    }

    L2_BSRXMV_WRAPPER(rocsparse_sbsrxmv, float)
    L2_BSRXMV_WRAPPER(rocsparse_dbsrxmv, double)
    L2_BSRXMV_WRAPPER(rocsparse_cbsrxmv, rocsparse_float_complex)
    L2_BSRXMV_WRAPPER(rocsparse_zbsrxmv, rocsparse_double_complex)
#undef L2_BSRXMV_WRAPPER

#define L2_BSRMV_WRAPPER(NAME, TYPE)                                                              \
    inline rocsparse_status bsrmv(rocsparse_handle          handle,                               \
                                  rocsparse_direction       dir,                                  \
                                  rocsparse_operation       trans,                                \
                                  int32_t                   mb,                                   \
                                  int32_t                   nb,                                   \
                                  int32_t                   nnzb,                                 \
                                  const TYPE*               alpha,                                \
                                  const rocsparse_mat_descr descr,                                \
                                  const TYPE*               val,                                  \
                                  const int32_t*            ptr,                                  \
                                  const int32_t*            ind,                                  \
                                  int32_t                   bd,                                   \
                                  rocsparse_mat_info        info,                                 \
                                  const TYPE*               x,                                    \
                                  const TYPE*               beta,                                 \
                                  TYPE*                     y)                                    \
    {                                                                                             \
        return NAME(                                                                              \
            handle, dir, trans, mb, nb, nnzb, alpha, descr, val, ptr, ind, bd, info, x, beta, y); \
    }

    L2_BSRMV_WRAPPER(rocsparse_sbsrmv, float)
    L2_BSRMV_WRAPPER(rocsparse_dbsrmv, double)
    L2_BSRMV_WRAPPER(rocsparse_cbsrmv, rocsparse_float_complex)
    L2_BSRMV_WRAPPER(rocsparse_zbsrmv, rocsparse_double_complex)
#undef L2_BSRMV_WRAPPER

#define L2_BSRMV_ANALYSIS_WRAPPER(NAME, TYPE)                                          \
    inline rocsparse_status bsrmv_analysis(rocsparse_handle          handle,           \
                                           rocsparse_direction       dir,              \
                                           rocsparse_operation       trans,            \
                                           int32_t                   mb,               \
                                           int32_t                   nb,               \
                                           int32_t                   nnzb,             \
                                           const rocsparse_mat_descr descr,            \
                                           const TYPE*               val,              \
                                           const int32_t*            ptr,              \
                                           const int32_t*            ind,              \
                                           int32_t                   bd,               \
                                           rocsparse_mat_info        info)             \
    {                                                                                  \
        return NAME(handle, dir, trans, mb, nb, nnzb, descr, val, ptr, ind, bd, info); \
    }

    L2_BSRMV_ANALYSIS_WRAPPER(rocsparse_sbsrmv_analysis, float)
    L2_BSRMV_ANALYSIS_WRAPPER(rocsparse_dbsrmv_analysis, double)
    L2_BSRMV_ANALYSIS_WRAPPER(rocsparse_cbsrmv_analysis, rocsparse_float_complex)
    L2_BSRMV_ANALYSIS_WRAPPER(rocsparse_zbsrmv_analysis, rocsparse_double_complex)
#undef L2_BSRMV_ANALYSIS_WRAPPER

    inline std::string
        describe(const char* what, int32_t rbd, int32_t cbd, int32_t bpr, int dir, int base)
    {
        std::ostringstream s;
        s << what << " rbd=" << rbd << " cbd=" << cbd << " bpr=" << bpr << " dir=" << dir
          << " base=" << base;
        return s.str();
    }

    //
    // gebsrmv: y = alpha * A * x + beta * y for a (rbd x cbd)-blocked matrix.
    //
    template <typename T>
    void check_gebsrmv(rocsparse_handle    handle,
                       rocsparse_direction dir,
                       int32_t             mb,
                       int32_t             nb,
                       int32_t             bpr,
                       int32_t             rbd,
                       int32_t             cbd,
                       int                 base,
                       T                   alpha,
                       T                   beta,
                       bool                device_mode)
    {
        SCOPED_TRACE(describe("gebsrmv", rbd, cbd, bpr, static_cast<int>(dir), base));

        const host_bsr<T>    A  = make_bsr<T>(mb, nb, bpr, rbd, cbd, dir, base);
        const std::vector<T> x  = make_vector<T>(static_cast<size_t>(nb) * cbd, 1);
        const std::vector<T> y0 = make_vector<T>(static_cast<size_t>(mb) * rbd, 5);
        std::vector<int32_t> rows;
        all_rows(mb, rows);
        const std::vector<T> want = bsr_gemv(A, rows, {}, alpha, x, beta, y0);

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(base)), rocsparse_status_success);

        device_vector<int32_t> dptr(A.ptr), dind(A.ind);
        device_vector<T>       dval(A.val), dx(x), dy(y0);
        device_vector<T>       dalpha(std::vector<T>(1, alpha)), dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dx.ptr && dy.ptr && dalpha.ptr
                    && dbeta.ptr);

        scoped_pointer_mode mode(
            handle, device_mode ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
        const T* palpha = device_mode ? dalpha.ptr : &alpha;
        const T* pbeta  = device_mode ? dbeta.ptr : &beta;
        ASSERT_EQ(gebsrmv(handle,
                          dir,
                          rocsparse_operation_none,
                          mb,
                          nb,
                          A.nnzb(),
                          palpha,
                          descr,
                          dval,
                          dptr,
                          dind,
                          rbd,
                          cbd,
                          dx,
                          pbeta,
                          dy),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "gebsrmv y"));
    }

    class Cov70L2Gebsrmv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep(const std::vector<int32_t>& rbds,
                   const std::vector<int32_t>& cbds,
                   const std::vector<int32_t>& bprs,
                   bool                        both_dirs)
        {
            for(int32_t rbd : rbds)
            {
                for(int32_t cbd : cbds)
                {
                    if(rbd == cbd)
                        continue;
                    for(int32_t bpr : bprs)
                    {
                        for(int d = 0; d < (both_dirs ? 2 : 1); ++d)
                        {
                            const rocsparse_direction dir
                                = (d == 0) ? rocsparse_direction_row : rocsparse_direction_column;
                            // Alternate the index base and the scalars so that every
                            // dispatch leaf sees both of them.
                            const int  base   = ((rbd + cbd + d) & 1);
                            const bool device = (((rbd * 3 + cbd + bpr) % 5) == 0);
                            check_gebsrmv<T>(handle,
                                             dir,
                                             5,
                                             bpr + 5,
                                             bpr,
                                             rbd,
                                             cbd,
                                             base,
                                             rocsparse_ut::scalar<T>(2.0f),
                                             rocsparse_ut::scalar<T>(-0.5f),
                                             device);
                            if(::testing::Test::HasFailure())
                                return;
                        }
                    }
                }
            }
        }
    };

    TEST_F(Cov70L2Gebsrmv, row_block_dim_1_to_4)
    {
        // Blocks per row 2, 10, 20 and 40 select the 4, 8, 16 and 32 lane kernels.
        const std::vector<int32_t> cbds
            = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20};
        sweep<float>({1, 2, 3, 4}, cbds, {2, 10, 20, 40}, true);
    }

    TEST_F(Cov70L2Gebsrmv, row_block_dim_5_to_8)
    {
        const std::vector<int32_t> cbds
            = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20};
        sweep<float>({5, 6, 7, 8}, cbds, {3}, true);
    }

    TEST_F(Cov70L2Gebsrmv, row_block_dim_9_to_16)
    {
        sweep<float>({9, 10, 11, 12, 13, 14, 15, 16}, {1, 3, 8, 17}, {3}, true);
    }

    TEST_F(Cov70L2Gebsrmv, row_block_dim_17_and_up)
    {
        // The general kernel picks its logical wavefront from col_block_dim:
        // <= 2, <= 4, <= 8, <= 16 and larger.
        sweep<float>({17, 20, 33}, {1, 2, 3, 4, 7, 8, 9, 16, 18}, {3}, true);
    }

    TEST_F(Cov70L2Gebsrmv, double_and_complex)
    {
        sweep<double>({1, 2, 3}, {2, 4, 5}, {2, 12}, true);
        sweep<double>({5, 7}, {3, 9}, {3}, true);
        sweep<rocsparse_float_complex>({2, 4, 6}, {3, 7}, {5}, true);
        sweep<rocsparse_double_complex>({1, 3, 8}, {5, 2}, {5}, true);
    }

    TEST_F(Cov70L2Gebsrmv, alpha_zero_beta_one_is_identity)
    {
        // alpha == 0 and beta == 1 leaves y untouched (device scalars bypass the
        // host side early return, so the kernels' own check is exercised too).
        for(const int32_t rbd : {1, 3, 6, 18})
        {
            const int32_t cbd = (rbd == 3) ? 2 : 5;
            for(const bool device : {false, true})
            {
                check_gebsrmv<float>(
                    handle, rocsparse_direction_row, 4, 9, 3, rbd, cbd, 0, 0.0f, 1.0f, device);
                check_gebsrmv<float>(
                    handle, rocsparse_direction_column, 4, 9, 3, rbd, cbd, 1, 0.0f, 0.0f, device);
            }
        }
    }

    //
    // bsrxmv / bsrmv with a square (bd x bd) block.
    //
    struct xmv_case
    {
        int32_t             bd;
        int32_t             bpr;
        rocsparse_direction dir;
        bool                masked;
        bool                device;
        int                 base;
    };

    template <typename T>
    void check_bsrxmv(rocsparse_handle handle, const xmv_case& c, T alpha, T beta)
    {
        SCOPED_TRACE(describe("bsrxmv", c.bd, c.bd, c.bpr, static_cast<int>(c.dir), c.base)
                     + (c.masked ? " masked" : " all rows"));

        const int32_t        mb = 7;
        const int32_t        nb = c.bpr + 4;
        const host_bsr<T>    A  = make_bsr<T>(mb, nb, c.bpr, c.bd, c.bd, c.dir, c.base);
        const std::vector<T> x  = make_vector<T>(static_cast<size_t>(nb) * c.bd, 2);
        const std::vector<T> y0 = make_vector<T>(static_cast<size_t>(mb) * c.bd, 9);

        // Row ends. The masked variant shortens the first block of the even rows
        // away, so the end pointer array (and not ptr[i + 1]) decides the range.
        std::vector<int32_t> end(mb);
        for(int32_t i = 0; i < mb; ++i)
        {
            end[i] = A.ptr[i + 1];
            if(c.masked && (i % 2 == 0) && A.ptr[i + 1] - A.ptr[i] > 1)
                end[i] -= 1;
        }

        // Rows processed: every row, or every other row in reverse order.
        std::vector<int32_t> mask;
        std::vector<int32_t> rows;
        if(c.masked)
        {
            for(int32_t i = mb - 1; i >= 0; i -= 2)
            {
                rows.push_back(i);
                mask.push_back(i + c.base);
            }
        }
        else
        {
            all_rows(mb, rows);
        }
        const std::vector<T> want = bsr_gemv(A, rows, end, alpha, x, beta, y0);

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);

        device_vector<int32_t> dptr(A.ptr), dind(A.ind), dend(end);
        device_vector<int32_t> dmask(mask.empty() ? std::vector<int32_t>(1, 0) : mask);
        device_vector<T>       dval(A.val), dx(x), dy(y0);
        device_vector<T>       dalpha(std::vector<T>(1, alpha)), dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dptr.ptr && dind.ptr && dend.ptr && dmask.ptr && dval.ptr && dx.ptr && dy.ptr
                    && dalpha.ptr && dbeta.ptr);

        scoped_pointer_mode mode(
            handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
        const T* palpha = c.device ? dalpha.ptr : &alpha;
        const T* pbeta  = c.device ? dbeta.ptr : &beta;
        ASSERT_EQ(bsrxmv(handle,
                         c.dir,
                         rocsparse_operation_none,
                         c.masked ? static_cast<int32_t>(mask.size()) : 0,
                         mb,
                         nb,
                         A.nnzb(),
                         palpha,
                         descr,
                         dval,
                         c.masked ? dmask.ptr : nullptr,
                         dptr,
                         dend,
                         dind,
                         c.bd,
                         dx,
                         pbeta,
                         dy),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "bsrxmv y"));
    }

    template <typename T>
    void
        check_bsrmv(rocsparse_handle handle, const xmv_case& c, T alpha, T beta, bool with_analysis)
    {
        SCOPED_TRACE(describe("bsrmv", c.bd, c.bd, c.bpr, static_cast<int>(c.dir), c.base));

        const int32_t        mb = 6;
        const int32_t        nb = c.bpr + 3;
        const host_bsr<T>    A  = make_bsr<T>(mb, nb, c.bpr, c.bd, c.bd, c.dir, c.base);
        const std::vector<T> x  = make_vector<T>(static_cast<size_t>(nb) * c.bd, 4);
        const std::vector<T> y0 = make_vector<T>(static_cast<size_t>(mb) * c.bd, 6);
        std::vector<int32_t> rows;
        all_rows(mb, rows);
        const std::vector<T> want = bsr_gemv(A, rows, {}, alpha, x, beta, y0);

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);
        mat_info info;
        ASSERT_NE(info.d, nullptr);

        device_vector<int32_t> dptr(A.ptr), dind(A.ind);
        device_vector<T>       dval(A.val), dx(x), dy(y0);
        device_vector<T>       dalpha(std::vector<T>(1, alpha)), dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dx.ptr && dy.ptr && dalpha.ptr
                    && dbeta.ptr);

        scoped_pointer_mode mode(
            handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
        const T* palpha = c.device ? dalpha.ptr : &alpha;
        const T* pbeta  = c.device ? dbeta.ptr : &beta;

        if(with_analysis)
        {
            ASSERT_EQ(bsrmv_analysis(handle,
                                     c.dir,
                                     rocsparse_operation_none,
                                     mb,
                                     nb,
                                     A.nnzb(),
                                     descr,
                                     dval,
                                     dptr,
                                     dind,
                                     c.bd,
                                     info),
                      rocsparse_status_success);
        }
        ASSERT_EQ(bsrmv(handle,
                        c.dir,
                        rocsparse_operation_none,
                        mb,
                        nb,
                        A.nnzb(),
                        palpha,
                        descr,
                        dval,
                        dptr,
                        dind,
                        c.bd,
                        with_analysis ? info.d : nullptr,
                        dx,
                        pbeta,
                        dy),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "bsrmv y"));
        if(with_analysis)
        {
            EXPECT_EQ(rocsparse_bsrmv_clear(handle, info), rocsparse_status_success);
        }
    }

    class Cov70L2Bsrxmv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep_xmv(const std::vector<int32_t>& bds, const std::vector<int32_t>& bprs)
        {
            for(int32_t bd : bds)
            {
                for(int32_t bpr : bprs)
                {
                    for(int d = 0; d < 2; ++d)
                    {
                        for(int masked = 0; masked < 2; ++masked)
                        {
                            xmv_case c;
                            c.bd  = bd;
                            c.bpr = bpr;
                            c.dir = d == 0 ? rocsparse_direction_row : rocsparse_direction_column;
                            c.masked = masked != 0;
                            c.device = ((bd + bpr + d + masked) % 3) == 0;
                            c.base   = (bd + masked) & 1;
                            check_bsrxmv<T>(handle,
                                            c,
                                            rocsparse_ut::scalar<T>(1.5f),
                                            rocsparse_ut::scalar<T>(0.25f));
                            if(::testing::Test::HasFailure())
                                return;
                        }
                    }
                }
            }
        }

        template <typename T>
        void sweep_mv(const std::vector<int32_t>& bds,
                      const std::vector<int32_t>& bprs,
                      bool                        with_analysis)
        {
            for(int32_t bd : bds)
            {
                for(int32_t bpr : bprs)
                {
                    for(int d = 0; d < 2; ++d)
                    {
                        xmv_case c;
                        c.bd     = bd;
                        c.bpr    = bpr;
                        c.dir    = d == 0 ? rocsparse_direction_row : rocsparse_direction_column;
                        c.masked = false;
                        c.device = ((bd + bpr + d) % 3) == 0;
                        c.base   = (bd + d) & 1;
                        check_bsrmv<T>(handle,
                                       c,
                                       rocsparse_ut::scalar<T>(2.0f),
                                       rocsparse_ut::scalar<T>(-1.0f),
                                       with_analysis);
                        if(::testing::Test::HasFailure())
                            return;
                    }
                }
            }
        }
    };

    TEST_F(Cov70L2Bsrxmv, general_kernel_wave32)
    {
        // Wavefront 32 (the real device): every block dimension is handled by the
        // general kernel. 1 and 2 use the tight 2x2 launch, 3..4 the 4x4 launch,
        // 5..8 the 8x8, 9..16 the 16x16 and everything above the 32x32 launch.
        sweep_xmv<float>({2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 32, 33}, {3});
        sweep_xmv<double>({2, 4, 8, 16, 20}, {9});
        sweep_xmv<rocsparse_float_complex>({3, 5, 10}, {4});
        sweep_xmv<rocsparse_double_complex>({2, 7, 17}, {4});
    }

    TEST_F(Cov70L2Bsrxmv, specialised_kernels_wave64_dispatch)
    {
        // Dimensions 2, 3 and 4: blocks per row 2, 10, 20 and 40 select the 4, 8,
        // 16 and 32 lane kernels.
        const scoped_wavefront wave(handle, 64);
        sweep_xmv<float>({2, 3, 4}, {2, 10, 20, 40});
        sweep_xmv<double>({2, 3, 4}, {3, 17});
        sweep_xmv<rocsparse_float_complex>({2, 3, 4}, {5});
        sweep_xmv<rocsparse_double_complex>({2, 3, 4}, {12});
    }

    TEST_F(Cov70L2Bsrxmv, specialised_kernels_wave64_dispatch_large_blocks)
    {
        const scoped_wavefront wave(handle, 64);
        // 16x16 and 17..32 do not depend on the wavefront size, float included. The
        // 5x5 and 8x8 kernels of double and complex data reduce through shared memory
        // without a barrier and need a real 64 lane wavefront, so they are not run.
        sweep_xmv<float>({16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, {3});
        sweep_xmv<double>({16, 19, 32}, {7});
        sweep_xmv<rocsparse_float_complex>({16, 17, 25}, {2});
        sweep_xmv<rocsparse_double_complex>({16, 18, 30}, {2});
    }

    TEST_F(Cov70L2Bsrxmv, float_5x5_wave64_dispatch)
    {
        // The single precision 8x8 kernel needs a real 64 lane wavefront.
        const scoped_wavefront wave(handle, 64);
        sweep_xmv<float>({5}, {3, 9});
    }

    TEST_F(Cov70L2Bsrxmv, argument_paths)
    {
        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        const float one = 1.0f;

        device_vector<int32_t> ptr(std::vector<int32_t>{0, 1}), ind(std::vector<int32_t>{0});
        device_vector<float>   val(std::vector<float>(4, 1.0f));
        device_vector<float>   x(std::vector<float>(2, 1.0f)), y(std::vector<float>(2, 1.0f));
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && x.ptr && y.ptr);

        // block_dim == 1 has no bsrxmv kernel.
        EXPECT_EQ(bsrxmv(handle,
                         rocsparse_direction_row,
                         rocsparse_operation_none,
                         0,
                         1,
                         1,
                         1,
                         &one,
                         descr,
                         val.ptr,
                         static_cast<const int32_t*>(nullptr),
                         ptr.ptr,
                         ptr.ptr + 1,
                         ind.ptr,
                         1,
                         x.ptr,
                         &one,
                         y.ptr),
                  rocsparse_status_not_implemented);
        // Transposition is not supported.
        EXPECT_EQ(bsrxmv(handle,
                         rocsparse_direction_row,
                         rocsparse_operation_transpose,
                         0,
                         1,
                         1,
                         1,
                         &one,
                         descr,
                         val.ptr,
                         static_cast<const int32_t*>(nullptr),
                         ptr.ptr,
                         ptr.ptr + 1,
                         ind.ptr,
                         2,
                         x.ptr,
                         &one,
                         y.ptr),
                  rocsparse_status_not_implemented);
        // alpha == 0 with beta == 1 returns before touching y (host pointer mode).
        const float zero = 0.0f;
        EXPECT_EQ(bsrxmv(handle,
                         rocsparse_direction_row,
                         rocsparse_operation_none,
                         0,
                         1,
                         1,
                         1,
                         &zero,
                         descr,
                         val.ptr,
                         static_cast<const int32_t*>(nullptr),
                         ptr.ptr,
                         ptr.ptr + 1,
                         ind.ptr,
                         2,
                         x.ptr,
                         &one,
                         y.ptr),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(std::vector<float>(2, 1.0f), to_host(y), "bsrxmv untouched y"));
    }

    TEST_F(Cov70L2Bsrxmv, empty_matrix_scales_y_by_beta)
    {
        // nb == 0: the matrix is not read, y (selected rows) is scaled by beta. The
        // scaling kernel reads beta from device memory.
        const int32_t            mb   = 5;
        const int32_t            bd   = 3;
        const std::vector<float> y0   = make_vector<float>(static_cast<size_t>(mb) * bd, 3);
        const float              beta = -2.0f;

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        device_vector<int32_t> ptr(std::vector<int32_t>(mb + 1, 0));
        device_vector<int32_t> mask(std::vector<int32_t>{4, 1});
        device_vector<float>   x(std::vector<float>(1, 0.0f));
        device_vector<float>   dbeta(std::vector<float>(1, beta));
        ASSERT_TRUE(ptr.ptr && mask.ptr && x.ptr && dbeta.ptr);

        scoped_pointer_mode mode(handle, rocsparse_pointer_mode_device);

        // All rows.
        {
            device_vector<float> y(y0);
            ASSERT_TRUE(y.ptr);
            ASSERT_EQ(bsrxmv(handle,
                             rocsparse_direction_row,
                             rocsparse_operation_none,
                             0,
                             mb,
                             0,
                             0,
                             static_cast<const float*>(nullptr),
                             descr,
                             static_cast<const float*>(nullptr),
                             static_cast<const int32_t*>(nullptr),
                             ptr.ptr,
                             ptr.ptr + 1,
                             static_cast<const int32_t*>(nullptr),
                             bd,
                             x.ptr,
                             dbeta.ptr,
                             y.ptr),
                      rocsparse_status_success);
            std::vector<float> want(y0);
            for(auto& v : want)
                v *= beta;
            EXPECT_TRUE(expect_near(want, to_host(y), "bsrxmv beta scaling, all rows"));
        }

        // Rows 4 and 1 only.
        {
            device_vector<float> y(y0);
            ASSERT_TRUE(y.ptr);
            ASSERT_EQ(bsrxmv(handle,
                             rocsparse_direction_row,
                             rocsparse_operation_none,
                             2,
                             mb,
                             0,
                             0,
                             static_cast<const float*>(nullptr),
                             descr,
                             static_cast<const float*>(nullptr),
                             mask.ptr,
                             ptr.ptr,
                             ptr.ptr + 1,
                             static_cast<const int32_t*>(nullptr),
                             bd,
                             x.ptr,
                             dbeta.ptr,
                             y.ptr),
                      rocsparse_status_success);
            std::vector<float> want(y0);
            for(int32_t r : {4, 1})
                for(int32_t k = 0; k < bd; ++k)
                    want[static_cast<size_t>(r) * bd + k] *= beta;
            EXPECT_TRUE(expect_near(want, to_host(y), "bsrxmv beta scaling, masked rows"));
        }
    }

    class Cov70L2Bsrmv : public Cov70L2Bsrxmv
    {
    };

    TEST_F(Cov70L2Bsrmv, general_kernel_wave32)
    {
        sweep_mv<float>({2, 3, 4, 5, 6, 8, 11, 16, 17, 32, 34}, {4}, false);
        sweep_mv<double>({3, 9}, {5}, false);
    }

    TEST_F(Cov70L2Bsrmv, specialised_kernels_wave64_dispatch)
    {
        const scoped_wavefront wave(handle, 64);
        sweep_mv<float>({2, 3, 4}, {2, 10, 20}, false);
        sweep_mv<float>({16, 17, 20, 25, 32}, {3}, false);
        // The 5x5 and 8x8 kernels of the other precisions reduce through shared memory
        // without a barrier; they need a real 64 lane wavefront and are not run here.
        sweep_mv<double>({2, 3, 4, 16, 18}, {6}, false);
        sweep_mv<rocsparse_float_complex>({3, 4}, {3}, false);
        sweep_mv<rocsparse_double_complex>({2, 17}, {3}, false);
    }

    TEST_F(Cov70L2Bsrmv, with_analysis)
    {
        // Analysis exists for block_dim == 1 (the CSR adaptive algorithm); for
        // larger blocks it records nothing and the general dispatch runs.
        sweep_mv<float>({1, 2, 4, 7}, {3, 40}, true);
        sweep_mv<double>({1, 3}, {5}, true);
        sweep_mv<rocsparse_float_complex>({1, 2}, {4}, true);
        sweep_mv<rocsparse_double_complex>({1, 3}, {4}, true);
    }

    TEST_F(Cov70L2Bsrmv, empty_matrix_scales_y_by_beta)
    {
        const std::vector<float> y0   = make_vector<float>(12, 1);
        const float              beta = 0.5f;
        const float              one  = 1.0f;
        mat_descr                descr;
        ASSERT_NE(descr.d, nullptr);
        device_vector<int32_t> ptr(std::vector<int32_t>(5, 0));
        device_vector<float>   x(std::vector<float>(1, 0.0f));
        device_vector<float>   y(y0);
        ASSERT_TRUE(ptr.ptr && x.ptr && y.ptr);

        // nnzb == 0: y = beta * y.
        ASSERT_EQ(bsrmv(handle,
                        rocsparse_direction_row,
                        rocsparse_operation_none,
                        4,
                        2,
                        0,
                        &one,
                        descr,
                        static_cast<const float*>(nullptr),
                        ptr.ptr,
                        static_cast<const int32_t*>(nullptr),
                        3,
                        nullptr,
                        x.ptr,
                        &beta,
                        y.ptr),
                  rocsparse_status_success);
        std::vector<float> want(y0);
        for(auto& v : want)
            v *= beta;
        EXPECT_TRUE(expect_near(want, to_host(y), "bsrmv beta scaling"));
    }
}
