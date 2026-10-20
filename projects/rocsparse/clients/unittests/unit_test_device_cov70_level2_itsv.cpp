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
// Device unit tests of the Jacobi iterative triangular solvers
// rocsparse_Xcsritsv_* and rocsparse_spitsv.
//
// The iteration solves op(A) y = alpha * x for a triangular system. For the
// diagonally dominant (or, for unit diagonals, strictly small off-diagonal)
// matrices used here it converges to the exact solution after at most m
// iterations, so every result is compared with a host substitution.
//
#include "unit_test_device_cov70_level2_common.hpp"

#include <sstream>
#include <string>

using namespace cov70l2;

namespace
{
    // ----- C API wrappers (one overload per precision) --------------------------
#define L2_ITSV_WRAPPERS(P, TYPE, REAL)                                                  \
    inline rocsparse_status itsv_buffer_size(rocsparse_handle          handle,           \
                                             rocsparse_operation       trans,            \
                                             int32_t                   m,                \
                                             int32_t                   nnz,              \
                                             const rocsparse_mat_descr descr,            \
                                             const TYPE*               val,              \
                                             const int32_t*            ptr,              \
                                             const int32_t*            ind,              \
                                             rocsparse_mat_info        info,             \
                                             size_t*                   buffer_size)      \
    {                                                                                    \
        return rocsparse_##P##csritsv_buffer_size(                                       \
            handle, trans, m, nnz, descr, val, ptr, ind, info, buffer_size);             \
    }                                                                                    \
    inline rocsparse_status itsv_analysis(rocsparse_handle          handle,              \
                                          rocsparse_operation       trans,               \
                                          int32_t                   m,                   \
                                          int32_t                   nnz,                 \
                                          const rocsparse_mat_descr descr,               \
                                          const TYPE*               val,                 \
                                          const int32_t*            ptr,                 \
                                          const int32_t*            ind,                 \
                                          rocsparse_mat_info        info,                \
                                          rocsparse_analysis_policy analysis,            \
                                          rocsparse_solve_policy    solve,               \
                                          void*                     buffer)              \
    {                                                                                    \
        return rocsparse_##P##csritsv_analysis(                                          \
            handle, trans, m, nnz, descr, val, ptr, ind, info, analysis, solve, buffer); \
    }                                                                                    \
    inline rocsparse_status itsv_solve(rocsparse_handle          handle,                 \
                                       int32_t*                  nmaxiter,               \
                                       int32_t                   nfreeiter,              \
                                       const REAL*               tol,                    \
                                       REAL*                     history,                \
                                       rocsparse_operation       trans,                  \
                                       int32_t                   m,                      \
                                       int32_t                   nnz,                    \
                                       const TYPE*               alpha,                  \
                                       const rocsparse_mat_descr descr,                  \
                                       const TYPE*               val,                    \
                                       const int32_t*            ptr,                    \
                                       const int32_t*            ind,                    \
                                       rocsparse_mat_info        info,                   \
                                       const TYPE*               x,                      \
                                       TYPE*                     y,                      \
                                       rocsparse_solve_policy    policy,                 \
                                       void*                     buffer,                 \
                                       bool                      ex)                     \
    {                                                                                    \
        if(ex)                                                                           \
            return rocsparse_##P##csritsv_solve_ex(handle,                               \
                                                   nmaxiter,                             \
                                                   nfreeiter,                            \
                                                   tol,                                  \
                                                   history,                              \
                                                   trans,                                \
                                                   m,                                    \
                                                   nnz,                                  \
                                                   alpha,                                \
                                                   descr,                                \
                                                   val,                                  \
                                                   ptr,                                  \
                                                   ind,                                  \
                                                   info,                                 \
                                                   x,                                    \
                                                   y,                                    \
                                                   policy,                               \
                                                   buffer);                              \
        return rocsparse_##P##csritsv_solve(handle,                                      \
                                            nmaxiter,                                    \
                                            tol,                                         \
                                            history,                                     \
                                            trans,                                       \
                                            m,                                           \
                                            nnz,                                         \
                                            alpha,                                       \
                                            descr,                                       \
                                            val,                                         \
                                            ptr,                                         \
                                            ind,                                         \
                                            info,                                        \
                                            x,                                           \
                                            y,                                           \
                                            policy,                                      \
                                            buffer);                                     \
    }

    L2_ITSV_WRAPPERS(s, float, float)
    L2_ITSV_WRAPPERS(d, double, double)
    L2_ITSV_WRAPPERS(c, rocsparse_float_complex, float)
    L2_ITSV_WRAPPERS(z, rocsparse_double_complex, double)
#undef L2_ITSV_WRAPPERS

    struct itsv_case
    {
        rocsparse_matrix_type mtype   = rocsparse_matrix_type_general;
        rocsparse_fill_mode   fill    = rocsparse_fill_mode_lower;
        rocsparse_diag_type   diag    = rocsparse_diag_type_non_unit;
        rocsparse_operation   trans   = rocsparse_operation_none;
        int                   base    = 0;
        bool                  device  = false;
        int32_t               nfree   = 0;
        bool                  tol     = true;
        bool                  hist    = true;
        bool                  ex      = true;
        int32_t               m       = 20;
        int32_t               per_row = 3;
    };

    inline std::string describe(const itsv_case& c)
    {
        std::ostringstream s;
        s << "csritsv type=" << c.mtype << " fill=" << c.fill << " diag=" << c.diag
          << " trans=" << c.trans << " base=" << c.base << " device=" << c.device
          << " nfree=" << c.nfree << " tol=" << c.tol << " hist=" << c.hist << " ex=" << c.ex;
        return s.str();
    }

    template <typename T>
    host_csr<T> make_matrix(const itsv_case& c)
    {
        const bool general  = (c.mtype == rocsparse_matrix_type_general);
        const bool lower    = (c.fill == rocsparse_fill_mode_lower);
        const bool non_unit = (c.diag == rocsparse_diag_type_non_unit);
        // A general matrix stores both triangles (and a diagonal that the unit
        // variant must ignore); a triangular matrix stores only its triangle.
        host_csr<T> A = make_triangular_pattern<T>(c.m,
                                                   c.per_row,
                                                   general ? true : lower,
                                                   general ? true : !lower,
                                                   non_unit || general,
                                                   c.base);
        if(!non_unit)
        {
            // Small off-diagonal entries keep the Jacobi iteration contractive.
            for(auto& v : A.val)
                v = v * rocsparse_ut::scalar<T>(0.125f);
        }
        return A;
    }

    template <typename T>
    void set_descr(const mat_descr& descr, const itsv_case& c)
    {
        ASSERT_EQ(rocsparse_set_mat_type(descr, c.mtype), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, c.fill), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, c.diag), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);
    }

    template <typename T>
    void run_csritsv(rocsparse_handle handle, const itsv_case& c)
    {
        using R = typename real_of<T>::type;
        SCOPED_TRACE(describe(c));

        const host_csr<T>    A     = make_matrix<T>(c);
        const int32_t        m     = A.m;
        const T              alpha = rocsparse_ut::scalar<T>(2.0f);
        const std::vector<T> x     = make_vector<T>(m, 3);
        const std::vector<T> want  = host_triangular_solve(A, c.fill, c.diag, c.trans, alpha, x);

        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        set_descr<T>(descr, c);
        if(::testing::Test::HasFatalFailure())
            return;

        device_csr<T>    dA(A);
        device_vector<T> dx(x), dy(std::vector<T>(m, T{})), dalpha(std::vector<T>(1, alpha));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr);

        size_t bytes = 0;
        ASSERT_EQ(itsv_buffer_size(
                      handle, c.trans, m, A.nnz(), descr, dA.val, dA.ptr, dA.ind, info, &bytes),
                  rocsparse_status_success);
        ASSERT_GT(bytes, size_t{0});
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dbuf.ptr);

        ASSERT_EQ(itsv_analysis(handle,
                                c.trans,
                                m,
                                A.nnz(),
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                info,
                                rocsparse_analysis_policy_force,
                                rocsparse_solve_policy_auto,
                                dbuf),
                  rocsparse_status_success);

        const int32_t  nmax_in = 60;
        int32_t        nmax    = nmax_in;
        const R        tol     = is_single<T>() ? R(1.0e-5) : R(1.0e-13);
        std::vector<R> history(nmax_in, R(-1));
        {
            scoped_pointer_mode mode(
                handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
            const T* palpha = c.device ? dalpha.ptr : &alpha;
            ASSERT_EQ(itsv_solve(handle,
                                 &nmax,
                                 c.nfree,
                                 c.tol ? &tol : nullptr,
                                 c.hist ? history.data() : nullptr,
                                 c.trans,
                                 m,
                                 A.nnz(),
                                 palpha,
                                 descr,
                                 dA.val,
                                 dA.ptr,
                                 dA.ind,
                                 info,
                                 dx,
                                 dy,
                                 rocsparse_solve_policy_auto,
                                 dbuf,
                                 c.ex),
                      rocsparse_status_success);
        }
        EXPECT_TRUE(expect_near(want, to_host(dy), "csritsv y"));

        if(c.hist)
            EXPECT_GE(history[0], R(0));
        if(c.tol && nmax < nmax_in)
        {
            // The loop stopped early: the recorded norm is within the tolerance.
            if(c.hist)
                EXPECT_LE(history[nmax], tol);
        }
        else if(c.tol)
        {
            // Never stopped early, so it ran all the iterations.
            EXPECT_EQ(nmax, nmax_in);
        }
        else
        {
            EXPECT_EQ(nmax, nmax_in);
        }

        rocsparse_int position = -2;
        EXPECT_EQ(rocsparse_csritsv_zero_pivot(handle, descr, info, &position),
                  rocsparse_status_success);
        EXPECT_EQ(position, -1);
        EXPECT_EQ(rocsparse_csritsv_clear(handle, descr, info), rocsparse_status_success);
    }

    class Cov70L2Csritsv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep(bool full)
        {
            for(const auto mtype :
                {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
                for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                    for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                        for(const auto trans : {rocsparse_operation_none,
                                                rocsparse_operation_transpose,
                                                rocsparse_operation_conjugate_transpose})
                        {
                            if(!full && trans == rocsparse_operation_transpose)
                                continue;
                            itsv_case c;
                            c.mtype     = mtype;
                            c.fill      = fill;
                            c.diag      = diag;
                            c.trans     = trans;
                            const int k = static_cast<int>(mtype) + 2 * static_cast<int>(fill)
                                          + 4 * static_cast<int>(diag)
                                          + 8 * static_cast<int>(trans);
                            c.base   = k & 1;
                            c.device = ((k >> 1) & 1) != 0;
                            c.nfree  = (k >> 2) % 3;
                            c.tol    = ((k >> 3) & 1) == 0;
                            c.hist   = ((k >> 1) % 3) != 0;
                            c.ex     = ((k >> 2) & 1) != 0 || c.nfree != 0;
                            run_csritsv<T>(handle, c);
                            if(::testing::Test::HasFailure())
                                return;
                        }
        }
    };

    TEST_F(Cov70L2Csritsv, solve_float)
    {
        sweep<float>(true);
    }

    TEST_F(Cov70L2Csritsv, solve_double)
    {
        sweep<double>(false);
    }

    TEST_F(Cov70L2Csritsv, solve_complex)
    {
        sweep<rocsparse_float_complex>(false);
        sweep<rocsparse_double_complex>(false);
    }

    TEST_F(Cov70L2Csritsv, solve_both_apis_and_reuse_policy)
    {
        // The legacy solve entry (no free iterations) next to solve_ex, and a second
        // analysis with the reuse policy that keeps the first one.
        for(const bool ex : {false, true})
        {
            itsv_case c;
            c.ex    = ex;
            c.fill  = rocsparse_fill_mode_upper;
            c.mtype = rocsparse_matrix_type_triangular;
            run_csritsv<double>(handle, c);

            const host_csr<double> A = make_matrix<double>(c);
            mat_descr              descr;
            mat_info               info;
            ASSERT_TRUE(descr.d && info.d);
            set_descr<double>(descr, c);
            device_csr<double> dA(A);
            ASSERT_TRUE(dA.ok());
            size_t bytes = 0;
            ASSERT_EQ(
                itsv_buffer_size(
                    handle, c.trans, A.m, A.nnz(), descr, dA.val, dA.ptr, dA.ind, info, &bytes),
                rocsparse_status_success);
            device_vector<char> dbuf(bytes);
            ASSERT_TRUE(dbuf.ptr);
            for(const auto policy : {rocsparse_analysis_policy_force,
                                     rocsparse_analysis_policy_reuse,
                                     rocsparse_analysis_policy_reuse,
                                     rocsparse_analysis_policy_force})
            {
                EXPECT_EQ(itsv_analysis(handle,
                                        c.trans,
                                        A.m,
                                        A.nnz(),
                                        descr,
                                        dA.val,
                                        dA.ptr,
                                        dA.ind,
                                        info,
                                        policy,
                                        rocsparse_solve_policy_auto,
                                        dbuf),
                          rocsparse_status_success);
            }
        }
    }

    // A matrix whose diagonal entry of row `p` is missing (`zero_value` false) or
    // stored as an explicit zero (`zero_value` true).
    template <typename T>
    void run_zero_pivot(rocsparse_handle      handle,
                        rocsparse_matrix_type mtype,
                        rocsparse_fill_mode   fill,
                        bool                  zero_value,
                        int32_t               p,
                        int                   base)
    {
        SCOPED_TRACE(::testing::Message()
                     << "zero pivot type=" << mtype << " fill=" << fill << " value=" << zero_value
                     << " p=" << p << " base=" << base);
        itsv_case c;
        c.mtype         = mtype;
        c.fill          = fill;
        c.base          = base;
        host_csr<T>   A = make_matrix<T>(c);
        const int32_t m = A.m;

        if(zero_value)
        {
            for(int32_t k = A.ptr[p] - A.base; k < A.ptr[p + 1] - A.base; ++k)
                if(A.ind[k] - A.base == p)
                    A.val[k] = T{};
        }
        else
        {
            host_csr<T> B;
            B.m    = A.m;
            B.n    = A.n;
            B.base = A.base;
            B.ptr.push_back(A.base);
            for(int32_t i = 0; i < A.m; ++i)
            {
                for(int32_t k = A.ptr[i] - A.base; k < A.ptr[i + 1] - A.base; ++k)
                {
                    if(i == p && A.ind[k] - A.base == p)
                        continue;
                    B.ind.push_back(A.ind[k]);
                    B.val.push_back(A.val[k]);
                }
                B.ptr.push_back(static_cast<int32_t>(B.ind.size()) + A.base);
            }
            A = B;
        }

        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        set_descr<T>(descr, c);
        const std::vector<T> x     = make_vector<T>(m, 3);
        const std::vector<T> y0    = make_vector<T>(m, 8);
        const T              alpha = rocsparse_ut::scalar<T>(1.0f);
        device_csr<T>        dA(A);
        device_vector<T>     dx(x), dy(y0);
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);

        size_t bytes = 0;
        ASSERT_EQ(itsv_buffer_size(
                      handle, c.trans, m, A.nnz(), descr, dA.val, dA.ptr, dA.ind, info, &bytes),
                  rocsparse_status_success);
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(itsv_analysis(handle,
                                c.trans,
                                m,
                                A.nnz(),
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                info,
                                rocsparse_analysis_policy_force,
                                rocsparse_solve_policy_auto,
                                dbuf),
                  rocsparse_status_success);

        using R      = typename real_of<T>::type;
        int32_t nmax = 30;
        const R tol  = R(1.0e-4);
        ASSERT_EQ(itsv_solve(handle,
                             &nmax,
                             0,
                             &tol,
                             nullptr,
                             c.trans,
                             m,
                             A.nnz(),
                             &alpha,
                             descr,
                             dA.val,
                             dA.ptr,
                             dA.ind,
                             info,
                             dx,
                             dy,
                             rocsparse_solve_policy_auto,
                             dbuf,
                             true),
                  rocsparse_status_success);

        // The solve is skipped when a pivot is zero: y is not modified.
        EXPECT_TRUE(expect_near(y0, to_host(dy), "csritsv y after zero pivot"));

        rocsparse_int position = -2;
        EXPECT_EQ(rocsparse_csritsv_zero_pivot(handle, descr, info, &position),
                  rocsparse_status_zero_pivot);
        EXPECT_EQ(position, p + base);
        EXPECT_EQ(rocsparse_csritsv_clear(handle, descr, info), rocsparse_status_success);
    }

    TEST_F(Cov70L2Csritsv, zero_pivot)
    {
        for(const auto mtype : {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                for(const bool zero_value : {false, true})
                {
                    run_zero_pivot<float>(handle, mtype, fill, zero_value, 7, 0);
                    run_zero_pivot<double>(handle, mtype, fill, zero_value, 3, 1);
                    run_zero_pivot<rocsparse_double_complex>(
                        handle, mtype, fill, zero_value, 11, 0);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(Cov70L2Csritsv, empty_matrices)
    {
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
                  rocsparse_status_success);

        // m == 0.
        {
            size_t bytes = 99;
            EXPECT_EQ(itsv_buffer_size(handle,
                                       rocsparse_operation_none,
                                       0,
                                       0,
                                       descr,
                                       static_cast<const float*>(nullptr),
                                       static_cast<const int32_t*>(nullptr),
                                       static_cast<const int32_t*>(nullptr),
                                       info,
                                       &bytes),
                      rocsparse_status_success);
            EXPECT_EQ(bytes, size_t{0});
            EXPECT_EQ(itsv_analysis(handle,
                                    rocsparse_operation_none,
                                    0,
                                    0,
                                    descr,
                                    static_cast<const float*>(nullptr),
                                    static_cast<const int32_t*>(nullptr),
                                    static_cast<const int32_t*>(nullptr),
                                    info,
                                    rocsparse_analysis_policy_force,
                                    rocsparse_solve_policy_auto,
                                    nullptr),
                      rocsparse_status_success);
            int32_t     nmax  = 5;
            const float alpha = 1.0f;
            for(const bool ex : {false, true})
                EXPECT_EQ(itsv_solve(handle,
                                     &nmax,
                                     0,
                                     nullptr,
                                     nullptr,
                                     rocsparse_operation_none,
                                     0,
                                     0,
                                     &alpha,
                                     descr,
                                     static_cast<const float*>(nullptr),
                                     static_cast<const int32_t*>(nullptr),
                                     static_cast<const int32_t*>(nullptr),
                                     info,
                                     static_cast<const float*>(nullptr),
                                     static_cast<float*>(nullptr),
                                     rocsparse_solve_policy_auto,
                                     nullptr,
                                     ex),
                          rocsparse_status_success);
            rocsparse_int position = -2;
            EXPECT_EQ(rocsparse_csritsv_zero_pivot(handle, descr, info, &position),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_csritsv_clear(handle, descr, info), rocsparse_status_success);
        }

        // nnz == 0 with m > 0: a unit diagonal solves to alpha * x, a non unit
        // diagonal is a (structurally) zero pivot at the first row.
        const int32_t            m     = 9;
        const std::vector<float> x     = make_vector<float>(m, 6);
        const float              alpha = 2.0f;
        host_csr<float>          A;
        A.m = A.n = m;
        A.ptr.assign(m + 1, 0);
        std::vector<float> scaled(x);
        for(auto& v : scaled)
            v *= alpha;

        for(const auto diag : {rocsparse_diag_type_unit, rocsparse_diag_type_non_unit})
            for(const auto mtype :
                {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
            {
                SCOPED_TRACE(::testing::Message() << "nnz=0 diag=" << diag << " type=" << mtype);
                mat_descr d2;
                mat_info  i2;
                ASSERT_TRUE(d2.d && i2.d);
                ASSERT_EQ(rocsparse_set_mat_type(d2, mtype), rocsparse_status_success);
                ASSERT_EQ(rocsparse_set_mat_diag_type(d2, diag), rocsparse_status_success);

                device_csr<float>    dA(A);
                device_vector<float> dx(x);
                ASSERT_TRUE(dA.ok() && dx.ptr);
                size_t bytes = 0;
                ASSERT_EQ(itsv_buffer_size(handle,
                                           rocsparse_operation_none,
                                           m,
                                           0,
                                           d2,
                                           dA.val,
                                           dA.ptr,
                                           dA.ind,
                                           i2,
                                           &bytes),
                          rocsparse_status_success);
                device_vector<char> dbuf(std::max<size_t>(bytes, 16));
                ASSERT_TRUE(dbuf.ptr);
                ASSERT_EQ(itsv_analysis(handle,
                                        rocsparse_operation_none,
                                        m,
                                        0,
                                        d2,
                                        dA.val,
                                        dA.ptr,
                                        dA.ind,
                                        i2,
                                        rocsparse_analysis_policy_force,
                                        rocsparse_solve_policy_auto,
                                        dbuf),
                          rocsparse_status_success);

                const float tol = 1.0e-6f;
                // y starts at zero, so one step is needed.
                {
                    device_vector<float> dy(std::vector<float>(m, 0.0f));
                    int32_t              nmax = 5;
                    ASSERT_EQ(itsv_solve(handle,
                                         &nmax,
                                         0,
                                         &tol,
                                         nullptr,
                                         rocsparse_operation_none,
                                         m,
                                         0,
                                         &alpha,
                                         d2,
                                         dA.val,
                                         dA.ptr,
                                         dA.ind,
                                         i2,
                                         dx,
                                         dy,
                                         rocsparse_solve_policy_auto,
                                         dbuf,
                                         true),
                              rocsparse_status_success);
                    if(diag == rocsparse_diag_type_unit)
                    {
                        EXPECT_EQ(nmax, 1);
                        EXPECT_TRUE(expect_near(scaled, to_host(dy), "csritsv unit nnz=0 y"));
                    }
                    else
                    {
                        EXPECT_TRUE(expect_near(
                            std::vector<float>(m, 0.0f), to_host(dy), "csritsv non unit nnz=0 y"));
                    }
                }
                // y already holds the solution: no step is needed.
                if(diag == rocsparse_diag_type_unit)
                {
                    device_vector<float> dy(scaled);
                    int32_t              nmax = 5;
                    for(const bool history : {false, true})
                    {
                        std::vector<float> hist(5, -1.0f);
                        ASSERT_EQ(itsv_solve(handle,
                                             &nmax,
                                             0,
                                             &tol,
                                             history ? hist.data() : nullptr,
                                             rocsparse_operation_none,
                                             m,
                                             0,
                                             &alpha,
                                             d2,
                                             dA.val,
                                             dA.ptr,
                                             dA.ind,
                                             i2,
                                             dx,
                                             dy,
                                             rocsparse_solve_policy_auto,
                                             dbuf,
                                             true),
                                  rocsparse_status_success);
                        EXPECT_EQ(nmax, 0);
                        EXPECT_TRUE(expect_near(scaled, to_host(dy), "csritsv unit nnz=0 kept y"));
                        if(history)
                            EXPECT_EQ(hist[0], 0.0f);
                        nmax = 5;
                    }
                }

                rocsparse_int          position = -2;
                const rocsparse_status st = rocsparse_csritsv_zero_pivot(handle, d2, i2, &position);
                if(diag == rocsparse_diag_type_unit)
                {
                    EXPECT_EQ(st, rocsparse_status_success);
                }
                else
                {
                    EXPECT_EQ(st, rocsparse_status_zero_pivot);
                    EXPECT_EQ(position, 0);
                }
            }
    }

    TEST_F(Cov70L2Csritsv, argument_checks)
    {
        const int32_t         m = 6;
        itsv_case             c;
        const host_csr<float> A = make_matrix<float>([&] {
            itsv_case k;
            k.m = m;
            return k;
        }());
        mat_descr             descr;
        mat_info              info;
        ASSERT_TRUE(descr.d && info.d);
        device_csr<float>    dA(A);
        device_vector<float> dx(make_vector<float>(m, 1)), dy(make_vector<float>(m, 2));
        size_t               bytes = 0;
        ASSERT_EQ(itsv_buffer_size(
                      handle, c.trans, m, A.nnz(), descr, dA.val, dA.ptr, dA.ind, info, &bytes),
                  rocsparse_status_success);
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dbuf.ptr);

        // ---- buffer size ----
        EXPECT_EQ(itsv_buffer_size(handle,
                                   rocsparse_operation_none,
                                   -1,
                                   A.nnz(),
                                   descr,
                                   dA.val,
                                   dA.ptr,
                                   dA.ind,
                                   info,
                                   &bytes),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(itsv_buffer_size(handle,
                                   rocsparse_operation_none,
                                   m,
                                   A.nnz(),
                                   descr,
                                   dA.val,
                                   dA.ptr,
                                   dA.ind,
                                   info,
                                   nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(itsv_buffer_size(handle,
                                   static_cast<rocsparse_operation>(77),
                                   m,
                                   A.nnz(),
                                   descr,
                                   dA.val,
                                   dA.ptr,
                                   dA.ind,
                                   info,
                                   &bytes),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(itsv_buffer_size(handle,
                                   rocsparse_operation_none,
                                   m,
                                   A.nnz(),
                                   descr,
                                   dA.val,
                                   nullptr,
                                   dA.ind,
                                   info,
                                   &bytes),
                  rocsparse_status_invalid_pointer);
        {
            mat_descr sym;
            ASSERT_EQ(rocsparse_set_mat_type(sym, rocsparse_matrix_type_symmetric),
                      rocsparse_status_success);
            EXPECT_EQ(itsv_buffer_size(handle,
                                       rocsparse_operation_none,
                                       m,
                                       A.nnz(),
                                       sym,
                                       dA.val,
                                       dA.ptr,
                                       dA.ind,
                                       info,
                                       &bytes),
                      rocsparse_status_not_implemented);
            mat_descr unsorted;
            ASSERT_EQ(rocsparse_set_mat_storage_mode(unsorted, rocsparse_storage_mode_unsorted),
                      rocsparse_status_success);
            EXPECT_EQ(itsv_buffer_size(handle,
                                       rocsparse_operation_none,
                                       m,
                                       A.nnz(),
                                       unsorted,
                                       dA.val,
                                       dA.ptr,
                                       dA.ind,
                                       info,
                                       &bytes),
                      rocsparse_status_requires_sorted_storage);
        }

        // ---- analysis ----
        EXPECT_EQ(itsv_analysis(handle,
                                rocsparse_operation_none,
                                m,
                                A.nnz(),
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                nullptr,
                                rocsparse_analysis_policy_force,
                                rocsparse_solve_policy_auto,
                                dbuf),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(itsv_analysis(handle,
                                rocsparse_operation_none,
                                m,
                                A.nnz(),
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                info,
                                static_cast<rocsparse_analysis_policy>(55),
                                rocsparse_solve_policy_auto,
                                dbuf),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(itsv_analysis(handle,
                                rocsparse_operation_none,
                                m,
                                A.nnz(),
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                info,
                                rocsparse_analysis_policy_force,
                                static_cast<rocsparse_solve_policy>(55),
                                dbuf),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(itsv_analysis(handle,
                                rocsparse_operation_none,
                                m,
                                A.nnz(),
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                info,
                                rocsparse_analysis_policy_force,
                                rocsparse_solve_policy_auto,
                                nullptr),
                  rocsparse_status_invalid_pointer);

        // ---- solve without analysis ----
        int32_t     nmax  = 3;
        const float alpha = 1.0f;
        for(const bool ex : {false, true})
        {
            mat_info fresh;
            ASSERT_TRUE(fresh.d);
            EXPECT_EQ(itsv_solve(handle,
                                 &nmax,
                                 0,
                                 nullptr,
                                 nullptr,
                                 rocsparse_operation_none,
                                 m,
                                 A.nnz(),
                                 &alpha,
                                 descr,
                                 dA.val,
                                 dA.ptr,
                                 dA.ind,
                                 fresh,
                                 dx,
                                 dy,
                                 rocsparse_solve_policy_auto,
                                 dbuf,
                                 ex),
                      rocsparse_status_invalid_pointer);
        }

        ASSERT_EQ(itsv_analysis(handle,
                                rocsparse_operation_none,
                                m,
                                A.nnz(),
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                info,
                                rocsparse_analysis_policy_force,
                                rocsparse_solve_policy_auto,
                                dbuf),
                  rocsparse_status_success);

        // ---- solve ----
        for(const bool ex : {false, true})
        {
            auto solve = [&](rocsparse_operation    trans,
                             int32_t                mm,
                             int32_t*               p_nmax,
                             int32_t                nfree,
                             const float*           p_alpha,
                             rocsparse_mat_descr    d,
                             const int32_t*         ptr,
                             const float*           px,
                             float*                 py,
                             rocsparse_solve_policy policy,
                             void*                  buffer) {
                return itsv_solve(handle,
                                  p_nmax,
                                  nfree,
                                  nullptr,
                                  nullptr,
                                  trans,
                                  mm,
                                  A.nnz(),
                                  p_alpha,
                                  d,
                                  dA.val,
                                  ptr,
                                  dA.ind,
                                  info,
                                  px,
                                  py,
                                  policy,
                                  buffer,
                                  ex);
            };
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_success);
            EXPECT_EQ(solve(static_cast<rocsparse_operation>(77),
                            m,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_value);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            dx,
                            dy,
                            static_cast<rocsparse_solve_policy>(55),
                            dbuf),
                      rocsparse_status_invalid_value);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            -1,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_size);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            nullptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            nullptr),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            nullptr,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            nullptr,
                            descr,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            nullptr,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            descr,
                            dA.ptr,
                            dx,
                            nullptr,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            nullptr,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_invalid_pointer);
            mat_descr sym;
            ASSERT_EQ(rocsparse_set_mat_type(sym, rocsparse_matrix_type_symmetric),
                      rocsparse_status_success);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            sym,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_not_implemented);
            mat_descr unsorted;
            ASSERT_EQ(rocsparse_set_mat_storage_mode(unsorted, rocsparse_storage_mode_unsorted),
                      rocsparse_status_success);
            EXPECT_EQ(solve(rocsparse_operation_none,
                            m,
                            &nmax,
                            0,
                            &alpha,
                            unsorted,
                            dA.ptr,
                            dx,
                            dy,
                            rocsparse_solve_policy_auto,
                            dbuf),
                      rocsparse_status_requires_sorted_storage);
        }
        {
            // negative free iteration count (solve_ex only)
            EXPECT_EQ(itsv_solve(handle,
                                 &nmax,
                                 -1,
                                 nullptr,
                                 nullptr,
                                 rocsparse_operation_none,
                                 m,
                                 A.nnz(),
                                 &alpha,
                                 descr,
                                 dA.val,
                                 dA.ptr,
                                 dA.ind,
                                 info,
                                 dx,
                                 dy,
                                 rocsparse_solve_policy_auto,
                                 dbuf,
                                 true),
                      rocsparse_status_invalid_size);
        }
    }

    // ----- generic API --------------------------------------------------------------
    inline const char* index_name(rocsparse_indextype t)
    {
        return t == rocsparse_indextype_i32 ? "i32" : "i64";
    }

    template <typename T, typename I, typename J>
    void run_spitsv(rocsparse_handle handle, const itsv_case& c)
    {
        using R = typename real_of<T>::type;
        SCOPED_TRACE(describe(c));
        SCOPED_TRACE(std::string("spitsv ") + index_name(rocsparse_ut::it_of<I>()) + "/"
                     + index_name(rocsparse_ut::it_of<J>()));

        const host_csr<T>    A     = make_matrix<T>(c);
        const int32_t        m     = A.m;
        const T              alpha = rocsparse_ut::scalar<T>(2.0f);
        const std::vector<T> x     = make_vector<T>(m, 3);
        const std::vector<T> want  = host_triangular_solve(A, c.fill, c.diag, c.trans, alpha, x);

        device_csr<T, I, J> dA(A);
        device_vector<T>    dx(x), dy(std::vector<T>(m, T{})), dalpha(std::vector<T>(1, alpha));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             m,
                                             m,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_ut::it_of<I>(),
                                             rocsparse_ut::it_of<J>(),
                                             to_base(c.base),
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_fill_mode, &c.fill, sizeof(c.fill)),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_diag_type, &c.diag, sizeof(c.diag)),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_matrix_type, &c.mtype, sizeof(c.mtype)),
                  rocsparse_status_success);

        const int32_t  nmax_in = 60;
        int32_t        nmax    = nmax_in;
        const R        tol     = is_single<T>() ? R(1.0e-5) : R(1.0e-13);
        std::vector<R> history(nmax_in, R(-1));

        scoped_pointer_mode mode(
            handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
        const void* palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha);

        auto call = [&](rocsparse_spitsv_stage stage, size_t* bytes, void* buffer) {
            return rocsparse_spitsv(handle,
                                    &nmax,
                                    c.tol ? &tol : nullptr,
                                    c.hist ? history.data() : nullptr,
                                    c.trans,
                                    palpha,
                                    mat.d,
                                    vx.d,
                                    vy.d,
                                    rocsparse_ut::dt_of<T>(),
                                    rocsparse_spitsv_alg_default,
                                    stage,
                                    bytes,
                                    buffer);
        };

        size_t bytes = 0;
        ASSERT_EQ(call(rocsparse_spitsv_stage_buffer_size, &bytes, nullptr),
                  rocsparse_status_success);
        ASSERT_GT(bytes, size_t{0});
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(call(rocsparse_spitsv_stage_preprocess, nullptr, dbuf), rocsparse_status_success);
        ASSERT_EQ(call(rocsparse_spitsv_stage_compute, nullptr, dbuf), rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "spitsv y"));
        if(c.tol && nmax < nmax_in && c.hist)
            EXPECT_LE(history[nmax], tol);
    }

    TEST(Cov70L2Spitsv, types_and_modes)
    {
        rocsparse_handle handle = nullptr;
        ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
        for(const auto mtype : {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                    for(const auto trans :
                        {rocsparse_operation_none, rocsparse_operation_transpose})
                    {
                        itsv_case c;
                        c.mtype     = mtype;
                        c.fill      = fill;
                        c.diag      = diag;
                        c.trans     = trans;
                        const int k = static_cast<int>(mtype) + 2 * static_cast<int>(fill)
                                      + 4 * static_cast<int>(diag) + 8 * static_cast<int>(trans);
                        c.base   = k & 1;
                        c.device = ((k >> 1) & 1) != 0;
                        c.tol    = ((k >> 3) & 1) == 0;
                        c.hist   = ((k >> 2) & 1) == 0;
                        run_spitsv<float, int32_t, int32_t>(handle, c);
                        run_spitsv<double, int64_t, int32_t>(handle, c);
                        run_spitsv<rocsparse_float_complex, int64_t, int64_t>(handle, c);
                        run_spitsv<rocsparse_double_complex, int32_t, int32_t>(handle, c);
                        if(::testing::Test::HasFailure())
                        {
                            EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
                            return;
                        }
                    }
        EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
    }

    TEST_F(Cov70L2Csritsv, spitsv_argument_checks)
    {
        const int32_t m = 6;
        itsv_case     c;
        c.m                     = m;
        const host_csr<float> A = make_matrix<float>(c);
        device_csr<float>     dA(A);
        device_vector<float>  dx(make_vector<float>(m, 1)), dy(make_vector<float>(m, 2));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             m,
                                             m,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        int32_t     nmax  = 4;
        const float alpha = 1.0f;
        size_t      bytes = 0;
        auto        call  = [&](rocsparse_handle       h,
                        int32_t*               p_nmax,
                        rocsparse_operation    trans,
                        const void*            p_alpha,
                        rocsparse_spmat_descr  a,
                        rocsparse_dnvec_descr  px,
                        rocsparse_dnvec_descr  py,
                        rocsparse_datatype     compute,
                        rocsparse_spitsv_alg   alg,
                        rocsparse_spitsv_stage stage,
                        size_t*                p_bytes) {
            return rocsparse_spitsv(h,
                                    p_nmax,
                                    nullptr,
                                    nullptr,
                                    trans,
                                    p_alpha,
                                    a,
                                    px,
                                    py,
                                    compute,
                                    alg,
                                    stage,
                                    p_bytes,
                                    nullptr);
        };
        const auto f32 = rocsparse_datatype_f32_r;
        const auto sz  = rocsparse_spitsv_stage_buffer_size;
        const auto alg = rocsparse_spitsv_alg_default;

        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_success);
        EXPECT_EQ(call(nullptr,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       nullptr,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       nullptr,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       nullptr,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       nullptr,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       nullptr,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       static_cast<rocsparse_operation>(77),
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       static_cast<rocsparse_datatype>(77),
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       static_cast<rocsparse_spitsv_alg>(77),
                       sz,
                       &bytes),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       static_cast<rocsparse_spitsv_stage>(77),
                       &bytes),
                  rocsparse_status_invalid_value);
        // Mixed precision is not supported.
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       rocsparse_datatype_f64_r,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_not_implemented);

        // Index type combinations that the solver does not provide.
        {
            device_csr<float, int32_t, int64_t> dB(A);
            ASSERT_TRUE(dB.ok());
            spmat_guard mixed;
            ASSERT_EQ(rocsparse_create_csr_descr(&mixed.d,
                                                 m,
                                                 m,
                                                 A.nnz(),
                                                 dB.ptr,
                                                 dB.ind,
                                                 dB.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i64,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            EXPECT_EQ(call(handle,
                           &nmax,
                           rocsparse_operation_none,
                           &alpha,
                           mixed.d,
                           vx.d,
                           vy.d,
                           f32,
                           alg,
                           sz,
                           &bytes),
                      rocsparse_status_not_implemented);
        }
        // Types other than the four floating point precisions.
        {
            spmat_guard       half;
            device_csr<float> dC(A);
            ASSERT_TRUE(dC.ok());
            ASSERT_EQ(rocsparse_create_csr_descr(&half.d,
                                                 m,
                                                 m,
                                                 A.nnz(),
                                                 dC.ptr,
                                                 dC.ind,
                                                 dC.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            dnvec_guard hx, hy;
            ASSERT_EQ(rocsparse_create_dnvec_descr(&hx.d, m, dx, rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_dnvec_descr(&hy.d, m, dy, rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            // compute type i32 differs from the matrix type: also not implemented.
            EXPECT_EQ(call(handle,
                           &nmax,
                           rocsparse_operation_none,
                           &alpha,
                           half.d,
                           hx.d,
                           hy.d,
                           rocsparse_datatype_i32_r,
                           alg,
                           sz,
                           &bytes),
                      rocsparse_status_not_implemented);
        }
    }
}
