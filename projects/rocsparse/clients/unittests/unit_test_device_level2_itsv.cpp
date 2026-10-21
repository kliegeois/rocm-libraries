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
#include "unit_test_device_level2_common.hpp"

using namespace level2;

namespace
{
    // The legacy solve (no free iterations) or, with `ex`, rocsparse_csritsv_solve_ex.
    template <typename T, typename... Args>
    rocsparse_status itsv_solve(
        bool ex, rocsparse_handle handle, rocsparse_int* nmax, rocsparse_int nfree, Args&&... args)
    {
        return ex ? csritsv_solve_ex<T>(handle, nmax, nfree, std::forward<Args>(args)...)
                  : csritsv_solve<T>(handle, nmax, std::forward<Args>(args)...);
    }

    class L2DeviceCsritsv : public rocsparse_ut::HandleTest
    {
    };

    TEST_F(L2DeviceCsritsv, empty_matrices)
    {
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
                  rocsparse_status_success);

        // m == 0.
        {
            size_t bytes = 99;
            EXPECT_EQ(csritsv_buffer_size<float>(handle,
                                                 rocsparse_operation_none,
                                                 0,
                                                 0,
                                                 descr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 info,
                                                 &bytes),
                      rocsparse_status_success);
            EXPECT_EQ(csritsv_analysis<float>(handle,
                                              rocsparse_operation_none,
                                              0,
                                              0,
                                              descr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              info,
                                              rocsparse_analysis_policy_force,
                                              rocsparse_solve_policy_auto,
                                              nullptr),
                      rocsparse_status_success);
            rocsparse_int nmax  = 5;
            const float   alpha = 1.0f;
            for(const bool ex : {false, true})
                EXPECT_EQ(itsv_solve<float>(ex,
                                            handle,
                                            &nmax,
                                            0,
                                            nullptr,
                                            nullptr,
                                            rocsparse_operation_none,
                                            0,
                                            0,
                                            &alpha,
                                            descr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            info,
                                            nullptr,
                                            nullptr,
                                            rocsparse_solve_policy_auto,
                                            nullptr),
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
                ASSERT_EQ(csritsv_buffer_size<float>(handle,
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
                ASSERT_EQ(csritsv_analysis<float>(handle,
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
                    rocsparse_int        nmax = 5;
                    ASSERT_EQ(itsv_solve<float>(true,
                                                handle,
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
                                                dbuf),
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
                // Without a tolerance the solve still runs: y = alpha * x for a unit
                // diagonal, with or without a history array. The step is exact, so one
                // iteration is reported and history[0] is the initial residual
                // ||alpha * x - 0||_inf. With an iteration limit <= 0 nothing is done and the
                // limit is returned unchanged.
                if(diag == rocsparse_diag_type_unit)
                {
                    float initial_residual = 0.0f;
                    for(const float v : scaled)
                        initial_residual = std::max(initial_residual, std::abs(v));
                    for(const bool ex : {false, true})
                        for(const bool history : {false, true})
                            for(const rocsparse_int limit : {5, 0, -1})
                            {
                                SCOPED_TRACE(::testing::Message()
                                             << "no tolerance, ex " << ex << ", history " << history
                                             << ", nmaxiter " << limit);
                                device_vector<float> dy(std::vector<float>(m, 0.0f));
                                std::vector<float>   hist(5, -1.0f);
                                rocsparse_int        nmax = limit;
                                ASSERT_EQ(itsv_solve<float>(ex,
                                                            handle,
                                                            &nmax,
                                                            0,
                                                            nullptr,
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
                                                            dbuf),
                                          rocsparse_status_success);
                                const bool stepped = (limit > 0);
                                EXPECT_EQ(nmax, stepped ? 1 : limit);
                                EXPECT_TRUE(
                                    expect_near(stepped ? scaled : std::vector<float>(m, 0.0f),
                                                to_host(dy),
                                                "csritsv unit nnz=0 no tol y"));
                                if(history)
                                {
                                    EXPECT_EQ(hist[0], stepped ? initial_residual : -1.0f);
                                    for(size_t i = 1; i < hist.size(); ++i)
                                        EXPECT_EQ(hist[i], -1.0f) << "history[" << i << "]";
                                }
                            }
                }
                // y already holds the solution: no step is needed.
                if(diag == rocsparse_diag_type_unit)
                {
                    device_vector<float> dy(scaled);
                    rocsparse_int        nmax = 5;
                    for(const bool history : {false, true})
                    {
                        std::vector<float> hist(5, -1.0f);
                        ASSERT_EQ(itsv_solve<float>(true,
                                                    handle,
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
                                                    dbuf),
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
}
