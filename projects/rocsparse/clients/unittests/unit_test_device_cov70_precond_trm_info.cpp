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

// Copies of a rocsparse_mat_info that carries triangular-solve analysis data
// (rocsparse::trm_info_t): the copy constructor when the destination is empty,
// the assignment when it already holds the data. The analysis of the
// transposed operations fills every array of the info (row map, diagonal
// indices and the transposed permutation / row pointer / column indices), and
// each copy is then used for the solve and compared with a dense reference.

#include "unit_test_device_cov70_precond_common.hpp"

#include <string>

using namespace cov70;

namespace
{
    // Dense triangle of `a` (lower or upper, with the diagonal) as a row-major matrix.
    std::vector<double> dense_triangle(const csr_matrix<double>& a, bool lower)
    {
        const int32_t       n = a.n;
        std::vector<double> t(static_cast<size_t>(n) * n, 0.0);
        for(int32_t i = 0; i < n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(lower ? (a.ind[p] <= i) : (a.ind[p] >= i))
                    t[static_cast<size_t>(i) * n + a.ind[p]] = a.val[p];
        return t;
    }

    // Solves op(T) y = rhs for the dense triangular matrix T.
    std::vector<double> solve_dense(const std::vector<double>& t,
                                    int32_t                    n,
                                    bool                       lower,
                                    bool                       transpose,
                                    const std::vector<double>& rhs)
    {
        const auto at = [&](int32_t i, int32_t j) {
            return transpose ? t[static_cast<size_t>(j) * n + i]
                             : t[static_cast<size_t>(i) * n + j];
        };
        // op(T) is lower triangular for (lower, no transpose) and (upper, transpose).
        const bool          forward = (lower != transpose);
        std::vector<double> y(static_cast<size_t>(n), 0.0);
        for(int32_t s = 0; s < n; ++s)
        {
            const int32_t i   = forward ? s : n - 1 - s;
            double        sum = rhs[i];
            for(int32_t j = forward ? 0 : i + 1; j < (forward ? i : n); ++j)
                sum -= at(i, j) * y[j];
            y[i] = sum / at(i, i);
        }
        return y;
    }

    struct trm_fixture
    {
        csr_matrix<double>           a;
        device_vector<rocsparse_int> row;
        device_vector<rocsparse_int> col;
        device_vector<double>        val;
        rocsparse_mat_descr          lower = nullptr;
        rocsparse_mat_descr          upper = nullptr;

        trm_fixture()
            : a(make_grid_matrix<double>(4, true))
        {
            upload(row, a.ptr);
            upload(col, a.ind);
            upload(val, a.val);
            EXPECT_EQ(rocsparse_create_mat_descr(&lower), rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_mat_descr(&upper), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_fill_mode(lower, rocsparse_fill_mode_lower),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_fill_mode(upper, rocsparse_fill_mode_upper),
                      rocsparse_status_success);
        }

        ~trm_fixture()
        {
            if(lower)
                (void)rocsparse_destroy_mat_descr(lower);
            if(upper)
                (void)rocsparse_destroy_mat_descr(upper);
        }

        trm_fixture(const trm_fixture&) = delete;
        trm_fixture& operator=(const trm_fixture&) = delete;
    };

    // Buffer size and analysis of one triangle / operation into `info`.
    void analyze(rocsparse_handle    handle,
                 trm_fixture&        f,
                 rocsparse_mat_info  info,
                 bool                lower,
                 rocsparse_operation trans,
                 size_t&             buffer_size,
                 void*               buffer)
    {
        rocsparse_mat_descr descr = lower ? f.lower : f.upper;
        size_t              size  = 0;
        ASSERT_EQ(rocsparse_dcsrsv_buffer_size(
                      handle, trans, f.a.n, f.a.nnz(), descr, f.val, f.row, f.col, info, &size),
                  rocsparse_status_success);
        if(buffer == nullptr)
        {
            buffer_size = std::max(buffer_size, size);
            return;
        }
        ASSERT_LE(size, buffer_size);
        ASSERT_EQ(rocsparse_dcsrsv_analysis(handle,
                                            trans,
                                            f.a.n,
                                            f.a.nnz(),
                                            descr,
                                            f.val,
                                            f.row,
                                            f.col,
                                            info,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            buffer),
                  rocsparse_status_success);
    }

    // Solve with the analysis stored in `info` and compare with the dense reference.
    void check_solve(rocsparse_handle    handle,
                     trm_fixture&        f,
                     rocsparse_mat_info  info,
                     bool                lower,
                     rocsparse_operation trans,
                     void*               buffer,
                     const std::string&  what)
    {
        SCOPED_TRACE(what);
        const int32_t       n = f.a.n;
        std::vector<double> rhs(static_cast<size_t>(n));
        for(int32_t i = 0; i < n; ++i)
            rhs[i] = 1.0 + 0.25 * (i % 5);
        device_vector<double> dx{rhs};
        device_vector<double> dy{static_cast<size_t>(n)};
        ASSERT_TRUE(dx.ptr && dy.ptr);
        const double alpha = 1.5;
        ASSERT_EQ(rocsparse_dcsrsv_solve(handle,
                                         trans,
                                         n,
                                         f.a.nnz(),
                                         &alpha,
                                         lower ? f.lower : f.upper,
                                         f.val,
                                         f.row,
                                         f.col,
                                         info,
                                         dx,
                                         dy,
                                         rocsparse_solve_policy_auto,
                                         buffer),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<double> scaled = rhs;
        for(double& v : scaled)
            v *= alpha;
        const auto want = solve_dense(
            dense_triangle(f.a, lower), n, lower, trans == rocsparse_operation_transpose, scaled);
        expect_close(to_host(dy), want, 10.0);
    }
}

class Cov70PrecondTrmInfo : public HandleTest
{
};

TEST_F(Cov70PrecondTrmInfo, copy_mat_info_with_triangular_analysis)
{
    for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
    {
        trm_fixture f;
        ASSERT_NE(f.lower, nullptr);

        rocsparse_mat_info src = nullptr;
        rocsparse_mat_info dst = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&dst), rocsparse_status_success);

        size_t buffer_size = 0;
        analyze(handle, f, src, true, trans, buffer_size, nullptr);
        analyze(handle, f, src, false, trans, buffer_size, nullptr);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        analyze(handle, f, src, true, trans, buffer_size, buffer.ptr);
        analyze(handle, f, src, false, trans, buffer_size, buffer.ptr);

        const std::string tag = (trans == rocsparse_operation_none) ? "none" : "transpose";
        check_solve(handle, f, src, true, trans, buffer.ptr, "src lower " + tag);
        check_solve(handle, f, src, false, trans, buffer.ptr, "src upper " + tag);

        // Empty destination: the trm info is copy constructed.
        ASSERT_EQ(rocsparse_copy_mat_info(dst, src), rocsparse_status_success);
        check_solve(handle, f, dst, true, trans, buffer.ptr, "copy lower " + tag);
        check_solve(handle, f, dst, false, trans, buffer.ptr, "copy upper " + tag);

        // Destination that already holds the data: the trm info is assigned.
        ASSERT_EQ(rocsparse_copy_mat_info(dst, src), rocsparse_status_success);
        check_solve(handle, f, dst, true, trans, buffer.ptr, "assigned lower " + tag);
        check_solve(handle, f, dst, false, trans, buffer.ptr, "assigned upper " + tag);

        // The source is unaffected.
        check_solve(handle, f, src, true, trans, buffer.ptr, "src again " + tag);

        for(const auto info : {dst, src})
        {
            EXPECT_EQ(rocsparse_csrsv_clear(handle, f.lower, info), rocsparse_status_success);
            EXPECT_EQ(rocsparse_csrsv_clear(handle, f.upper, info), rocsparse_status_success);
        }
        EXPECT_EQ(rocsparse_destroy_mat_info(dst), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    }
}

TEST_F(Cov70PrecondTrmInfo, copy_mat_info_with_ilu0_analysis)
{
    // The ILU(0) analysis keeps its data in a trm info too (diagonal indices and row map).
    trm_fixture f;
    ASSERT_NE(f.lower, nullptr);
    rocsparse_mat_info src = nullptr;
    rocsparse_mat_info dst = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_info(&dst), rocsparse_status_success);

    size_t size = 0;
    ASSERT_EQ(rocsparse_dcsrilu0_buffer_size(
                  handle, f.a.n, f.a.nnz(), f.lower, f.val, f.row, f.col, src, &size),
              rocsparse_status_success);
    device_vector<char> buffer{size ? size : size_t{1}};
    ASSERT_TRUE(buffer.ptr);
    ASSERT_EQ(rocsparse_dcsrilu0_analysis(handle,
                                          f.a.n,
                                          f.a.nnz(),
                                          f.lower,
                                          f.val,
                                          f.row,
                                          f.col,
                                          src,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          buffer),
              rocsparse_status_success);

    for(int pass = 0; pass < 2; ++pass)
    {
        SCOPED_TRACE("copy " + std::to_string(pass));
        ASSERT_EQ(rocsparse_copy_mat_info(dst, src), rocsparse_status_success);

        // Factorize with the copied analysis and compare with the exact ILU(0).
        device_vector<double> dilu{f.a.val};
        ASSERT_TRUE(dilu.ptr);
        ASSERT_EQ(rocsparse_dcsrilu0(handle,
                                     f.a.n,
                                     f.a.nnz(),
                                     f.lower,
                                     dilu,
                                     f.row,
                                     f.col,
                                     dst,
                                     rocsparse_solve_policy_auto,
                                     buffer),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_close(to_host(dilu), host_ilu0(f.a));
    }

    EXPECT_EQ(rocsparse_csrilu0_clear(handle, dst), rocsparse_status_success);
    EXPECT_EQ(rocsparse_csrilu0_clear(handle, src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(dst), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
}
