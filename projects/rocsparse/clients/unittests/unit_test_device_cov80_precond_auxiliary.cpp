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

// Public auxiliary routines: the number of non-zeros reported for the ELL, Block ELL and
// BSR matrix descriptors, and the copy of a matrix info structure that holds the analysis
// of an iterative triangular solve (the copy must solve like the original).

#include "unit_test_device_cov70_precond_common.hpp"

#include <vector>

using namespace cov70;

class Cov80PrecondAuxiliary : public HandleTest
{
};

namespace
{
    struct spmat_guard
    {
        rocsparse_spmat_descr descr = nullptr;

        spmat_guard() = default;

        ~spmat_guard()
        {
            if(descr)
                (void)rocsparse_destroy_spmat_descr(descr);
        }

        spmat_guard(const spmat_guard&) = delete;
        spmat_guard& operator=(const spmat_guard&) = delete;
    };
}

TEST_F(Cov80PrecondAuxiliary, spmat_get_nnz_of_ell_block_ell_and_bsr_descriptors)
{
    // The descriptors keep the pointers without reading them.
    device_vector<int32_t> ind{std::vector<int32_t>(256, 0)};
    device_vector<int32_t> ptr{std::vector<int32_t>(32, 0)};
    device_vector<float>   val{std::vector<float>(512, 0.0f)};
    ASSERT_TRUE(ind.ptr && ptr.ptr && val.ptr);

    // ELL: width entries per row.
    {
        spmat_guard   A;
        const int64_t rows = 10, cols = 12, width = 3;
        ASSERT_EQ(rocsparse_create_ell_descr(&A.descr,
                                             rows,
                                             cols,
                                             ind.ptr,
                                             val.ptr,
                                             width,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_nnz(A.descr, &nnz), rocsparse_status_success);
        EXPECT_GE(nnz, width);
        EXPECT_LE(nnz, rows * width);
    }

    // Block ELL.
    {
        spmat_guard A;
        ASSERT_EQ(rocsparse_create_bell_descr(&A.descr,
                                              16,
                                              16,
                                              rocsparse_direction_row,
                                              4,
                                              2,
                                              ind.ptr,
                                              val.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_nnz(A.descr, &nnz), rocsparse_status_success);
        EXPECT_GT(nnz, 0);
    }

    // BSR: the number of non-zeros counts the entries of the blocks.
    {
        spmat_guard   A;
        const int64_t brows = 4, bcols = 4, bnnz = 6, block_dim = 3;
        ASSERT_EQ(rocsparse_create_bsr_descr(&A.descr,
                                             brows,
                                             bcols,
                                             bnnz,
                                             rocsparse_direction_row,
                                             block_dim,
                                             ptr.ptr,
                                             ind.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_nnz(A.descr, &nnz), rocsparse_status_success);
        EXPECT_GE(nnz, bnnz);
        EXPECT_LE(nnz, bnnz * block_dim * block_dim);
    }

    // A missing descriptor is refused.
    {
        int64_t nnz = 0;
        EXPECT_NE(rocsparse_spmat_get_nnz(nullptr, &nnz), rocsparse_status_success);
    }
}

TEST_F(Cov80PrecondAuxiliary, copy_mat_info_keeps_the_iterative_triangular_solve_analysis)
{
    // Lower triangular matrix, strictly diagonally dominant.
    const rocsparse_int        m = 24;
    std::vector<rocsparse_int> ptr{0};
    std::vector<rocsparse_int> ind;
    std::vector<float>         val;
    for(rocsparse_int i = 0; i < m; ++i)
    {
        for(rocsparse_int j = std::max<rocsparse_int>(0, i - 2); j <= i; ++j)
        {
            ind.push_back(j);
            val.push_back(j == i ? 4.0f : (j == i - 1 ? -1.0f : 0.5f));
        }
        ptr.push_back(static_cast<rocsparse_int>(ind.size()));
    }
    const rocsparse_int nnz = static_cast<rocsparse_int>(ind.size());

    device_vector<rocsparse_int> dptr{ptr};
    device_vector<rocsparse_int> dind{ind};
    device_vector<float>         dval{val};
    ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);
    rocsparse_mat_info src = nullptr, dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);

    const auto trans       = rocsparse_operation_none;
    size_t     buffer_size = 0;
    const auto st          = rocsparse_scsritsv_buffer_size(
        handle, trans, m, nnz, descr, dval.ptr, dptr.ptr, dind.ptr, src, &buffer_size);
    if(st == rocsparse_status_not_implemented)
    {
        (void)rocsparse_destroy_mat_info(src);
        (void)rocsparse_destroy_mat_info(dest);
        (void)rocsparse_destroy_mat_descr(descr);
        GTEST_SKIP() << "iterative triangular solve is not implemented here";
    }
    ASSERT_EQ(st, rocsparse_status_success);
    device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
    ASSERT_TRUE(buffer.ptr);
    ASSERT_EQ(rocsparse_scsritsv_analysis(handle,
                                          trans,
                                          m,
                                          nnz,
                                          descr,
                                          dval.ptr,
                                          dptr.ptr,
                                          dind.ptr,
                                          src,
                                          rocsparse_analysis_policy_force,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // The first copy creates what the destination lacks, the second one overwrites it.
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Host forward substitution of L y = alpha x.
    const float        alpha = 2.0f;
    std::vector<float> x(static_cast<size_t>(m));
    for(rocsparse_int i = 0; i < m; ++i)
        x[static_cast<size_t>(i)] = 1.0f + 0.25f * static_cast<float>(i % 5);
    std::vector<double> want(static_cast<size_t>(m), 0.0);
    for(rocsparse_int i = 0; i < m; ++i)
    {
        double sum  = alpha * static_cast<double>(x[static_cast<size_t>(i)]);
        double diag = 1.0;
        for(rocsparse_int k = ptr[static_cast<size_t>(i)]; k < ptr[static_cast<size_t>(i) + 1]; ++k)
        {
            const auto j = ind[static_cast<size_t>(k)];
            if(j == i)
                diag = val[static_cast<size_t>(k)];
            else
                sum -= static_cast<double>(val[static_cast<size_t>(k)])
                       * want[static_cast<size_t>(j)];
        }
        want[static_cast<size_t>(i)] = sum / diag;
    }

    for(const auto info : {src, dest})
    {
        device_vector<float> dx{x};
        device_vector<float> dy{std::vector<float>(static_cast<size_t>(m), 0.0f)};
        ASSERT_TRUE(dx.ptr && dy.ptr);
        rocsparse_int      nmaxiter = 200;
        const float        tol      = 1.0e-6f;
        std::vector<float> history(200, 0.0f);
        ASSERT_EQ(rocsparse_scsritsv_solve(handle,
                                           &nmaxiter,
                                           &tol,
                                           history.data(),
                                           trans,
                                           m,
                                           nnz,
                                           &alpha,
                                           descr,
                                           dval.ptr,
                                           dptr.ptr,
                                           dind.ptr,
                                           info,
                                           dx.ptr,
                                           dy.ptr,
                                           rocsparse_solve_policy_auto,
                                           buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto got = to_host(dy);
        for(rocsparse_int i = 0; i < m; ++i)
        {
            EXPECT_NEAR(got[static_cast<size_t>(i)],
                        static_cast<float>(want[static_cast<size_t>(i)]),
                        1.0e-4f
                            * (1.0f + std::abs(static_cast<float>(want[static_cast<size_t>(i)]))))
                << "row " << i << (info == dest ? " (copy)" : " (original)");
        }
    }

    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}
