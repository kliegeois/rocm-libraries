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

// Fill mode checks of CSC matrices through rocsparse_Xcheck_matrix_csc and
// rocsparse_check_spmat, and of BSC matrices through rocsparse_Xcheck_matrix_gebsc.

#include "unit_test_device_conv_common.hpp"

#include <string>

using namespace convut;

namespace
{
    using ConvSpmat = ConvTest;

    // ------------------------------------------------------------------
    // check_spmat
    // ------------------------------------------------------------------

    struct Attributes
    {
        rocsparse_matrix_type  type    = rocsparse_matrix_type_general;
        rocsparse_fill_mode    fill    = rocsparse_fill_mode_lower;
        rocsparse_storage_mode storage = rocsparse_storage_mode_sorted;
    };

    constexpr auto chk_size = rocsparse_check_spmat_stage_buffer_size;
    constexpr auto chk_cmp  = rocsparse_check_spmat_stage_compute;

    // Runs both stages of rocsparse_check_spmat; returns the status of the compute stage
    // and the reported data status.
    rocsparse_status run_check(rocsparse_handle       handle,
                               rocsparse_spmat_descr  descr,
                               const Attributes&      at,
                               rocsparse_data_status* ds)
    {
        const auto set = [&](rocsparse_spmat_attribute a, const auto& v) {
            EXPECT_EQ(rocsparse_spmat_set_attribute(descr, a, &v, sizeof(v)), st_ok);
        };
        set(rocsparse_spmat_matrix_type, at.type);
        set(rocsparse_spmat_fill_mode, at.fill);
        set(rocsparse_spmat_storage_mode, at.storage);
        size_t size = 0;
        auto   st   = rocsparse_check_spmat(handle, descr, ds, chk_size, &size, nullptr);
        if(st != st_ok)
            return st;
        DevBuf<char> temp(size);
        st = rocsparse_check_spmat(handle, descr, ds, chk_cmp, &size, temp);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return st;
    }

    // 3 x 3 lower (L) and upper (U) triangular matrices in CSC format, checked as symmetric
    // and as triangular matrices with either fill mode. A matrix is valid only for the fill
    // mode of the triangle it stores.
    TEST_F(ConvSpmat, CheckMatrixCscFillMode)
    {
        const I                   m = 3, nnz = 6;
        const std::vector<I>      l_ptr = {0, 3, 5, 6}, u_ptr = {0, 1, 3, 6};
        const std::vector<I>      l_ind = {0, 1, 2, 1, 2, 2}, u_ind = {0, 0, 1, 0, 1, 2};
        const std::vector<double> l_val = {1, 2, 4, 3, 5, 6}, u_val = {1, 2, 3, 4, 5, 6};
        for(auto type : {rocsparse_matrix_type_symmetric, rocsparse_matrix_type_triangular})
            for(auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                for(bool lower : {true, false})
                {
                    SCOPED_TRACE("type=" + std::to_string(type) + " fill=" + std::to_string(fill)
                                 + (lower ? " L" : " U"));
                    const auto     want = lower == (fill == rocsparse_fill_mode_lower)
                                              ? rocsparse_data_status_success
                                              : rocsparse_data_status_invalid_fill;
                    DevBuf<I>      d_ptr(lower ? l_ptr : u_ptr), d_ind(lower ? l_ind : u_ind);
                    DevBuf<double> d_val(lower ? l_val : u_val);
                    const auto     sorted = rocsparse_storage_mode_sorted;
                    size_t         size   = 0;
                    ASSERT_EQ(rocsparse_dcheck_matrix_csc_buffer_size(handle,
                                                                      m,
                                                                      m,
                                                                      nnz,
                                                                      d_val,
                                                                      d_ptr,
                                                                      d_ind,
                                                                      base0,
                                                                      type,
                                                                      fill,
                                                                      sorted,
                                                                      &size),
                              st_ok);
                    DevBuf<char> temp(size);
                    auto         ds = bad_ds;
                    ASSERT_EQ(rocsparse_dcheck_matrix_csc(handle,
                                                          m,
                                                          m,
                                                          nnz,
                                                          d_val,
                                                          d_ptr,
                                                          d_ind,
                                                          base0,
                                                          type,
                                                          fill,
                                                          sorted,
                                                          &ds,
                                                          temp),
                              st_ok);
                    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
                    EXPECT_EQ(ds, want);

                    const auto itype = sizeof(I) == 4 ? i32 : rocsparse_indextype_i64;
                    Spmat      descr;
                    ASSERT_EQ(rocsparse_create_csc_descr(&descr,
                                                         m,
                                                         m,
                                                         nnz,
                                                         d_ptr,
                                                         d_ind,
                                                         d_val,
                                                         itype,
                                                         itype,
                                                         base0,
                                                         rocsparse_datatype_f64_r),
                              st_ok);
                    ds = bad_ds;
                    EXPECT_EQ(run_check(handle, descr, {type, fill}, &ds), st_ok);
                    EXPECT_EQ(ds, want);
                }
    }

    // check_matrix_gebsc

    // The block patterns of the matrices above in BSC format, with 1 x 1 and 2 x 2 blocks,
    // both block directions and both storage modes.
    TEST_F(ConvSpmat, CheckMatrixGebscFillMode)
    {
        const auto dirs     = {rocsparse_direction_row, rocsparse_direction_column};
        const auto storages = {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted};
        const I    mb = 3, nnzb = 6;
        const std::vector<I> l_ptr = {0, 3, 5, 6}, u_ptr = {0, 1, 3, 6};
        const std::vector<I> l_ind = {0, 1, 2, 1, 2, 2}, u_ind = {0, 0, 1, 0, 1, 2};
        for(I bd : {1, 2})
            for(auto dir : dirs)
                for(auto storage : storages)
                    for(auto type :
                        {rocsparse_matrix_type_symmetric, rocsparse_matrix_type_triangular})
                        for(auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                            for(bool lower : {true, false})
                            {
                                SCOPED_TRACE("block_dim=" + std::to_string(bd)
                                             + " dir=" + std::to_string(dir)
                                             + " storage=" + std::to_string(storage)
                                             + " type=" + std::to_string(type) + " fill="
                                             + std::to_string(fill) + (lower ? " L" : " U"));
                                const auto want = lower == (fill == rocsparse_fill_mode_lower)
                                                      ? rocsparse_data_status_success
                                                      : rocsparse_data_status_invalid_fill;
                                std::vector<double> val(nnzb * bd * bd);
                                for(size_t k = 0; k < val.size(); ++k)
                                    val[k] = double(k + 1);
                                DevBuf<I> d_ptr(lower ? l_ptr : u_ptr),
                                    d_ind(lower ? l_ind : u_ind);
                                DevBuf<double> d_val(val);
                                size_t         size = 0;
                                ASSERT_EQ(rocsparse_dcheck_matrix_gebsc_buffer_size(handle,
                                                                                    dir,
                                                                                    mb,
                                                                                    mb,
                                                                                    nnzb,
                                                                                    bd,
                                                                                    bd,
                                                                                    d_val,
                                                                                    d_ptr,
                                                                                    d_ind,
                                                                                    base0,
                                                                                    type,
                                                                                    fill,
                                                                                    storage,
                                                                                    &size),
                                          st_ok);
                                DevBuf<char> temp(size);
                                auto         ds = bad_ds;
                                ASSERT_EQ(rocsparse_dcheck_matrix_gebsc(handle,
                                                                        dir,
                                                                        mb,
                                                                        mb,
                                                                        nnzb,
                                                                        bd,
                                                                        bd,
                                                                        d_val,
                                                                        d_ptr,
                                                                        d_ind,
                                                                        base0,
                                                                        type,
                                                                        fill,
                                                                        storage,
                                                                        &ds,
                                                                        temp),
                                          st_ok);
                                EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
                                EXPECT_EQ(ds, want);
                            }
    }
} // namespace
