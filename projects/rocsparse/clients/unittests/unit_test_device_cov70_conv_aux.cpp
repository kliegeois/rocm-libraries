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

// Device unit tests for the generic-API descriptor and auxiliary entry points
// (const descriptor constructors and getters, nnz / batch accessors, status
// strings, color-info and HYB copies, dnmat_copy_data). Every test checks the
// returned status and the values read back through the public getters.

#include "unit_test_device_cov70_conv_common.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Only the pointers are stored by the const descriptors, so host arrays are
    // enough for the constructor/getter round trips.
    struct Buf
    {
        std::vector<int32_t> i32 = std::vector<int32_t>(64, 0);
        std::vector<int32_t> j32 = std::vector<int32_t>(64, 0);
        std::vector<float>   v32 = std::vector<float>(64, 0.0f);
    };

    TEST(Cov70ConvAux, StatusNameAndDescription)
    {
        struct Entry
        {
            rocsparse_status status;
            const char*      name;
            const char*      description;
        };
        const Entry entries[] = {
            {rocsparse_status_success,
             "rocsparse_status_success",
             "rocsparse operation was successful"},
            {rocsparse_status_invalid_handle,
             "rocsparse_status_invalid_handle",
             "handle not initialized, invalid or null"},
            {rocsparse_status_not_implemented,
             "rocsparse_status_not_implemented",
             "function is not implemented"},
            {rocsparse_status_invalid_pointer,
             "rocsparse_status_invalid_pointer",
             "invalid pointer parameter"},
            {rocsparse_status_invalid_size,
             "rocsparse_status_invalid_size",
             "invalid size parameter"},
            {rocsparse_status_memory_error,
             "rocsparse_status_memory_error",
             "failed memory allocation, copy, dealloc"},
            {rocsparse_status_internal_error,
             "rocsparse_status_internal_error",
             "other internal library failure"},
            {rocsparse_status_invalid_value,
             "rocsparse_status_invalid_value",
             "invalid value parameter"},
            {rocsparse_status_arch_mismatch,
             "rocsparse_status_arch_mismatch",
             "device arch is not supported"},
            {rocsparse_status_zero_pivot, "rocsparse_status_zero_pivot", "encountered zero pivot"},
            {rocsparse_status_not_initialized,
             "rocsparse_status_not_initialized",
             "descriptor has not been initialized"},
            {rocsparse_status_type_mismatch,
             "rocsparse_status_type_mismatch",
             "index types do not match"},
            {rocsparse_status_requires_sorted_storage,
             "rocsparse_status_requires_sorted_storage",
             "sorted storage required"},
            {rocsparse_status_thrown_exception,
             "rocsparse_status_thrown_exception",
             "exception being thrown"},
            {rocsparse_status_continue,
             "rocsparse_status_continue",
             "nothing preventing function to proceed"},
        };
        for(const Entry& e : entries)
        {
            EXPECT_STREQ(rocsparse_get_status_name(e.status), e.name);
            EXPECT_STREQ(rocsparse_get_status_description(e.status), e.description);
        }
        EXPECT_STREQ(rocsparse_get_status_name(static_cast<rocsparse_status>(1000)),
                     "Unrecognized status code");
        EXPECT_STREQ(rocsparse_get_status_description(static_cast<rocsparse_status>(1000)),
                     "Unrecognized status code");
    }

    TEST(Cov70ConvAux, ConstSpvecRoundTrip)
    {
        Buf                         b;
        rocsparse_const_spvec_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_const_spvec_descr(&descr,
                                                     10,
                                                     4,
                                                     b.i32.data(),
                                                     b.v32.data(),
                                                     rocsparse_indextype_i32,
                                                     rocsparse_index_base_one,
                                                     rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        int64_t              size = 0, nnz = 0;
        const void*          idx  = nullptr;
        const void*          val  = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_zero;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_spvec_get(descr, &size, &nnz, &idx, &val, &it, &base, &dt),
                  rocsparse_status_success);
        EXPECT_EQ(size, 10);
        EXPECT_EQ(nnz, 4);
        EXPECT_EQ(idx, static_cast<const void*>(b.i32.data()));
        EXPECT_EQ(val, static_cast<const void*>(b.v32.data()));
        EXPECT_EQ(it, rocsparse_indextype_i32);
        EXPECT_EQ(base, rocsparse_index_base_one);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);

        const void* vals = nullptr;
        ASSERT_EQ(rocsparse_const_spvec_get_values(descr, &vals), rocsparse_status_success);
        EXPECT_EQ(vals, static_cast<const void*>(b.v32.data()));
        EXPECT_EQ(rocsparse_destroy_spvec_descr(const_cast<rocsparse_spvec_descr>(descr)),
                  rocsparse_status_success);
    }

    TEST(Cov70ConvAux, ConstSpvecInvalidArguments)
    {
        Buf                         b;
        rocsparse_const_spvec_descr descr  = nullptr;
        const auto                  create = [&](rocsparse_const_spvec_descr* d,
                                int64_t                      size,
                                int64_t                      nnz,
                                const void*                  idx,
                                const void*                  val,
                                rocsparse_indextype          it,
                                rocsparse_index_base         base,
                                rocsparse_datatype           dt) {
            return rocsparse_create_const_spvec_descr(d, size, nnz, idx, val, it, base, dt);
        };
        const rocsparse_indextype  it   = rocsparse_indextype_i32;
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const rocsparse_datatype   dt   = rocsparse_datatype_f32_r;

        EXPECT_EQ(create(nullptr, 4, 2, b.i32.data(), b.v32.data(), it, base, dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&descr, -1, 0, b.i32.data(), b.v32.data(), it, base, dt),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(create(&descr, 4, -1, b.i32.data(), b.v32.data(), it, base, dt),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(create(&descr, 4, 5, b.i32.data(), b.v32.data(), it, base, dt),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(create(&descr, 4, 2, nullptr, b.v32.data(), it, base, dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&descr, 4, 2, b.i32.data(), nullptr, it, base, dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&descr,
                         4,
                         2,
                         b.i32.data(),
                         b.v32.data(),
                         static_cast<rocsparse_indextype>(100),
                         base,
                         dt),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(create(&descr,
                         4,
                         2,
                         b.i32.data(),
                         b.v32.data(),
                         it,
                         static_cast<rocsparse_index_base>(100),
                         dt),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(create(&descr,
                         4,
                         2,
                         b.i32.data(),
                         b.v32.data(),
                         it,
                         base,
                         static_cast<rocsparse_datatype>(100)),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(descr, nullptr);
    }

    TEST(Cov70ConvAux, ConstCooRoundTrip)
    {
        Buf                         b;
        rocsparse_const_spmat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_const_coo_descr(&descr,
                                                   6,
                                                   7,
                                                   3,
                                                   b.i32.data(),
                                                   b.j32.data(),
                                                   b.v32.data(),
                                                   rocsparse_indextype_i32,
                                                   rocsparse_index_base_one,
                                                   rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t              rows = 0, cols = 0, nnz = 0;
        const void *         ri = nullptr, *ci = nullptr, *v = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_zero;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_coo_get(descr, &rows, &cols, &nnz, &ri, &ci, &v, &it, &base, &dt),
                  rocsparse_status_success);
        EXPECT_EQ(rows, 6);
        EXPECT_EQ(cols, 7);
        EXPECT_EQ(nnz, 3);
        EXPECT_EQ(ri, static_cast<const void*>(b.i32.data()));
        EXPECT_EQ(ci, static_cast<const void*>(b.j32.data()));
        EXPECT_EQ(v, static_cast<const void*>(b.v32.data()));
        EXPECT_EQ(it, rocsparse_indextype_i32);
        EXPECT_EQ(base, rocsparse_index_base_one);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);

        rocsparse_format fmt = rocsparse_format_csr;
        EXPECT_EQ(rocsparse_spmat_get_format(descr, &fmt), rocsparse_status_success);
        EXPECT_EQ(fmt, rocsparse_format_coo);
        int64_t nnz_get = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_get), rocsparse_status_success);
        EXPECT_EQ(nnz_get, 3);

        EXPECT_EQ(rocsparse_destroy_spmat_descr(const_cast<rocsparse_spmat_descr>(descr)),
                  rocsparse_status_success);
    }

    TEST(Cov70ConvAux, ConstCooInvalidArguments)
    {
        Buf                         b;
        rocsparse_const_spmat_descr descr  = nullptr;
        const auto                  create = [&](rocsparse_const_spmat_descr* d,
                                int64_t                      rows,
                                int64_t                      cols,
                                int64_t                      nnz,
                                const void*                  ri,
                                const void*                  ci,
                                const void*                  v) {
            return rocsparse_create_const_coo_descr(d,
                                                    rows,
                                                    cols,
                                                    nnz,
                                                    ri,
                                                    ci,
                                                    v,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_index_base_zero,
                                                    rocsparse_datatype_f32_r);
        };
        const int32_t* i = b.i32.data();
        const float*   v = b.v32.data();
        EXPECT_EQ(create(nullptr, 3, 3, 2, i, i, v), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&descr, -1, 3, 2, i, i, v), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&descr, 3, -1, 2, i, i, v), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&descr, 3, 3, -1, i, i, v), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&descr, 3, 3, 10, i, i, v), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&descr, 3, 3, 2, nullptr, i, v), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&descr, 3, 3, 2, i, nullptr, v), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&descr, 3, 3, 2, i, i, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_create_const_coo_descr(&descr,
                                                   3,
                                                   3,
                                                   2,
                                                   i,
                                                   i,
                                                   v,
                                                   static_cast<rocsparse_indextype>(100),
                                                   rocsparse_index_base_zero,
                                                   rocsparse_datatype_f32_r),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(descr, nullptr);
    }

    TEST(Cov70ConvAux, ConstCscRoundTrip)
    {
        Buf                         b;
        rocsparse_const_spmat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_const_csc_descr(&descr,
                                                   5,
                                                   4,
                                                   3,
                                                   b.i32.data(),
                                                   b.j32.data(),
                                                   b.v32.data(),
                                                   rocsparse_indextype_i32,
                                                   rocsparse_indextype_i32,
                                                   rocsparse_index_base_zero,
                                                   rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t              rows = 0, cols = 0, nnz = 0;
        const void *         cp = nullptr, *ri = nullptr, *v = nullptr;
        rocsparse_indextype  cpt  = rocsparse_indextype_i64;
        rocsparse_indextype  rit  = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_one;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_csc_get(
                      descr, &rows, &cols, &nnz, &cp, &ri, &v, &cpt, &rit, &base, &dt),
                  rocsparse_status_success);
        EXPECT_EQ(rows, 5);
        EXPECT_EQ(cols, 4);
        EXPECT_EQ(nnz, 3);
        EXPECT_EQ(cp, static_cast<const void*>(b.i32.data()));
        EXPECT_EQ(ri, static_cast<const void*>(b.j32.data()));
        EXPECT_EQ(v, static_cast<const void*>(b.v32.data()));
        EXPECT_EQ(cpt, rocsparse_indextype_i32);
        EXPECT_EQ(rit, rocsparse_indextype_i32);
        EXPECT_EQ(base, rocsparse_index_base_zero);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);

        rocsparse_format fmt = rocsparse_format_csr;
        EXPECT_EQ(rocsparse_spmat_get_format(descr, &fmt), rocsparse_status_success);
        EXPECT_EQ(fmt, rocsparse_format_csc);

        EXPECT_EQ(rocsparse_destroy_spmat_descr(const_cast<rocsparse_spmat_descr>(descr)),
                  rocsparse_status_success);

        // Invalid arguments.
        const auto create = [&](rocsparse_const_spmat_descr* d,
                                int64_t                      rows_,
                                int64_t                      cols_,
                                int64_t                      nnz_,
                                const void*                  cptr,
                                const void*                  rind,
                                const void*                  val) {
            return rocsparse_create_const_csc_descr(d,
                                                    rows_,
                                                    cols_,
                                                    nnz_,
                                                    cptr,
                                                    rind,
                                                    val,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_index_base_zero,
                                                    rocsparse_datatype_f32_r);
        };
        const int32_t*              i   = b.i32.data();
        const float*                x   = b.v32.data();
        rocsparse_const_spmat_descr bad = nullptr;
        EXPECT_EQ(create(nullptr, 3, 3, 2, i, i, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, -1, 3, 2, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 3, -1, 2, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 3, 3, -1, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 3, 3, 10, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 3, 3, 2, nullptr, i, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, 3, 3, 2, i, nullptr, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, 3, 3, 2, i, i, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(bad, nullptr);
    }

    TEST(Cov70ConvAux, ConstBsrRoundTripAndNnz)
    {
        Buf                         b;
        rocsparse_const_spmat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_const_bsr_descr(&descr,
                                                   3,
                                                   4,
                                                   5,
                                                   rocsparse_direction_column,
                                                   2,
                                                   b.i32.data(),
                                                   b.j32.data(),
                                                   b.v32.data(),
                                                   rocsparse_indextype_i32,
                                                   rocsparse_indextype_i32,
                                                   rocsparse_index_base_one,
                                                   rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t              brows = 0, bcols = 0, bnnz = 0, bdim = 0;
        rocsparse_direction  dir = rocsparse_direction_row;
        const void *         rp = nullptr, *ci = nullptr, *v = nullptr;
        rocsparse_indextype  rpt  = rocsparse_indextype_i64;
        rocsparse_indextype  cit  = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_zero;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(
            rocsparse_const_bsr_get(
                descr, &brows, &bcols, &bnnz, &dir, &bdim, &rp, &ci, &v, &rpt, &cit, &base, &dt),
            rocsparse_status_success);
        EXPECT_EQ(brows, 3);
        EXPECT_EQ(bcols, 4);
        EXPECT_EQ(bnnz, 5);
        EXPECT_EQ(dir, rocsparse_direction_column);
        EXPECT_EQ(bdim, 2);
        EXPECT_EQ(rp, static_cast<const void*>(b.i32.data()));
        EXPECT_EQ(ci, static_cast<const void*>(b.j32.data()));
        EXPECT_EQ(v, static_cast<const void*>(b.v32.data()));
        EXPECT_EQ(rpt, rocsparse_indextype_i32);
        EXPECT_EQ(cit, rocsparse_indextype_i32);
        EXPECT_EQ(base, rocsparse_index_base_one);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);

        // The nnz of a BSR matrix is the number of blocks times block_dim^2.
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr, &nnz), rocsparse_status_success);
        EXPECT_EQ(nnz, 5 * 2 * 2);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(const_cast<rocsparse_spmat_descr>(descr)),
                  rocsparse_status_success);

        // Invalid arguments.
        const int32_t*              i      = b.i32.data();
        const float*                x      = b.v32.data();
        rocsparse_const_spmat_descr bad    = nullptr;
        const auto                  create = [&](rocsparse_const_spmat_descr* d,
                                int64_t                      brows_,
                                int64_t                      bcols_,
                                int64_t                      bnnz_,
                                rocsparse_direction          dir_,
                                int64_t                      bd,
                                const void*                  rptr,
                                const void*                  cind,
                                const void*                  val) {
            return rocsparse_create_const_bsr_descr(d,
                                                    brows_,
                                                    bcols_,
                                                    bnnz_,
                                                    dir_,
                                                    bd,
                                                    rptr,
                                                    cind,
                                                    val,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_index_base_zero,
                                                    rocsparse_datatype_f32_r);
        };
        const auto row = rocsparse_direction_row;
        EXPECT_EQ(create(nullptr, 2, 2, 2, row, 2, i, i, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, -1, 2, 2, row, 2, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 2, -1, 2, row, 2, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 2, 2, -1, row, 2, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 2, 2, 5, row, 2, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 2, 2, 2, static_cast<rocsparse_direction>(100), 2, i, i, x),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(create(&bad, 2, 2, 2, row, -1, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 2, 2, 2, row, 0, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 2, 2, 2, row, 2, nullptr, i, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, 2, 2, 2, row, 2, i, nullptr, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, 2, 2, 2, row, 2, i, i, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(bad, nullptr);
    }

    TEST(Cov70ConvAux, ConstBellRoundTripAndNnz)
    {
        Buf                         b;
        rocsparse_const_spmat_descr descr = nullptr;
        // rows = 6, cols = 8, block_dim = 2, ell_cols = 4 -> 3 block rows.
        ASSERT_EQ(rocsparse_create_const_bell_descr(&descr,
                                                    6,
                                                    8,
                                                    rocsparse_direction_row,
                                                    2,
                                                    4,
                                                    b.i32.data(),
                                                    b.v32.data(),
                                                    rocsparse_indextype_i32,
                                                    rocsparse_index_base_zero,
                                                    rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t              rows = 0, cols = 0, bdim = 0, ell_cols = 0;
        rocsparse_direction  dir = rocsparse_direction_column;
        const void *         ci = nullptr, *v = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_one;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_bell_get(
                      descr, &rows, &cols, &dir, &bdim, &ell_cols, &ci, &v, &it, &base, &dt),
                  rocsparse_status_success);
        EXPECT_EQ(rows, 6);
        EXPECT_EQ(cols, 8);
        EXPECT_EQ(dir, rocsparse_direction_row);
        EXPECT_EQ(bdim, 2);
        EXPECT_EQ(ell_cols, 4);
        EXPECT_EQ(ci, static_cast<const void*>(b.i32.data()));
        EXPECT_EQ(v, static_cast<const void*>(b.v32.data()));
        EXPECT_EQ(it, rocsparse_indextype_i32);
        EXPECT_EQ(base, rocsparse_index_base_zero);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);

        // Pins the library definition: ell_cols * rows * block_dim^2.
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr, &nnz), rocsparse_status_success);
        EXPECT_EQ(nnz, ell_cols * rows * bdim * bdim);

        // The number of non-zeros of a blocked ELL matrix cannot be set.
        EXPECT_EQ(rocsparse_spmat_set_nnz(const_cast<rocsparse_spmat_descr>(descr), 3),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(const_cast<rocsparse_spmat_descr>(descr)),
                  rocsparse_status_success);

        // Invalid arguments.
        const int32_t*              i      = b.i32.data();
        const float*                x      = b.v32.data();
        rocsparse_const_spmat_descr bad    = nullptr;
        const auto                  create = [&](rocsparse_const_spmat_descr* d,
                                int64_t                      rows_,
                                int64_t                      cols_,
                                rocsparse_direction          dir_,
                                int64_t                      bd,
                                int64_t                      ec,
                                const void*                  cind,
                                const void*                  val) {
            return rocsparse_create_const_bell_descr(d,
                                                     rows_,
                                                     cols_,
                                                     dir_,
                                                     bd,
                                                     ec,
                                                     cind,
                                                     val,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_index_base_zero,
                                                     rocsparse_datatype_f32_r);
        };
        const auto row = rocsparse_direction_row;
        EXPECT_EQ(create(nullptr, 4, 4, row, 2, 2, i, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, -1, 4, row, 2, 2, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 4, -1, row, 2, 2, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 4, 4, static_cast<rocsparse_direction>(100), 2, 2, i, x),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(create(&bad, 4, 4, row, -1, 2, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 4, 4, row, 0, 2, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 4, 4, row, 2, -1, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 4, 4, row, 2, 5, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 4, 4, row, 2, 2, nullptr, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, 4, 4, row, 2, 2, i, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(bad, nullptr);
    }

    TEST(Cov70ConvAux, ConstSellRoundTrip)
    {
        Buf                         b;
        rocsparse_const_spmat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_const_sell_descr(&descr,
                                                    8,
                                                    9,
                                                    10,
                                                    4,
                                                    16,
                                                    b.i32.data(),
                                                    b.j32.data(),
                                                    b.v32.data(),
                                                    rocsparse_indextype_i32,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_index_base_one,
                                                    rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t              rows = 0, cols = 0, nnz = 0, slice = 0, colval = 0;
        const void *         so = nullptr, *ci = nullptr, *v = nullptr;
        rocsparse_indextype  sot  = rocsparse_indextype_i64;
        rocsparse_indextype  cit  = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_zero;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(
            rocsparse_const_sell_get(
                descr, &rows, &cols, &nnz, &slice, &colval, &so, &ci, &v, &sot, &cit, &base, &dt),
            rocsparse_status_success);
        EXPECT_EQ(rows, 8);
        EXPECT_EQ(cols, 9);
        EXPECT_EQ(nnz, 10);
        EXPECT_EQ(slice, 4);
        EXPECT_EQ(colval, 16);
        EXPECT_EQ(so, static_cast<const void*>(b.i32.data()));
        EXPECT_EQ(ci, static_cast<const void*>(b.j32.data()));
        EXPECT_EQ(v, static_cast<const void*>(b.v32.data()));
        EXPECT_EQ(sot, rocsparse_indextype_i32);
        EXPECT_EQ(cit, rocsparse_indextype_i32);
        EXPECT_EQ(base, rocsparse_index_base_one);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);

        // The nnz of a SELL matrix can be queried and updated.
        int64_t got = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr, &got), rocsparse_status_success);
        EXPECT_EQ(got, 10);
        EXPECT_EQ(rocsparse_spmat_set_nnz(const_cast<rocsparse_spmat_descr>(descr), 7),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr, &got), rocsparse_status_success);
        EXPECT_EQ(got, 7);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(const_cast<rocsparse_spmat_descr>(descr)),
                  rocsparse_status_success);

        // Invalid arguments.
        const int32_t*              i      = b.i32.data();
        const float*                x      = b.v32.data();
        rocsparse_const_spmat_descr bad    = nullptr;
        const auto                  create = [&](rocsparse_const_spmat_descr* d,
                                int64_t                      rows_,
                                int64_t                      cols_,
                                int64_t                      nnz_,
                                int64_t                      ss,
                                int64_t                      cvs,
                                const void*                  off,
                                const void*                  cind,
                                const void*                  val) {
            return rocsparse_create_const_sell_descr(d,
                                                     rows_,
                                                     cols_,
                                                     nnz_,
                                                     ss,
                                                     cvs,
                                                     off,
                                                     cind,
                                                     val,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_index_base_zero,
                                                     rocsparse_datatype_f32_r);
        };
        EXPECT_EQ(create(nullptr, 8, 8, 4, 4, 8, i, i, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, -1, 8, 4, 4, 8, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, -1, 4, 4, 8, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, 8, -1, 4, 8, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, 8, 4, -1, 8, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, 8, 4, 0, 8, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, 8, 4, 4, -1, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, 8, 9, 4, 8, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, 8, 4, 9, 8, i, i, x), rocsparse_status_invalid_size);
        EXPECT_EQ(create(&bad, 8, 8, 4, 4, 8, nullptr, i, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, 8, 8, 4, 4, 8, i, nullptr, x), rocsparse_status_invalid_pointer);
        EXPECT_EQ(create(&bad, 8, 8, 4, 4, 8, i, i, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(bad, nullptr);
    }

    TEST(Cov70ConvAux, ConstSellGetInvalidPointers)
    {
        Buf                         b;
        rocsparse_const_spmat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_const_sell_descr(&descr,
                                                    4,
                                                    4,
                                                    4,
                                                    4,
                                                    4,
                                                    b.i32.data(),
                                                    b.j32.data(),
                                                    b.v32.data(),
                                                    rocsparse_indextype_i32,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_index_base_zero,
                                                    rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        int64_t              rows = 0, cols = 0, nnz = 0, slice = 0, colval = 0;
        const void *         so = nullptr, *ci = nullptr, *v = nullptr;
        rocsparse_indextype  sot  = rocsparse_indextype_i32;
        rocsparse_indextype  cit  = rocsparse_indextype_i32;
        rocsparse_index_base base = rocsparse_index_base_zero;
        rocsparse_datatype   dt   = rocsparse_datatype_f32_r;
        EXPECT_EQ(
            rocsparse_const_sell_get(
                nullptr, &rows, &cols, &nnz, &slice, &colval, &so, &ci, &v, &sot, &cit, &base, &dt),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_const_sell_get(
                descr, nullptr, &cols, &nnz, &slice, &colval, &so, &ci, &v, &sot, &cit, &base, &dt),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_const_sell_get(
                descr, &rows, nullptr, &nnz, &slice, &colval, &so, &ci, &v, &sot, &cit, &base, &dt),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_const_sell_get(descr,
                                           &rows,
                                           &cols,
                                           nullptr,
                                           &slice,
                                           &colval,
                                           &so,
                                           &ci,
                                           &v,
                                           &sot,
                                           &cit,
                                           &base,
                                           &dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_const_sell_get(
                descr, &rows, &cols, &nnz, nullptr, &colval, &so, &ci, &v, &sot, &cit, &base, &dt),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_const_sell_get(
                descr, &rows, &cols, &nnz, &slice, nullptr, &so, &ci, &v, &sot, &cit, &base, &dt),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_const_sell_get(descr,
                                           &rows,
                                           &cols,
                                           &nnz,
                                           &slice,
                                           &colval,
                                           nullptr,
                                           &ci,
                                           &v,
                                           &sot,
                                           &cit,
                                           &base,
                                           &dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_const_sell_get(descr,
                                           &rows,
                                           &cols,
                                           &nnz,
                                           &slice,
                                           &colval,
                                           &so,
                                           nullptr,
                                           &v,
                                           &sot,
                                           &cit,
                                           &base,
                                           &dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_const_sell_get(descr,
                                           &rows,
                                           &cols,
                                           &nnz,
                                           &slice,
                                           &colval,
                                           &so,
                                           &ci,
                                           nullptr,
                                           &sot,
                                           &cit,
                                           &base,
                                           &dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_const_sell_get(descr,
                                           &rows,
                                           &cols,
                                           &nnz,
                                           &slice,
                                           &colval,
                                           &so,
                                           &ci,
                                           &v,
                                           nullptr,
                                           &cit,
                                           &base,
                                           &dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_const_sell_get(descr,
                                           &rows,
                                           &cols,
                                           &nnz,
                                           &slice,
                                           &colval,
                                           &so,
                                           &ci,
                                           &v,
                                           &sot,
                                           nullptr,
                                           &base,
                                           &dt),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_const_sell_get(
                descr, &rows, &cols, &nnz, &slice, &colval, &so, &ci, &v, &sot, &cit, nullptr, &dt),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_const_sell_get(descr,
                                           &rows,
                                           &cols,
                                           &nnz,
                                           &slice,
                                           &colval,
                                           &so,
                                           &ci,
                                           &v,
                                           &sot,
                                           &cit,
                                           &base,
                                           nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(const_cast<rocsparse_spmat_descr>(descr)),
                  rocsparse_status_success);
    }

    TEST(Cov70ConvAux, BellSetPointers)
    {
        Buf                   b;
        std::vector<int32_t>  other_idx(64, 1);
        std::vector<float>    other_val(64, 2.0f);
        rocsparse_spmat_descr descr = nullptr;
        // 4 rows, block_dim 2 -> 2 block rows, ell_cols 4 -> 2 blocks per block row.
        ASSERT_EQ(rocsparse_create_bell_descr(&descr,
                                              4,
                                              6,
                                              rocsparse_direction_row,
                                              2,
                                              4,
                                              b.i32.data(),
                                              b.v32.data(),
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        EXPECT_EQ(rocsparse_bell_set_pointers(nullptr, other_idx.data(), other_val.data()),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_bell_set_pointers(descr, nullptr, other_val.data()),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_bell_set_pointers(descr, other_idx.data(), nullptr),
                  rocsparse_status_invalid_pointer);
        ASSERT_EQ(rocsparse_bell_set_pointers(descr, other_idx.data(), other_val.data()),
                  rocsparse_status_success);

        int64_t              rows = 0, cols = 0, bdim = 0, ell_cols = 0;
        rocsparse_direction  dir = rocsparse_direction_column;
        void *               ci = nullptr, *v = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_one;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_bell_get(
                      descr, &rows, &cols, &dir, &bdim, &ell_cols, &ci, &v, &it, &base, &dt),
                  rocsparse_status_success);
        EXPECT_EQ(ci, static_cast<void*>(other_idx.data()));
        EXPECT_EQ(v, static_cast<void*>(other_val.data()));
        EXPECT_EQ(rows, 4);
        EXPECT_EQ(cols, 6);
        EXPECT_EQ(bdim, 2);
        EXPECT_EQ(ell_cols, 4);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
    }

    TEST(Cov70ConvAux, EllNnzAndStridedBatch)
    {
        Buf                   b;
        rocsparse_spmat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                             5,
                                             7,
                                             b.i32.data(),
                                             b.v32.data(),
                                             3,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        // ELL: nnz is ell_width * rows and cannot be set.
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr, &nnz), rocsparse_status_success);
        EXPECT_EQ(nnz, 3 * 5);
        EXPECT_EQ(rocsparse_spmat_set_nnz(descr, 4), rocsparse_status_invalid_value);

        EXPECT_EQ(rocsparse_ell_set_strided_batch(nullptr, 2, 10),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_ell_set_strided_batch(descr, 0, 10), rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_ell_set_strided_batch(descr, 2, -1), rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_ell_set_strided_batch(descr, 1, 5), rocsparse_status_invalid_value);
        ASSERT_EQ(rocsparse_ell_set_strided_batch(descr, 3, 20), rocsparse_status_success);
        rocsparse_int batch_count = 0;
        EXPECT_EQ(rocsparse_spmat_get_strided_batch(descr, &batch_count), rocsparse_status_success);
        EXPECT_EQ(batch_count, 3);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
    }

    TEST(Cov70ConvAux, SpmatSetNnzPerFormat)
    {
        Buf                   b;
        rocsparse_spmat_descr csr = nullptr, csc = nullptr, coo = nullptr, bsr = nullptr,
                              aos = nullptr;
        ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                             4,
                                             4,
                                             8,
                                             b.i32.data(),
                                             b.j32.data(),
                                             b.v32.data(),
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csc_descr(&csc,
                                             4,
                                             4,
                                             8,
                                             b.i32.data(),
                                             b.j32.data(),
                                             b.v32.data(),
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_descr(&coo,
                                             4,
                                             4,
                                             8,
                                             b.i32.data(),
                                             b.j32.data(),
                                             b.v32.data(),
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(&bsr,
                                             2,
                                             2,
                                             4,
                                             rocsparse_direction_row,
                                             3,
                                             b.i32.data(),
                                             b.j32.data(),
                                             b.v32.data(),
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&aos,
                                                 4,
                                                 4,
                                                 8,
                                                 b.i32.data(),
                                                 b.v32.data(),
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        const struct
        {
            rocsparse_spmat_descr d;
            int64_t               scale;
        } cases[] = {{csr, 1}, {csc, 1}, {coo, 1}, {aos, 1}, {bsr, 3 * 3}};
        for(const auto& c : cases)
        {
            ASSERT_EQ(rocsparse_spmat_set_nnz(c.d, 3), rocsparse_status_success);
            int64_t nnz = 0;
            ASSERT_EQ(rocsparse_spmat_get_nnz(c.d, &nnz), rocsparse_status_success);
            EXPECT_EQ(nnz, 3 * c.scale);
            EXPECT_EQ(rocsparse_spmat_set_nnz(c.d, -1), rocsparse_status_invalid_size);
        }
        EXPECT_EQ(rocsparse_spmat_set_nnz(nullptr, 3), rocsparse_status_invalid_pointer);
        int64_t out = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(nullptr, &out), rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spmat_get_nnz(csr, nullptr), rocsparse_status_invalid_pointer);

        for(rocsparse_spmat_descr d : {csr, csc, coo, bsr, aos})
        {
            EXPECT_EQ(rocsparse_destroy_spmat_descr(d), rocsparse_status_success);
        }
    }

#ifdef ROCSPARSE_WITH_SPMAT_SCALE
    TEST_F(HandleTest, Cov70ConvAuxDnvecScalar)
    {
        float alpha = 2.5f;
        // Read-only descriptor, host pointer mode.
        rocsparse_dnvec_descr descr = nullptr;
        ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &descr,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f32_r,
                                                      &alpha,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_success);
        int64_t            size = 0;
        const void*        vals = nullptr;
        rocsparse_datatype dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_dnvec_get(descr, &size, &vals, &dt), rocsparse_status_success);
        EXPECT_EQ(size, 1);
        EXPECT_EQ(vals, static_cast<const void*>(&alpha));
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);

        // Read-write descriptor in device pointer mode: values must alias const_values.
        descr = nullptr;
        ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &descr,
                                                      rocsparse_pointer_mode_device,
                                                      rocsparse_datatype_f32_r,
                                                      &alpha,
                                                      &alpha,
                                                      nullptr),
                  rocsparse_status_success);
        void* rw = nullptr;
        ASSERT_EQ(rocsparse_dnvec_get(descr, &size, &rw, &dt), rocsparse_status_success);
        EXPECT_EQ(size, 1);
        EXPECT_EQ(rw, static_cast<void*>(&alpha));
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);

        // Invalid arguments.
        float other = 1.0f;
        descr       = nullptr;
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(nullptr,
                                                      &descr,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f32_r,
                                                      &alpha,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      nullptr,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f32_r,
                                                      &alpha,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &descr,
                                                      static_cast<rocsparse_pointer_mode>(100),
                                                      rocsparse_datatype_f32_r,
                                                      &alpha,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &descr,
                                                      rocsparse_pointer_mode_host,
                                                      static_cast<rocsparse_datatype>(100),
                                                      &alpha,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &descr,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f32_r,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &descr,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f32_r,
                                                      &alpha,
                                                      &other,
                                                      nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(descr, nullptr);
    }
#endif

    TEST(Cov70ConvAux, ConstDnmatRowOrderLd)
    {
        std::vector<float>          v(64, 1.0f);
        rocsparse_const_dnmat_descr descr = nullptr;
        // Row order: ld must be at least cols.
        EXPECT_EQ(rocsparse_create_const_dnmat_descr(
                      &descr, 3, 5, 4, v.data(), rocsparse_datatype_f32_r, rocsparse_order_row),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(descr, nullptr);
        ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                      &descr, 3, 5, 6, v.data(), rocsparse_datatype_f32_r, rocsparse_order_row),
                  rocsparse_status_success);
        int64_t            rows = 0, cols = 0, ld = 0;
        const void*        vals = nullptr;
        rocsparse_datatype dt   = rocsparse_datatype_f64_r;
        rocsparse_order    ord  = rocsparse_order_column;
        ASSERT_EQ(rocsparse_const_dnmat_get(descr, &rows, &cols, &ld, &vals, &dt, &ord),
                  rocsparse_status_success);
        EXPECT_EQ(rows, 3);
        EXPECT_EQ(cols, 5);
        EXPECT_EQ(ld, 6);
        EXPECT_EQ(vals, static_cast<const void*>(v.data()));
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);
        EXPECT_EQ(ord, rocsparse_order_row);
        EXPECT_EQ(rocsparse_destroy_dnmat_descr(const_cast<rocsparse_dnmat_descr>(descr)),
                  rocsparse_status_success);
    }

    TEST(Cov70ConvAux, CopyColorInfo)
    {
        rocsparse_color_info a = nullptr, b = nullptr;
        ASSERT_EQ(rocsparse_create_color_info(&a), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_color_info(&b), rocsparse_status_success);
        EXPECT_EQ(rocsparse_copy_color_info(nullptr, a), rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_copy_color_info(a, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_copy_color_info(a, a), rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_copy_color_info(a, b), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_color_info(a), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_color_info(b), rocsparse_status_success);
    }
} // namespace
