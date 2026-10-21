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

// Generic-API descriptor and auxiliary entry points: nnz / batch accessors and dense
// matrix copy and transpose. The checks pin the statuses documented in the public headers.

#include "unit_test_device_conv_common.hpp"

#include <vector>

using namespace convut;

namespace
{
    using ConvDescr = ConvTest;

    // Only the pointers are stored by the const descriptors, so host arrays are enough.
    struct Buf
    {
        int32_t i[64]{};
        int32_t j[64]{};
        float   v[64]{};
    };

    TEST_F(ConvDescr, EllNnzAndStridedBatch)
    {
        Buf   b;
        Spmat d;
        ASSERT_EQ(rocsparse_create_ell_descr(&d, 5, 7, b.i, b.v, 3, i32, base0, f32), st_ok);
        // ELL: nnz is ell_width * rows and cannot be set.
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &nnz), st_ok);
        EXPECT_EQ(nnz, 3 * 5);
        EXPECT_EQ(rocsparse_spmat_set_nnz(d, 4), st_val);

        EXPECT_EQ(rocsparse_ell_set_strided_batch(nullptr, 2, 10), st_ptr);
        // An invalid batch count or stride is an invalid value.
        EXPECT_EQ(rocsparse_ell_set_strided_batch(d, 0, 10), st_val);
        EXPECT_EQ(rocsparse_ell_set_strided_batch(d, -1, 10), st_val);
        EXPECT_EQ(rocsparse_ell_set_strided_batch(d, 2, -1), st_val);
        // A positive stride with a batch count of one is accepted.
        EXPECT_EQ(rocsparse_ell_set_strided_batch(d, 1, 5), st_ok);
        ASSERT_EQ(rocsparse_ell_set_strided_batch(d, 3, 20), st_ok);
        rocsparse_int batch_count = 0;
        EXPECT_EQ(rocsparse_spmat_get_strided_batch(d, &batch_count), st_ok);
        EXPECT_EQ(batch_count, 3);
    }

    TEST_F(ConvDescr, StridedBatchSettersAndGetters)
    {
        Buf   b;
        Spmat coo, csr, csc;
        Dnmat dnmat;
        Dnvec dnvec;
        ASSERT_EQ(rocsparse_create_coo_descr(&coo, 5, 7, 4, b.i, b.j, b.v, i32, base0, f32), st_ok);
        ASSERT_EQ(rocsparse_create_csr_descr(&csr, 5, 7, 4, b.i, b.j, b.v, i32, i32, base0, f32),
                  st_ok);
        ASSERT_EQ(rocsparse_create_csc_descr(&csc, 5, 7, 4, b.i, b.j, b.v, i32, i32, base0, f32),
                  st_ok);
        ASSERT_EQ(rocsparse_create_dnmat_descr(&dnmat, 5, 7, 5, b.v, f32, rocsparse_order_column),
                  st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&dnvec, 5, b.v, f32), st_ok);

        // A batch count that is not positive or a negative stride is an invalid value.
        for(rocsparse_int bad_count : {0, -1})
        {
            EXPECT_EQ(rocsparse_spmat_set_strided_batch(csr, bad_count), st_val);
            EXPECT_EQ(rocsparse_coo_set_strided_batch(coo, bad_count, 10), st_val);
            EXPECT_EQ(rocsparse_csr_set_strided_batch(csr, bad_count, 6, 4), st_val);
            EXPECT_EQ(rocsparse_csc_set_strided_batch(csc, bad_count, 8, 4), st_val);
            EXPECT_EQ(rocsparse_dnmat_set_strided_batch(dnmat, bad_count, 35), st_val);
            EXPECT_EQ(rocsparse_dnvec_set_strided_batch(dnvec, bad_count, 5), st_val);
        }
        EXPECT_EQ(rocsparse_coo_set_strided_batch(coo, 2, -1), st_val);
        EXPECT_EQ(rocsparse_csr_set_strided_batch(csr, 2, -1, 4), st_val);
        EXPECT_EQ(rocsparse_csr_set_strided_batch(csr, 2, 6, -1), st_val);
        EXPECT_EQ(rocsparse_csc_set_strided_batch(csc, 2, -1, 4), st_val);
        EXPECT_EQ(rocsparse_csc_set_strided_batch(csc, 2, 8, -1), st_val);
        EXPECT_EQ(rocsparse_dnmat_set_strided_batch(dnmat, 2, -1), st_val);

        // A null output of a getter is an invalid pointer.
        rocsparse_int batch_count  = 0;
        int64_t       batch_stride = 0;
        EXPECT_EQ(rocsparse_spmat_get_strided_batch(csr, nullptr), st_ptr);
        EXPECT_EQ(rocsparse_dnmat_get_strided_batch(dnmat, nullptr, &batch_stride), st_ptr);
        EXPECT_EQ(rocsparse_dnmat_get_strided_batch(dnmat, &batch_count, nullptr), st_ptr);
        EXPECT_EQ(rocsparse_dnvec_get_strided_batch(dnvec, nullptr, &batch_stride), st_ptr);
        EXPECT_EQ(rocsparse_dnvec_get_strided_batch(dnvec, &batch_count, nullptr), st_ptr);
    }
} // namespace

// rocsparse_dnmat_copy_data and rocsparse_dnmat_transpose (only declared with
// ROCSPARSE_WITH_TRSM_REFACTORING): argument validation. The scalar checks need
// ROCSPARSE_WITH_SPMAT_SCALE, which provides rocsparse_dnvec_descr_create_scalar.
#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
namespace
{
    constexpr auto col = rocsparse_order_column;

    TEST_F(ConvDescr, DnmatCopyInvalid)
    {
        DevBuf<float>  df(std::vector<float>(64, 1.0f)), df2(std::vector<float>(64, 1.0f));
        DevBuf<double> dd(std::vector<double>(64, 1.0));
        Dnmat          src, dst, wide, tall, dst_double;
        const auto     mk = [&](Dnmat& d, int64_t m, int64_t n, void* p, rocsparse_datatype dt) {
            ASSERT_EQ(rocsparse_create_dnmat_descr(&d, m, n, m, p, dt, col), st_ok);
        };
        mk(src, 4, 3, df, f32);
        mk(dst, 4, 3, df2, f32);
        mk(wide, 4, 4, df2, f32);
        mk(tall, 5, 3, df2, f32);
        mk(dst_double, 4, 3, dd, rocsparse_datatype_f64_r);

        const auto copy
            = [&](rocsparse_dnvec_descr a, rocsparse_dnmat_descr s, rocsparse_dnmat_descr d) {
                  return rocsparse_dnmat_copy_data(handle, a, s, d, nullptr);
              };
        EXPECT_EQ(rocsparse_dnmat_copy_data(nullptr, nullptr, src, dst, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(copy(nullptr, nullptr, dst), st_ptr);
        EXPECT_EQ(copy(nullptr, src, nullptr), st_ptr);
        // Different data types (mixed precision) are not implemented.
        EXPECT_EQ(copy(nullptr, src, dst_double), rocsparse_status_not_implemented);
        EXPECT_EQ(copy(nullptr, src, tall), st_val);
        EXPECT_EQ(copy(nullptr, src, wide), st_val);
        // Different batch counts.
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 2, 12), st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 3, 12), st_ok);
        EXPECT_EQ(copy(nullptr, src, dst), st_val);
        // A source batch count of one with a zero stride is broadcast.
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 1, 0), st_ok);
        EXPECT_EQ(copy(nullptr, src, dst), st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 1, 0), st_ok);

#ifdef ROCSPARSE_WITH_SPMAT_SCALE
        const auto scalar = [&](Dnvec& a, rocsparse_datatype dt, const void* p) {
            ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(
                          handle, &a, rocsparse_pointer_mode_host, dt, p, nullptr, nullptr),
                      st_ok);
        };
        // A scalar of another data type (mixed precision) is not implemented.
        double alpha_d = 2.0;
        Dnvec  alpha_double;
        scalar(alpha_double, rocsparse_datatype_f64_r, &alpha_d);
        EXPECT_EQ(copy(alpha_double, src, dst), rocsparse_status_not_implemented);
        // The scalar batch count must match the target or be one with a zero stride.
        float alpha_f[3] = {2.0f, 3.0f, 4.0f};
        Dnvec alpha;
        scalar(alpha, f32, alpha_f);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(alpha, 2, 1), st_ok);
        EXPECT_EQ(copy(alpha, src, dst), st_val);
        // The same checks when the orders differ.
        Dnmat dst_row;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&dst_row, 4, 3, 3, df2, f32, rocsparse_order_row),
                  st_ok);
        EXPECT_EQ(copy(alpha_double, src, dst_row), rocsparse_status_not_implemented);
        EXPECT_EQ(copy(alpha, src, dst_row), st_val);
#endif
    }

    TEST_F(ConvDescr, DnmatTransposeInvalid)
    {
        DevBuf<float>  df(std::vector<float>(64, 1.0f)), df2(std::vector<float>(64, 1.0f));
        DevBuf<double> dd(std::vector<double>(64, 1.0));
        Dnmat          src, dst, same, dst_double;
        const auto     mk = [&](Dnmat& d, int64_t m, int64_t n, void* p, rocsparse_datatype dt) {
            ASSERT_EQ(rocsparse_create_dnmat_descr(&d, m, n, m, p, dt, col), st_ok);
        };
        mk(src, 4, 3, df, f32);
        mk(dst, 3, 4, df2, f32);
        mk(same, 4, 3, df2, f32);
        mk(dst_double, 3, 4, dd, rocsparse_datatype_f64_r);

        const auto transpose
            = [&](rocsparse_dnvec_descr a, rocsparse_dnmat_descr s, rocsparse_dnmat_descr d) {
                  return rocsparse_dnmat_transpose(handle, a, s, d, nullptr);
              };
        EXPECT_EQ(rocsparse_dnmat_transpose(nullptr, nullptr, src, dst, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(transpose(nullptr, nullptr, dst), st_ptr);
        EXPECT_EQ(transpose(nullptr, src, nullptr), st_ptr);
        // Different data types (mixed precision) are not implemented.
        EXPECT_EQ(transpose(nullptr, src, dst_double), rocsparse_status_not_implemented);
        // The target must have the transposed dimensions.
        EXPECT_EQ(transpose(nullptr, src, same), st_val);
        // Different batch counts.
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 2, 12), st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 3, 12), st_ok);
        EXPECT_EQ(transpose(nullptr, src, dst), st_val);
        // A source batch count of one with a zero stride is broadcast.
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 1, 0), st_ok);
        EXPECT_EQ(transpose(nullptr, src, dst), st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 1, 0), st_ok);

#ifdef ROCSPARSE_WITH_SPMAT_SCALE
        const auto scalar = [&](Dnvec& a, rocsparse_datatype dt, const void* p) {
            ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(
                          handle, &a, rocsparse_pointer_mode_host, dt, p, nullptr, nullptr),
                      st_ok);
        };
        // A scalar of another data type (mixed precision) is not implemented.
        double alpha_d = 2.0;
        Dnvec  alpha_double;
        scalar(alpha_double, rocsparse_datatype_f64_r, &alpha_d);
        EXPECT_EQ(transpose(alpha_double, src, dst), rocsparse_status_not_implemented);
        // The scalar batch count must match the target or be one with a zero stride.
        float alpha_f[3] = {2.0f, 3.0f, 4.0f};
        Dnvec alpha;
        scalar(alpha, f32, alpha_f);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(alpha, 2, 1), st_ok);
        EXPECT_EQ(transpose(alpha, src, dst), st_val);
        // The same checks when the orders differ.
        Dnmat dst_row;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&dst_row, 3, 4, 4, df2, f32, rocsparse_order_row),
                  st_ok);
        EXPECT_EQ(transpose(alpha_double, src, dst_row), rocsparse_status_not_implemented);
        EXPECT_EQ(transpose(alpha, src, dst_row), st_val);
#endif
    }
} // namespace
#endif // ROCSPARSE_WITH_TRSM_REFACTORING
