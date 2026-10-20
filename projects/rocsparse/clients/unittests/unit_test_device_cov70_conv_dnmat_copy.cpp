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

// Device unit tests for rocsparse_dnmat_copy_data: contiguous copies, padded
// (2D) copies, batched layouts, scaled copies through the kernel path in both
// pointer modes and the argument validation. Each test reads the whole target
// buffer back and checks every logical element and every padding element.

#include "unit_test_device_cov70_conv_common.hpp"

#include <cstdint>
#include <vector>

using namespace rocsparse_ut;
using namespace cov70conv;

namespace
{
    struct Layout
    {
        rocsparse_order order;
        int64_t         rows;
        int64_t         cols;
        int64_t         ld;
        int             batch;
        int64_t         stride;

        size_t size() const
        {
            const int64_t outer = (order == rocsparse_order_column) ? cols : rows;
            return static_cast<size_t>((batch - 1) * stride + ld * outer);
        }
        size_t index(int b, int64_t i, int64_t j) const
        {
            return static_cast<size_t>(
                b * stride + ((order == rocsparse_order_column) ? (i + ld * j) : (i * ld + j)));
        }
    };

    enum class AlphaKind
    {
        none,
        host,
        device,
        device_batched
    };

    // Copies src -> dst through the public API and verifies every element.
    template <typename T>
    void run_copy(rocsparse_handle handle,
                  const Layout&    src,
                  const Layout&    dst,
                  AlphaKind        alpha_kind)
    {
        // Source values depend on the logical position only.
        const int      src_batches = src.batch;
        std::vector<T> hsrc(src.size(), make_val<T>(-7, -3));
        for(int b = 0; b < src_batches; ++b)
            for(int64_t i = 0; i < src.rows; ++i)
                for(int64_t j = 0; j < src.cols; ++j)
                    hsrc[src.index(b, i, j)] = make_val<T>(
                        static_cast<int>(1 + b * 100 + i * 10 + j), static_cast<int>(j - i));

        const T        sentinel = make_val<T>(-99, 99);
        std::vector<T> hdst(dst.size(), sentinel);

        device_vector<T> dsrc(hsrc), ddst(hdst);
        ASSERT_NE(dsrc.ptr, nullptr);
        ASSERT_NE(ddst.ptr, nullptr);

        rocsparse_dnmat_descr dm_src = nullptr, dm_dst = nullptr;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &dm_src, src.rows, src.cols, src.ld, dsrc.ptr, dt_of<T>(), src.order),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &dm_dst, dst.rows, dst.cols, dst.ld, ddst.ptr, dt_of<T>(), dst.order),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dm_src, src.batch, src.stride),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dm_dst, dst.batch, dst.stride),
                  rocsparse_status_success);

        rocsparse_dnvec_descr alpha = nullptr;
        std::vector<T>        halpha;
        if(alpha_kind == AlphaKind::host || alpha_kind == AlphaKind::device)
        {
            halpha = {make_val<T>(2, 1)};
        }
        else if(alpha_kind == AlphaKind::device_batched)
        {
            for(int b = 0; b < dst.batch; ++b)
                halpha.push_back(make_val<T>(2 + b, 1));
        }
        device_vector<T> dalpha(halpha);
        if(alpha_kind == AlphaKind::host)
        {
            ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                          &alpha,
                                                          rocsparse_pointer_mode_host,
                                                          dt_of<T>(),
                                                          halpha.data(),
                                                          nullptr,
                                                          nullptr),
                      rocsparse_status_success);
        }
        else if(alpha_kind != AlphaKind::none)
        {
            ASSERT_NE(dalpha.ptr, nullptr);
            ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                          &alpha,
                                                          rocsparse_pointer_mode_device,
                                                          dt_of<T>(),
                                                          dalpha.ptr,
                                                          nullptr,
                                                          nullptr),
                      rocsparse_status_success);
            if(alpha_kind == AlphaKind::device_batched)
            {
                ASSERT_EQ(rocsparse_dnvec_set_strided_batch(alpha, dst.batch, 1),
                          rocsparse_status_success);
            }
        }

        ASSERT_EQ(rocsparse_dnmat_copy_data(handle, alpha, dm_src, dm_dst, nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<T> got = to_host(ddst);
        std::vector<T>       expected(dst.size(), sentinel);
        for(int b = 0; b < dst.batch; ++b)
        {
            const int sb = (src.batch == 1) ? 0 : b;
            T         a  = make_val<T>(1, 0);
            if(alpha_kind == AlphaKind::host || alpha_kind == AlphaKind::device)
                a = halpha[0];
            else if(alpha_kind == AlphaKind::device_batched)
                a = halpha[static_cast<size_t>(b)];
            for(int64_t i = 0; i < dst.rows; ++i)
                for(int64_t j = 0; j < dst.cols; ++j)
                    expected[dst.index(b, i, j)] = a * hsrc[src.index(sb, i, j)];
        }
        ASSERT_EQ(got.size(), expected.size());
        for(size_t k = 0; k < got.size(); ++k)
        {
            EXPECT_EQ(got[k], expected[k]) << "element " << k;
        }

        EXPECT_EQ(rocsparse_destroy_dnmat_descr(dm_src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnmat_descr(dm_dst), rocsparse_status_success);
        if(alpha)
        {
            EXPECT_EQ(rocsparse_destroy_dnvec_descr(alpha), rocsparse_status_success);
        }
    }

    template <typename T>
    void run_all_layouts(rocsparse_handle handle)
    {
        const int64_t M = 5, N = 4;
        const int     B   = 3;
        const auto    col = rocsparse_order_column;
        const auto    row = rocsparse_order_row;

        // Single matrix, packed and padded.
        run_copy<T>(handle, {col, M, N, M, 1, 0}, {col, M, N, M, 1, 0}, AlphaKind::none);
        run_copy<T>(handle, {col, M, N, M + 3, 1, 0}, {col, M, N, M + 2, 1, 0}, AlphaKind::none);
        run_copy<T>(handle, {row, M, N, N, 1, 0}, {row, M, N, N + 1, 1, 0}, AlphaKind::none);

        // Batched, contiguous matrices (stride is a multiple of the matrix size).
        run_copy<T>(handle, {col, M, N, M, B, M * N}, {col, M, N, M, B, M * N}, AlphaKind::none);
        // Batched, contiguous matrices, padded target stride.
        run_copy<T>(
            handle, {col, M, N, M, B, M * N}, {col, M, N, M, B, M * N + 5}, AlphaKind::none);
        run_copy<T>(handle, {row, M, N, N, B, M * N}, {row, M, N, N, B, M * N}, AlphaKind::none);

        // Batched, interleaved matrices.
        run_copy<T>(handle, {col, M, N, M * B, B, M}, {col, M, N, M * B, B, M}, AlphaKind::none);
        run_copy<T>(
            handle, {col, M, N, M * B + 2, B, M}, {col, M, N, M * B + 2, B, M}, AlphaKind::none);

        // Source and target use different batch layouts, which needs the kernel.
        run_copy<T>(handle, {col, M, N, M, B, M * N}, {col, M, N, M * B, B, M}, AlphaKind::none);
        // One source matrix broadcast into every batch of the target.
        run_copy<T>(handle, {col, M, N, M, 1, 0}, {col, M, N, M, B, M * N}, AlphaKind::none);

        // Scaled copies take the kernel.
        run_copy<T>(handle, {col, M, N, M, 1, 0}, {col, M, N, M, 1, 0}, AlphaKind::host);
        run_copy<T>(handle, {col, M, N, M + 1, 1, 0}, {col, M, N, M, 1, 0}, AlphaKind::device);
        run_copy<T>(handle, {row, M, N, N, 1, 0}, {row, M, N, N + 2, 1, 0}, AlphaKind::host);
        run_copy<T>(handle, {col, M, N, M, B, M * N}, {col, M, N, M, B, M * N}, AlphaKind::device);
        run_copy<T>(
            handle, {col, M, N, M, B, M * N}, {col, M, N, M, B, M * N}, AlphaKind::device_batched);
        run_copy<T>(handle, {col, M, N, M, 1, 0}, {col, M, N, M, B, M * N}, AlphaKind::host);

        // Different orders go through the order switch.
        run_copy<T>(handle, {col, M, N, M, 1, 0}, {row, M, N, N, 1, 0}, AlphaKind::none);
        run_copy<T>(handle, {row, M, N, N, 1, 0}, {col, M, N, M, 1, 0}, AlphaKind::host);
    }

    TEST_F(HandleTest, Cov70ConvDnmatCopyDataFloat)
    {
        run_all_layouts<float>(handle);
    }
    TEST_F(HandleTest, Cov70ConvDnmatCopyDataDouble)
    {
        run_all_layouts<double>(handle);
    }
    TEST_F(HandleTest, Cov70ConvDnmatCopyDataFloatComplex)
    {
        run_all_layouts<rocsparse_float_complex>(handle);
    }
    TEST_F(HandleTest, Cov70ConvDnmatCopyDataDoubleComplex)
    {
        run_all_layouts<rocsparse_double_complex>(handle);
    }

    TEST_F(HandleTest, Cov70ConvDnmatCopyDataInvalid)
    {
        std::vector<float>    hf(64, 1.0f);
        std::vector<double>   hd(64, 1.0);
        device_vector<float>  df(hf), df2(hf);
        device_vector<double> dd(hd);
        ASSERT_NE(df.ptr, nullptr);
        ASSERT_NE(df2.ptr, nullptr);
        ASSERT_NE(dd.ptr, nullptr);

        const auto            col = rocsparse_order_column;
        rocsparse_dnmat_descr src = nullptr, dst = nullptr, dst_wide = nullptr, dst_tall = nullptr,
                              dst_double = nullptr;
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&src, 4, 3, 4, df.ptr, rocsparse_datatype_f32_r, col),
            rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&dst, 4, 3, 4, df2.ptr, rocsparse_datatype_f32_r, col),
            rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &dst_wide, 4, 4, 4, df2.ptr, rocsparse_datatype_f32_r, col),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &dst_tall, 5, 3, 5, df2.ptr, rocsparse_datatype_f32_r, col),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &dst_double, 4, 3, 4, dd.ptr, rocsparse_datatype_f64_r, col),
                  rocsparse_status_success);

        EXPECT_EQ(rocsparse_dnmat_copy_data(nullptr, nullptr, src, dst, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, nullptr, nullptr, dst, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, nullptr, src, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        // Different data types.
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, nullptr, src, dst_double, nullptr),
                  rocsparse_status_not_implemented);
        // Different shapes.
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, nullptr, src, dst_tall, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, nullptr, src, dst_wide, nullptr),
                  rocsparse_status_invalid_value);

        // Different batch counts.
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 2, 12), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 3, 12), rocsparse_status_success);
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, nullptr, src, dst, nullptr),
                  rocsparse_status_invalid_value);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 1, 0), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 1, 0), rocsparse_status_success);

        // The scalar must have the data type of the matrices.
        double                alpha_d = 2.0;
        rocsparse_dnvec_descr alpha   = nullptr;
        ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &alpha,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f64_r,
                                                      &alpha_d,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, alpha, src, dst, nullptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(alpha), rocsparse_status_success);

        // The scalar batch count must match the target or be one with a zero stride.
        float alpha_f[3] = {2.0f, 3.0f, 4.0f};
        alpha            = nullptr;
        ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &alpha,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f32_r,
                                                      alpha_f,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(alpha, 2, 1), rocsparse_status_success);
        EXPECT_EQ(rocsparse_dnmat_copy_data(handle, alpha, src, dst, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(alpha), rocsparse_status_success);

        for(rocsparse_dnmat_descr d : {src, dst, dst_wide, dst_tall, dst_double})
        {
            EXPECT_EQ(rocsparse_destroy_dnmat_descr(d), rocsparse_status_success);
        }
    }
} // namespace
