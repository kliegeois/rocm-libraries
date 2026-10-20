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

// Internal dense helpers of the library (dense transpose and back transpose, datatype
// utilities) and the dense matrix transposition of the public
// auxiliary interface (rocsparse_dnmat_transpose). Every data movement is checked
// against a host reference; the matrices hold small integers so that the products by
// the scalars are exact.

#include "unit_test_device_cov70_precond_common.hpp"

#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_dense_transpose.hpp"
#include "rocsparse_dense_transpose_back.hpp"

#include <complex>
#include <string>

using namespace cov70;

namespace
{
    template <typename F>
    void for_all_types(F&& f)
    {
        f(float{});
        f(double{});
        f(rocsparse_float_complex{});
        f(rocsparse_double_complex{});
    }

    // Small integer valued entry (the imaginary part only for complex types).
    template <typename T>
    T entry(int64_t i, int64_t j, int64_t b)
    {
        return make_value<T>(static_cast<double>((i * 7 + j * 3 + b * 5) % 11 - 5),
                             static_cast<double>((i * 5 + j * 2 + b) % 7 - 3));
    }

    template <typename T>
    T product(T a, T b)
    {
        return from_cd<T>(to_cd(a) * to_cd(b));
    }

    // The scalar of batch b: 1 or a small integer.
    template <typename T>
    T scalar_of(int64_t b)
    {
        return make_value<T>(static_cast<double>(b + 2), is_cplx<T> ? -1.0 : 0.0);
    }

    constexpr double exact = 1.0e-6;

    struct dim_pair
    {
        int64_t m, n;
    };

    constexpr dim_pair shapes[] = {{37, 53}, {32, 32}, {65, 1}, {1, 40}, {100, 33}, {9, 70}};
}

class Cov80PrecondDense : public HandleTest
{
};

TEST_F(Cov80PrecondDense, datatype_utilities)
{
    EXPECT_EQ(rocsparse::get_datatype<_Float16>(), rocsparse_datatype_f16_r);
    EXPECT_EQ(rocsparse::get_datatype<rocsparse_bfloat16>(), rocsparse_datatype_bf16_r);
    EXPECT_EQ(rocsparse::get_datatype<float>(), rocsparse_datatype_f32_r);
    EXPECT_EQ(rocsparse::get_datatype<double>(), rocsparse_datatype_f64_r);
    EXPECT_EQ(rocsparse::get_datatype<rocsparse_float_complex>(), rocsparse_datatype_f32_c);
    EXPECT_EQ(rocsparse::get_datatype<rocsparse_double_complex>(), rocsparse_datatype_f64_c);

    const struct
    {
        rocsparse_datatype type;
        size_t             size;
        bool               complex;
    } types[] = {{rocsparse_datatype_i32_r, sizeof(int32_t), false},
                 {rocsparse_datatype_u32_r, sizeof(uint32_t), false},
                 {rocsparse_datatype_i8_r, sizeof(int8_t), false},
                 {rocsparse_datatype_u8_r, sizeof(uint8_t), false},
                 {rocsparse_datatype_f16_r, 2, false},
                 {rocsparse_datatype_bf16_r, 2, false},
                 {rocsparse_datatype_f32_r, sizeof(float), false},
                 {rocsparse_datatype_f64_r, sizeof(double), false},
                 {rocsparse_datatype_f32_c, 2 * sizeof(float), true},
                 {rocsparse_datatype_f64_c, 2 * sizeof(double), true}};
    for(const auto& t : types)
    {
        SCOPED_TRACE("datatype " + std::to_string(static_cast<int>(t.type)));
        EXPECT_EQ(rocsparse::datatype_sizeof(t.type), t.size);
        EXPECT_EQ(rocsparse::datatype_is_complex(t.type), t.complex);
    }
}

TEST_F(Cov80PrecondDense, dense_transpose_with_host_and_device_scalars)
{
    for_all_types([&](auto tag) {
        using T          = decltype(tag);
        const auto dtype = rocsparse::get_datatype<T>();
        for(const auto& s : shapes)
        {
            for(const int64_t batch : {int64_t{1}, int64_t{3}})
            {
                SCOPED_TRACE("m " + std::to_string(s.m) + " n " + std::to_string(s.n) + " batch "
                             + std::to_string(batch));
                const int64_t  lda = s.m + 3, ldb = s.n + 2;
                const int64_t  a_stride = lda * s.n, b_stride = ldb * s.m;
                std::vector<T> a(static_cast<size_t>(batch * a_stride));
                for(int64_t b = 0; b < batch; ++b)
                    for(int64_t j = 0; j < s.n; ++j)
                        for(int64_t i = 0; i < lda; ++i)
                            a[static_cast<size_t>(b * a_stride + i + lda * j)] = entry<T>(i, j, b);
                const std::vector<T> sentinel(static_cast<size_t>(batch * b_stride),
                                              make_value<T>(-99.0, is_cplx<T> ? 98.0 : 0.0));

                // Scalars of the batch: one per entry of the batch in device mode and one
                // for the whole batch in host mode.
                std::vector<T> alphas(static_cast<size_t>(batch));
                for(int64_t b = 0; b < batch; ++b)
                    alphas[static_cast<size_t>(b)] = scalar_of<T>(b);

                for(const bool host_mode : {true, false})
                {
                    SCOPED_TRACE(host_mode ? "host scalar" : "device scalar");
                    device_vector<T> da{a};
                    device_vector<T> db{sentinel};
                    device_vector<T> dalpha{alphas};
                    ASSERT_TRUE(da.ptr && db.ptr && dalpha.ptr);
                    const T host_alpha = alphas[0];

                    ASSERT_EQ(
                        rocsparse::dense_transpose_strided_batched(
                            handle,
                            host_mode ? rocsparse_pointer_mode_host : rocsparse_pointer_mode_device,
                            batch,
                            s.m,
                            s.n,
                            dtype,
                            host_mode ? static_cast<const void*>(&host_alpha)
                                      : static_cast<const void*>(dalpha.ptr),
                            1,
                            dtype,
                            da.ptr,
                            lda,
                            a_stride,
                            dtype,
                            db.ptr,
                            ldb,
                            b_stride),
                        rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                    std::vector<T> want = sentinel;
                    for(int64_t b = 0; b < batch; ++b)
                        for(int64_t i = 0; i < s.m; ++i)
                            for(int64_t j = 0; j < s.n; ++j)
                                want[static_cast<size_t>(b * b_stride + j + ldb * i)] = product(
                                    host_mode ? host_alpha : alphas[static_cast<size_t>(b)],
                                    a[static_cast<size_t>(b * a_stride + i + lda * j)]);
                    expect_close(to_host(db), want, exact);
                }
            }
        }
    });
}

TEST_F(Cov80PrecondDense, dense_transpose_single_matrix_entry_points)
{
    for_all_types([&](auto tag) {
        using T              = decltype(tag);
        const auto     dtype = rocsparse::get_datatype<T>();
        const int64_t  m = 41, n = 29, lda = m + 1, ldb = n + 4;
        std::vector<T> a(static_cast<size_t>(lda * n));
        for(int64_t j = 0; j < n; ++j)
            for(int64_t i = 0; i < lda; ++i)
                a[static_cast<size_t>(i + lda * j)] = entry<T>(i, j, 2);
        const std::vector<T> sentinel(static_cast<size_t>(ldb * m),
                                      make_value<T>(-99.0, is_cplx<T> ? 98.0 : 0.0));
        const T              alpha = scalar_of<T>(3);

        auto expected = [&](T scale) {
            std::vector<T> want = sentinel;
            for(int64_t i = 0; i < m; ++i)
                for(int64_t j = 0; j < n; ++j)
                    want[static_cast<size_t>(j + ldb * i)]
                        = product(scale, a[static_cast<size_t>(i + lda * j)]);
            return want;
        };

        device_vector<T> da{a};
        ASSERT_TRUE(da.ptr);

        // No scalar: a plain transposition.
        {
            device_vector<T> db{sentinel};
            ASSERT_TRUE(db.ptr);
            ASSERT_EQ(
                rocsparse::dense_transpose(handle, m, n, dtype, da.ptr, lda, dtype, db.ptr, ldb),
                rocsparse_status_success);
            expect_close(to_host(db), expected(make_value<T>(1.0)), exact);
        }

        // Scalar in host or device memory.
        for(const bool host_mode : {true, false})
        {
            SCOPED_TRACE(host_mode ? "host scalar" : "device scalar");
            device_vector<T> db{sentinel};
            device_vector<T> dalpha{std::vector<T>{alpha}};
            ASSERT_TRUE(db.ptr && dalpha.ptr);
            ASSERT_EQ(rocsparse::dense_transpose(handle,
                                                 host_mode ? rocsparse_pointer_mode_host
                                                           : rocsparse_pointer_mode_device,
                                                 m,
                                                 n,
                                                 dtype,
                                                 host_mode ? static_cast<const void*>(&alpha)
                                                           : static_cast<const void*>(dalpha.ptr),
                                                 dtype,
                                                 da.ptr,
                                                 lda,
                                                 dtype,
                                                 db.ptr,
                                                 ldb),
                      rocsparse_status_success);
            expect_close(to_host(db), expected(alpha), exact);
        }

        // Typed entry point (the scalar lives on the device).
        {
            device_vector<T> db{sentinel};
            device_vector<T> dalpha{std::vector<T>{alpha}};
            ASSERT_TRUE(db.ptr && dalpha.ptr);
            ASSERT_EQ(
                rocsparse::dense_transpose(handle, m, n, dalpha.ptr, da.ptr, lda, db.ptr, ldb),
                rocsparse_status_success);
            expect_close(to_host(db), expected(alpha), exact);
        }
    });

    // Types the kernels do not handle are rejected before any launch.
    device_vector<float> x{std::vector<float>(64, 1.0f)};
    device_vector<float> y{std::vector<float>(64, 0.0f)};
    ASSERT_TRUE(x.ptr && y.ptr);
    for(const auto bad : {rocsparse_datatype_i8_r,
                          rocsparse_datatype_u8_r,
                          rocsparse_datatype_i32_r,
                          rocsparse_datatype_f16_r,
                          rocsparse_datatype_bf16_r})
    {
        EXPECT_NE(rocsparse::dense_transpose(handle, 8, 8, bad, x.ptr, 8, bad, y.ptr, 8),
                  rocsparse_status_success);
        EXPECT_NE(rocsparse::dense_transpose_back(handle, 8, 8, bad, x.ptr, 8, bad, y.ptr, 8),
                  rocsparse_status_success);
    }
}

TEST_F(Cov80PrecondDense, dense_transpose_back_of_batches)
{
    for_all_types([&](auto tag) {
        using T          = decltype(tag);
        const auto dtype = rocsparse::get_datatype<T>();
        for(const auto& s : shapes)
        {
            for(const int64_t batch : {int64_t{1}, int64_t{3}})
            {
                SCOPED_TRACE("m " + std::to_string(s.m) + " n " + std::to_string(s.n) + " batch "
                             + std::to_string(batch));
                // A holds the transposed matrix (n x m), B receives the matrix (m x n).
                const int64_t  lda = s.n + 2, ldb = s.m + 3;
                const int64_t  a_stride = lda * s.m, b_stride = ldb * s.n;
                std::vector<T> a(static_cast<size_t>(batch * a_stride));
                for(int64_t b = 0; b < batch; ++b)
                    for(int64_t j = 0; j < s.m; ++j)
                        for(int64_t i = 0; i < lda; ++i)
                            a[static_cast<size_t>(b * a_stride + i + lda * j)] = entry<T>(i, j, b);
                const std::vector<T> sentinel(static_cast<size_t>(batch * b_stride),
                                              make_value<T>(-99.0, is_cplx<T> ? 98.0 : 0.0));
                device_vector<T>     da{a};
                device_vector<T>     db{sentinel};
                ASSERT_TRUE(da.ptr && db.ptr);

                if(batch == 1)
                {
                    ASSERT_EQ(rocsparse::dense_transpose_back(
                                  handle, s.m, s.n, dtype, da.ptr, lda, dtype, db.ptr, ldb),
                              rocsparse_status_success);
                }
                else
                {
                    ASSERT_EQ(rocsparse::dense_transpose_back_strided_batched(handle,
                                                                              batch,
                                                                              s.m,
                                                                              s.n,
                                                                              dtype,
                                                                              da.ptr,
                                                                              lda,
                                                                              a_stride,
                                                                              dtype,
                                                                              db.ptr,
                                                                              ldb,
                                                                              b_stride),
                              rocsparse_status_success);
                }
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                std::vector<T> want = sentinel;
                for(int64_t b = 0; b < batch; ++b)
                    for(int64_t r = 0; r < s.m; ++r)
                        for(int64_t c = 0; c < s.n; ++c)
                            want[static_cast<size_t>(b * b_stride + r + ldb * c)]
                                = a[static_cast<size_t>(b * a_stride + c + lda * r)];
                expect_close(to_host(db), want, exact);
            }
        }

        // The typed entry point.
        const int64_t  m = 21, n = 34;
        std::vector<T> a(static_cast<size_t>(n * m));
        for(int64_t r = 0; r < m; ++r)
            for(int64_t c = 0; c < n; ++c)
                a[static_cast<size_t>(c + n * r)] = entry<T>(c, r, 4);
        device_vector<T> da{a};
        device_vector<T> db{std::vector<T>(static_cast<size_t>(m * n))};
        ASSERT_TRUE(da.ptr && db.ptr);
        ASSERT_EQ(rocsparse::dense_transpose_back(handle, m, n, da.ptr, n, db.ptr, m),
                  rocsparse_status_success);
        std::vector<T> want(static_cast<size_t>(m * n));
        for(int64_t r = 0; r < m; ++r)
            for(int64_t c = 0; c < n; ++c)
                want[static_cast<size_t>(r + m * c)] = a[static_cast<size_t>(c + n * r)];
        expect_close(to_host(db), want, exact);
    });
}

namespace
{
    // A dense matrix of the public interface with its device storage.
    template <typename T>
    struct dn_matrix
    {
        int64_t               rows, cols, ld, batch_stride, batch_count;
        rocsparse_order       order;
        std::vector<T>        host;
        device_vector<T>      device;
        rocsparse_dnmat_descr descr = nullptr;

        dn_matrix(int64_t               r,
                  int64_t               c,
                  rocsparse_order       o,
                  int64_t               batch,
                  const std::vector<T>* values = nullptr)
            : rows(r)
            , cols(c)
            , batch_count(batch)
            , order(o)
        {
            ld           = (o == rocsparse_order_column ? r : c) + 2;
            batch_stride = ld * (o == rocsparse_order_column ? c : r);
            host.assign(static_cast<size_t>(batch * batch_stride), make_value<T>(-99.0));
            if(values)
                host = *values;
            upload(device, host);
            EXPECT_EQ(rocsparse_create_dnmat_descr(
                          &descr, rows, cols, ld, device.ptr, rocsparse::get_datatype<T>(), order),
                      rocsparse_status_success);
            if(batch > 1)
                EXPECT_EQ(rocsparse_dnmat_set_strided_batch(
                              descr, static_cast<rocsparse_int>(batch), batch_stride),
                          rocsparse_status_success);
        }

        ~dn_matrix()
        {
            if(descr)
                (void)rocsparse_destroy_dnmat_descr(descr);
        }

        dn_matrix(const dn_matrix&) = delete;
        dn_matrix& operator=(const dn_matrix&) = delete;

        size_t position(int64_t b, int64_t i, int64_t j) const
        {
            return static_cast<size_t>(
                b * batch_stride + (order == rocsparse_order_column ? i + ld * j : i * ld + j));
        }
    };

    // The vector of the scalars of a batch.
    template <typename T>
    struct dn_scalars
    {
        std::vector<T>        host;
        device_vector<T>      device;
        rocsparse_dnvec_descr descr = nullptr;

        dn_scalars(int64_t batch, rocsparse_datatype type)
        {
            host.resize(static_cast<size_t>(batch));
            for(int64_t b = 0; b < batch; ++b)
                host[static_cast<size_t>(b)] = scalar_of<T>(b);
            upload(device, host);
            EXPECT_EQ(rocsparse_create_dnvec_descr(&descr, batch, device.ptr, type),
                      rocsparse_status_success);
            if(batch > 1)
                EXPECT_EQ(
                    rocsparse_dnvec_set_strided_batch(descr, static_cast<rocsparse_int>(batch), 1),
                    rocsparse_status_success);
        }

        ~dn_scalars()
        {
            if(descr)
                (void)rocsparse_destroy_dnvec_descr(descr);
        }

        dn_scalars(const dn_scalars&) = delete;
        dn_scalars& operator=(const dn_scalars&) = delete;
    };

    template <typename T>
    std::vector<T> fill_matrix(const dn_matrix<T>& m)
    {
        std::vector<T> values = m.host;
        for(int64_t b = 0; b < m.batch_count; ++b)
            for(int64_t i = 0; i < m.rows; ++i)
                for(int64_t j = 0; j < m.cols; ++j)
                    values[m.position(b, i, j)] = entry<T>(i, j, b);
        return values;
    }
}

TEST_F(Cov80PrecondDense, dnmat_transpose_of_every_order_and_type)
{
    for_all_types([&](auto tag) {
        using T          = decltype(tag);
        const auto dtype = rocsparse::get_datatype<T>();
        for(const auto source_order : {rocsparse_order_column, rocsparse_order_row})
        {
            for(const auto target_order : {rocsparse_order_column, rocsparse_order_row})
            {
                for(const int64_t batch : {int64_t{1}, int64_t{3}})
                {
                    for(const bool with_alpha : {false, true})
                    {
                        SCOPED_TRACE(
                            std::string(source_order == rocsparse_order_row ? "row" : "col")
                            + " -> " + (target_order == rocsparse_order_row ? "row" : "col")
                            + " batch " + std::to_string(batch)
                            + (with_alpha ? " alpha" : " no alpha"));
                        const int64_t m = 37, n = 53;

                        dn_matrix<T>         probe(m, n, source_order, batch);
                        const std::vector<T> filled = fill_matrix(probe);
                        dn_matrix<T>         src(m, n, source_order, batch, &filled);
                        dn_matrix<T>         target(n, m, target_order, batch);
                        dn_scalars<T>        alpha(batch, dtype);
                        rocsparse_error*     p_error = nullptr;

                        ASSERT_EQ(rocsparse_dnmat_transpose(handle,
                                                            with_alpha ? alpha.descr : nullptr,
                                                            src.descr,
                                                            target.descr,
                                                            p_error),
                                  rocsparse_status_success);
                        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                        std::vector<T> got  = to_host(target.device);
                        std::vector<T> want = target.host;
                        for(int64_t b = 0; b < batch; ++b)
                            for(int64_t i = 0; i < m; ++i)
                                for(int64_t j = 0; j < n; ++j)
                                    want[target.position(b, j, i)]
                                        = with_alpha ? product(alpha.host[static_cast<size_t>(b)],
                                                               filled[src.position(b, i, j)])
                                                     : filled[src.position(b, i, j)];
                        expect_close(got, want, exact);
                    }
                }
            }
        }
    });
}

TEST_F(Cov80PrecondDense, dnmat_transpose_rejects_invalid_arguments)
{
    using T                  = double;
    rocsparse_error* p_error = nullptr;

    for(const auto source_order : {rocsparse_order_column, rocsparse_order_row})
    {
        for(const auto target_order : {rocsparse_order_column, rocsparse_order_row})
        {
            SCOPED_TRACE(std::string(source_order == rocsparse_order_row ? "row" : "col") + " -> "
                         + (target_order == rocsparse_order_row ? "row" : "col"));
            dn_matrix<T>  source(6, 4, source_order, 1);
            dn_matrix<T>  target(4, 6, target_order, 1);
            dn_scalars<T> alpha(1, rocsparse_datatype_f64_r);

            // Arguments.
            EXPECT_EQ(rocsparse_dnmat_transpose(
                          nullptr, alpha.descr, source.descr, target.descr, p_error),
                      rocsparse_status_invalid_handle);
            EXPECT_EQ(
                rocsparse_dnmat_transpose(handle, alpha.descr, nullptr, target.descr, p_error),
                rocsparse_status_invalid_pointer);
            EXPECT_EQ(
                rocsparse_dnmat_transpose(handle, alpha.descr, source.descr, nullptr, p_error),
                rocsparse_status_invalid_pointer);

            // A target of other dimensions.
            dn_matrix<T> wrong(4, 7, target_order, 1);
            EXPECT_EQ(
                rocsparse_dnmat_transpose(handle, alpha.descr, source.descr, wrong.descr, p_error),
                rocsparse_status_invalid_value);

            // Mixed precision.
            dn_matrix<float> single_target(4, 6, target_order, 1);
            EXPECT_EQ(rocsparse_dnmat_transpose(
                          handle, alpha.descr, source.descr, single_target.descr, p_error),
                      rocsparse_status_not_implemented);
            dn_scalars<float> single_alpha(1, rocsparse_datatype_f32_r);
            EXPECT_NE(rocsparse_dnmat_transpose(
                          handle, single_alpha.descr, source.descr, target.descr, p_error),
                      rocsparse_status_success);

            // Batches that do not match.
            dn_matrix<T> batched_source(6, 4, source_order, 3);
            dn_matrix<T> batched_target(4, 6, target_order, 2);
            EXPECT_NE(rocsparse_dnmat_transpose(
                          handle, alpha.descr, batched_source.descr, batched_target.descr, p_error),
                      rocsparse_status_success);
            dn_scalars<T> batched_alpha(2, rocsparse_datatype_f64_r);
            dn_matrix<T>  matching_target(4, 6, target_order, 3);
            EXPECT_NE(rocsparse_dnmat_transpose(handle,
                                                batched_alpha.descr,
                                                batched_source.descr,
                                                matching_target.descr,
                                                p_error),
                      rocsparse_status_success);
        }
    }
}
