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

// csr2dense / csc2dense / dense2csr / dense2csc: every value type, both index bases,
// empty matrices and the pointer checks that depend on the number of entries.

#include "unit_test_cov80_conv_common.hpp"

#include <type_traits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    using cov80conv::DevBuf;

    class Descr
    {
        rocsparse_mat_descr m_d = nullptr;

    public:
        explicit Descr(int base)
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&m_d), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_index_base(m_d, static_cast<rocsparse_index_base>(base)),
                      rocsparse_status_success);
        }
        ~Descr()
        {
            if(m_d != nullptr)
            {
                EXPECT_EQ(rocsparse_destroy_mat_descr(m_d), rocsparse_status_success);
            }
        }
        Descr(const Descr&) = delete;
        Descr& operator=(const Descr&) = delete;
               operator rocsparse_mat_descr() const
        {
            return m_d;
        }
    };

    template <typename T>
    struct Api;

#define COV80_DENSE_API(T__, P__)                                              \
    template <>                                                                \
    struct Api<T__>                                                            \
    {                                                                          \
        static rocsparse_status csr2dense(rocsparse_handle          h,         \
                                          rocsparse_int             m,         \
                                          rocsparse_int             n,         \
                                          const rocsparse_mat_descr d,         \
                                          const T__*                v,         \
                                          const rocsparse_int*      p,         \
                                          const rocsparse_int*      c,         \
                                          T__*                      a,         \
                                          rocsparse_int             ld)        \
        {                                                                      \
            return rocsparse_##P__##csr2dense(h, m, n, d, v, p, c, a, ld);     \
        }                                                                      \
        static rocsparse_status csc2dense(rocsparse_handle          h,         \
                                          rocsparse_int             m,         \
                                          rocsparse_int             n,         \
                                          const rocsparse_mat_descr d,         \
                                          const T__*                v,         \
                                          const rocsparse_int*      p,         \
                                          const rocsparse_int*      c,         \
                                          T__*                      a,         \
                                          rocsparse_int             ld)        \
        {                                                                      \
            return rocsparse_##P__##csc2dense(h, m, n, d, v, p, c, a, ld);     \
        }                                                                      \
        static rocsparse_status dense2csr(rocsparse_handle          h,         \
                                          rocsparse_int             m,         \
                                          rocsparse_int             n,         \
                                          const rocsparse_mat_descr d,         \
                                          const T__*                a,         \
                                          rocsparse_int             ld,        \
                                          const rocsparse_int*      nz,        \
                                          T__*                      v,         \
                                          rocsparse_int*            p,         \
                                          rocsparse_int*            c)         \
        {                                                                      \
            return rocsparse_##P__##dense2csr(h, m, n, d, a, ld, nz, v, p, c); \
        }                                                                      \
        static rocsparse_status dense2csc(rocsparse_handle          h,         \
                                          rocsparse_int             m,         \
                                          rocsparse_int             n,         \
                                          const rocsparse_mat_descr d,         \
                                          const T__*                a,         \
                                          rocsparse_int             ld,        \
                                          const rocsparse_int*      nz,        \
                                          T__*                      v,         \
                                          rocsparse_int*            p,         \
                                          rocsparse_int*            c)         \
        {                                                                      \
            return rocsparse_##P__##dense2csc(h, m, n, d, a, ld, nz, v, p, c); \
        }                                                                      \
    }

    COV80_DENSE_API(float, s);
    COV80_DENSE_API(double, d);
    COV80_DENSE_API(rocsparse_float_complex, c);
    COV80_DENSE_API(rocsparse_double_complex, z);
#undef COV80_DENSE_API

    template <typename T>
    T make_value(int k)
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return static_cast<T>(1 + k);
        }
        else
        {
            using R = std::conditional_t<std::is_same_v<T, rocsparse_float_complex>, float, double>;
            return T(static_cast<R>(1 + k), static_cast<R>(-k));
        }
    }

    // The test matrix is square (so that a CSC pointer array is also long enough for the
    // checks that read its last element).
    constexpr rocsparse_int s_m = 7, s_n = 7, s_ld = 9;

    template <typename T>
    struct Matrices
    {
        std::vector<T>             dense; // column major, leading dimension s_ld
        std::vector<rocsparse_int> csr_ptr, csr_ind, csc_ptr, csc_ind, per_row, per_col;
        std::vector<T>             csr_val, csc_val;
    };

    template <typename T>
    Matrices<T> make_matrices(int base)
    {
        Matrices<T> a;
        a.dense.assign(s_ld * s_n, T{});
        int k = 0;
        for(rocsparse_int j = 0; j < s_n; ++j)
        {
            for(rocsparse_int i = 0; i < s_m; ++i)
            {
                if((3 * i + 2 * j) % 4 == 0 || i == j)
                {
                    a.dense[i + j * s_ld] = make_value<T>(k++);
                }
            }
        }
        const auto is_set
            = [&](rocsparse_int i, rocsparse_int j) { return (3 * i + 2 * j) % 4 == 0 || i == j; };
        a.csr_ptr.push_back(base);
        for(rocsparse_int i = 0; i < s_m; ++i)
        {
            rocsparse_int cnt = 0;
            for(rocsparse_int j = 0; j < s_n; ++j)
            {
                if(is_set(i, j))
                {
                    a.csr_ind.push_back(j + base);
                    a.csr_val.push_back(a.dense[i + j * s_ld]);
                    ++cnt;
                }
            }
            a.per_row.push_back(cnt);
            a.csr_ptr.push_back(base + static_cast<rocsparse_int>(a.csr_ind.size()));
        }
        a.csc_ptr.push_back(base);
        for(rocsparse_int j = 0; j < s_n; ++j)
        {
            rocsparse_int cnt = 0;
            for(rocsparse_int i = 0; i < s_m; ++i)
            {
                if(is_set(i, j))
                {
                    a.csc_ind.push_back(i + base);
                    a.csc_val.push_back(a.dense[i + j * s_ld]);
                    ++cnt;
                }
            }
            a.per_col.push_back(cnt);
            a.csc_ptr.push_back(base + static_cast<rocsparse_int>(a.csc_ind.size()));
        }
        return a;
    }

    template <typename T>
    void run_dense(rocsparse_handle handle)
    {
        for(int base : {0, 1})
        {
            SCOPED_TRACE("base " + std::to_string(base));
            const Matrices<T>     a   = make_matrices<T>(base);
            const rocsparse_int   nnz = static_cast<rocsparse_int>(a.csr_ind.size());
            Descr                 descr(base);
            DevBuf<rocsparse_int> d_rptr(a.csr_ptr), d_rind(a.csr_ind), d_cptr(a.csc_ptr),
                d_cind(a.csc_ind);
            DevBuf<T> d_rval(a.csr_val), d_cval(a.csc_val);

            // Sparse to dense.
            {
                DevBuf<T> d_a(std::vector<T>{}, a.dense.size());
                ASSERT_EQ(
                    Api<T>::csr2dense(handle, s_m, s_n, descr, d_rval, d_rptr, d_rind, d_a, s_ld),
                    rocsparse_status_success);
                EXPECT_EQ(d_a.read(a.dense.size()), a.dense);
            }
            {
                DevBuf<T> d_a(std::vector<T>{}, a.dense.size());
                ASSERT_EQ(
                    Api<T>::csc2dense(handle, s_m, s_n, descr, d_cval, d_cptr, d_cind, d_a, s_ld),
                    rocsparse_status_success);
                EXPECT_EQ(d_a.read(a.dense.size()), a.dense);
            }

            // Dense to sparse.
            {
                DevBuf<T>             d_a(a.dense);
                DevBuf<rocsparse_int> d_nz(a.per_row);
                DevBuf<rocsparse_int> o_ptr(std::vector<rocsparse_int>{}, s_m + 1);
                DevBuf<rocsparse_int> o_ind(std::vector<rocsparse_int>{}, nnz);
                DevBuf<T>             o_val(std::vector<T>{}, nnz);
                ASSERT_EQ(Api<T>::dense2csr(
                              handle, s_m, s_n, descr, d_a, s_ld, d_nz, o_val, o_ptr, o_ind),
                          rocsparse_status_success);
                EXPECT_EQ(o_ptr.read(s_m + 1), a.csr_ptr);
                EXPECT_EQ(o_ind.read(nnz), a.csr_ind);
                EXPECT_EQ(o_val.read(nnz), a.csr_val);

                // The arrays are required when there are entries.
                EXPECT_EQ(Api<T>::dense2csr(handle,
                                            s_m,
                                            s_n,
                                            descr,
                                            d_a,
                                            s_ld,
                                            d_nz,
                                            static_cast<T*>(nullptr),
                                            o_ptr,
                                            static_cast<rocsparse_int*>(nullptr)),
                          rocsparse_status_invalid_pointer);
            }
            {
                DevBuf<T>             d_a(a.dense);
                DevBuf<rocsparse_int> d_nz(a.per_col);
                DevBuf<rocsparse_int> o_ptr(std::vector<rocsparse_int>{}, s_n + 1);
                DevBuf<rocsparse_int> o_ind(std::vector<rocsparse_int>{}, nnz);
                DevBuf<T>             o_val(std::vector<T>{}, nnz);
                ASSERT_EQ(Api<T>::dense2csc(
                              handle, s_m, s_n, descr, d_a, s_ld, d_nz, o_val, o_ptr, o_ind),
                          rocsparse_status_success);
                EXPECT_EQ(o_ptr.read(s_n + 1), a.csc_ptr);
                EXPECT_EQ(o_ind.read(nnz), a.csc_ind);
                EXPECT_EQ(o_val.read(nnz), a.csc_val);

                EXPECT_EQ(Api<T>::dense2csc(handle,
                                            s_m,
                                            s_n,
                                            descr,
                                            d_a,
                                            s_ld,
                                            d_nz,
                                            static_cast<T*>(nullptr),
                                            o_ptr,
                                            static_cast<rocsparse_int*>(nullptr)),
                          rocsparse_status_invalid_pointer);
            }

            // The values and indices of the sparse input are required.
            {
                DevBuf<T> d_a(std::vector<T>{}, a.dense.size());
                EXPECT_EQ(Api<T>::csr2dense(handle,
                                            s_m,
                                            s_n,
                                            descr,
                                            static_cast<const T*>(nullptr),
                                            d_rptr,
                                            d_rind,
                                            d_a,
                                            s_ld),
                          rocsparse_status_invalid_pointer);
                EXPECT_EQ(Api<T>::csr2dense(handle,
                                            s_m,
                                            s_n,
                                            descr,
                                            d_rval,
                                            d_rptr,
                                            static_cast<const rocsparse_int*>(nullptr),
                                            d_a,
                                            s_ld),
                          rocsparse_status_invalid_pointer);
                EXPECT_EQ(Api<T>::csc2dense(handle,
                                            s_m,
                                            s_n,
                                            descr,
                                            static_cast<const T*>(nullptr),
                                            d_cptr,
                                            d_cind,
                                            d_a,
                                            s_ld),
                          rocsparse_status_invalid_pointer);
                EXPECT_EQ(Api<T>::csc2dense(handle,
                                            s_m,
                                            s_n,
                                            descr,
                                            d_cval,
                                            d_cptr,
                                            static_cast<const rocsparse_int*>(nullptr),
                                            d_a,
                                            s_ld),
                          rocsparse_status_invalid_pointer);
            }
        }
    }

    template <typename T>
    void run_dense_empty(rocsparse_handle handle)
    {
        Descr                 descr(0);
        DevBuf<T>             d_a(std::vector<T>{}, 16), d_v(std::vector<T>{}, 16);
        DevBuf<rocsparse_int> d_p(std::vector<rocsparse_int>(8, 0)),
            d_c(std::vector<rocsparse_int>{}, 8), d_nz(std::vector<rocsparse_int>(8, 0));
        const rocsparse_int dims[][2] = {{0, 4}, {4, 0}};
        for(const auto& d : dims)
        {
            SCOPED_TRACE("m " + std::to_string(d[0]) + " n " + std::to_string(d[1]));
            const rocsparse_int ld = std::max<rocsparse_int>(d[0], 1);
            EXPECT_EQ(Api<T>::csr2dense(handle, d[0], d[1], descr, d_v, d_p, d_c, d_a, ld),
                      rocsparse_status_success);
            EXPECT_EQ(Api<T>::csc2dense(handle, d[0], d[1], descr, d_v, d_p, d_c, d_a, ld),
                      rocsparse_status_success);
            EXPECT_EQ(Api<T>::dense2csr(handle, d[0], d[1], descr, d_a, ld, d_nz, d_v, d_p, d_c),
                      rocsparse_status_success);
            EXPECT_EQ(Api<T>::dense2csc(handle, d[0], d[1], descr, d_a, ld, d_nz, d_v, d_p, d_c),
                      rocsparse_status_success);
        }
    }

    TEST_F(HandleTest, Cov80ConvDenseFloat)
    {
        run_dense<float>(handle);
        run_dense_empty<float>(handle);
    }
    TEST_F(HandleTest, Cov80ConvDenseDouble)
    {
        run_dense<double>(handle);
        run_dense_empty<double>(handle);
    }
    TEST_F(HandleTest, Cov80ConvDenseFloatComplex)
    {
        run_dense<rocsparse_float_complex>(handle);
        run_dense_empty<rocsparse_float_complex>(handle);
    }
    TEST_F(HandleTest, Cov80ConvDenseDoubleComplex)
    {
        run_dense<rocsparse_double_complex>(handle);
        run_dense_empty<rocsparse_double_complex>(handle);
    }
}
