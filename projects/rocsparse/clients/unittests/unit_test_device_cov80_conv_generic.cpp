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

// rocsparse_sparse_to_dense for the CSR, CSC and COO formats (documented), both index
// types, both index bases and both orders of the dense matrix.

#include "unit_test_cov80_conv_common.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    using cov80conv::DevBuf;

    template <typename I>
    constexpr rocsparse_indextype index_type()
    {
        return sizeof(I) == 4 ? rocsparse_indextype_i32 : rocsparse_indextype_i64;
    }

    // Destroys the dense matrix descriptor on scope exit.
    class DnmatGuard
    {
        rocsparse_dnmat_descr& m_ref;

    public:
        explicit DnmatGuard(rocsparse_dnmat_descr& ref)
            : m_ref(ref)
        {
        }
        ~DnmatGuard()
        {
            if(m_ref != nullptr)
            {
                EXPECT_EQ(rocsparse_destroy_dnmat_descr(m_ref), rocsparse_status_success);
            }
        }
        DnmatGuard(const DnmatGuard&) = delete;
        DnmatGuard& operator=(const DnmatGuard&) = delete;
    };

    constexpr rocsparse_int s_m = 6, s_n = 5;

    // Host dense matrix, column major with leading dimension s_m. An entry is stored
    // when (3 i + 2 j) % 4 == 0.
    std::vector<float> reference_dense()
    {
        std::vector<float> dense(s_m * s_n, 0.0f);
        for(rocsparse_int j = 0; j < s_n; ++j)
        {
            for(rocsparse_int i = 0; i < s_m; ++i)
            {
                if((3 * i + 2 * j) % 4 == 0)
                {
                    dense[i + j * s_m] = 1.0f + i + 10.0f * j;
                }
            }
        }
        return dense;
    }

    template <typename I, typename J>
    void run_case(rocsparse_handle handle, rocsparse_format format, int base, rocsparse_order order)
    {
        SCOPED_TRACE("format " + std::to_string(static_cast<int>(format)) + " base "
                     + std::to_string(base) + " index bytes " + std::to_string(sizeof(I)) + "/"
                     + std::to_string(sizeof(J)) + " row order "
                     + std::to_string(order == rocsparse_order_row));

        const std::vector<float> dense = reference_dense();

        // The three sparse representations of the same matrix.
        std::vector<I>     ptr; // CSR row pointers or CSC column pointers
        std::vector<J>     ind; // CSR column indices or CSC row indices
        std::vector<J>     rows; // COO
        std::vector<float> val;
        if(format == rocsparse_format_csc)
        {
            ptr.push_back(base);
            for(rocsparse_int j = 0; j < s_n; ++j)
            {
                for(rocsparse_int i = 0; i < s_m; ++i)
                {
                    if(dense[i + j * s_m] != 0.0f)
                    {
                        ind.push_back(i + base);
                        val.push_back(dense[i + j * s_m]);
                    }
                }
                ptr.push_back(static_cast<I>(ind.size()) + base);
            }
        }
        else
        {
            ptr.push_back(base);
            for(rocsparse_int i = 0; i < s_m; ++i)
            {
                for(rocsparse_int j = 0; j < s_n; ++j)
                {
                    if(dense[i + j * s_m] != 0.0f)
                    {
                        ind.push_back(j + base);
                        rows.push_back(i + base);
                        val.push_back(dense[i + j * s_m]);
                    }
                }
                ptr.push_back(static_cast<I>(ind.size()) + base);
            }
        }
        const int64_t nnz = static_cast<int64_t>(val.size());
        ASSERT_GT(nnz, 0);

        DevBuf<I>     d_ptr(ptr);
        DevBuf<J>     d_ind(ind), d_rows(rows);
        DevBuf<float> d_val(val);

        rocsparse_spmat_descr a = nullptr;
        if(format == rocsparse_format_csr)
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&a,
                                                 s_m,
                                                 s_n,
                                                 nnz,
                                                 d_ptr,
                                                 d_ind,
                                                 d_val,
                                                 index_type<I>(),
                                                 index_type<J>(),
                                                 static_cast<rocsparse_index_base>(base),
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
        }
        else if(format == rocsparse_format_csc)
        {
            ASSERT_EQ(rocsparse_create_csc_descr(&a,
                                                 s_m,
                                                 s_n,
                                                 nnz,
                                                 d_ptr,
                                                 d_ind,
                                                 d_val,
                                                 index_type<I>(),
                                                 index_type<J>(),
                                                 static_cast<rocsparse_index_base>(base),
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
        }
        else
        {
            // COO has a single index type.
            ASSERT_EQ(rocsparse_create_coo_descr(&a,
                                                 s_m,
                                                 s_n,
                                                 nnz,
                                                 d_rows,
                                                 d_ind,
                                                 d_val,
                                                 index_type<J>(),
                                                 static_cast<rocsparse_index_base>(base),
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
        }
        cov80conv::SpmatGuard a_guard(a);

        // The dense output, in the requested order.
        const int64_t      ld = (order == rocsparse_order_column) ? s_m : s_n;
        std::vector<float> expected(s_m * s_n, 0.0f);
        for(rocsparse_int j = 0; j < s_n; ++j)
        {
            for(rocsparse_int i = 0; i < s_m; ++i)
            {
                const int64_t at = (order == rocsparse_order_column) ? i + j * ld : j + i * ld;
                expected[at]     = dense[i + j * s_m];
            }
        }
        DevBuf<float>         d_dense(std::vector<float>{}, expected.size());
        rocsparse_dnmat_descr b = nullptr;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &b, s_m, s_n, ld, d_dense, rocsparse_datatype_f32_r, order),
                  rocsparse_status_success);
        DnmatGuard b_guard(b);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_sparse_to_dense(
                      handle, a, b, rocsparse_sparse_to_dense_alg_default, &buffer_size, nullptr),
                  rocsparse_status_success);
        DevBuf<char> d_buffer(std::vector<char>{}, buffer_size);
        ASSERT_EQ(rocsparse_sparse_to_dense(
                      handle, a, b, rocsparse_sparse_to_dense_alg_default, &buffer_size, d_buffer),
                  rocsparse_status_success);
        EXPECT_EQ(d_dense.read(expected.size()), expected);
    }

    template <typename I, typename J>
    void run_format(rocsparse_handle handle, rocsparse_format format)
    {
        for(int base : {0, 1})
        {
            for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
            {
                run_case<I, J>(handle, format, base, order);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvSparseToDenseCsr)
    {
        run_format<int32_t, int32_t>(handle, rocsparse_format_csr);
        run_format<int64_t, int64_t>(handle, rocsparse_format_csr);
        run_format<int64_t, int32_t>(handle, rocsparse_format_csr);
    }

    TEST_F(HandleTest, Cov80ConvSparseToDenseCsc)
    {
        run_format<int32_t, int32_t>(handle, rocsparse_format_csc);
        run_format<int64_t, int64_t>(handle, rocsparse_format_csc);
        run_format<int64_t, int32_t>(handle, rocsparse_format_csc);
    }

    TEST_F(HandleTest, Cov80ConvSparseToDenseCoo)
    {
        run_format<int32_t, int32_t>(handle, rocsparse_format_coo);
        run_format<int64_t, int64_t>(handle, rocsparse_format_coo);
    }
}
