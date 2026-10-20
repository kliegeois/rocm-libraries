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
// Helpers shared by the unit_test_device_cov80_level2_*.cpp tests: irregular
// host matrices, host conversions to the COO, ELL and sliced ELL formats and
// precision overloaded wrappers of the legacy level2 routines.
//
#pragma once

#include "unit_test_device_cov70_level2_common.hpp"

#include <set>

namespace cov80l2
{
    using namespace cov70l2;

    // A sorted (m x n) CSR matrix with rows of varying length: row i holds
    // (i * 5 + 3) % 9 entries (none if empty_every > 0 and i % empty_every == 0),
    // row 1 holds `long_row` entries when long_row > 0. Values are never zero.
    // The columns of a row are distinct.
    template <typename T>
    inline host_csr<T>
        make_matrix(int32_t m, int32_t n, int base, int32_t long_row = 0, int32_t empty_every = 0)
    {
        host_csr<T> a;
        a.m    = m;
        a.n    = n;
        a.base = 0;
        a.ptr.push_back(0);
        int64_t counter = 0;
        for(int32_t i = 0; i < m; ++i)
        {
            int32_t len = (i * 5 + 3) % 9;
            if(empty_every > 0 && i % empty_every == 0)
                len = 0;
            if(long_row > 0 && i == 1)
                len = long_row;
            len = std::min(len, n);
            std::set<int32_t> cols;
            int32_t           j    = (n > 0) ? (i * 13) % n : 0;
            const int32_t     step = 1 + (i % 4) * 2;
            while(static_cast<int32_t>(cols.size()) < len)
            {
                if(!cols.insert(j % n).second)
                    ++j;
                else
                    j += step;
            }
            for(int32_t c : cols)
            {
                a.ind.push_back(c);
                a.val.push_back(value_of<T>(counter++, true));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        rebase(a, base);
        return a;
    }

    template <typename T>
    inline int32_t max_row_length(const host_csr<T>& a)
    {
        int32_t w = 0;
        for(int32_t i = 0; i < a.m; ++i)
            w = std::max(w, a.ptr[i + 1] - a.ptr[i]);
        return w;
    }

    template <typename T>
    struct host_coo
    {
        std::vector<int32_t> row;
        std::vector<int32_t> col;
        std::vector<T>       val;
    };

    // Row major sorted COO entries with the index base of `a`.
    template <typename T>
    inline host_coo<T> to_coo(const host_csr<T>& a)
    {
        host_coo<T> c;
        for(int32_t i = 0; i < a.m; ++i)
            for(int32_t k = a.ptr[i] - a.base; k < a.ptr[i + 1] - a.base; ++k)
            {
                c.row.push_back(i + a.base);
                c.col.push_back(a.ind[k]);
                c.val.push_back(a.val[k]);
            }
        return c;
    }

    template <typename T>
    struct host_ell
    {
        int32_t              width = 0;
        std::vector<int32_t> col; // column major: entry p of row i at p * m + i
        std::vector<T>       val;
    };

    template <typename T>
    inline host_ell<T> to_ell(const host_csr<T>& a)
    {
        host_ell<T> e;
        e.width = max_row_length(a);
        e.col.assign(static_cast<size_t>(e.width) * a.m, -1);
        e.val.assign(static_cast<size_t>(e.width) * a.m, T{});
        for(int32_t i = 0; i < a.m; ++i)
            for(int32_t k = a.ptr[i] - a.base, p = 0; k < a.ptr[i + 1] - a.base; ++k, ++p)
            {
                e.col[static_cast<size_t>(p) * a.m + i] = a.ind[k];
                e.val[static_cast<size_t>(p) * a.m + i] = a.val[k];
            }
        return e;
    }

    template <typename T>
    struct host_sell
    {
        int32_t              slice_size = 0;
        int32_t              colval     = 0;
        std::vector<int32_t> offsets; // nslices + 1, with the index base of the matrix
        std::vector<int32_t> col;
        std::vector<T>       val;
    };

    template <typename T>
    inline host_sell<T> to_sell(const host_csr<T>& a, int32_t slice_size)
    {
        host_sell<T> s;
        s.slice_size                 = slice_size;
        const int32_t        nslices = (a.m - 1) / slice_size + 1;
        std::vector<int32_t> w(nslices, 0);
        s.offsets.assign(nslices + 1, 0);
        for(int32_t sl = 0; sl < nslices; ++sl)
        {
            for(int32_t r = sl * slice_size; r < std::min(a.m, (sl + 1) * slice_size); ++r)
                w[sl] = std::max(w[sl], a.ptr[r + 1] - a.ptr[r]);
            s.offsets[sl + 1] = s.offsets[sl] + w[sl] * slice_size;
        }
        s.colval = s.offsets[nslices];
        s.col.assign(s.colval, -1);
        s.val.assign(s.colval, T{});
        for(int32_t r = 0; r < a.m; ++r)
        {
            const int32_t sl = r / slice_size;
            const int32_t l  = r % slice_size;
            for(int32_t k = a.ptr[r] - a.base, p = 0; k < a.ptr[r + 1] - a.base; ++k, ++p)
            {
                s.col[s.offsets[sl] + p * slice_size + l] = a.ind[k];
                s.val[s.offsets[sl] + p * slice_size + l] = a.val[k];
            }
        }
        for(auto& o : s.offsets)
            o += a.base;
        return s;
    }

    inline size_t trans_size(rocsparse_operation trans, int32_t m, int32_t n, bool input)
    {
        const bool none = (trans == rocsparse_operation_none);
        return static_cast<size_t>((input == none) ? n : m);
    }

    inline std::vector<int> to_int(const std::vector<int32_t>& v)
    {
        return std::vector<int>(v.begin(), v.end());
    }

    // ----- precision overloaded wrappers -------------------------------------
#define COV80_COOMV(P, TYPE)                                                                 \
    inline rocsparse_status coomv(rocsparse_handle          handle,                          \
                                  rocsparse_operation       trans,                           \
                                  rocsparse_int             m,                               \
                                  rocsparse_int             n,                               \
                                  rocsparse_int             nnz,                             \
                                  const TYPE*               alpha,                           \
                                  const rocsparse_mat_descr descr,                           \
                                  const TYPE*               val,                             \
                                  const rocsparse_int*      row,                             \
                                  const rocsparse_int*      col,                             \
                                  const TYPE*               x,                               \
                                  const TYPE*               beta,                            \
                                  TYPE*                     y)                               \
    {                                                                                        \
        return rocsparse_##P##coomv(                                                         \
            handle, trans, m, n, nnz, alpha, descr, val, row, col, x, beta, y);              \
    }                                                                                        \
    inline rocsparse_status ellmv(rocsparse_handle          handle,                          \
                                  rocsparse_operation       trans,                           \
                                  rocsparse_int             m,                               \
                                  rocsparse_int             n,                               \
                                  const TYPE*               alpha,                           \
                                  const rocsparse_mat_descr descr,                           \
                                  const TYPE*               val,                             \
                                  const rocsparse_int*      col,                             \
                                  rocsparse_int             width,                           \
                                  const TYPE*               x,                               \
                                  const TYPE*               beta,                            \
                                  TYPE*                     y)                               \
    {                                                                                        \
        return rocsparse_##P##ellmv(                                                         \
            handle, trans, m, n, alpha, descr, val, col, width, x, beta, y);                 \
    }                                                                                        \
    inline rocsparse_status hybmv(rocsparse_handle          handle,                          \
                                  rocsparse_operation       trans,                           \
                                  const TYPE*               alpha,                           \
                                  const rocsparse_mat_descr descr,                           \
                                  const rocsparse_hyb_mat   hyb,                             \
                                  const TYPE*               x,                               \
                                  const TYPE*               beta,                            \
                                  TYPE*                     y)                               \
    {                                                                                        \
        return rocsparse_##P##hybmv(handle, trans, alpha, descr, hyb, x, beta, y);           \
    }                                                                                        \
    inline rocsparse_status csr2hyb(rocsparse_handle          handle,                        \
                                    rocsparse_int             m,                             \
                                    rocsparse_int             n,                             \
                                    const rocsparse_mat_descr descr,                         \
                                    const TYPE*               val,                           \
                                    const rocsparse_int*      ptr,                           \
                                    const rocsparse_int*      col,                           \
                                    rocsparse_hyb_mat         hyb,                           \
                                    rocsparse_int             width,                         \
                                    rocsparse_hyb_partition   part)                          \
    {                                                                                        \
        return rocsparse_##P##csr2hyb(handle, m, n, descr, val, ptr, col, hyb, width, part); \
    }                                                                                        \
    inline rocsparse_status gemvi_buffer_size(rocsparse_handle    handle,                    \
                                              rocsparse_operation trans,                     \
                                              rocsparse_int       m,                         \
                                              rocsparse_int       n,                         \
                                              rocsparse_int       nnz,                       \
                                              size_t*             size,                      \
                                              const TYPE*)                                   \
    {                                                                                        \
        return rocsparse_##P##gemvi_buffer_size(handle, trans, m, n, nnz, size);             \
    }                                                                                        \
    inline rocsparse_status gemvi(rocsparse_handle     handle,                               \
                                  rocsparse_operation  trans,                                \
                                  rocsparse_int        m,                                    \
                                  rocsparse_int        n,                                    \
                                  const TYPE*          alpha,                                \
                                  const TYPE*          A,                                    \
                                  rocsparse_int        lda,                                  \
                                  rocsparse_int        nnz,                                  \
                                  const TYPE*          x_val,                                \
                                  const rocsparse_int* x_ind,                                \
                                  const TYPE*          beta,                                 \
                                  TYPE*                y,                                    \
                                  rocsparse_index_base base,                                 \
                                  void*                buffer)                               \
    {                                                                                        \
        return rocsparse_##P##gemvi(                                                         \
            handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, base, buffer);   \
    }

    COV80_COOMV(s, float)
    COV80_COOMV(d, double)
    COV80_COOMV(c, rocsparse_float_complex)
    COV80_COOMV(z, rocsparse_double_complex)
#undef COV80_COOMV

    struct hyb_guard
    {
        rocsparse_hyb_mat d = nullptr;
        hyb_guard()
        {
            if(rocsparse_create_hyb_mat(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~hyb_guard()
        {
            if(d)
                (void)rocsparse_destroy_hyb_mat(d);
        }
        hyb_guard(const hyb_guard&) = delete;
        hyb_guard& operator=(const hyb_guard&) = delete;
    };

    // RAII wrapper of the v2 spmv descriptor.
    struct spmv_descr_guard
    {
        rocsparse_spmv_descr d = nullptr;
        spmv_descr_guard()
        {
            if(rocsparse_create_spmv_descr(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~spmv_descr_guard()
        {
            if(d)
                (void)rocsparse_destroy_spmv_descr(d);
        }
        spmv_descr_guard(const spmv_descr_guard&) = delete;
        spmv_descr_guard& operator=(const spmv_descr_guard&) = delete;
    };

    template <typename V>
    inline rocsparse_status set_input(rocsparse_handle     handle,
                                      rocsparse_spmv_descr d,
                                      rocsparse_spmv_input input,
                                      const V&             value)
    {
        return rocsparse_spmv_set_input(handle, d, input, &value, sizeof(V), nullptr);
    }
}
