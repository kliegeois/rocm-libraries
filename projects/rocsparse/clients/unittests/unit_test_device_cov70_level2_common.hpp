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

//
// Shared helpers of the cov70 level2 device unit tests: deterministic data,
// host sparse containers, RAII descriptors and host reference products.
//
#pragma once

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace cov70l2
{
    using rocsparse_ut::device_vector;
    using rocsparse_ut::to_host;

    // ----- scalar traits ----------------------------------------------------
    template <typename T>
    struct real_of
    {
        using type = T;
    };
    template <>
    struct real_of<rocsparse_float_complex>
    {
        using type = float;
    };
    template <>
    struct real_of<rocsparse_double_complex>
    {
        using type = double;
    };

    template <typename T>
    constexpr bool is_single()
    {
        return std::is_same<typename real_of<T>::type, float>::value;
    }

    // Comparison tolerance used for products and solves of the small, exactly
    // representable data generated below.
    template <typename T>
    constexpr double tolerance()
    {
        return is_single<T>() ? 1.0e-4 : 1.0e-11;
    }

    template <typename T>
    inline double magnitude(const T& v)
    {
        return static_cast<double>(std::abs(v));
    }

    // Deterministic values: multiples of 1/4 in [-1, 1]. Never zero for odd
    // `n` so that they can be used for diagonal entries when `nonzero` is set.
    template <typename T>
    inline T value_of(int64_t n, bool nonzero = false)
    {
        int64_t k = (n * 7 + 3) % 9 - 4;
        if(nonzero && k == 0)
            k = 3;
        return rocsparse_ut::scalar<T>(static_cast<float>(k) / 4.0f);
    }
    template <>
    inline rocsparse_float_complex value_of<rocsparse_float_complex>(int64_t n, bool nonzero)
    {
        const float im = static_cast<float>((n * 5 + 1) % 7 - 3) / 4.0f;
        return rocsparse_float_complex(value_of<float>(n, nonzero), im);
    }
    template <>
    inline rocsparse_double_complex value_of<rocsparse_double_complex>(int64_t n, bool nonzero)
    {
        const double im = static_cast<double>((n * 5 + 1) % 7 - 3) / 4.0;
        return rocsparse_double_complex(value_of<double>(n, nonzero), im);
    }

    // ----- comparison -------------------------------------------------------
    // Compares two vectors element by element. Reports the first mismatch (and
    // the count of mismatches) once, so a wrong kernel does not flood the log.
    template <typename T>
    inline bool expect_near(const std::vector<T>& want,
                            const std::vector<T>& got,
                            const char*           what,
                            double                tol_scale = 1.0)
    {
        if(want.size() != got.size())
        {
            ADD_FAILURE() << what << ": size " << got.size() << " != " << want.size();
            return false;
        }
        const double tol   = tolerance<T>() * tol_scale;
        size_t       bad   = 0;
        size_t       first = 0;
        for(size_t i = 0; i < want.size(); ++i)
        {
            const double err = magnitude(want[i] - got[i]);
            if(!(err <= tol * (1.0 + magnitude(want[i]))))
            {
                if(bad == 0)
                    first = i;
                ++bad;
            }
        }
        if(bad > 0)
        {
            ADD_FAILURE() << what << ": " << bad << " mismatches, first at " << first
                          << " want=" << want[first] << " got=" << got[first];
            return false;
        }
        return true;
    }

    // ----- RAII descriptors -------------------------------------------------
    struct mat_descr
    {
        rocsparse_mat_descr d = nullptr;
        mat_descr()
        {
            if(rocsparse_create_mat_descr(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~mat_descr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
        mat_descr(const mat_descr&) = delete;
        mat_descr& operator=(const mat_descr&) = delete;
                   operator rocsparse_mat_descr() const
        {
            return d;
        }
    };

    struct mat_info
    {
        rocsparse_mat_info d = nullptr;
        mat_info()
        {
            if(rocsparse_create_mat_info(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~mat_info()
        {
            if(d)
                (void)rocsparse_destroy_mat_info(d);
        }
        mat_info(const mat_info&) = delete;
        mat_info& operator=(const mat_info&) = delete;
                  operator rocsparse_mat_info() const
        {
            return d;
        }
    };

    struct handle_guard
    {
        rocsparse_handle h = nullptr;
        handle_guard()
        {
            if(rocsparse_create_handle(&h) != rocsparse_status_success)
                h = nullptr;
        }
        ~handle_guard()
        {
            if(h)
                (void)rocsparse_destroy_handle(h);
        }
        handle_guard(const handle_guard&) = delete;
        handle_guard& operator=(const handle_guard&) = delete;
                      operator rocsparse_handle() const
        {
            return h;
        }
    };

    struct scoped_wavefront
    {
        rocsparse_handle handle;
        int              saved;

        scoped_wavefront(rocsparse_handle h, int wavefront)
            : handle(h)
            , saved(h->wavefront_size)
        {
            handle->wavefront_size = wavefront;
        }

        ~scoped_wavefront()
        {
            handle->wavefront_size = saved;
        }

        scoped_wavefront(const scoped_wavefront&) = delete;
        scoped_wavefront& operator=(const scoped_wavefront&) = delete;
    };

    struct scoped_pointer_mode
    {
        rocsparse_handle       handle;
        rocsparse_pointer_mode saved;

        scoped_pointer_mode(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
            , saved(h->pointer_mode)
        {
            EXPECT_EQ(rocsparse_set_pointer_mode(handle, mode), rocsparse_status_success);
        }

        ~scoped_pointer_mode()
        {
            (void)rocsparse_set_pointer_mode(handle, saved);
        }

        scoped_pointer_mode(const scoped_pointer_mode&) = delete;
        scoped_pointer_mode& operator=(const scoped_pointer_mode&) = delete;
    };

    struct spmat_guard
    {
        rocsparse_spmat_descr d = nullptr;
        ~spmat_guard()
        {
            if(d)
                (void)rocsparse_destroy_spmat_descr(d);
        }
    };

    struct dnvec_guard
    {
        rocsparse_dnvec_descr d = nullptr;
        ~dnvec_guard()
        {
            if(d)
                (void)rocsparse_destroy_dnvec_descr(d);
        }
    };

    inline rocsparse_index_base to_base(int base)
    {
        return base == 0 ? rocsparse_index_base_zero : rocsparse_index_base_one;
    }

    // ----- host containers --------------------------------------------------
    template <typename T>
    struct host_csr
    {
        int32_t              m    = 0;
        int32_t              n    = 0;
        int                  base = 0;
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;

        int32_t nnz() const
        {
            return static_cast<int32_t>(ind.size());
        }
    };

    // Applies a zero-based pattern to the requested index base.
    template <typename T>
    inline void rebase(host_csr<T>& a, int base)
    {
        if(a.base == base)
            return;
        const int d = base - a.base;
        for(auto& p : a.ptr)
            p += d;
        for(auto& i : a.ind)
            i += d;
        a.base = base;
    }

    template <typename T>
    inline T conj_of(const T& v)
    {
        return v;
    }
    inline rocsparse_float_complex conj_of(const rocsparse_float_complex& v)
    {
        return rocsparse_float_complex(std::real(v), -std::imag(v));
    }
    inline rocsparse_double_complex conj_of(const rocsparse_double_complex& v)
    {
        return rocsparse_double_complex(std::real(v), -std::imag(v));
    }

    // y = alpha * op(A) * x + beta * y for a host CSR matrix; the rows
    // [0, m) are all used. `trans` is none / transpose / conjugate_transpose.
    template <typename T>
    inline std::vector<T> csr_gemv(const host_csr<T>&    a,
                                   rocsparse_operation   trans,
                                   T                     alpha,
                                   const std::vector<T>& x,
                                   T                     beta,
                                   const std::vector<T>& y0)
    {
        std::vector<T> y(y0);
        for(auto& v : y)
            v = beta * v;
        for(int32_t i = 0; i < a.m; ++i)
        {
            for(int32_t k = a.ptr[i] - a.base; k < a.ptr[i + 1] - a.base; ++k)
            {
                const int32_t j = a.ind[k] - a.base;
                if(trans == rocsparse_operation_none)
                    y[i] = y[i] + alpha * a.val[k] * x[j];
                else if(trans == rocsparse_operation_transpose)
                    y[j] = y[j] + alpha * a.val[k] * x[i];
                else
                    y[j] = y[j] + alpha * conj_of(a.val[k]) * x[i];
            }
        }
        return y;
    }

    // A diagonally dominant, sorted n x n pattern with `per_row` off-diagonal
    // candidates per row on each side. The diagonal is present (non-zero) and
    // dominant, so triangular solves are well conditioned.
    //   lower / upper: keep only that triangle (plus the diagonal),
    //   both:          keep the full pattern.
    // `unit` omits the diagonal entries (unit diagonal storage).
    template <typename T>
    inline host_csr<T> make_triangular_pattern(int32_t n,
                                               int32_t per_row,
                                               bool    lower,
                                               bool    upper,
                                               bool    with_diag,
                                               int     base        = 0,
                                               int32_t start_shift = 1)
    {
        host_csr<T> a;
        a.m    = n;
        a.n    = n;
        a.base = 0;
        a.ptr.push_back(0);
        int64_t counter = 0;
        for(int32_t i = 0; i < n; ++i)
        {
            std::vector<int32_t> cols;
            if(lower)
            {
                for(int32_t k = 1; k <= per_row; ++k)
                {
                    const int32_t j = i - k * start_shift;
                    if(j >= 0)
                        cols.push_back(j);
                }
            }
            if(with_diag)
                cols.push_back(i);
            if(upper)
            {
                for(int32_t k = 1; k <= per_row; ++k)
                {
                    const int32_t j = i + k * start_shift;
                    if(j < n)
                        cols.push_back(j);
                }
            }
            std::sort(cols.begin(), cols.end());
            for(int32_t j : cols)
            {
                a.ind.push_back(j);
                if(j == i)
                {
                    // dominant diagonal, real part positive
                    T d = value_of<T>(counter++, true);
                    d   = rocsparse_ut::scalar<T>(static_cast<float>(2 * per_row + 2)) + d;
                    a.val.push_back(d);
                }
                else
                {
                    T v = value_of<T>(counter++, true);
                    a.val.push_back(v);
                }
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        rebase(a, base);
        return a;
    }

    template <typename J, typename S>
    inline std::vector<J> convert_vector(const std::vector<S>& v)
    {
        return std::vector<J>(v.begin(), v.end());
    }

    // Device copy of a host CSR matrix with row pointer type I and column
    // index type J. The column index array is allocated with one guard element
    // (value -7, never a valid column) on each side: the triangular-solve
    // analysis kernels read the entry that precedes or follows an empty row.
    template <typename T, typename I = int32_t, typename J = int32_t>
    struct device_csr
    {
        device_vector<I> ptr;
        device_vector<J> ind_buf;
        device_vector<T> val;
        J*               ind = nullptr;

        static std::vector<J> padded(const std::vector<int32_t>& v)
        {
            std::vector<J> p(v.size() + 2, static_cast<J>(-7));
            for(size_t i = 0; i < v.size(); ++i)
                p[i + 1] = static_cast<J>(v[i]);
            return p;
        }

        explicit device_csr(const host_csr<T>& a)
            : ptr(convert_vector<I>(a.ptr))
            , ind_buf(padded(a.ind))
            , val(a.val.empty() ? std::vector<T>(1, T{}) : a.val)
        {
            ind = ind_buf.ptr ? ind_buf.ptr + 1 : nullptr;
        }

        bool ok() const
        {
            return ptr.ptr && ind && val.ptr;
        }
    };

    template <typename T>
    inline std::vector<T> make_vector(size_t n, int64_t seed)
    {
        std::vector<T> v(n);
        for(size_t i = 0; i < n; ++i)
            v[i] = value_of<T>(static_cast<int64_t>(i) + seed);
        return v;
    }

    // ----- host reference ---------------------------------------------------------
    // Solves op(A) y = alpha * x with the triangle `fill` of A (and a unit or the
    // stored diagonal) by substitution on a dense copy.
    template <typename T>
    std::vector<T> host_triangular_solve(const host_csr<T>&    A,
                                         rocsparse_fill_mode   fill,
                                         rocsparse_diag_type   diag,
                                         rocsparse_operation   trans,
                                         T                     alpha,
                                         const std::vector<T>& x)
    {
        const int32_t  n = A.m;
        std::vector<T> M(static_cast<size_t>(n) * n, T{});
        for(int32_t i = 0; i < n; ++i)
        {
            for(int32_t k = A.ptr[i] - A.base; k < A.ptr[i + 1] - A.base; ++k)
            {
                const int32_t j = A.ind[k] - A.base;
                if(j == i)
                    M[static_cast<size_t>(i) * n + j] = A.val[k];
                else if(fill == rocsparse_fill_mode_lower ? (j < i) : (j > i))
                    M[static_cast<size_t>(i) * n + j] = A.val[k];
            }
            if(diag == rocsparse_diag_type_unit)
                M[static_cast<size_t>(i) * n + i] = rocsparse_ut::scalar<T>(1.0f);
        }

        std::vector<T> op(M);
        if(trans != rocsparse_operation_none)
        {
            for(int32_t r = 0; r < n; ++r)
                for(int32_t c = 0; c < n; ++c)
                {
                    const T v = M[static_cast<size_t>(c) * n + r];
                    op[static_cast<size_t>(r) * n + c]
                        = (trans == rocsparse_operation_conjugate_transpose) ? conj_of(v) : v;
                }
        }

        const bool lower_after
            = (fill == rocsparse_fill_mode_lower) == (trans == rocsparse_operation_none);
        std::vector<T> y(n, T{});
        for(int32_t s = 0; s < n; ++s)
        {
            const int32_t r   = lower_after ? s : n - 1 - s;
            T             sum = alpha * x[r];
            if(lower_after)
            {
                for(int32_t c = 0; c < r; ++c)
                    sum = sum - op[static_cast<size_t>(r) * n + c] * y[c];
            }
            else
            {
                for(int32_t c = r + 1; c < n; ++c)
                    sum = sum - op[static_cast<size_t>(r) * n + c] * y[c];
            }
            y[r] = sum / op[static_cast<size_t>(r) * n + r];
        }
        return y;
    }

    // ----- block (BSR / GEBSR) matrices ---------------------------------------
    template <typename T>
    struct host_bsr
    {
        int32_t              mb   = 0;
        int32_t              nb   = 0;
        int32_t              rbd  = 0;
        int32_t              cbd  = 0;
        int                  base = 0;
        rocsparse_direction  dir  = rocsparse_direction_row;
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;

        int32_t nnzb() const
        {
            return static_cast<int32_t>(ind.size());
        }

        // Entry (r, c) of block j (relative to the stored layout).
        const T& at(int32_t j, int32_t r, int32_t c) const
        {
            const size_t blk = static_cast<size_t>(rbd) * cbd;
            return (dir == rocsparse_direction_row) ? val[blk * j + r * cbd + c]
                                                    : val[blk * j + c * rbd + r];
        }
    };

    // `bpr` distinct sorted block columns in every block row.
    template <typename T>
    inline host_bsr<T> make_bsr(int32_t             mb,
                                int32_t             nb,
                                int32_t             bpr,
                                int32_t             rbd,
                                int32_t             cbd,
                                rocsparse_direction dir,
                                int                 base = 0)
    {
        host_bsr<T> a;
        a.mb  = mb;
        a.nb  = nb;
        a.rbd = rbd;
        a.cbd = cbd;
        a.dir = dir;
        a.ptr.push_back(0);
        const int32_t stride = std::max<int32_t>(1, nb / bpr);
        int64_t       seed   = 0;
        for(int32_t i = 0; i < mb; ++i)
        {
            std::vector<int32_t> cols;
            const int32_t        start = (i * 5) % nb;
            for(int32_t k = 0; k < bpr; ++k)
                cols.push_back((start + k * stride) % nb);
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            for(int32_t c : cols)
            {
                a.ind.push_back(c);
                for(int32_t e = 0; e < rbd * cbd; ++e)
                    a.val.push_back(value_of<T>(seed++));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        if(base != 0)
        {
            for(auto& p : a.ptr)
                p += base;
            for(auto& c : a.ind)
                c += base;
            a.base = base;
        }
        return a;
    }

    // y[row range] = alpha * A_row * x + beta * y for the listed block rows.
    // `end_ptr` (when not empty) replaces ptr[i + 1] as the row end.
    template <typename T>
    inline std::vector<T> bsr_gemv(const host_bsr<T>&          a,
                                   const std::vector<int32_t>& rows,
                                   const std::vector<int32_t>& end_ptr,
                                   T                           alpha,
                                   const std::vector<T>&       x,
                                   T                           beta,
                                   const std::vector<T>&       y0)
    {
        std::vector<T> y(y0);
        for(int32_t i : rows)
        {
            const int32_t b = a.ptr[i] - a.base;
            const int32_t e = (end_ptr.empty() ? a.ptr[i + 1] : end_ptr[i]) - a.base;
            for(int32_t r = 0; r < a.rbd; ++r)
            {
                T sum = T{};
                for(int32_t j = b; j < e; ++j)
                {
                    const int32_t col = a.ind[j] - a.base;
                    for(int32_t c = 0; c < a.cbd; ++c)
                        sum = sum + a.at(j, r, c) * x[static_cast<size_t>(col) * a.cbd + c];
                }
                y[static_cast<size_t>(i) * a.rbd + r]
                    = beta * y[static_cast<size_t>(i) * a.rbd + r] + alpha * sum;
            }
        }
        return y;
    }

    inline bool all_rows(int32_t m, std::vector<int32_t>& rows)
    {
        rows.resize(m);
        for(int32_t i = 0; i < m; ++i)
            rows[i] = i;
        return true;
    }
} // namespace cov70l2
