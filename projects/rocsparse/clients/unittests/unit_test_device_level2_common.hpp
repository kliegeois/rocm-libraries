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
// Helpers shared by the level2 device unit tests: deterministic data, host sparse
// containers and formats, RAII descriptors, precision dispatch of the C API and
// host reference products.
//
#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace level2
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

    // Deterministic values: multiples of 1/16 in [-1.875, 1.875]. The real part has
    // period 61 and the imaginary part period 53 (both prime), so the sequence
    // does not repeat for any block size or row length used by the tests, and
    // an index that is off by a small multiple of a block dimension reads a
    // different value. All sums and products of these values (a few thousand
    // terms at most) are exact in single precision, which keeps the results of
    // atomic reductions independent of the summation order.
    // The value is zero for one `n` in 61 (a deliberate sprinkling of zeros);
    // with `nonzero` set the zero is replaced, so the value can be used for a
    // diagonal entry.
    template <typename T>
    inline T value_of(int64_t n, bool nonzero = false)
    {
        int64_t k = (n * 37 + 11) % 61 - 30;
        if(nonzero && k == 0)
            k = 7;
        return rocsparse_ut::scalar<T>(static_cast<float>(k) / 16.0f);
    }
    template <>
    inline rocsparse_float_complex value_of<rocsparse_float_complex>(int64_t n, bool nonzero)
    {
        const float im = static_cast<float>((n * 23 + 5) % 53 - 26) / 16.0f;
        return rocsparse_float_complex(value_of<float>(n, nonzero), im);
    }
    template <>
    inline rocsparse_double_complex value_of<rocsparse_double_complex>(int64_t n, bool nonzero)
    {
        const double im = static_cast<double>((n * 23 + 5) % 53 - 26) / 16.0;
        return rocsparse_double_complex(value_of<double>(n, nonzero), im);
    }

    // ----- comparison -------------------------------------------------------
    // Compares two vectors element by element. Reports the first mismatch (and
    // the count of mismatches) once, so a wrong kernel does not flood the log.
    template <typename T>
    inline bool expect_near(const std::vector<T>& want, const std::vector<T>& got, const char* what)
    {
        if(want.size() != got.size())
        {
            ADD_FAILURE() << what << ": size " << got.size() << " != " << want.size();
            return false;
        }
        const double tol   = tolerance<T>();
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
    // Owns a descriptor created by `create(&d)`; a failed creation leaves d null.
    template <typename D, auto create, auto destroy>
    struct created
    {
        D d = nullptr;
        created()
        {
            if(create(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~created()
        {
            if(d)
                (void)destroy(d);
        }
        created(const created&) = delete;
        created& operator=(const created&) = delete;
                 operator D() const
        {
            return d;
        }
    };

    // Owns a descriptor that the test creates itself (`create_*_descr(&x.d, ...)`).
    template <typename D, auto destroy>
    struct owned
    {
        D d                 = nullptr;
        owned()             = default;
        owned(const owned&) = delete;
        owned& operator=(const owned&) = delete;
        ~owned()
        {
            if(d)
                (void)destroy(d);
        }
        operator D() const
        {
            return d;
        }
    };

    using sptrsv_descr_guard = created<rocsparse_sptrsv_descr,
                                       rocsparse_create_sptrsv_descr,
                                       rocsparse_destroy_sptrsv_descr>;
    using spmat_guard        = owned<rocsparse_spmat_descr, rocsparse_destroy_spmat_descr>;
    using dnvec_guard        = owned<rocsparse_dnvec_descr, rocsparse_destroy_dnvec_descr>;

    struct scoped_pointer_mode
    {
        rocsparse_handle       handle;
        rocsparse_pointer_mode saved = rocsparse_pointer_mode_host;

        scoped_pointer_mode(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
        {
            EXPECT_EQ(rocsparse_get_pointer_mode(handle, &saved), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_pointer_mode(handle, mode), rocsparse_status_success);
        }

        scoped_pointer_mode(rocsparse_handle h, bool device)
            : scoped_pointer_mode(
                h, device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host)
        {
        }

        ~scoped_pointer_mode()
        {
            (void)rocsparse_set_pointer_mode(handle, saved);
        }

        scoped_pointer_mode(const scoped_pointer_mode&) = delete;
        scoped_pointer_mode& operator=(const scoped_pointer_mode&) = delete;
    };

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

    // A diagonally dominant, sorted n x n pattern with `per_row` off-diagonal
    // candidates per row on each side. The diagonal is present (non-zero) and
    // dominant, so triangular solves are well conditioned.
    //   lower / upper: keep only that triangle (plus the diagonal),
    //   both:          keep the full pattern.
    // `with_diag` false omits the diagonal entries (the pattern of a matrix with an
    // implicit unit diagonal).
    // Every off-diagonal entry is non-zero. The magnitude of the diagonal exceeds
    // the sum of the magnitudes of one triangle by a wide margin.
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
                    d   = rocsparse_ut::scalar<T>(static_cast<float>(4 * per_row + 2)) + d;
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
} // namespace level2
