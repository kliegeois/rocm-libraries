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

// Shared helpers of the preconditioner device tests: device upload, typed s / d / c / z
// calls, scalar type utilities, the scalar CSR matrix and the exact ILU(0) host reference.

#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace precond_ut
{
    using namespace rocsparse_ut;

    // Replaces the content of a device vector.
    template <typename T>
    void upload(device_vector<T>& d, const std::vector<T>& host)
    {
        if(d.ptr)
            (void)hipFree(d.ptr);
        d.ptr = nullptr;
        d.n   = host.size();
        if(hipMalloc(&d.ptr, d.n * sizeof(T)) != hipSuccess)
        {
            d.ptr = nullptr;
            return;
        }
        (void)hipMemcpy(d.ptr, host.data(), d.n * sizeof(T), hipMemcpyHostToDevice);
    }

    // Index array for the entry points that take rocsparse_int (64-bit with rocsparse_ILP64).
    inline std::vector<rocsparse_int> as_rocsparse_int(const std::vector<int32_t>& v)
    {
        return std::vector<rocsparse_int>(v.begin(), v.end());
    }

    // Calls the s / d / c / z variant of a routine matching T: the four names are passed
    // as function pointers and only the matching call is instantiated.
    template <typename T, typename S, typename D, typename C, typename Z, typename... Args>
    rocsparse_status typed_call(S s, D d, C c, Z z, Args&&... args)
    {
        if constexpr(std::is_same_v<T, float>)
            return s(std::forward<Args>(args)...);
        else if constexpr(std::is_same_v<T, double>)
            return d(std::forward<Args>(args)...);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return c(std::forward<Args>(args)...);
        else
            return z(std::forward<Args>(args)...);
    }

#define PRECOND_CALL(T, NAME, ...) \
    ::precond_ut::typed_call<T>(   \
        rocsparse_s##NAME, rocsparse_d##NAME, rocsparse_c##NAME, rocsparse_z##NAME, __VA_ARGS__)

    template <typename T>
    using real_of = std::conditional_t<
        std::is_same_v<T, double> || std::is_same_v<T, rocsparse_double_complex>,
        double,
        float>;

    template <typename T>
    constexpr bool is_cplx
        = std::is_same_v<T, rocsparse_float_complex> || std::is_same_v<T, rocsparse_double_complex>;

    template <typename T>
    double re(T v)
    {
        return static_cast<double>(std::real(v));
    }

    template <typename T>
    double im(T v)
    {
        return static_cast<double>(std::imag(v));
    }

    template <typename T>
    T make_value(double r, double i = 0.0)
    {
        if constexpr(std::is_same_v<T, float>)
            return static_cast<float>(r);
        else if constexpr(std::is_same_v<T, double>)
            return r;
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_float_complex(static_cast<float>(r), static_cast<float>(i));
        else
            return rocsparse_double_complex(r, i);
    }

    template <typename T>
    std::complex<double> to_cd(T v)
    {
        return std::complex<double>(re(v), im(v));
    }

    template <typename T>
    T from_cd(std::complex<double> v)
    {
        return make_value<T>(v.real(), v.imag());
    }

    template <typename T>
    constexpr bool is_single_precision
        = std::is_same_v<T, float> || std::is_same_v<T, rocsparse_float_complex>;

    // Per-entry relative tolerance of a direct factorization or solve against a
    // double precision host reference: about ten times the largest error measured
    // on gfx1201 over every test using it (float/cfloat: 3.1e-7, double/cdouble:
    // 1.6e-15).
    template <typename T>
    double tolerance()
    {
        return is_single_precision<T> ? 3.0e-6 : 2.0e-14;
    }

    // Unit roundoff scale of the value type (epsilon of the real type).
    template <typename T>
    constexpr double epsilon_of()
    {
        return is_single_precision<T> ? 1.1920929e-7 : 2.220446049250313e-16;
    }

    // Entries below this fraction of the largest reference entry are compared with
    // an absolute allowance (they may be exact zeros or the result of cancellation).
    constexpr double small_entry_fraction = 1.0e-3;

    // Element-wise comparison: |got - want| <= rel_tol * max(|want|, floor) for every
    // entry (complex modulus), with floor = small_entry_fraction * max |want|.
    template <typename T>
    void expect_close(const std::vector<T>& got,
                      const std::vector<T>& want,
                      double                rel_tol = tolerance<T>())
    {
        ASSERT_EQ(got.size(), want.size());
        // The reference is computed on the host from generated data: a non-finite value
        // means the generated matrix is broken (not positive definite, zero pivot), not that
        // the library is wrong. Report that instead of "nan vs nan" for every entry.
        for(size_t i = 0; i < want.size(); ++i)
        {
            if(!std::isfinite(re(want[i])) || !std::isfinite(im(want[i])))
            {
                ADD_FAILURE() << "host reference is not finite at entry " << i << " ("
                              << re(want[i]) << ", " << im(want[i])
                              << "): the generated test matrix is not positive definite / has a "
                                 "zero pivot, the comparison is meaningless";
                return;
            }
        }
        double max_want = 0.0;
        for(const auto& w : want)
            max_want = std::max(max_want, std::abs(to_cd(w)));
        const double floor = small_entry_fraction * max_want;
        for(size_t i = 0; i < got.size(); ++i)
        {
            const double w = std::abs(to_cd(want[i]));
            EXPECT_LE(std::abs(to_cd(got[i]) - to_cd(want[i])), rel_tol * std::max(w, floor))
                << "entry " << i << " got (" << re(got[i]) << ", " << im(got[i]) << ") want ("
                << re(want[i]) << ", " << im(want[i]) << ")";
        }
    }

    // ------------------------------------------------------------------
    // Scalar CSR matrices
    // ------------------------------------------------------------------
    template <typename T>
    struct csr_matrix
    {
        int32_t              n = 0;
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;

        int32_t nnz() const
        {
            return static_cast<int32_t>(ind.size());
        }
    };

    // boost_tol < 0 disables the numeric boost. Otherwise a pivot with
    // |pivot| <= boost_tol is replaced by |boost_val| * pivot / |pivot|.
    template <typename T>
    std::vector<T>
        host_ilu0(const csr_matrix<T>& a, double boost_tol = -1.0, double boost_val = 0.0)
    {
        std::vector<std::complex<double>> v(a.val.size());
        for(size_t i = 0; i < v.size(); ++i)
            v[i] = to_cd(a.val[i]);
        std::vector<int32_t> diag(static_cast<size_t>(a.n), -1);
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] == i)
                    diag[i] = p;

        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1] && a.ind[p] < i; ++p)
            {
                const int32_t k = a.ind[p];
                v[p] /= v[diag[k]];
                for(int32_t q = p + 1; q < a.ptr[i + 1]; ++q)
                {
                    for(int32_t r = a.ptr[k]; r < a.ptr[k + 1]; ++r)
                    {
                        if(a.ind[r] == a.ind[q] && a.ind[r] > k)
                        {
                            v[q] -= v[p] * v[r];
                            break;
                        }
                    }
                }
            }
            if(boost_tol >= 0.0 && diag[i] >= 0)
            {
                const double mag = std::abs(v[diag[i]]);
                if(mag <= boost_tol)
                {
                    v[diag[i]] = (mag > 0.0) ? std::abs(boost_val) * (v[diag[i]] / mag)
                                             : std::complex<double>(std::abs(boost_val), 0.0);
                }
            }
        }
        std::vector<T> out(v.size());
        for(size_t i = 0; i < v.size(); ++i)
            out[i] = from_cd<T>(v[i]);
        return out;
    }

    template <typename T>
    int32_t max_row_length(const csr_matrix<T>& a)
    {
        int32_t longest = 0;
        for(int32_t i = 0; i < a.n; ++i)
            longest = std::max(longest, a.ptr[i + 1] - a.ptr[i]);
        return longest;
    }
} // namespace precond_ut
