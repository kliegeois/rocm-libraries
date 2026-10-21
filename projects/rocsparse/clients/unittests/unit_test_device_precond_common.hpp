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

// Shared helpers of the preconditioner device tests: scalar type utilities, test-matrix
// builders, host references (exact ILU(0), IC(0) and ILDL^T(0) on a scalar CSR matrix)
// and the harness of the generic descriptor API (spilu0 / spic0 / spildlt0).

#pragma once

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
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

    // Overrides a handle member for the lifetime of the guard.
    template <typename T>
    struct ScopedOverride
    {
        T& member;
        T  saved;

        ScopedOverride(T& m, T value)
            : member(m)
            , saved(m)
        {
            member = value;
        }

        ~ScopedOverride()
        {
            member = saved;
        }

        ScopedOverride(const ScopedOverride&) = delete;
        ScopedOverride& operator=(const ScopedOverride&) = delete;
    };

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

    // Tolerance for results that are sums over rows of `row_length` entries (inner sums
    // of IC(0)/ILU(0), the pivot sum of a fat row). The rounding error of such a sum grows
    // with its length and depends on the summation order, which differs between the host
    // (sequential), the wave32 and the wave64 kernels (lane partial sums plus a tree), so a
    // fixed bound measured on one wave size is not portable.
    // Measured (order perturbed host sums: forward, reverse, 64 lane tree, pairwise, in
    // float and double against a long double reference, fat rows of 80..381 entries): the
    // worst per-entry relative error is 0.063 * eps * row_length (sequential float; the
    // tree orders stay below 0.006 * eps * row_length). The bound below is eps * row_length,
    // about 16x the worst order and never below the fixed tolerance() that covers short
    // rows.
    template <typename T>
    double tolerance(int32_t row_length)
    {
        return std::max(tolerance<T>(), epsilon_of<T>() * static_cast<double>(row_length));
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

    // 5-point stencil on a k x k grid. The matrix is Hermitian and strictly
    // diagonally dominant, so ILU(0) and IC(0) both exist without pivoting.
    // `skew` perturbs the lower triangle so that the matrix is not symmetric
    // (ILU(0) only).
    template <typename T>
    csr_matrix<T> make_grid_matrix(int32_t k, bool skew)
    {
        csr_matrix<T> a;
        a.n = k * k;
        a.ptr.push_back(0);
        for(int32_t y = 0; y < k; ++y)
        {
            for(int32_t x = 0; x < k; ++x)
            {
                const int32_t i   = y * k + x;
                const auto    add = [&](int32_t j, double r) {
                    a.ind.push_back(j);
                    const double s = (skew && j < i) ? 1.15 : 1.0;
                    // Hermitian imaginary part: im(i, j) = -im(j, i).
                    const double w    = (std::abs(j - i) == 1) ? 1.0 : 0.5;
                    const double imag = (is_cplx<T> && j != i) ? (j > i ? 0.1 * w : -0.1 * w) : 0.0;
                    a.val.push_back(make_value<T>(r * s, imag));
                };
                if(y > 0)
                    add(i - k, -1.0);
                if(x > 0)
                    add(i - 1, -1.0);
                add(i, 5.0);
                if(x + 1 < k)
                    add(i + 1, -1.0);
                if(y + 1 < k)
                    add(i + k, -1.0);
                a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
            }
        }
        return a;
    }

    // Precondition of the generators below: the longest row does not have exactly
    // 32, 64, 128, 256 or 512 entries (the slot counts of the csric0 / csrildlt0
    // hash tables on a wavefront of 32).
    inline bool is_hash_table_size(int32_t row_length)
    {
        return row_length == 32 || row_length == 64 || row_length == 128 || row_length == 256
               || row_length == 512;
    }

    inline void require_safe_row_length(int32_t row_length)
    {
        if(is_hash_table_size(row_length))
        {
            throw std::invalid_argument("row length " + std::to_string(row_length)
                                        + " equals a csric0/csrildlt0 hash table size");
        }
    }

    // Lower-triangular matrix (stored pattern includes the diagonal) whose last
    // row holds `fat` entries and every other row `band` + 1 entries
    // (r - band .. r). The longest row, `fat`, selects between the hash kernels
    // (fat <= 512) and the binary-search kernels (fat > 512). With band >= 2 the
    // IC(0) / ILDLT(0) inner sums of the last row have several terms.
    // `imag` adds an imaginary part to the off-diagonal entries of complex types.
    template <typename T>
    csr_matrix<T> make_fat_row_matrix(int32_t fat, bool imag = true, int32_t band = 1)
    {
        require_safe_row_length(fat);
        csr_matrix<T> a;
        a.n = fat + 1;
        a.ptr.push_back(0);
        for(int32_t r = 0; r < a.n; ++r)
        {
            const int32_t first = (r == a.n - 1) ? a.n - fat : std::max(0, r - band);
            for(int32_t c = first; c <= r; ++c)
            {
                a.ind.push_back(c);
                if(c == r)
                    a.val.push_back(make_value<T>(4.0));
                else
                    a.val.push_back(
                        make_value<T>(-0.05 * (1.0 + 0.1 * ((r + 2 * c) % 3)),
                                      (imag && is_cplx<T>) ? 0.01 * ((r - c) % 3) : 0.0));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        return a;
    }

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

    // IC(0) of the lower triangle (A ~ L * L^H). Entries above the diagonal
    // are returned unchanged.
    template <typename T>
    std::vector<T> host_ic0(const csr_matrix<T>& a)
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
            const int32_t begin = a.ptr[i];
            for(int32_t j = begin; j < diag[i]; ++j)
            {
                const int32_t        c = a.ind[j];
                std::complex<double> s = 0.0;
                for(int32_t p = begin; p < j; ++p)
                {
                    for(int32_t q = a.ptr[c]; q < diag[c]; ++q)
                    {
                        if(a.ind[q] == a.ind[p])
                        {
                            s += v[p] * std::conj(v[q]);
                            break;
                        }
                    }
                }
                v[j] = (v[j] - s) / v[diag[c]];
            }
            std::complex<double> d = v[diag[i]];
            for(int32_t j = begin; j < diag[i]; ++j)
                d -= v[j] * std::conj(v[j]);
            v[diag[i]] = std::sqrt(d.real());
        }
        std::vector<T> out(v.size());
        for(size_t i = 0; i < v.size(); ++i)
            out[i] = from_cd<T>(v[i]);
        return out;
    }

    // Incomplete LDL^H of the lower triangle: A ~ L * D * L^H with a unit lower
    // L (strictly lower part stored) and a real D stored on the diagonal.
    // boost_tol < 0 disables the numeric boost; otherwise d_i with |d_i| <= boost_tol
    // is replaced by boost_val.
    template <typename T>
    std::vector<T>
        host_ildlt0(const csr_matrix<T>& a, double boost_tol = -1.0, double boost_val = 0.0)
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
            const int32_t begin    = a.ptr[i];
            double        diag_sum = 0.0;
            for(int32_t j = begin; j < diag[i]; ++j)
            {
                const int32_t        c   = a.ind[j];
                const double         d_c = v[diag[c]].real();
                std::complex<double> s   = 0.0;
                for(int32_t p = begin; p < j; ++p)
                {
                    for(int32_t q = a.ptr[c]; q < diag[c]; ++q)
                    {
                        if(a.ind[q] == a.ind[p])
                        {
                            const double d_k = v[diag[a.ind[p]]].real();
                            s += v[p] * (d_k * std::conj(v[q]));
                            break;
                        }
                    }
                }
                v[j] = (v[j] - s) / d_c;
                diag_sum += std::norm(v[j]) * d_c;
            }
            const double d = v[diag[i]].real() - diag_sum;
            v[diag[i]]     = (boost_tol >= 0.0 && std::abs(d) <= boost_tol) ? boost_val : d;
        }
        std::vector<T> out(v.size());
        for(size_t i = 0; i < v.size(); ++i)
            out[i] = from_cd<T>(v[i]);
        return out;
    }

    // Residual max_{(i,j) in lower pattern} | (L*D*L^H)(i,j) - A(i,j) |, L unit lower.
    template <typename T>
    double ildlt_pattern_residual(const csr_matrix<T>& a, const std::vector<T>& f)
    {
        std::vector<int32_t> diag(static_cast<size_t>(a.n), -1);
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] == i)
                    diag[i] = p;
        double res = 0.0;
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1] && a.ind[p] <= i; ++p)
            {
                const int32_t        j = a.ind[p];
                std::complex<double> s = 0.0;
                for(int32_t q = a.ptr[i]; q < a.ptr[i + 1] && a.ind[q] <= j; ++q)
                {
                    const int32_t        k   = a.ind[q];
                    std::complex<double> lik = (k == i) ? 1.0 : to_cd(f[q]);
                    std::complex<double> ljk = 0.0;
                    if(k == j)
                    {
                        ljk = 1.0;
                    }
                    else
                    {
                        for(int32_t r = a.ptr[j]; r < diag[j]; ++r)
                        {
                            if(a.ind[r] == k)
                            {
                                ljk = to_cd(f[r]);
                                break;
                            }
                        }
                    }
                    s += lik * to_cd(f[diag[k]]).real() * std::conj(ljk);
                }
                res = std::max(res, std::abs(s - to_cd(a.val[p])));
            }
        }
        return res;
    }

    // The L factor (entries on or below the scalar diagonal) of a factorized
    // CSR matrix, with every other entry replaced by the original value so
    // that only the factor part is compared.
    template <typename T>
    std::vector<T> keep_lower_only(const csr_matrix<T>& a, const std::vector<T>& factor)
    {
        std::vector<T> out = a.val;
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] <= i)
                    out[p] = factor[p];
        return out;
    }

    // Residual max_{(i,j) in pattern} | (L*U)(i,j) - A(i,j) | of an in-place LU
    // factorization of a CSR matrix (unit lower L). ILU(0) reproduces A exactly
    // on its sparsity pattern.
    template <typename T>
    double ilu_pattern_residual(const csr_matrix<T>& a, const std::vector<T>& lu)
    {
        double res = 0.0;
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
            {
                const int32_t        j = a.ind[p];
                std::complex<double> s = 0.0;
                for(int32_t q = a.ptr[i]; q < a.ptr[i + 1] && a.ind[q] <= std::min(i, j); ++q)
                {
                    const int32_t k = a.ind[q];
                    // L(i,k) * U(k,j); the diagonal of L is one.
                    std::complex<double> l = (k == i) ? 1.0 : to_cd(lu[q]);
                    for(int32_t r = a.ptr[k]; r < a.ptr[k + 1]; ++r)
                    {
                        if(a.ind[r] == j)
                        {
                            s += l * to_cd(lu[r]);
                            break;
                        }
                    }
                }
                res = std::max(res, std::abs(s - to_cd(a.val[p])));
            }
        }
        return res;
    }

    // Residual max_{(i,j) in lower pattern} | (L*L^H)(i,j) - A(i,j) | of an IC(0)
    // factorization.
    template <typename T>
    double ic_pattern_residual(const csr_matrix<T>& a, const std::vector<T>& l)
    {
        double res = 0.0;
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1] && a.ind[p] <= i; ++p)
            {
                const int32_t        j = a.ind[p];
                std::complex<double> s = 0.0;
                for(int32_t q = a.ptr[i]; q < a.ptr[i + 1] && a.ind[q] <= j; ++q)
                {
                    const int32_t k = a.ind[q];
                    for(int32_t r = a.ptr[j]; r < a.ptr[j + 1] && a.ind[r] <= k; ++r)
                    {
                        if(a.ind[r] == k)
                        {
                            s += to_cd(l[q]) * std::conj(to_cd(l[r]));
                            break;
                        }
                    }
                }
                res = std::max(res, std::abs(s - to_cd(a.val[p])));
            }
        }
        return res;
    }

    template <typename T>
    double max_abs_value(const csr_matrix<T>& a)
    {
        double m = 0.0;
        for(const auto& v : a.val)
            m = std::max(m, std::abs(to_cd(v)));
        return m;
    }

    // Residual of a factorization on the pattern, relative to the largest entry of A:
    // about ten times the largest measured on gfx1201 (float/cfloat 2.0e-7, double/cdouble
    // 2.8e-16).
    // For long rows the bound grows like eps * row_length, see tolerance(row_length).
    template <typename T>
    double residual_tolerance(int32_t row_length = 0)
    {
        const double fixed = is_single_precision<T> ? 2.0e-6 : 3.0e-15;
        return std::max(fixed, epsilon_of<T>() * static_cast<double>(row_length));
    }

    template <typename T>
    int32_t max_row_length(const csr_matrix<T>& a)
    {
        int32_t longest = 0;
        for(int32_t i = 0; i < a.n; ++i)
            longest = std::max(longest, a.ptr[i + 1] - a.ptr[i]);
        return longest;
    }

    template <typename T>
    void expect_small_residual(double residual, const csr_matrix<T>& a)
    {
        EXPECT_TRUE(std::isfinite(residual)) << "factor is not finite";
        EXPECT_LE(residual, residual_tolerance<T>(max_row_length(a)) * max_abs_value(a));
    }

    // ------------------------------------------------------------------
    // Generic descriptor API (spilu0 / spic0 / spildlt0) harness
    // ------------------------------------------------------------------
    enum class Kind
    {
        ilu,
        ic,
        ildlt
    };
    enum class In
    {
        alg,
        policy,
        dtype,
        boost_enable,
        boost_tol,
        boost_val,
        sing_tol
    };
    enum class Stage
    {
        analysis,
        compute
    };

    template <Kind K>
    struct api;

#define PRECOND_API(KIND, NAME)                                                                 \
    template <>                                                                                 \
    struct api<KIND>                                                                            \
    {                                                                                           \
        using descr_t = rocsparse_##NAME##_descr;                                               \
        static rocsparse_status create(rocsparse_handle handle, descr_t* descr)                 \
        {                                                                                       \
            return rocsparse_##NAME##_descr_create(handle, descr, nullptr);                     \
        }                                                                                       \
        static rocsparse_status destroy(rocsparse_handle handle, descr_t descr)                 \
        {                                                                                       \
            return rocsparse_##NAME##_descr_destroy(handle, descr, nullptr);                    \
        }                                                                                       \
        static rocsparse_##NAME##_stage stage(Stage s)                                          \
        {                                                                                       \
            return (s == Stage::analysis) ? rocsparse_##NAME##_stage_analysis                   \
                                          : rocsparse_##NAME##_stage_compute;                   \
        }                                                                                       \
        static rocsparse_##NAME##_input input(In in)                                            \
        {                                                                                       \
            static const rocsparse_##NAME##_input table[]                                       \
                = {rocsparse_##NAME##_input_alg,                                                \
                   rocsparse_##NAME##_input_analysis_policy,                                    \
                   rocsparse_##NAME##_input_compute_datatype,                                   \
                   rocsparse_##NAME##_input_boost_enable,                                       \
                   rocsparse_##NAME##_input_boost_tolerance,                                    \
                   rocsparse_##NAME##_input_boost_value,                                        \
                   rocsparse_##NAME##_input_singularity_tolerance};                             \
            return table[static_cast<int>(in)];                                                 \
        }                                                                                       \
        static rocsparse_status set(rocsparse_handle         handle,                            \
                                    descr_t                  descr,                             \
                                    rocsparse_##NAME##_input in,                                \
                                    const void*              data,                              \
                                    size_t                   size)                              \
        {                                                                                       \
            return rocsparse_##NAME##_set_input(handle, descr, in, data, size, nullptr);        \
        }                                                                                       \
        static rocsparse_status buffer_size(rocsparse_handle            handle,                 \
                                            descr_t                     descr,                  \
                                            rocsparse_const_spmat_descr A,                      \
                                            rocsparse_const_spmat_descr P,                      \
                                            rocsparse_##NAME##_stage    stage,                  \
                                            size_t*                     size)                   \
        {                                                                                       \
            return rocsparse_##NAME##_buffer_size(handle, descr, A, P, stage, size, nullptr);   \
        }                                                                                       \
        static rocsparse_status run(rocsparse_handle            handle,                         \
                                    descr_t                     descr,                          \
                                    rocsparse_const_spmat_descr A,                              \
                                    rocsparse_spmat_descr       P,                              \
                                    rocsparse_##NAME##_stage    stage,                          \
                                    size_t                      size,                           \
                                    void*                       buffer)                         \
        {                                                                                       \
            return rocsparse_##NAME(handle, descr, A, P, stage, size, buffer, nullptr);         \
        }                                                                                       \
        static rocsparse_status                                                                 \
            get(rocsparse_handle handle, descr_t descr, bool position, void* data, size_t size) \
        {                                                                                       \
            return rocsparse_##NAME##_get_output(                                               \
                handle,                                                                         \
                descr,                                                                          \
                position ? rocsparse_##NAME##_output_singularity_position                       \
                         : rocsparse_##NAME##_output_singularity,                               \
                data,                                                                           \
                size,                                                                           \
                nullptr);                                                                       \
        }                                                                                       \
    }

    PRECOND_API(Kind::ilu, spilu0);
    PRECOND_API(Kind::ic, spic0);
    PRECOND_API(Kind::ildlt, spildlt0);
#undef PRECOND_API

    template <Kind K>
    rocsparse_status set_value(rocsparse_handle         handle,
                               typename api<K>::descr_t descr,
                               In                       in,
                               const void*              data,
                               size_t                   size)
    {
        return api<K>::set(handle, descr, api<K>::input(in), data, size);
    }

    struct run_options
    {
        rocsparse_analysis_policy policy       = rocsparse_analysis_policy_reuse;
        bool                      boost        = false;
        double                    boost_tol    = 0.0;
        double                    boost_val    = 1.0;
        bool                      set_sing_tol = false;
        double                    sing_tol     = 0.0;
        bool                      tol_after    = false; // set the tolerance after the analysis
        size_t device_memory = 0; // > 0: size of the internal buffer of the handle
    };

    template <typename T>
    struct run_result
    {
        rocsparse_status                   analysis_status = rocsparse_status_internal_error;
        rocsparse_status                   compute_status  = rocsparse_status_internal_error;
        std::vector<T>                     values;
        std::vector<rocsparse_singularity> singularity;
        std::vector<int64_t>               position;
    };

    // Destroys the descriptor when a failed ASSERT leaves the function early.
    template <Kind K>
    struct descr_guard
    {
        rocsparse_handle         handle;
        typename api<K>::descr_t descr = nullptr;

        explicit descr_guard(rocsparse_handle h)
            : handle(h)
        {
        }

        ~descr_guard()
        {
            if(descr)
                (void)api<K>::destroy(handle, descr);
        }

        descr_guard(const descr_guard&) = delete;
        descr_guard& operator=(const descr_guard&) = delete;
    };

    // Analysis then compute through the generic API on matrix `A`. `val` is the
    // device value array (nvals entries, all batches) that holds the result.
    template <Kind K, typename T>
    void run_factorization(rocsparse_handle        handle,
                           rocsparse_spmat_descr   A,
                           const device_vector<T>& val,
                           int                     batch,
                           const run_options&      opt,
                           run_result<T>&          out)
    {
        descr_guard<K> guard(handle);
        auto&          descr = guard.descr;
        // The library only ever sees a smaller buffer than the real one.
        std::unique_ptr<ScopedOverride<size_t>> shrink;
        if(opt.device_memory > 0)
            shrink
                = std::make_unique<ScopedOverride<size_t>>(handle->buffer_size, opt.device_memory);
        ASSERT_EQ(api<K>::create(handle, &descr), rocsparse_status_success);

        const rocsparse_datatype dtype = dt_of<T>();
        ASSERT_EQ((set_value<K>(handle, descr, In::dtype, &dtype, sizeof(dtype))),
                  rocsparse_status_success);
        const int32_t alg = 0;
        ASSERT_EQ((set_value<K>(handle, descr, In::alg, &alg, sizeof(int32_t))),
                  rocsparse_status_success);
        ASSERT_EQ((set_value<K>(handle, descr, In::policy, &opt.policy, sizeof(opt.policy))),
                  rocsparse_status_success);

        const int32_t boost_enable  = opt.boost ? 1 : 0;
        const double  boost_tol     = opt.boost_tol;
        const T       boost_val     = make_value<T>(opt.boost_val, 0.0);
        const void*   boost_val_ptr = &boost_val;
        const double  sing_tol      = opt.sing_tol;
        const void*   sing_tol_ptr  = &sing_tol;
        if(opt.boost)
        {
            ASSERT_EQ(
                (set_value<K>(handle, descr, In::boost_enable, &boost_enable, sizeof(int32_t))),
                rocsparse_status_success);
            ASSERT_EQ((set_value<K>(handle, descr, In::boost_tol, &boost_tol, sizeof(double))),
                      rocsparse_status_success);
            ASSERT_EQ((set_value<K>(handle, descr, In::boost_val, boost_val_ptr, sizeof(void*))),
                      rocsparse_status_success);
        }
        if(opt.set_sing_tol && !opt.tol_after)
        {
            ASSERT_EQ((set_value<K>(handle, descr, In::sing_tol, sing_tol_ptr, sizeof(void*))),
                      rocsparse_status_success);
        }

        size_t analysis_size = 0;
        ASSERT_EQ(api<K>::buffer_size(
                      handle, descr, A, A, api<K>::stage(Stage::analysis), &analysis_size),
                  rocsparse_status_success);
        device_vector<char> analysis_buffer{analysis_size ? analysis_size : size_t{1}};
        ASSERT_TRUE(analysis_buffer.ptr);
        out.analysis_status = api<K>::run(handle,
                                          descr,
                                          A,
                                          A,
                                          api<K>::stage(Stage::analysis),
                                          analysis_size,
                                          analysis_size ? analysis_buffer.ptr : nullptr);
        ASSERT_EQ(out.analysis_status, rocsparse_status_success);

        if(opt.set_sing_tol && opt.tol_after)
        {
            ASSERT_EQ((set_value<K>(handle, descr, In::sing_tol, sing_tol_ptr, sizeof(void*))),
                      rocsparse_status_success);
        }

        size_t compute_size = 0;
        ASSERT_EQ(
            api<K>::buffer_size(handle, descr, A, A, api<K>::stage(Stage::compute), &compute_size),
            rocsparse_status_success);
        device_vector<char> compute_buffer{compute_size ? compute_size : size_t{1}};
        ASSERT_TRUE(compute_buffer.ptr);

        out.compute_status = api<K>::run(handle,
                                         descr,
                                         A,
                                         A,
                                         api<K>::stage(Stage::compute),
                                         compute_size,
                                         compute_size ? compute_buffer.ptr : nullptr);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        out.values = to_host(val);

        // The header makes each output an array of batch_count elements, but does not say
        // whether the data size covers the array or one element: the array size is tried
        // first, then the size of one element. A sentinel after the `batch` entries proves
        // that no more is written.
        const auto get_output = [&](bool position, void* data, size_t element_size) {
            rocsparse_status st = api<K>::get(
                handle, descr, position, data, static_cast<size_t>(batch) * element_size);
            if(st != rocsparse_status_success && batch > 1)
                st = api<K>::get(handle, descr, position, data, element_size);
            return st;
        };
        const auto    sing_sentinel = static_cast<rocsparse_singularity>(-123);
        const int64_t pos_sentinel  = -987654321;
        out.singularity.assign(static_cast<size_t>(batch) + 1, rocsparse_singularity_numeric_near);
        out.singularity.back() = sing_sentinel;
        out.position.assign(static_cast<size_t>(batch) + 1, -2);
        out.position.back() = pos_sentinel;
        ASSERT_EQ(get_output(false, out.singularity.data(), sizeof(rocsparse_singularity)),
                  rocsparse_status_success);
        ASSERT_EQ(get_output(true, out.position.data(), sizeof(int64_t)), rocsparse_status_success);
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        EXPECT_EQ(out.singularity.back(), sing_sentinel);
        EXPECT_EQ(out.position.back(), pos_sentinel);
        out.singularity.pop_back();
        out.position.pop_back();

        EXPECT_EQ(api<K>::destroy(handle, descr), rocsparse_status_success);
        descr = nullptr;
    }

    // CSR matrix on the device, optionally batched (each batch entry scales the
    // base values by 1 + 0.25 * b).
    template <typename T, typename I, typename J>
    struct dev_csr
    {
        csr_matrix<T>              base;
        std::vector<csr_matrix<T>> batches;
        device_vector<I>           row;
        device_vector<J>           col;
        device_vector<T>           val;
        rocsparse_spmat_descr      A = nullptr;

        dev_csr()               = default;
        dev_csr(const dev_csr&) = delete;
        dev_csr& operator=(const dev_csr&) = delete;

        ~dev_csr()
        {
            if(A)
                (void)rocsparse_destroy_spmat_descr(A);
        }

        // `m` is zero based; the arrays handed to the library are shifted by one
        // when `index_base` is one.
        void init(const csr_matrix<T>& m,
                  int                  batch,
                  bool                 shared_row_ptr,
                  rocsparse_index_base index_base = rocsparse_index_base_zero)
        {
            const I shift = (index_base == rocsparse_index_base_one) ? 1 : 0;
            base          = m;
            batches.clear();
            std::vector<I> hrow;
            std::vector<J> hcol;
            std::vector<T> hval;
            for(int b = 0; b < batch; ++b)
            {
                csr_matrix<T> mb = m;
                const double  s  = 1.0 + 0.25 * b;
                for(auto& v : mb.val)
                    v = make_value<T>(re(v) * s, im(v) * s);
                batches.push_back(mb);
                if(b == 0 || !shared_row_ptr)
                {
                    for(const auto p : m.ptr)
                        hrow.push_back(static_cast<I>(p) + shift);
                }
                for(const auto c : m.ind)
                    hcol.push_back(static_cast<J>(c) + static_cast<J>(shift));
                hval.insert(hval.end(), mb.val.begin(), mb.val.end());
            }
            upload(row, hrow);
            upload(col, hcol);
            upload(val, hval);
            ASSERT_TRUE(row.ptr && col.ptr && val.ptr);
            ASSERT_EQ(rocsparse_create_csr_descr(&A,
                                                 m.n,
                                                 m.n,
                                                 m.nnz(),
                                                 row.ptr,
                                                 col.ptr,
                                                 val.ptr,
                                                 it_of<I>(),
                                                 it_of<J>(),
                                                 index_base,
                                                 dt_of<T>()),
                      rocsparse_status_success);
            if(batch > 1)
            {
                ASSERT_EQ(rocsparse_csr_set_strided_batch(
                              A, batch, shared_row_ptr ? 0 : m.n + 1, m.nnz()),
                          rocsparse_status_success);
            }
        }
    };

    template <typename F>
    void for_all_types(F&& f)
    {
        f(float{});
        f(double{});
        f(rocsparse_float_complex{});
        f(rocsparse_double_complex{});
    }

    // Expected factor of a scalar matrix for each kind. For IC(0) only the
    // lower triangle is a factor; the entries above it are not compared. The IC(0)
    // reference has no numeric boost (its semantics are not documented).
    template <Kind K, typename T>
    std::vector<T> reference(const csr_matrix<T>& a, const run_options& opt)
    {
        const double boost_tol = opt.boost ? opt.boost_tol : -1.0;
        if constexpr(K == Kind::ilu)
            return host_ilu0(a, boost_tol, opt.boost_val);
        else if constexpr(K == Kind::ic)
        {
            if(opt.boost)
                throw std::invalid_argument("no IC(0) reference with a numeric boost");
            return keep_lower_only(a, host_ic0(a));
        }
        else
            return host_ildlt0(a, boost_tol, opt.boost_val);
    }

    // Compares one factorized scalar matrix with the host reference and checks the residual.
    template <Kind K, typename T>
    void check_factor(const csr_matrix<T>&  a,
                      const std::vector<T>& got_in,
                      const run_options&    opt,
                      const std::string&    what)
    {
        SCOPED_TRACE(what);
        std::vector<T> got = got_in;
        if constexpr(K == Kind::ic)
            got = keep_lower_only(a, got_in);
        expect_close(got, reference<K>(a, opt), tolerance<T>(max_row_length(a)));
        if(!opt.boost)
        {
            double res = 0.0;
            if constexpr(K == Kind::ilu)
                res = ilu_pattern_residual(a, got);
            else if constexpr(K == Kind::ic)
                res = ic_pattern_residual(a, got);
            else
                res = ildlt_pattern_residual(a, got);
            expect_small_residual(res, a);
        }
    }

    // Factorization of one scalar CSR matrix through the generic API, compared
    // with the host reference. `result` (optional) receives the raw outputs.
    template <Kind K, typename T, typename I = int32_t, typename J = int32_t>
    void check_csr_matrix(rocsparse_handle     handle,
                          const csr_matrix<T>& m,
                          const run_options&   opt,
                          const std::string&   what,
                          rocsparse_index_base base   = rocsparse_index_base_zero,
                          run_result<T>*       result = nullptr)
    {
        SCOPED_TRACE(what);
        dev_csr<T, I, J> d;
        d.init(m, 1, true, base);
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<K, T>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        check_factor<K>(d.base, out.values, opt, what);
        EXPECT_EQ(out.singularity[0], rocsparse_singularity_none);
        EXPECT_EQ(out.position[0], -1);
        if(result != nullptr)
            *result = out;
    }

    // Position of the diagonal entry of `row` in the CSR arrays.
    template <typename T>
    size_t diagonal_position(const csr_matrix<T>& a, int32_t row)
    {
        for(int32_t p = a.ptr[row]; p < a.ptr[row + 1]; ++p)
            if(a.ind[p] == row)
                return static_cast<size_t>(p);
        return static_cast<size_t>(-1);
    }
} // namespace precond_ut
