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

// Shared helpers of the cov70 preconditioner device tests: scalar type
// utilities, test-matrix builders and host references (exact ILU(0) and IC(0)
// on a scalar CSR matrix, plus the scalar expansion of a BSR matrix).

#pragma once

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cov70
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

    // Makes the handle report another wavefront size for the numeric call.
    struct ScopedWavefront
    {
        rocsparse_handle handle;
        int              saved;

        ScopedWavefront(rocsparse_handle h, int wavefront)
            : handle(h)
            , saved(h->wavefront_size)
        {
            handle->wavefront_size = wavefront;
        }

        ~ScopedWavefront()
        {
            handle->wavefront_size = saved;
        }

        ScopedWavefront(const ScopedWavefront&) = delete;
        ScopedWavefront& operator=(const ScopedWavefront&) = delete;
    };

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
    double tolerance()
    {
        return (std::is_same_v<T, float> || std::is_same_v<T, rocsparse_float_complex>) ? 2.0e-4
                                                                                        : 1.0e-9;
    }

    // Element-wise comparison of every entry (real and imaginary parts).
    template <typename T>
    void expect_close(const std::vector<T>& got, const std::vector<T>& want, double scale_tol = 1.0)
    {
        ASSERT_EQ(got.size(), want.size());
        const double tol = tolerance<T>() * scale_tol;
        for(size_t i = 0; i < got.size(); ++i)
        {
            const double scale
                = std::max(1.0, std::max(std::abs(re(want[i])), std::abs(im(want[i]))));
            EXPECT_NEAR(re(got[i]), re(want[i]), tol * scale) << "re " << i;
            EXPECT_NEAR(im(got[i]), im(want[i]), tol * scale) << "im " << i;
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

    // Exact ILU(0) in place (unit L below the diagonal, U on and above it).
    // Lower-triangular matrix (stored pattern includes the diagonal) whose last
    // row holds `fat` entries and every other row two (r-1, r). The row-length
    // statistic that selects between hash and binary-search kernels is `fat`.
    template <typename T>
    csr_matrix<T> make_fat_row_matrix(int32_t fat, bool imag = true)
    {
        csr_matrix<T> a;
        a.n = fat + 1;
        a.ptr.push_back(0);
        for(int32_t r = 0; r < a.n; ++r)
        {
            const int32_t first = (r == a.n - 1) ? a.n - fat : std::max(0, r - 1);
            for(int32_t c = first; c <= r; ++c)
            {
                a.ind.push_back(c);
                if(c == r)
                    a.val.push_back(make_value<T>(4.0));
                else
                    a.val.push_back(
                        make_value<T>(-0.05, (imag && is_cplx<T>) ? 0.01 * ((r - c) % 3) : 0.0));
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
    template <typename T>
    std::vector<T> host_ildlt0(const csr_matrix<T>& a)
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
            v[diag[i]] = v[diag[i]].real() - diag_sum;
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

    // ------------------------------------------------------------------
    // BSR matrices (stored as dense blocks) and their scalar expansion
    // ------------------------------------------------------------------
    template <typename T>
    struct bsr_matrix
    {
        int32_t              mb        = 0;
        int32_t              block_dim = 0;
        rocsparse_direction  dir       = rocsparse_direction_row;
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;

        int32_t nnzb() const
        {
            return static_cast<int32_t>(ind.size());
        }
    };

    // Block tridiagonal pattern (blocks I-1, I, I+1 present when `bandwidth`
    // is 1; only the diagonal block when it is 0, block I-1 .. I when `lower`).
    // The scalar matrix is Hermitian and strictly diagonally dominant;
    // `skew` breaks the symmetry of the lower triangle (ILU(0) only).
    template <typename T>
    bsr_matrix<T> make_block_matrix(
        int32_t mb, int32_t block_dim, rocsparse_direction dir, bool skew, bool lower_only = false)
    {
        bsr_matrix<T> m;
        m.mb        = mb;
        m.block_dim = block_dim;
        m.dir       = dir;
        m.ptr.push_back(0);
        const int32_t b = block_dim;
        for(int32_t r = 0; r < mb; ++r)
        {
            for(int32_t c = std::max(0, r - 1); c <= std::min(mb - 1, r + (lower_only ? 0 : 1));
                ++c)
                m.ind.push_back(c);
            m.ptr.push_back(static_cast<int32_t>(m.ind.size()));
        }
        m.val.assign(static_cast<size_t>(m.nnzb()) * b * b, make_value<T>(0.0));
        for(int32_t r = 0; r < mb; ++r)
        {
            for(int32_t k = m.ptr[r]; k < m.ptr[r + 1]; ++k)
            {
                const int32_t c = m.ind[k];
                for(int32_t i = 0; i < b; ++i)
                {
                    for(int32_t j = 0; j < b; ++j)
                    {
                        const int32_t gi = r * b + i;
                        const int32_t gj = c * b + j;
                        double        rv = 0.0;
                        double        iv = 0.0;
                        if(gi == gj)
                        {
                            rv = 4.0 * b;
                        }
                        else
                        {
                            rv = -0.3 * (1.0 + 0.1 * ((gi + gj) % 3));
                            iv = is_cplx<T> ? 0.02 * (gi - gj) : 0.0;
                            if(skew && gi > gj)
                                rv *= 1.2;
                        }
                        const size_t idx = (dir == rocsparse_direction_row)
                                               ? static_cast<size_t>(i) * b + j
                                               : static_cast<size_t>(j) * b + i;
                        m.val[static_cast<size_t>(k) * b * b + idx] = make_value<T>(rv, iv);
                    }
                }
            }
        }
        return m;
    }

    template <typename T>
    csr_matrix<T> expand_bsr(const bsr_matrix<T>& m)
    {
        const int32_t b = m.block_dim;
        csr_matrix<T> a;
        a.n = m.mb * b;
        a.ptr.push_back(0);
        for(int32_t r = 0; r < m.mb; ++r)
        {
            for(int32_t i = 0; i < b; ++i)
            {
                for(int32_t k = m.ptr[r]; k < m.ptr[r + 1]; ++k)
                {
                    for(int32_t j = 0; j < b; ++j)
                    {
                        const size_t idx = (m.dir == rocsparse_direction_row)
                                               ? static_cast<size_t>(i) * b + j
                                               : static_cast<size_t>(j) * b + i;
                        a.ind.push_back(m.ind[k] * b + j);
                        a.val.push_back(m.val[static_cast<size_t>(k) * b * b + idx]);
                    }
                }
                a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
            }
        }
        return a;
    }

    // Scatters scalar CSR values (of the expansion) back into BSR block storage.
    template <typename T>
    std::vector<T> collapse_bsr(const bsr_matrix<T>& m, const std::vector<T>& scalar_val)
    {
        const int32_t  b = m.block_dim;
        std::vector<T> out(m.val.size());
        size_t         pos = 0;
        for(int32_t r = 0; r < m.mb; ++r)
        {
            for(int32_t i = 0; i < b; ++i)
            {
                for(int32_t k = m.ptr[r]; k < m.ptr[r + 1]; ++k)
                {
                    for(int32_t j = 0; j < b; ++j)
                    {
                        const size_t idx = (m.dir == rocsparse_direction_row)
                                               ? static_cast<size_t>(i) * b + j
                                               : static_cast<size_t>(j) * b + i;
                        out[static_cast<size_t>(k) * b * b + idx] = scalar_val[pos++];
                    }
                }
            }
        }
        return out;
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
} // namespace cov70
