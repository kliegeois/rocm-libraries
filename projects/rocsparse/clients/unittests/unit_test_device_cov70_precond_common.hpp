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
#include <stdexcept>
#include <string>
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

    // The CSR hash kernels of csric0 and csrildlt0 keep the entries of the current
    // row in a shared-memory table of 32, 64, 128, 256 or 512 slots (wavefront 32:
    // table size = 32 * HASH, HASH picked from the longest row with
    // max_nnz <= 32 / 64 / 128 / 256 / 512). A row of EXACTLY that many entries
    // fills the table and the lookup of a column that is not in the row then probes
    // forever: the kernel never returns and the GPU hangs. No test may build a matrix
    // whose longest row has 32, 64, 128, 256 or 512 entries; the generators below
    // throw instead of launching such a matrix (csrilu0 switches to the binary
    // search kernel from 512 entries and is not affected, but it is kept away from
    // these lengths as well).
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
                                        + " fills the csric0/csrildlt0 hash table and hangs");
        }
    }

    // Lower-triangular matrix (stored pattern includes the diagonal) whose last
    // row holds `fat` entries and every other row `band` + 1 entries
    // (r - band .. r). The longest row, `fat`, selects between the hash kernels
    // (fat <= 512) and the binary-search kernels (fat > 512). With band >= 2 the
    // IC(0) / ILDLT(0) inner sums of the last row have several terms.
    // `imag` adds an imaginary part to the off-diagonal entries of complex types
    // (the complex IC(0) kernels conjugate the wrong operand, so IC(0) tests on
    // complex types must pass imag = false: the products are then real).
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

    // Lower band matrix of order n: row r holds the columns max(0, r - w) .. r, so the
    // longest row has min(n, w + 1) entries and every IC(0) / ILDLT(0) inner sum
    // runs over up to w terms. The diagonal grows with w so that the symmetric
    // matrix (the lower triangle mirrored) stays positive definite. See make_fat_row_matrix for `imag` and the row
    // lengths that must be avoided.
    template <typename T>
    csr_matrix<T> make_lower_band_matrix(int32_t n, int32_t w, bool imag = true)
    {
        require_safe_row_length(std::min(n, w + 1));
        csr_matrix<T> a;
        a.n = n;
        a.ptr.push_back(0);
        for(int32_t r = 0; r < n; ++r)
        {
            for(int32_t c = std::max(0, r - w); c <= r; ++c)
            {
                a.ind.push_back(c);
                if(c == r)
                    a.val.push_back(
                        make_value<T>(1.0 + 0.15 * std::min(w, n - 1) + 0.25 * (r % 3)));
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

    // Fills the dense blocks of a BSR matrix whose pattern (ptr / ind) is set. The
    // scalar matrix is strictly diagonally dominant for the block tridiagonal
    // pattern and Hermitian (its lower triangle is what IC(0) reads); `skew` breaks
    // the symmetry of the lower triangle (ILU(0) only).
    template <typename T>
    void fill_block_values(bsr_matrix<T>& m, bool skew)
    {
        const int32_t b = m.block_dim;
        m.val.assign(static_cast<size_t>(m.nnzb()) * b * b, make_value<T>(0.0));
        for(int32_t r = 0; r < m.mb; ++r)
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
                        const size_t idx = (m.dir == rocsparse_direction_row)
                                               ? static_cast<size_t>(i) * b + j
                                               : static_cast<size_t>(j) * b + i;
                        m.val[static_cast<size_t>(k) * b * b + idx] = make_value<T>(rv, iv);
                    }
                }
            }
        }
    }

    // Block tridiagonal pattern: block row I holds the blocks I-1, I and I+1.
    template <typename T>
    bsr_matrix<T>
        make_block_matrix(int32_t mb, int32_t block_dim, rocsparse_direction dir, bool skew)
    {
        bsr_matrix<T> m;
        m.mb        = mb;
        m.block_dim = block_dim;
        m.dir       = dir;
        m.ptr.push_back(0);
        for(int32_t r = 0; r < mb; ++r)
        {
            for(int32_t c = std::max(0, r - 1); c <= std::min(mb - 1, r + 1); ++c)
                m.ind.push_back(c);
            m.ptr.push_back(static_cast<int32_t>(m.ind.size()));
        }
        fill_block_values(m, skew);
        return m;
    }

    // Lower block pattern whose last block row holds `fat` blocks (the others the
    // blocks I-1 and I): the longest block row selects the bsric0 kernel family
    // (max_nnzb <= 32 unrolled, 33..128 plain, above 128 general).
    template <typename T>
    bsr_matrix<T> make_fat_block_row_matrix(int32_t fat, int32_t block_dim, rocsparse_direction dir)
    {
        bsr_matrix<T> m;
        m.mb        = fat + 1;
        m.block_dim = block_dim;
        m.dir       = dir;
        m.ptr.push_back(0);
        for(int32_t r = 0; r < m.mb; ++r)
        {
            const int32_t first = (r == m.mb - 1) ? m.mb - fat : std::max(0, r - 1);
            for(int32_t c = first; c <= r; ++c)
                m.ind.push_back(c);
            m.ptr.push_back(static_cast<int32_t>(m.ind.size()));
        }
        fill_block_values(m, false);
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
    template <typename T>
    double residual_tolerance()
    {
        return is_single_precision<T> ? 2.0e-6 : 3.0e-15;
    }

    template <typename T>
    void expect_small_residual(double residual, const csr_matrix<T>& a)
    {
        EXPECT_LE(residual, residual_tolerance<T>() * max_abs_value(a));
    }
} // namespace cov70
