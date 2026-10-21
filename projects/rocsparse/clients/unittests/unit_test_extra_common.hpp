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
// Shared helpers for the extra-routine device unit tests: small host reference
// types (CSR / BSR matrices computed in double-complex precision) with a
// deterministic matrix generator and the BSR geam reference, RAII wrappers
// for the generic-API descriptors, and output comparison helpers.
//
// Every test built on these helpers uploads a small matrix, calls the public C
// API, and compares the COMPLETE output with a host reference.
//
#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <set>
#include <vector>

namespace extra_ut
{
    using cplx = std::complex<double>;

    // ----- value conversions ------------------------------------------------
    template <typename T>
    inline cplx to_c(T v)
    {
        return cplx(static_cast<double>(v), 0.0);
    }
    template <>
    inline cplx to_c<rocsparse_float_complex>(rocsparse_float_complex v)
    {
        return cplx(std::real(v), std::imag(v));
    }
    template <>
    inline cplx to_c<rocsparse_double_complex>(rocsparse_double_complex v)
    {
        return cplx(std::real(v), std::imag(v));
    }

    template <typename T>
    inline T from_c(cplx c)
    {
        return static_cast<T>(c.real());
    }
    template <>
    inline rocsparse_float_complex from_c<rocsparse_float_complex>(cplx c)
    {
        return rocsparse_float_complex(static_cast<float>(c.real()), static_cast<float>(c.imag()));
    }
    template <>
    inline rocsparse_double_complex from_c<rocsparse_double_complex>(cplx c)
    {
        return rocsparse_double_complex(c.real(), c.imag());
    }

    template <typename T>
    struct is_complex_t
    {
        static constexpr bool value = false;
    };
    template <>
    struct is_complex_t<rocsparse_float_complex>
    {
        static constexpr bool value = true;
    };
    template <>
    struct is_complex_t<rocsparse_double_complex>
    {
        static constexpr bool value = true;
    };

    // The test data are small integers (and exactly representable fractions),
    // so sums and products are computed exactly in float and double: results
    // are compared exactly (tolerance 0). The few tests whose result is not
    // exactly representable (e.g. deep triangular solves) pass an explicit,
    // measured and justified tolerance.

    // Round a value to what T can represent, so the host reference uses exactly
    // the numbers the device sees.
    template <typename T>
    inline cplx round_to(cplx c)
    {
        return to_c<T>(from_c<T>(c));
    }

    // ----- deterministic generator -----------------------------------------
    struct Rng
    {
        uint64_t s;
        explicit Rng(uint64_t seed = 1)
            : s(seed * 6364136223846793005ULL + 1442695040888963407ULL)
        {
        }
        uint32_t next()
        {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<uint32_t>(s >> 33);
        }
        int64_t range(int64_t lo, int64_t hi) // inclusive
        {
            return lo + static_cast<int64_t>(next() % static_cast<uint64_t>(hi - lo + 1));
        }
    };

    // Non-zero small value (so that sums/products are exactly representable).
    template <typename T>
    inline cplx rand_val(Rng& rng)
    {
        const double re = static_cast<double>(rng.range(1, 4)) * (rng.range(0, 1) ? 1.0 : -1.0);
        const double im = is_complex_t<T>::value ? static_cast<double>(rng.range(1, 3))
                                                       * (rng.range(0, 1) ? 1.0 : -1.0)
                                                 : 0.0;
        return cplx(re, im);
    }

    // ----- host sparse matrix (CSR, zero based, double complex values) -----
    struct Csr
    {
        int64_t              m = 0, n = 0;
        std::vector<int64_t> ptr; // m + 1
        std::vector<int64_t> col;
        std::vector<cplx>    val;
        int64_t              nnz() const
        {
            return static_cast<int64_t>(col.size());
        }
    };

    // Matrix with the given number of non-zeros in each row (sorted, distinct
    // columns, random positions).
    template <typename T>
    inline Csr csr_with_row_lengths(int64_t m, int64_t n, const std::vector<int64_t>& len, Rng& rng)
    {
        Csr A;
        A.m = m;
        A.n = n;
        A.ptr.assign(static_cast<size_t>(m) + 1, 0);
        for(int64_t i = 0; i < m; ++i)
        {
            const int64_t     l = std::min<int64_t>(len[static_cast<size_t>(i)], n);
            std::set<int64_t> cols;
            if(l == n)
            {
                for(int64_t j = 0; j < n; ++j)
                    cols.insert(j);
            }
            else
            {
                while(static_cast<int64_t>(cols.size()) < l)
                    cols.insert(rng.range(0, n - 1));
            }
            for(int64_t j : cols)
            {
                A.col.push_back(j);
                A.val.push_back(round_to<T>(rand_val<T>(rng)));
            }
            A.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(A.col.size());
        }
        return A;
    }

    template <typename T>
    inline Csr random_csr(int64_t m, int64_t n, int64_t max_row_nnz, Rng& rng)
    {
        std::vector<int64_t> len(static_cast<size_t>(m));
        for(auto& l : len)
            l = rng.range(0, max_row_nnz);
        return csr_with_row_lengths<T>(m, n, len, rng);
    }

    // ----- comparison ------------------------------------------------------
    // |got - want| <= tol * (1 + |want|); tol == 0 is an exact comparison.
    template <typename T>
    inline bool close(T got, cplx want, double tol = 0.0)
    {
        const cplx   g = to_c<T>(got);
        const double e = std::abs(g - want);
        return std::isfinite(e) && e <= tol * (1.0 + std::abs(want));
    }

    template <typename T>
    inline void expect_vec_equal(const std::vector<T>&    got,
                                 const std::vector<cplx>& want,
                                 const char*              what,
                                 double                   tol = 0.0)
    {
        ASSERT_GE(got.size(), want.size()) << what;
        int bad = 0;
        for(size_t i = 0; i < want.size() && bad < 5; ++i)
            if(!close<T>(got[i], want[i], tol))
            {
                ++bad;
                ADD_FAILURE() << what << "[" << i << "] = " << to_c<T>(got[i]) << " expected "
                              << want[i];
            }
    }

    template <typename I>
    inline void expect_idx_equal(const std::vector<I>&       got,
                                 const std::vector<int64_t>& want,
                                 int64_t                     base,
                                 const char*                 what)
    {
        ASSERT_GE(got.size(), want.size()) << what;
        int bad = 0;
        for(size_t i = 0; i < want.size() && bad < 5; ++i)
            if(static_cast<int64_t>(got[i]) - base != want[i])
            {
                ++bad;
                ADD_FAILURE() << what << "[" << i << "] = " << static_cast<int64_t>(got[i]) - base
                              << " expected " << want[i];
            }
    }

    // The geam headers do not promise sorted rows in C: sort every row of a
    // read-back C by column before comparing (each column carries bb values).
    template <typename I, typename J, typename T>
    inline void sort_rows(const std::vector<I>& ptr,
                          int64_t               base,
                          std::vector<J>&       col,
                          std::vector<T>&       val,
                          int64_t               bb = 1)
    {
        for(size_t i = 0; i + 1 < ptr.size(); ++i)
        {
            const int64_t b = static_cast<int64_t>(ptr[i]) - base;
            const int64_t e = static_cast<int64_t>(ptr[i + 1]) - base;
            if(b < 0 || e - b < 2 || e > static_cast<int64_t>(col.size())
               || e * bb > static_cast<int64_t>(val.size()))
                continue;
            std::vector<std::pair<J, std::vector<T>>> row;
            for(int64_t p = b; p < e; ++p)
                row.emplace_back(col[static_cast<size_t>(p)],
                                 std::vector<T>(val.begin() + p * bb, val.begin() + (p + 1) * bb));
            std::stable_sort(row.begin(), row.end(), [](const auto& x, const auto& y) {
                return x.first < y.first;
            });
            for(int64_t p = b; p < e; ++p)
            {
                const auto& r               = row[static_cast<size_t>(p - b)];
                col[static_cast<size_t>(p)] = r.first;
                std::copy(r.second.begin(), r.second.end(), val.begin() + p * bb);
            }
        }
    }

    // ----- movable device buffer -----------------------------------------
    template <typename T>
    struct DevBuf
    {
        T*     ptr = nullptr;
        size_t n   = 0;
        DevBuf()   = default;
        explicit DevBuf(const std::vector<T>& h)
            : n(std::max<size_t>(h.size(), 1))
        {
            std::vector<T> tmp(n);
            std::copy(h.begin(), h.end(), tmp.begin());
            if(hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
            {
                ADD_FAILURE() << "hipMalloc of " << n * sizeof(T) << " bytes failed";
                ptr = nullptr;
                return;
            }
            EXPECT_EQ(hipMemcpy(ptr, tmp.data(), n * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
        }
        explicit DevBuf(size_t count)
            : n(std::max<size_t>(count, 1))
        {
            if(hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
            {
                ADD_FAILURE() << "hipMalloc of " << n * sizeof(T) << " bytes failed";
                ptr = nullptr;
            }
            else
                EXPECT_EQ(hipMemset(ptr, 0, n * sizeof(T)), hipSuccess);
        }
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;
        DevBuf(DevBuf&& o) noexcept
            : ptr(o.ptr)
            , n(o.n)
        {
            o.ptr = nullptr;
            o.n   = 0;
        }
        DevBuf& operator=(DevBuf&& o) noexcept
        {
            if(this != &o)
            {
                if(ptr)
                    (void)hipFree(ptr);
                ptr   = o.ptr;
                n     = o.n;
                o.ptr = nullptr;
                o.n   = 0;
            }
            return *this;
        }
        ~DevBuf()
        {
            if(ptr)
                (void)hipFree(ptr);
        }
        std::vector<T> host() const
        {
            std::vector<T> h(n);
            if(ptr)
                (void)hipMemcpy(h.data(), ptr, n * sizeof(T), hipMemcpyDeviceToHost);
            return h;
        }
    };

    // ----- uploads ---------------------------------------------------------
    // Device buffers always hold at least one element so the pointers stay valid.
    template <typename T>
    inline DevBuf<T> up_val(const std::vector<cplx>& v)
    {
        std::vector<T> h(v.size());
        for(size_t i = 0; i < v.size(); ++i)
            h[i] = from_c<T>(v[i]);
        return DevBuf<T>(h);
    }

    template <typename I>
    inline DevBuf<I> up_idx(const std::vector<int64_t>& v, int64_t base = 0)
    {
        std::vector<I> h(v.size());
        for(size_t i = 0; i < v.size(); ++i)
            h[i] = static_cast<I>(v[i] + base);
        return DevBuf<I>(h);
    }

    // ----- RAII descriptors -------------------------------------------------
    struct SpmatH
    {
        rocsparse_spmat_descr p = nullptr;
        SpmatH()                = default;
        SpmatH(const SpmatH&)   = delete;
        SpmatH& operator=(const SpmatH&) = delete;
        ~SpmatH()
        {
            if(p)
                (void)rocsparse_destroy_spmat_descr(p);
        }
    };

    // RAII owner of a bare SpGEAM descriptor.
    struct SpgeamDescrH
    {
        rocsparse_spgeam_descr d = nullptr;
        SpgeamDescrH()
        {
            (void)rocsparse_create_spgeam_descr(&d);
        }
        SpgeamDescrH(const SpgeamDescrH&) = delete;
        SpgeamDescrH& operator=(const SpgeamDescrH&) = delete;
        ~SpgeamDescrH()
        {
            if(d)
                (void)rocsparse_destroy_spgeam_descr(d);
        }
    };

    // Sets every input of an SpGEAM descriptor (default algorithm, operations
    // none unless given). alpha / beta are the scalar pointers of the handle's
    // pointer mode.
    inline void spgeam_set_inputs(rocsparse_handle       handle,
                                  rocsparse_spgeam_descr d,
                                  rocsparse_datatype     scalar_type,
                                  rocsparse_datatype     compute_type,
                                  const void*            alpha,
                                  const void*            beta,
                                  rocsparse_operation    op_A = rocsparse_operation_none,
                                  rocsparse_operation    op_B = rocsparse_operation_none)
    {
        const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
        auto put = [&](rocsparse_spgeam_input which, const void* p, size_t n) {
            ASSERT_EQ(rocsparse_spgeam_set_input(handle, d, which, p, n, nullptr),
                      rocsparse_status_success);
        };
        put(rocsparse_spgeam_input_alg, &alg, sizeof(alg));
        put(rocsparse_spgeam_input_scalar_datatype, &scalar_type, sizeof(scalar_type));
        put(rocsparse_spgeam_input_compute_datatype, &compute_type, sizeof(compute_type));
        put(rocsparse_spgeam_input_operation_A, &op_A, sizeof(op_A));
        put(rocsparse_spgeam_input_operation_B, &op_B, sizeof(op_B));
        put(rocsparse_spgeam_input_scalar_alpha, alpha, sizeof(void*));
        put(rocsparse_spgeam_input_scalar_beta, beta, sizeof(void*));
    }

    struct PointerModeGuard
    {
        rocsparse_handle       handle = nullptr;
        rocsparse_pointer_mode prev   = rocsparse_pointer_mode_host;
        PointerModeGuard(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
        {
            (void)rocsparse_get_pointer_mode(handle, &prev);
            (void)rocsparse_set_pointer_mode(handle, mode);
        }
        PointerModeGuard(const PointerModeGuard&) = delete;
        PointerModeGuard& operator=(const PointerModeGuard&) = delete;
        ~PointerModeGuard()
        {
            (void)rocsparse_set_pointer_mode(handle, prev);
        }
    };

    inline rocsparse_pointer_mode mode_of(bool device)
    {
        return device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host;
    }

    // Create a CSR descriptor from a host matrix on given index types / base.
    template <typename T, typename I, typename J>
    inline void make_csr_descr(SpmatH&              out,
                               const Csr&           A,
                               DevBuf<I>&           dptr,
                               DevBuf<J>&           dcol,
                               DevBuf<T>&           dval,
                               rocsparse_index_base base = rocsparse_index_base_zero)
    {
        dptr = up_idx<I>(A.ptr, static_cast<int64_t>(base));
        dcol = up_idx<J>(A.col, static_cast<int64_t>(base));
        dval = up_val<T>(A.val);
        ASSERT_TRUE(dptr.ptr && dcol.ptr && dval.ptr);
        ASSERT_EQ(rocsparse_create_csr_descr(&out.p,
                                             A.m,
                                             A.n,
                                             A.nnz(),
                                             dptr.ptr,
                                             dcol.ptr,
                                             dval.ptr,
                                             rocsparse_ut::it_of<I>(),
                                             rocsparse_ut::it_of<J>(),
                                             base,
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
    }

    // ----- host block sparse matrix (BSR, zero based) ------------------------
    // Blocks are stored row-major (logical layout); bsr_pack() / bsr_unpack()
    // convert to and from the device layout of a given block direction.
    struct Bsr
    {
        int64_t              mb = 0, nb = 0, bd = 1;
        std::vector<int64_t> ptr; // mb + 1
        std::vector<int64_t> col;
        std::vector<cplx>    val; // nnzb * bd * bd
        int64_t              nnzb() const
        {
            return static_cast<int64_t>(col.size());
        }
    };

    inline Bsr bsr_from_csr(const Csr& A)
    {
        Bsr B;
        B.mb  = A.m;
        B.nb  = A.n;
        B.bd  = 1;
        B.ptr = A.ptr;
        B.col = A.col;
        B.val = A.val;
        return B;
    }

    // C = alpha * A + beta * B (block pattern = union of both patterns).
    inline Bsr bsr_geam(cplx alpha, const Bsr& A, cplx beta, const Bsr& B)
    {
        const int64_t bd = A.bd;
        Bsr           C;
        C.mb = A.mb;
        C.nb = A.nb;
        C.bd = bd;
        C.ptr.assign(static_cast<size_t>(A.mb) + 1, 0);
        for(int64_t i = 0; i < A.mb; ++i)
        {
            std::vector<std::pair<int64_t, std::vector<cplx>>> row;
            auto                                               add_row = [&](const Bsr& M, cplx s) {
                for(int64_t p = M.ptr[static_cast<size_t>(i)];
                    p < M.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                {
                    const int64_t j = M.col[static_cast<size_t>(p)];
                    auto          it = std::find_if(
                        row.begin(), row.end(), [&](const auto& e) { return e.first == j; });
                    if(it == row.end())
                    {
                        row.emplace_back(j, std::vector<cplx>(static_cast<size_t>(bd * bd)));
                        it = row.end() - 1;
                    }
                    for(int64_t e = 0; e < bd * bd; ++e)
                        it->second[static_cast<size_t>(e)]
                            += s * M.val[static_cast<size_t>(p * bd * bd + e)];
                }
            };
            add_row(A, alpha);
            add_row(B, beta);
            std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            for(const auto& e : row)
            {
                C.col.push_back(e.first);
                C.val.insert(C.val.end(), e.second.begin(), e.second.end());
            }
            C.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(C.col.size());
        }
        return C;
    }
}
