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
// types (CSR / BSR matrices computed in double-complex precision) with the
// BSR gemm reference, RAII wrappers for the generic-API descriptors, and
// output comparison helpers.
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
#include <map>
#include <string>
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

    // ----- comparison ------------------------------------------------------
    // |got - want| <= tol * (1 + |want|); tol == 0 is an exact comparison.
    template <typename T>
    inline bool close(T got, cplx want, double tol = 0.0)
    {
        const cplx   g = to_c<T>(got);
        const double e = std::abs(g - want);
        return std::isfinite(e) && e <= tol * (1.0 + std::abs(want));
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

    // Expected status of a call: an exact status, or any_error where the
    // public header documents a failure but not which status is returned.
    struct AnyError
    {
    };
    constexpr AnyError any_error{};
    struct Expected
    {
        bool             any;
        rocsparse_status status;
        Expected(rocsparse_status s)
            : any(false)
            , status(s)
        {
        }
        Expected(AnyError)
            : any(true)
            , status(rocsparse_status_success)
        {
        }
    };

    inline ::testing::AssertionResult status_is(rocsparse_status got, Expected want)
    {
        if(want.any ? got != rocsparse_status_success : got == want.status)
            return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
               << "status " << got << ", expected "
               << (want.any ? std::string("an error") : std::to_string(want.status));
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

    template <typename T>
    inline std::vector<T> bsr_pack(const Bsr& B, rocsparse_direction dir)
    {
        const int64_t  bd = B.bd;
        std::vector<T> out(B.val.size());
        for(int64_t p = 0; p < B.nnzb(); ++p)
            for(int64_t r = 0; r < bd; ++r)
                for(int64_t c = 0; c < bd; ++c)
                {
                    const size_t dst = static_cast<size_t>(
                        p * bd * bd + (dir == rocsparse_direction_row ? r * bd + c : c * bd + r));
                    out[dst] = from_c<T>(B.val[static_cast<size_t>(p * bd * bd + r * bd + c)]);
                }
        return out;
    }

    template <typename T>
    inline std::vector<cplx>
        bsr_unpack(const std::vector<T>& dev, int64_t nnzb, int64_t bd, rocsparse_direction dir)
    {
        std::vector<cplx> out(static_cast<size_t>(nnzb * bd * bd));
        for(int64_t p = 0; p < nnzb; ++p)
            for(int64_t r = 0; r < bd; ++r)
                for(int64_t c = 0; c < bd; ++c)
                    out[static_cast<size_t>(p * bd * bd + r * bd + c)]
                        = to_c<T>(dev[static_cast<size_t>(
                            p * bd * bd
                            + (dir == rocsparse_direction_row ? r * bd + c : c * bd + r))]);
        return out;
    }

    // C = alpha * A * B + beta * D, where `mul` / `add` select the terms.
    inline Bsr bsr_gemm(bool       mul,
                        cplx       alpha,
                        const Bsr& A,
                        const Bsr& B,
                        bool       add,
                        cplx       beta,
                        const Bsr& D,
                        int64_t    mb,
                        int64_t    nb)
    {
        const int64_t bd = D.bd;
        Bsr           C;
        C.mb = mb;
        C.nb = nb;
        C.bd = bd;
        C.ptr.assign(static_cast<size_t>(mb) + 1, 0);
        for(int64_t i = 0; i < mb; ++i)
        {
            std::map<int64_t, std::vector<cplx>> row;
            auto                                 slot = [&](int64_t j) -> std::vector<cplx>& {
                auto& blk = row[j];
                if(blk.empty())
                    blk.resize(static_cast<size_t>(bd * bd));
                return blk;
            };
            if(mul)
            {
                for(int64_t pa = A.ptr[static_cast<size_t>(i)];
                    pa < A.ptr[static_cast<size_t>(i) + 1];
                    ++pa)
                {
                    const int64_t kk = A.col[static_cast<size_t>(pa)];
                    for(int64_t pb = B.ptr[static_cast<size_t>(kk)];
                        pb < B.ptr[static_cast<size_t>(kk) + 1];
                        ++pb)
                    {
                        std::vector<cplx>& blk = slot(B.col[static_cast<size_t>(pb)]);
                        for(int64_t r = 0; r < bd; ++r)
                            for(int64_t c = 0; c < bd; ++c)
                                for(int64_t t = 0; t < bd; ++t)
                                    blk[static_cast<size_t>(r * bd + c)]
                                        += alpha
                                           * A.val[static_cast<size_t>(pa * bd * bd + r * bd + t)]
                                           * B.val[static_cast<size_t>(pb * bd * bd + t * bd + c)];
                    }
                }
            }
            if(add)
            {
                for(int64_t p = D.ptr[static_cast<size_t>(i)];
                    p < D.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                {
                    std::vector<cplx>& blk = slot(D.col[static_cast<size_t>(p)]);
                    for(int64_t e = 0; e < bd * bd; ++e)
                        blk[static_cast<size_t>(e)]
                            += beta * D.val[static_cast<size_t>(p * bd * bd + e)];
                }
            }
            for(const auto& e : row)
            {
                C.col.push_back(e.first);
                C.val.insert(C.val.end(), e.second.begin(), e.second.end());
            }
            C.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(C.col.size());
        }
        return C;
    }

    inline Bsr empty_bsr(int64_t mb, int64_t nb, int64_t bd)
    {
        Bsr E;
        E.mb  = mb;
        E.nb  = nb;
        E.bd  = bd;
        E.ptr = std::vector<int64_t>(static_cast<size_t>(mb) + 1, 0);
        return E;
    }

    // Compares a computed BSR matrix with the reference. With `sorted` the
    // blocks of every row have to come out sorted by column (guaranteed by the
    // documentation of csrgemm: "the output will be sorted"); the
    // documentation of bsrgemm and of the generic spgemm gives no such
    // guarantee, so there the blocks of a row are matched by column index and
    // the order the library emits them in is not checked.
    template <typename T, typename I, typename J>
    void expect_bsr_equal(const std::vector<I>& ptr,
                          const std::vector<J>& col,
                          const std::vector<T>& val_dev,
                          const Bsr&            want,
                          rocsparse_direction   dir,
                          int64_t               base,
                          bool                  sorted = false)
    {
        bool ptr_ok = ptr.size() >= want.ptr.size();
        for(size_t i = 0; ptr_ok && i < want.ptr.size(); ++i)
            ptr_ok = (static_cast<int64_t>(ptr[i]) - base == want.ptr[i]);
        if(!ptr_ok)
        {
            expect_idx_equal<I>(ptr, want.ptr, base, "row_ptr_C");
            return;
        }
        const int64_t bb      = want.bd * want.bd;
        const auto    got_val = bsr_unpack<T>(val_dev, want.nnzb(), want.bd, dir);
        for(int64_t i = 0; i < want.mb; ++i)
        {
            const int64_t        b = want.ptr[static_cast<size_t>(i)];
            const int64_t        e = want.ptr[static_cast<size_t>(i) + 1];
            std::vector<int64_t> order(static_cast<size_t>(e - b));
            for(int64_t p = b; p < e; ++p)
                order[static_cast<size_t>(p - b)] = p;
            if(!sorted)
                std::sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
                    return col[static_cast<size_t>(x)] < col[static_cast<size_t>(y)];
                });
            for(int64_t q = 0; q < e - b; ++q)
            {
                const int64_t p = order[static_cast<size_t>(q)];
                const int64_t w = b + q; // the reference is sorted by column
                if(static_cast<int64_t>(col[static_cast<size_t>(p)]) - base
                   != want.col[static_cast<size_t>(w)])
                {
                    ADD_FAILURE() << "row " << i << " block " << q << " column "
                                  << (col[static_cast<size_t>(p)] - base) << " expected "
                                  << want.col[static_cast<size_t>(w)];
                    return;
                }
                for(int64_t t = 0; t < bb; ++t)
                {
                    if(!close<T>(from_c<T>(got_val[static_cast<size_t>(p * bb + t)]),
                                 want.val[static_cast<size_t>(w * bb + t)]))
                    {
                        ADD_FAILURE()
                            << "row " << i << " block col " << want.col[static_cast<size_t>(w)]
                            << " entry " << t << " = " << got_val[static_cast<size_t>(p * bb + t)]
                            << " expected " << want.val[static_cast<size_t>(w * bb + t)];
                        return;
                    }
                }
            }
        }
    }
}
