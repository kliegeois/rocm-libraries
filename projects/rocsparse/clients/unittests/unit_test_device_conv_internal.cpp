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

// Device unit tests for the internal conversion routines that the public API tests do not
// reach directly: convert_array (numeric datatype pairs) and dnvec_transfer_from, and
// rocsparse_sparse_to_sparse between data types that convert_array cannot convert.

#include "unit_test_device_conv_common.hpp"

#include "rocsparse_convert_array.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <utility>

using namespace convut;

namespace
{
    using ConvInternal = ConvTest;

    // Redirects std::cout and std::cerr: routines log the pairs that have no kernel.
    class Silence
    {
        std::ostringstream m_out, m_err;
        std::streambuf *   m_old_out, *m_old_err;

    public:
        Silence()
            : m_old_out(std::cout.rdbuf(m_out.rdbuf()))
            , m_old_err(std::cerr.rdbuf(m_err.rdbuf()))
        {
        }
        ~Silence()
        {
            std::cout.rdbuf(m_old_out);
            std::cerr.rdbuf(m_old_err);
        }
    };

    // ------------------------------------------------------------------
    // convert_array
    // ------------------------------------------------------------------

    constexpr rocsparse_datatype s_datatypes[] = {rocsparse_datatype_i8_r,
                                                  rocsparse_datatype_u8_r,
                                                  rocsparse_datatype_i32_r,
                                                  rocsparse_datatype_u32_r,
                                                  rocsparse_datatype_f16_r,
                                                  rocsparse_datatype_bf16_r,
                                                  rocsparse_datatype_f32_r,
                                                  rocsparse_datatype_f64_r,
                                                  rocsparse_datatype_f32_c,
                                                  rocsparse_datatype_f64_c};

    template <typename D, typename S>
    std::vector<D> convert_values(rocsparse_handle      handle,
                                  rocsparse_datatype    dd,
                                  rocsparse_datatype    ds,
                                  const std::vector<S>& in)
    {
        DevBuf<S> src(in);
        DevBuf<D> dst(in.size());
        EXPECT_EQ(rocsparse::convert_array(handle, in.size(), dd, dst.get(), ds, src.get()), st_ok);
        return dst.read();
    }

    // The values 1..n stored as datatype dt, as bytes; they are exact in every datatype.
    std::vector<char> iota_bytes(rocsparse_datatype dt, int n)
    {
        std::vector<char> b;
        const auto        put = [&](auto x) {
            const char* p = reinterpret_cast<const char*>(&x);
            b.insert(b.end(), p, p + sizeof(x));
        };
        for(int v = 1; v <= n; ++v)
        {
            switch(dt)
            {
            case rocsparse_datatype_i8_r:
                put(int8_t(v));
                break;
            case rocsparse_datatype_u8_r:
                put(uint8_t(v));
                break;
            case rocsparse_datatype_i32_r:
                put(int32_t(v));
                break;
            case rocsparse_datatype_u32_r:
                put(uint32_t(v));
                break;
            case rocsparse_datatype_f16_r:
                put(scalar<_Float16>(v));
                break;
            case rocsparse_datatype_bf16_r:
                put(scalar<rocsparse_bfloat16>(v));
                break;
            case rocsparse_datatype_f32_r:
                put(scalar<float>(v));
                break;
            case rocsparse_datatype_f64_r:
                put(scalar<double>(v));
                break;
            case rocsparse_datatype_f32_c:
                put(scalar<rocsparse_float_complex>(v));
                break;
            case rocsparse_datatype_f64_c:
                put(scalar<rocsparse_double_complex>(v));
                break;
            }
        }
        return b;
    }

    TEST_F(ConvInternal, ConvertArrayDatatypes)
    {
        constexpr size_t n   = 4;
        constexpr auto   f64 = rocsparse_datatype_f64_r;
        constexpr auto   c32 = rocsparse_datatype_f32_c, c64 = rocsparse_datatype_f64_c;
        using cf = rocsparse_float_complex;
        using cd = rocsparse_double_complex;
        Silence                                                 silence;
        const std::pair<rocsparse_datatype, rocsparse_datatype> converted[]
            = {{f64, f32}, {f32, f64}, {c32, f32}, {c32, f64}, {c32, c64}, {c64, c32}};
        for(auto t : s_datatypes)
            for(auto s : s_datatypes)
            {
                SCOPED_TRACE(testing::Message() << "target " << t << " source " << s);
                const std::vector<char> want = iota_bytes(t, n);
                DevBuf<char> src(iota_bytes(s, n)), dst(std::vector<char>(want.size(), -1));
                const rocsparse_status st
                    = rocsparse::convert_array(handle, n, t, dst.get(), s, src.get());
                EXPECT_EQ(rocsparse::convert_array(handle, 0, t, dst.get(), s, src.get()), st_ok);
                const bool must_convert
                    = t == s
                      || std::find(std::begin(converted), std::end(converted), std::make_pair(t, s))
                             != std::end(converted);
                // Other pairs are rejected and leave the target untouched.
                EXPECT_EQ(st, must_convert ? st_ok : rocsparse_status_not_implemented);
                EXPECT_EQ(dst.read(), st == st_ok ? want : std::vector<char>(want.size(), -1));
            }
        const std::vector<float> hf{1.0f, -2.0f, 3.5f, 0.25f};
        EXPECT_EQ(convert_values<double>(handle, f64, f32, hf),
                  (std::vector<double>{1.0, -2.0, 3.5, 0.25}));
        EXPECT_EQ(convert_values<float>(handle, f32, f64, std::vector<double>{1.25, 3.0}),
                  (std::vector<float>{1.25f, 3.0f}));
        EXPECT_EQ(convert_values<cf>(handle, c32, f32, hf),
                  (std::vector<cf>{{1, 0}, {-2, 0}, {3.5f, 0}, {0.25f, 0}}));
        const std::vector<cf> hc{{1.0f, 2.0f}, {-3.0f, 0.5f}};
        EXPECT_EQ(convert_values<cd>(handle, c64, c32, hc),
                  (std::vector<cd>{{1.0, 2.0}, {-3.0, 0.5}}));
        EXPECT_EQ(convert_values<cf>(handle, c32, c64, std::vector<cd>{{1.0, 2.0}}),
                  (std::vector<cf>{{1.0f, 2.0f}}));
    }

    TEST_F(ConvInternal, DnvecTransferFrom)
    {
        Silence                          silence;
        constexpr size_t                 n = 4;
        DevBuf<float>                    d_f32(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        DevBuf<double>                   d_f64(n);
        DevBuf<rocsparse_float_complex>  d_c32(n);
        DevBuf<rocsparse_double_complex> d_c64(n);
        Dnvec                            vf32, vf64, vc32, vc64, vshort;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vf32, n, d_f32, rocsparse_datatype_f32_r), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vf64, n, d_f64, rocsparse_datatype_f64_r), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vc32, n, d_c32, rocsparse_datatype_f32_c), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vc64, n, d_c64, rocsparse_datatype_f64_c), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vshort, n - 1, d_f32, rocsparse_datatype_f32_r),
                  st_ok);
        DevBuf<int32_t> d_i32(std::vector<int32_t>{5, 6, 7, 8});
        Dnvec           vi32;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vi32, n, d_i32, rocsparse_datatype_i32_r), st_ok);

        EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, nullptr, vf32), st_ptr);
        EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, nullptr), st_ptr);
        EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, vshort), st_size);
        // A real target from a complex source is rejected.
        EXPECT_REJECTED(rocsparse::dnvec_transfer_from(handle, vf64, vc32));
        EXPECT_REJECTED(rocsparse::dnvec_transfer_from(handle, vf32, vc64));

        // Real from real, complex from real, complex from complex.
        ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, vf32), st_ok);
        EXPECT_EQ(d_f64.read(), (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
        // A pair without a conversion kernel is rejected and leaves the target untouched.
        EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf32, vi32),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(d_f32.read(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
        ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vc32, vf64), st_ok);
        ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vc64, vc32), st_ok);
        EXPECT_EQ(d_c64.read(),
                  (std::vector<rocsparse_double_complex>{
                      {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}}));
    }

    // ------------------------------------------------------------------
    // sparse_to_sparse
    // ------------------------------------------------------------------

    // Descriptor creation, then analysis and compute; returns the first status that is not
    // success (an unsupported conversion may be rejected by any of the calls).
    rocsparse_status sparse_to_sparse(rocsparse_handle            handle,
                                      rocsparse_const_spmat_descr src,
                                      rocsparse_spmat_descr       dst)
    {
        S2sDescr         s2s;
        rocsparse_status st = rocsparse_create_sparse_to_sparse_descr(
            &s2s, src, dst, rocsparse_sparse_to_sparse_alg_default);
        for(auto stage :
            {rocsparse_sparse_to_sparse_stage_analysis, rocsparse_sparse_to_sparse_stage_compute})
        {
            size_t size = 0;
            if(st == st_ok)
                st = rocsparse_sparse_to_sparse_buffer_size(handle, s2s, src, dst, stage, &size);
            if(st != st_ok)
                break;
            DevBuf<char> buffer(size);
            st = rocsparse_sparse_to_sparse(handle, s2s, src, dst, stage, size, buffer);
        }
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return st;
    }

    // A 3 x 3 matrix with f32 values converted to CSR and COO targets with i32 values,
    // a pair without a conversion kernel. The conversion is rejected, and the target
    // arrays keep their contents. CSC -> CSR has no value conversion and rejects any
    // data type change.
    TEST_F(ConvInternal, SparseToSparseMixedDatatypes)
    {
        Silence                    silence;
        constexpr int64_t          m = 3, nnz = 4;
        const std::vector<int32_t> ptr = {0, 1, 3, 4}, row = {0, 1, 1, 2}, col = {0, 0, 2, 1};
        DevBuf<int32_t>            s_ptr(ptr), s_row(row), s_col(col);
        DevBuf<float>              s_val(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        const std::vector<int32_t> guard(m + 1, -7);
        const auto                 create = [&](Spmat&             d,
                                rocsparse_format   f,
                                void*              p,
                                void*              r,
                                void*              c,
                                void*              v,
                                rocsparse_datatype dt) {
            return f == rocsparse_format_csr
                                       ? rocsparse_create_csr_descr(&d, m, m, nnz, p, c, v, i32, i32, base0, dt)
                                   : f == rocsparse_format_csc
                                       ? rocsparse_create_csc_descr(&d, m, m, nnz, p, c, v, i32, i32, base0, dt)
                                       : rocsparse_create_coo_descr(&d, m, m, nnz, r, c, v, i32, base0, dt);
        };
        const std::pair<rocsparse_format, rocsparse_format> pairs[]
            = {{rocsparse_format_csr, rocsparse_format_csr},
               {rocsparse_format_coo, rocsparse_format_csr},
               {rocsparse_format_csr, rocsparse_format_coo},
               {rocsparse_format_csc, rocsparse_format_csr}};
        for(const auto& [sf, tf] : pairs)
        {
            SCOPED_TRACE(testing::Message() << "source " << sf << " target " << tf);
            DevBuf<int32_t> t_ptr(guard), t_row(guard), t_col(guard), t_val(guard);
            Spmat           source, target;
            ASSERT_EQ(create(source, sf, s_ptr, s_row, s_col, s_val, f32), st_ok);
            ASSERT_EQ(create(target, tf, t_ptr, t_row, t_col, t_val, rocsparse_datatype_i32_r),
                      st_ok);
            EXPECT_EQ(sparse_to_sparse(handle, source, target), rocsparse_status_not_implemented);
            EXPECT_EQ(t_ptr.read(), guard);
            EXPECT_EQ(t_row.read(), guard);
            EXPECT_EQ(t_col.read(), guard);
            EXPECT_EQ(t_val.read(), guard);
        }
    }
} // namespace
