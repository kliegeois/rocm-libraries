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
// Device unit tests for library/src/extra/rocsparse_spmat_scale.cpp.
//
// Every supported sparse format is scaled in place and out of place, with a
// host and a device alpha, for the four value types. The whole value array of
// the target is compared with alpha * source, and the source is checked to be
// unmodified. The argument-check paths are driven with deliberately mismatched
// descriptors and must return the documented statuses.
//
#include "unit_test_cov70_extra_common.hpp"

#ifdef ROCSPARSE_WITH_SPMAT_SCALE

using namespace cov70;

namespace
{
    enum class Fmt
    {
        coo,
        coo_aos,
        csr,
        csc,
        bsr,
        ell,
        bell,
        sell
    };

    const Fmt all_formats[]
        = {Fmt::coo, Fmt::coo_aos, Fmt::csr, Fmt::csc, Fmt::bsr, Fmt::ell, Fmt::bell, Fmt::sell};

    // Index storage (never touched by spmat_scale) and the length of the value
    // array of the matrix created last (cross-checked against value_length).
    struct Layout
    {
        std::vector<DevBuf<int32_t>> idx;
        int64_t                      val_len = 0;

        int32_t* add(const std::vector<int32_t>& h)
        {
            idx.emplace_back(h);
            return idx.back().ptr;
        }
    };

    rocsparse_status create_matrix(Fmt                fmt,
                                   rocsparse_datatype dt,
                                   SpmatH&            out,
                                   Layout&            lay,
                                   void*              val,
                                   int64_t            nnz_override = -1,
                                   int64_t            bdim         = 2)
    {
        const rocsparse_indextype  it   = rocsparse_indextype_i32;
        const rocsparse_index_base base = rocsparse_index_base_zero;
        switch(fmt)
        {
        case Fmt::coo:
        {
            const int64_t nnz = nnz_override >= 0 ? nnz_override : 5;
            lay.val_len       = nnz;
            return rocsparse_create_coo_descr(&out.p,
                                              4,
                                              4,
                                              nnz,
                                              lay.add({0, 1, 1, 2, 3}),
                                              lay.add({0, 0, 3, 2, 1}),
                                              val,
                                              it,
                                              base,
                                              dt);
        }
        case Fmt::coo_aos:
        {
            const int64_t nnz = nnz_override >= 0 ? nnz_override : 5;
            lay.val_len       = nnz;
            return rocsparse_create_coo_aos_descr(
                &out.p, 4, 4, nnz, lay.add({0, 0, 1, 0, 1, 3, 2, 2, 3, 1}), val, it, base, dt);
        }
        case Fmt::csr:
        {
            const int64_t nnz = nnz_override >= 0 ? nnz_override : 5;
            lay.val_len       = nnz;
            return rocsparse_create_csr_descr(&out.p,
                                              4,
                                              4,
                                              nnz,
                                              lay.add({0, 1, 3, 4, 5}),
                                              lay.add({0, 0, 3, 2, 1}),
                                              val,
                                              it,
                                              it,
                                              base,
                                              dt);
        }
        case Fmt::csc:
        {
            const int64_t nnz = nnz_override >= 0 ? nnz_override : 5;
            lay.val_len       = nnz;
            return rocsparse_create_csc_descr(&out.p,
                                              4,
                                              4,
                                              nnz,
                                              lay.add({0, 2, 3, 4, 5}),
                                              lay.add({0, 1, 3, 2, 1}),
                                              val,
                                              it,
                                              it,
                                              base,
                                              dt);
        }
        case Fmt::bsr:
        {
            const int64_t bnnz = nnz_override >= 0 ? nnz_override : 3;
            lay.val_len        = bnnz * bdim * bdim;
            return rocsparse_create_bsr_descr(&out.p,
                                              2,
                                              2,
                                              bnnz,
                                              rocsparse_direction_row,
                                              bdim,
                                              lay.add({0, 2, 3}),
                                              lay.add({0, 1, 1}),
                                              val,
                                              it,
                                              it,
                                              base,
                                              dt);
        }
        case Fmt::ell:
        {
            const int64_t width = nnz_override >= 0 ? nnz_override : 2;
            lay.val_len         = 4 * width;
            std::vector<int32_t> col(static_cast<size_t>(4 * width));
            for(size_t i = 0; i < col.size(); ++i)
                col[i] = static_cast<int32_t>(i % 4);
            return rocsparse_create_ell_descr(&out.p, 4, 4, lay.add(col), val, width, it, base, dt);
        }
        case Fmt::bell:
        {
            lay.val_len = 8;
            return rocsparse_create_bell_descr(
                &out.p, 4, 4, rocsparse_direction_row, bdim, 2, lay.add({0, 1}), val, it, base, dt);
        }
        case Fmt::sell:
        {
            // 4 rows, slices of 2 rows, each slice 2 wide: 8 stored entries.
            const int64_t slice = nnz_override >= 0 ? nnz_override : 2;
            lay.val_len         = 8;
            return rocsparse_create_sell_descr(&out.p,
                                               4,
                                               4,
                                               7,
                                               slice,
                                               8,
                                               lay.add({0, 4, 8, 8, 8}),
                                               lay.add({0, 1, 2, 3, 0, 1, 2, 3}),
                                               val,
                                               it,
                                               it,
                                               base,
                                               dt);
        }
        }
        return rocsparse_status_invalid_value;
    }

    // Length of the value array of the fixed matrix built by create_matrix
    // (checked against Layout::val_len, which create_matrix derives from the
    // arguments it passes to the library).
    size_t value_length(Fmt fmt)
    {
        switch(fmt)
        {
        case Fmt::coo:
        case Fmt::coo_aos:
        case Fmt::csr:
        case Fmt::csc:
            return 5;
        case Fmt::bsr:
            return 12;
        case Fmt::ell:
        case Fmt::bell:
        case Fmt::sell:
            return 8;
        }
        return 0;
    }

    template <typename T>
    void run_scale(rocsparse_handle handle, Fmt fmt, bool device_alpha, bool in_place)
    {
        SCOPED_TRACE(std::string("fmt=") + std::to_string(static_cast<int>(fmt)) + " device_alpha="
                     + std::to_string(device_alpha) + " in_place=" + std::to_string(in_place));

        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();

        const size_t len = value_length(fmt);

        std::vector<cplx> src(len), dst_init(len);
        for(size_t i = 0; i < len; ++i)
        {
            src[i] = round_to<T>(cplx(static_cast<double>(i) + 1.0, -static_cast<double>(i % 3)));
            dst_init[i] = round_to<T>(cplx(-99.0, 5.0));
        }

        DevBuf<T> d_src = up_val<T>(src);
        DevBuf<T> d_dst = up_val<T>(dst_init);
        ASSERT_TRUE(d_src.ptr && d_dst.ptr);

        Layout lay;
        SpmatH source, target;
        ASSERT_EQ(create_matrix(fmt, dt, source, lay, d_src.ptr), rocsparse_status_success);
        ASSERT_EQ(static_cast<size_t>(lay.val_len), len);
        if(in_place)
        {
            // Same value storage through a second descriptor.
            ASSERT_EQ(create_matrix(fmt, dt, target, lay, d_src.ptr), rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(create_matrix(fmt, dt, target, lay, d_dst.ptr), rocsparse_status_success);
        }

        // The index arrays of both matrices must not be touched.
        std::vector<std::vector<int32_t>> idx_before;
        for(const auto& b : lay.idx)
            idx_before.push_back(b.host());

        const cplx alpha_c = cplx(2.0, -1.0);
        Scalar<T>  alpha_s(alpha_c, device_alpha);
        DnvecH     alpha;
        ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(
                      handle, &alpha.p, mode_of(device_alpha), dt, alpha_s.get(), nullptr, nullptr),
                  rocsparse_status_success);

        ASSERT_EQ(rocsparse_spmat_scale(handle, alpha.p, source.p, target.p, nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_EQ(lay.idx.size(), idx_before.size());
        for(size_t i = 0; i < lay.idx.size(); ++i)
            EXPECT_EQ(lay.idx[i].host(), idx_before[i]) << "index array " << i << " modified";

        const cplx        a = to_c<T>(alpha_s.h);
        std::vector<cplx> want(len);
        for(size_t i = 0; i < len; ++i)
            want[i] = a * src[i];

        if(in_place)
        {
            expect_vec_equal<T>(d_src.host(), want, "in-place values");
            // The distinct output buffer is not involved.
            expect_vec_equal<T>(d_dst.host(), dst_init, "unused buffer");
        }
        else
        {
            expect_vec_equal<T>(d_dst.host(), want, "scaled values");
            expect_vec_equal<T>(d_src.host(), src, "source values");
        }
    }

    template <typename T>
    void run_all_formats(rocsparse_handle handle)
    {
        for(Fmt f : all_formats)
            for(bool dev : {false, true})
                for(bool in_place : {false, true})
                {
                    run_scale<T>(handle, f, dev, in_place);
                    if(::testing::Test::HasFatalFailure())
                        return;
                }
    }

    struct SpmatScaleCov70 : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(SpmatScaleCov70, formats_float)
{
    run_all_formats<float>(handle);
}
TEST_F(SpmatScaleCov70, formats_double)
{
    run_all_formats<double>(handle);
}
TEST_F(SpmatScaleCov70, formats_float_complex)
{
    run_all_formats<rocsparse_float_complex>(handle);
}
TEST_F(SpmatScaleCov70, formats_double_complex)
{
    run_all_formats<rocsparse_double_complex>(handle);
}

// An empty value array is a successful no-op and leaves the buffers alone.
TEST_F(SpmatScaleCov70, empty_matrix_is_noop)
{
    const rocsparse_datatype dt = rocsparse_datatype_f32_r;
    DevBuf<float>            d_src(std::vector<float>{1.f, 2.f});
    DevBuf<float>            d_dst(std::vector<float>{7.f, 8.f});
    Layout                   lay;
    SpmatH                   source, target;
    ASSERT_EQ(create_matrix(Fmt::csr, dt, source, lay, d_src.ptr, 0), rocsparse_status_success);
    ASSERT_EQ(create_matrix(Fmt::csr, dt, target, lay, d_dst.ptr, 0), rocsparse_status_success);
    const float alpha = 3.f;
    DnvecH      a;
    ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(
                  handle, &a.p, rocsparse_pointer_mode_host, dt, &alpha, nullptr, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, source.p, target.p, nullptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(d_src.host(), (std::vector<float>{1.f, 2.f}));
    EXPECT_EQ(d_dst.host(), (std::vector<float>{7.f, 8.f}));
}

// The scalar descriptor can also be created read-write (values != nullptr).
TEST_F(SpmatScaleCov70, read_write_scalar_descriptor)
{
    const rocsparse_datatype dt = rocsparse_datatype_f64_r;
    std::vector<double>      src{1.0, 2.0, 3.0, 4.0, 5.0};
    DevBuf<double>           d_src(src);
    Layout                   lay;
    SpmatH                   source;
    ASSERT_EQ(create_matrix(Fmt::coo, dt, source, lay, d_src.ptr), rocsparse_status_success);
    double alpha = -0.5;
    DnvecH a;
    ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(
                  handle, &a.p, rocsparse_pointer_mode_host, dt, &alpha, &alpha, nullptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmat_scale(handle, a.p, source.p, source.p, nullptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    const std::vector<double> got = d_src.host();
    for(size_t i = 0; i < src.size(); ++i)
        EXPECT_EQ(got[i], -0.5 * src[i]) << i;
}

// ---------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------
TEST_F(SpmatScaleCov70, argument_errors)
{
    const rocsparse_datatype dt = rocsparse_datatype_f32_r;
    DevBuf<float>            v1(size_t{64}), v2(size_t{64});
    Layout                   l1, l2;

    SpmatH csr, csr2, coo;
    ASSERT_EQ(create_matrix(Fmt::csr, dt, csr, l1, v1.ptr), rocsparse_status_success);
    ASSERT_EQ(create_matrix(Fmt::csr, dt, csr2, l1, v2.ptr), rocsparse_status_success);
    ASSERT_EQ(create_matrix(Fmt::coo, dt, coo, l2, v2.ptr), rocsparse_status_success);

    const float alpha = 2.f;
    DnvecH      a;
    ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(
                  handle, &a.p, rocsparse_pointer_mode_host, dt, &alpha, nullptr, nullptr),
              rocsparse_status_success);

    // null handle / pointers
    EXPECT_EQ(rocsparse_spmat_scale(nullptr, a.p, csr.p, csr2.p, nullptr),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_spmat_scale(handle, nullptr, csr.p, csr2.p, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, nullptr, csr2.p, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, csr.p, nullptr, nullptr),
              rocsparse_status_invalid_pointer);

    // different formats
    EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, csr.p, coo.p, nullptr),
              rocsparse_status_not_implemented);

    // different nnz / rows / cols
    {
        SpmatH other;
        Layout lo;
        ASSERT_EQ(create_matrix(Fmt::csr, dt, other, lo, v2.ptr, 4), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, csr.p, other.p, nullptr),
                  rocsparse_status_invalid_size);
    }
    {
        SpmatH               other;
        std::vector<int32_t> ptr{0, 1, 3, 4};
        DevBuf<int32_t>      dptr(ptr);
        DevBuf<int32_t>      dcol(std::vector<int32_t>{0, 0, 3, 2, 1});
        ASSERT_EQ(rocsparse_create_csr_descr(&other.p,
                                             3,
                                             4,
                                             5,
                                             dptr.ptr,
                                             dcol.ptr,
                                             v2.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, csr.p, other.p, nullptr),
                  rocsparse_status_invalid_size); // rows differ
        SpmatH wide;
        ASSERT_EQ(rocsparse_create_csr_descr(&wide.p,
                                             4,
                                             5,
                                             5,
                                             l1.idx[0].ptr,
                                             l1.idx[1].ptr,
                                             v2.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, csr.p, wide.p, nullptr),
                  rocsparse_status_invalid_size); // cols differ
    }

    // different data type
    {
        SpmatH         dbl;
        Layout         ld;
        DevBuf<double> vd(size_t{64});
        ASSERT_EQ(create_matrix(Fmt::csr, rocsparse_datatype_f64_r, dbl, ld, vd.ptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, csr.p, dbl.p, nullptr),
                  rocsparse_status_type_mismatch);
    }

    // format specific layout mismatches
    {
        SpmatH        b2, b3;
        Layout        lb2, lb3;
        DevBuf<float> vb(size_t{64});
        ASSERT_EQ(create_matrix(Fmt::bsr, dt, b2, lb2, vb.ptr, -1, 2), rocsparse_status_success);
        ASSERT_EQ(create_matrix(Fmt::bsr, dt, b3, lb3, vb.ptr, -1, 3), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, b2.p, b3.p, nullptr),
                  rocsparse_status_invalid_size);
    }
    {
        SpmatH        e2, e3;
        Layout        le2, le3;
        DevBuf<float> ve(size_t{64});
        ASSERT_EQ(create_matrix(Fmt::ell, dt, e2, le2, ve.ptr, 2), rocsparse_status_success);
        ASSERT_EQ(create_matrix(Fmt::ell, dt, e3, le3, ve.ptr, 3), rocsparse_status_success);
        // The nnz of an ELL matrix is rows * width, so a different width is
        // already rejected by the nnz comparison (the width comparison itself
        // cannot be reached through the public API).
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, e2.p, e3.p, nullptr),
                  rocsparse_status_invalid_size);
    }
    {
        SpmatH        b2, b4;
        Layout        lb2, lb4;
        DevBuf<float> vb(size_t{64});
        ASSERT_EQ(create_matrix(Fmt::bell, dt, b2, lb2, vb.ptr, -1, 2), rocsparse_status_success);
        ASSERT_EQ(create_matrix(Fmt::bell, dt, b4, lb4, vb.ptr, -1, 4), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, b2.p, b4.p, nullptr),
                  rocsparse_status_invalid_size); // block dimension differs
        // Same block dimension, different number of ELL columns.
        SpmatH b2_wide;
        ASSERT_EQ(rocsparse_create_bell_descr(&b2_wide.p,
                                              4,
                                              4,
                                              rocsparse_direction_row,
                                              2,
                                              4,
                                              lb2.idx[0].ptr,
                                              vb.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              dt),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, b2.p, b2_wide.p, nullptr),
                  rocsparse_status_invalid_size);
    }
    {
        // SELL: same shape and nnz, different slice size / colval size
        SpmatH        s1, s2, s3;
        Layout        ls1, ls2, ls3;
        DevBuf<float> vs(size_t{64});
        ASSERT_EQ(create_matrix(Fmt::sell, dt, s1, ls1, vs.ptr, 2), rocsparse_status_success);
        ASSERT_EQ(create_matrix(Fmt::sell, dt, s2, ls2, vs.ptr, 4), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, s1.p, s2.p, nullptr),
                  rocsparse_status_invalid_size);
        ASSERT_EQ(rocsparse_create_sell_descr(&s3.p,
                                              4,
                                              4,
                                              7,
                                              2,
                                              10,
                                              ls1.idx[0].ptr,
                                              ls1.idx[1].ptr,
                                              vs.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              dt),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, s1.p, s3.p, nullptr),
                  rocsparse_status_invalid_size);
    }

    // batched matrices are not supported
    {
        SpmatH batched, plain;
        Layout lb, lp;
        ASSERT_EQ(create_matrix(Fmt::csr, dt, batched, lb, v1.ptr), rocsparse_status_success);
        ASSERT_EQ(create_matrix(Fmt::csr, dt, plain, lp, v2.ptr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(batched.p, 2, 0, 0), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, batched.p, plain.p, nullptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, plain.p, batched.p, nullptr),
                  rocsparse_status_not_implemented);
    }

    // alpha: wrong size, batched, wrong type
    {
        DevBuf<float> hv(size_t{2});
        DnvecH        vec2;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vec2.p, 2, hv.ptr, dt), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, vec2.p, csr.p, csr2.p, nullptr),
                  rocsparse_status_not_implemented);

        DnvecH batched_alpha;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&batched_alpha.p, 1, hv.ptr, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(batched_alpha.p, 2, 1),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, batched_alpha.p, csr.p, csr2.p, nullptr),
                  rocsparse_status_not_implemented);

        DnvecH       wrong_type;
        const double dalpha = 2.0;
        ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &wrong_type.p,
                                                      rocsparse_pointer_mode_host,
                                                      rocsparse_datatype_f64_r,
                                                      &dalpha,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmat_scale(handle, wrong_type.p, csr.p, csr2.p, nullptr),
                  rocsparse_status_type_mismatch);
    }

    // The scalar descriptor constructor rejects bad arguments.
    {
        DnvecH bad;
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(
                      handle, nullptr, rocsparse_pointer_mode_host, dt, &alpha, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(
                      handle, &bad.p, rocsparse_pointer_mode_host, dt, nullptr, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                      &bad.p,
                                                      static_cast<rocsparse_pointer_mode>(7),
                                                      dt,
                                                      &alpha,
                                                      nullptr,
                                                      nullptr),
                  rocsparse_status_invalid_value);
    }
}

// Value types that the scaling kernel does not implement are rejected after
// the (matching) argument checks pass.
TEST_F(SpmatScaleCov70, unsupported_value_type)
{
    const rocsparse_datatype dt = rocsparse_datatype_f16_r;
    DevBuf<_Float16>         v(size_t{16});
    Layout                   lay;
    SpmatH                   source;
    ASSERT_EQ(create_matrix(Fmt::csr, dt, source, lay, v.ptr), rocsparse_status_success);
    DevBuf<_Float16> av(std::vector<_Float16>{static_cast<_Float16>(2.f)});
    DnvecH           a;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&a.p, 1, av.ptr, dt), rocsparse_status_success);
    EXPECT_EQ(rocsparse_spmat_scale(handle, a.p, source.p, source.p, nullptr),
              rocsparse_status_not_implemented);
}

#endif // ROCSPARSE_WITH_SPMAT_SCALE
