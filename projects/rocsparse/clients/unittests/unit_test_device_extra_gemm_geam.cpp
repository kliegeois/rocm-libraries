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
// Device unit tests that execute the host dispatch of
//   library/src/extra/rocsparse_spgeam.cpp
//
// Calls go through the public C API. rocsparse_spgeam.cpp is also compiled into
// this executable (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES), so that copy is what
// runs. 2x2 sums for the type / index / operation switches.
//
#include "unit_test_extra_common.hpp"

#include <vector>

using namespace rocsparse_ut;
using extra_ut::PointerModeGuard;
using extra_ut::spgeam_set_inputs;
using extra_ut::SpgeamDescrH;
using extra_ut::SpmatH;

namespace
{
    using extra_ut::Bsr;
    using extra_ut::bsr_from_csr;
    using extra_ut::cplx;
    using extra_ut::Csr;
    using extra_ut::DevBuf;

    // A = [1, 2; 0, 3] (first) or B = [4, 0; 5, 6]. Complex types get
    // imaginary parts, so a dropped conjugation or imaginary part shows.
    template <typename T>
    Csr pair_operand(bool first)
    {
        const double im = extra_ut::is_complex_t<T>::value ? 1.0 : 0.0;
        Csr          M;
        M.m = M.n = 2;
        M.ptr     = first ? std::vector<int64_t>{0, 2, 3} : std::vector<int64_t>{0, 1, 3};
        M.col     = first ? std::vector<int64_t>{0, 1, 1} : std::vector<int64_t>{0, 0, 1};
        M.val     = first ? std::vector<cplx>{1.0, cplx(2.0, im), 3.0}
                          : std::vector<cplx>{4.0, cplx(5.0, -2.0 * im), 6.0};
        return M;
    }

    template <typename I, typename J, typename T>
    void run_spgeam(rocsparse_handle handle, bool device_scalars)
    {
        // C = 2 * A + 3 * B with the pair_operand matrices.
        const rocsparse_datatype   dt   = dt_of<T>();
        const rocsparse_indextype  it   = it_of<I>();
        const rocsparse_indextype  jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const Csr                  hA   = pair_operand<T>(true);
        const Csr                  hB   = pair_operand<T>(false);
        DevBuf<I>                  dA_ptr, dB_ptr;
        DevBuf<J>                  dA_col, dB_col;
        DevBuf<T>                  dA_val, dB_val;
        SpmatH                     A, B;
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(A, hA, dA_ptr, dA_col, dA_val)));
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(B, hB, dB_ptr, dB_col, dB_val)));
        const Bsr want = extra_ut::bsr_geam(2.0, bsr_from_csr(hA), 3.0, bsr_from_csr(hB));

        const T          alpha = scalar<T>(2.f);
        const T          beta  = scalar<T>(3.f);
        device_vector<T> d_alpha{std::vector<T>{alpha}};
        device_vector<T> d_beta{std::vector<T>{beta}};
        ASSERT_TRUE(d_alpha.ptr && d_beta.ptr);

        SpgeamDescrH                 holder;
        const rocsparse_spgeam_descr descr = holder.d;
        ASSERT_NE(descr, nullptr);
        ASSERT_NO_FATAL_FAILURE(
            spgeam_set_inputs(handle,
                              descr,
                              dt,
                              dt,
                              device_scalars ? static_cast<const void*>(d_alpha.ptr) : &alpha,
                              device_scalars ? static_cast<const void*>(d_beta.ptr) : &beta));

        size_t bsz = 0;
        ASSERT_EQ(
            rocsparse_spgeam_buffer_size(
                handle, descr, A.p, B.p, nullptr, rocsparse_spgeam_stage_analysis, &bsz, nullptr),
            rocsparse_status_success);
        device_vector<char> tmp{bsz ? bsz : size_t{1}};
        ASSERT_TRUE(tmp.ptr);
        // mat_C == nullptr takes csrgeam_nnz_core_with_allocation.
        ASSERT_EQ(rocsparse_spgeam(handle,
                                   descr,
                                   A.p,
                                   B.p,
                                   nullptr,
                                   rocsparse_spgeam_stage_analysis,
                                   bsz,
                                   tmp.ptr,
                                   nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        int64_t nnz_C = 0;
        ASSERT_EQ(rocsparse_spgeam_get_output(
                      handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(nnz_C), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(nnz_C, want.nnzb());

        device_vector<I> dC_ptr{size_t{3}};
        device_vector<J> dC_col{static_cast<size_t>(std::max<int64_t>(nnz_C, 1))};
        device_vector<T> dC_val{static_cast<size_t>(std::max<int64_t>(nnz_C, 1))};
        ASSERT_TRUE(dC_ptr.ptr && dC_col.ptr && dC_val.ptr);
        SpmatH C;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 2, 2, nnz_C, dC_ptr.ptr, dC_col.ptr, dC_val.ptr, it, jt, base, dt),
                  rocsparse_status_success);

        PointerModeGuard mode(
            handle, device_scalars ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
        bsz = 0;
        ASSERT_EQ(rocsparse_spgeam_buffer_size(
                      handle, descr, A.p, B.p, C.p, rocsparse_spgeam_stage_compute, &bsz, nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam(handle,
                                   descr,
                                   A.p,
                                   B.p,
                                   C.p,
                                   rocsparse_spgeam_stage_compute,
                                   bsz,
                                   tmp.ptr,
                                   nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        extra_ut::expect_bsr_equal<T, I, J>(
            to_host(dC_ptr), to_host(dC_col), to_host(dC_val), want, rocsparse_direction_row, 0);
    }

    // Generic spgeam with a transposed or conjugate transposed operand: rejected with
    // not_implemented at every stage, both when the operation is set before the analysis
    // and when it is changed on a descriptor whose analysis ran with operation none.
    // C is not written, and once the operations are none again the compute stage gives
    // the reference sum.
    template <typename I, typename J, typename T>
    void run_transposed_spgeam(rocsparse_handle    handle,
                               bool                device_scalars,
                               rocsparse_operation op_A,
                               rocsparse_operation op_B)
    {
        SCOPED_TRACE(testing::Message()
                     << "op_A " << op_A << " op_B " << op_B << " device " << device_scalars);
        const rocsparse_status     ni   = rocsparse_status_not_implemented;
        const rocsparse_datatype   dt   = dt_of<T>();
        const rocsparse_indextype  it   = it_of<I>();
        const rocsparse_indextype  jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const Csr                  hA   = pair_operand<T>(true);
        const Csr                  hB   = pair_operand<T>(false);
        DevBuf<I>                  dA_ptr, dB_ptr;
        DevBuf<J>                  dA_col, dB_col;
        DevBuf<T>                  dA_val, dB_val;
        SpmatH                     A, B;
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(A, hA, dA_ptr, dA_col, dA_val)));
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(B, hB, dB_ptr, dB_col, dB_val)));

        const T          alpha = scalar<T>(2.f);
        const T          beta  = scalar<T>(3.f);
        device_vector<T> d_alpha{std::vector<T>{alpha}};
        device_vector<T> d_beta{std::vector<T>{beta}};
        ASSERT_TRUE(d_alpha.ptr && d_beta.ptr);
        const void*      a = device_scalars ? static_cast<const void*>(d_alpha.ptr) : &alpha;
        const void*      b = device_scalars ? static_cast<const void*>(d_beta.ptr) : &beta;
        PointerModeGuard mode(
            handle, device_scalars ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);

        // C of the untransposed sum, its values filled with a sentinel.
        const int64_t        nnz_C = 4;
        device_vector<I>     dC_ptr{std::vector<I>{0, 2, 4}};
        device_vector<J>     dC_col{std::vector<J>{0, 1, 0, 1}};
        const std::vector<T> sentinel(static_cast<size_t>(nnz_C), scalar<T>(-7.f));
        device_vector<T>     dC_val{sentinel};
        ASSERT_TRUE(dC_ptr.ptr && dC_col.ptr && dC_val.ptr);
        SpmatH C;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 2, 2, nnz_C, dC_ptr.ptr, dC_col.ptr, dC_val.ptr, it, jt, base, dt),
                  rocsparse_status_success);

        auto expect_c_untouched = [&]() {
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            const std::vector<T> got = to_host(dC_val);
            for(size_t k = 0; k < got.size(); ++k)
                EXPECT_TRUE(extra_ut::close<T>(got[k], extra_ut::to_c<T>(sentinel[k])))
                    << "C value " << k << " was written";
        };

        device_vector<char> tmp{size_t{4096}};
        ASSERT_TRUE(tmp.ptr);
        const rocsparse_spgeam_stage analyses[] = {rocsparse_spgeam_stage_analysis,
                                                   rocsparse_spgeam_stage_symbolic_analysis,
                                                   rocsparse_spgeam_stage_numeric_analysis};
        const rocsparse_spgeam_stage computes[] = {rocsparse_spgeam_stage_compute,
                                                   rocsparse_spgeam_stage_symbolic_compute,
                                                   rocsparse_spgeam_stage_numeric_compute};

        // The operations are set before the first stage.
        {
            SpgeamDescrH holder;
            ASSERT_NE(holder.d, nullptr);
            ASSERT_NO_FATAL_FAILURE(spgeam_set_inputs(handle, holder.d, dt, dt, a, b, op_A, op_B));
            for(auto stage : analyses)
            {
                size_t bsz = 0;
                EXPECT_EQ(rocsparse_spgeam_buffer_size(
                              handle, holder.d, A.p, B.p, nullptr, stage, &bsz, nullptr),
                          ni)
                    << "buffer_size stage " << stage;
                EXPECT_EQ(rocsparse_spgeam(
                              handle, holder.d, A.p, B.p, nullptr, stage, 0, tmp.ptr, nullptr),
                          ni)
                    << "stage " << stage;
            }
            for(auto stage : computes)
                EXPECT_EQ(
                    rocsparse_spgeam(handle, holder.d, A.p, B.p, C.p, stage, 0, tmp.ptr, nullptr),
                    ni)
                    << "stage " << stage;
            expect_c_untouched();
        }

        // The analysis runs with operation none, then the operations change.
        {
            SpgeamDescrH holder;
            ASSERT_NE(holder.d, nullptr);
            ASSERT_NO_FATAL_FAILURE(spgeam_set_inputs(handle, holder.d, dt, dt, a, b));
            size_t bsz = 0;
            ASSERT_EQ(rocsparse_spgeam_buffer_size(handle,
                                                   holder.d,
                                                   A.p,
                                                   B.p,
                                                   nullptr,
                                                   rocsparse_spgeam_stage_analysis,
                                                   &bsz,
                                                   nullptr),
                      rocsparse_status_success);
            device_vector<char> buf{bsz ? bsz : size_t{1}};
            ASSERT_TRUE(buf.ptr);
            ASSERT_EQ(rocsparse_spgeam(handle,
                                       holder.d,
                                       A.p,
                                       B.p,
                                       nullptr,
                                       rocsparse_spgeam_stage_analysis,
                                       bsz,
                                       buf.ptr,
                                       nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                                 holder.d,
                                                 rocsparse_spgeam_input_operation_A,
                                                 &op_A,
                                                 sizeof(op_A),
                                                 nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                                 holder.d,
                                                 rocsparse_spgeam_input_operation_B,
                                                 &op_B,
                                                 sizeof(op_B),
                                                 nullptr),
                      rocsparse_status_success);
            for(auto stage : computes)
            {
                EXPECT_EQ(rocsparse_spgeam_buffer_size(
                              handle, holder.d, A.p, B.p, C.p, stage, &bsz, nullptr),
                          ni)
                    << "buffer_size stage " << stage;
                EXPECT_EQ(
                    rocsparse_spgeam(handle, holder.d, A.p, B.p, C.p, stage, 0, tmp.ptr, nullptr),
                    ni)
                    << "stage " << stage;
            }
            expect_c_untouched();

            // The rejected calls left the descriptor usable.
            const rocsparse_operation none = rocsparse_operation_none;
            for(auto input :
                {rocsparse_spgeam_input_operation_A, rocsparse_spgeam_input_operation_B})
                ASSERT_EQ(rocsparse_spgeam_set_input(
                              handle, holder.d, input, &none, sizeof(none), nullptr),
                          rocsparse_status_success);
            bsz = 0;
            ASSERT_EQ(
                rocsparse_spgeam_buffer_size(
                    handle, holder.d, A.p, B.p, C.p, rocsparse_spgeam_stage_compute, &bsz, nullptr),
                rocsparse_status_success);
            ASSERT_LE(bsz, buf.n);
            ASSERT_EQ(rocsparse_spgeam(handle,
                                       holder.d,
                                       A.p,
                                       B.p,
                                       C.p,
                                       rocsparse_spgeam_stage_compute,
                                       bsz,
                                       buf.ptr,
                                       nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            const Bsr want = extra_ut::bsr_geam(2.0, bsr_from_csr(hA), 3.0, bsr_from_csr(hB));
            extra_ut::expect_bsr_equal<T, I, J>(to_host(dC_ptr),
                                                to_host(dC_col),
                                                to_host(dC_val),
                                                want,
                                                rocsparse_direction_row,
                                                0);
        }
    }

    class ExtraGemmGeam : public HandleTest
    {
    };
}

TEST_F(ExtraGemmGeam, spgeam_type_index_scalars)
{
    run_spgeam<int32_t, int32_t, float>(handle, false);
    run_spgeam<int32_t, int32_t, float>(handle, true);
    run_spgeam<int32_t, int32_t, double>(handle, false);
    run_spgeam<int32_t, int32_t, rocsparse_float_complex>(handle, false);
    run_spgeam<int32_t, int32_t, rocsparse_double_complex>(handle, false);
    run_spgeam<int64_t, int32_t, float>(handle, false);
    run_spgeam<int64_t, int32_t, double>(handle, false);
    run_spgeam<int64_t, int64_t, float>(handle, false);
    run_spgeam<int64_t, int64_t, rocsparse_float_complex>(handle, false);
    run_spgeam<int64_t, int32_t, rocsparse_double_complex>(handle, true);

    // Only operation none is supported: the others return not_implemented.
    const rocsparse_operation N = rocsparse_operation_none;
    const rocsparse_operation T = rocsparse_operation_transpose;
    const rocsparse_operation C = rocsparse_operation_conjugate_transpose;
    run_transposed_spgeam<int32_t, int32_t, float>(handle, false, T, N);
    run_transposed_spgeam<int32_t, int32_t, float>(handle, false, N, T);
    run_transposed_spgeam<int32_t, int32_t, float>(handle, false, T, C);
    run_transposed_spgeam<int32_t, int32_t, rocsparse_float_complex>(handle, false, C, C);
    run_transposed_spgeam<int64_t, int32_t, double>(handle, false, T, T);
    run_transposed_spgeam<int64_t, int64_t, rocsparse_float_complex>(handle, true, N, C);
}
