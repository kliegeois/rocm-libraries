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
//   library/src/extra/rocsparse_spgemm.cpp
//
// Public C API only, so the library (not a second copy of the .cpp) is what
// runs. 2x2 products (CSR and BSR) for the operation checks.
//
#include "unit_test_extra_common.hpp"

#include <optional>
#include <type_traits>
#include <vector>

using namespace rocsparse_ut;
using extra_ut::any_error;
using extra_ut::SpmatH;
using extra_ut::status_is;

namespace rocsparse_ut
{
    template <>
    inline rocsparse_datatype dt_of<_Float16>()
    {
        return rocsparse_datatype_f16_r;
    }
    template <>
    inline rocsparse_datatype dt_of<rocsparse_bfloat16>()
    {
        return rocsparse_datatype_bf16_r;
    }
    template <>
    inline _Float16 scalar<_Float16>(float v)
    {
        return static_cast<_Float16>(v);
    }
    template <>
    inline rocsparse_bfloat16 scalar<rocsparse_bfloat16>(float v)
    {
        return rocsparse_bfloat16(v);
    }
}

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

    // One generic spgemm problem C = alpha * A * B + beta * D and the stages
    // that are issued on it.
    struct SpgemmFlow
    {
        rocsparse_handle   handle;
        rocsparse_datatype dt;
        const void*        alpha;
        const void*        beta;
        SpmatH             A, B, D, C;
        size_t             bsz = 0;

        rocsparse_status stage(rocsparse_spgemm_stage s, void* buffer)
        {
            return rocsparse_spgemm(handle,
                                    rocsparse_operation_none,
                                    rocsparse_operation_none,
                                    alpha,
                                    A.p,
                                    B.p,
                                    beta,
                                    D.p,
                                    C.p,
                                    dt,
                                    rocsparse_spgemm_alg_default,
                                    s,
                                    &bsz,
                                    buffer);
        }

        // Buffer size, nnz and C's arrays: after this, `compute` is the next
        // stage. Returns the status of the first failing stage.
        template <typename I, typename J, typename T>
        rocsparse_status prepare(device_vector<I>&                   C_ptr,
                                 std::optional<device_vector<J>>&    C_col,
                                 std::optional<device_vector<T>>&    C_val,
                                 std::optional<device_vector<char>>& buffer)
        {
            auto st = stage(rocsparse_spgemm_stage_buffer_size, nullptr);
            if(st != rocsparse_status_success)
                return st;
            buffer.emplace(bsz ? bsz : size_t{1});
            st = stage(rocsparse_spgemm_stage_nnz, buffer->ptr);
            if(st != rocsparse_status_success)
                return st;
            int64_t rows = 0, cols = 0, nnz = 0;
            st = rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz);
            if(st != rocsparse_status_success)
                return st;
            C_col.emplace(static_cast<size_t>(nnz > 0 ? nnz : 1));
            C_val.emplace(static_cast<size_t>(nnz > 0 ? nnz : 1));
            return rocsparse_csr_set_pointers(C.p, C_ptr.ptr, C_col->ptr, C_val->ptr);
        }
    };

    // C = 2 * A * B with the non-symmetric pair_operand matrices.
    template <typename I, typename J, typename T>
    void run_pair_spgemm(rocsparse_handle handle, bool symbolic)
    {
        const rocsparse_datatype   dt   = dt_of<T>();
        const rocsparse_indextype  it   = it_of<I>();
        const rocsparse_indextype  jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const Csr                  hA   = pair_operand<T>(true);
        const Csr                  hB   = pair_operand<T>(false);
        DevBuf<I>                  Ap, Bp;
        DevBuf<J>                  Ac, Bc;
        DevBuf<T>                  Av, Bv;
        device_vector<I>           Cp{size_t{3}};
        ASSERT_TRUE(Cp.ptr);

        SpgemmFlow f{handle, dt};
        const T    alpha = scalar<T>(2.f);
        const T    beta  = scalar<T>(0.f);
        f.alpha          = &alpha;
        f.beta           = &beta;
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(f.A, hA, Ap, Ac, Av)));
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(f.B, hB, Bp, Bc, Bv)));
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &f.D.p, 0, 0, 0, nullptr, nullptr, nullptr, it, jt, base, dt),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_csr_descr(&f.C.p, 2, 2, 0, Cp.ptr, nullptr, nullptr, it, jt, base, dt),
            rocsparse_status_success);

        std::optional<device_vector<J>>    Cc;
        std::optional<device_vector<T>>    Cv;
        std::optional<device_vector<char>> tmp;
        ASSERT_EQ((f.prepare<I, J, T>(Cp, Cc, Cv, tmp)), rocsparse_status_success);
        ASSERT_EQ(
            f.stage(symbolic ? rocsparse_spgemm_stage_symbolic : rocsparse_spgemm_stage_compute,
                    tmp->ptr),
            rocsparse_status_success);
        if(symbolic)
            ASSERT_EQ(f.stage(rocsparse_spgemm_stage_numeric, tmp->ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Bsr want = extra_ut::bsr_gemm(true,
                                            2.0,
                                            bsr_from_csr(hA),
                                            bsr_from_csr(hB),
                                            false,
                                            0.0,
                                            extra_ut::empty_bsr(2, 2, 1),
                                            2,
                                            2);
        extra_ut::expect_bsr_equal<T, I, J>(
            to_host(Cp), to_host(*Cc), to_host(*Cv), want, rocsparse_direction_row, 0);
    }

    // Generic spgemm with a transposed or conjugate transposed operand: documented as
    // not implemented, at the first stage and at every later stage of a flow whose
    // first stages ran with operation none. C is not written.
    template <typename I, typename J, typename T>
    void run_transposed_spgemm(rocsparse_handle    handle,
                               rocsparse_operation trans_A,
                               rocsparse_operation trans_B)
    {
        SCOPED_TRACE(testing::Message() << "trans_A " << trans_A << " trans_B " << trans_B);
        const rocsparse_datatype   dt   = dt_of<T>();
        const rocsparse_indextype  it   = it_of<I>();
        const rocsparse_indextype  jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const Csr                  hA   = pair_operand<T>(true);
        const Csr                  hB   = pair_operand<T>(false);
        DevBuf<I>                  Ap, Bp;
        DevBuf<J>                  Ac, Bc;
        DevBuf<T>                  Av, Bv;
        device_vector<I>           Cp{size_t{3}};
        ASSERT_TRUE(Cp.ptr);

        SpgemmFlow f{handle, dt};
        const T    alpha = scalar<T>(2.f);
        const T    beta  = scalar<T>(0.f);
        f.alpha          = &alpha;
        f.beta           = &beta;
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(f.A, hA, Ap, Ac, Av)));
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(f.B, hB, Bp, Bc, Bv)));
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &f.D.p, 0, 0, 0, nullptr, nullptr, nullptr, it, jt, base, dt),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_csr_descr(&f.C.p, 2, 2, 0, Cp.ptr, nullptr, nullptr, it, jt, base, dt),
            rocsparse_status_success);

        auto stage_t = [&](rocsparse_spgemm_stage s, void* buffer) {
            return rocsparse_spgemm(handle,
                                    trans_A,
                                    trans_B,
                                    f.alpha,
                                    f.A.p,
                                    f.B.p,
                                    f.beta,
                                    f.D.p,
                                    f.C.p,
                                    dt,
                                    rocsparse_spgemm_alg_default,
                                    s,
                                    &f.bsz,
                                    buffer);
        };
        EXPECT_EQ(stage_t(rocsparse_spgemm_stage_buffer_size, nullptr),
                  rocsparse_status_not_implemented);

        std::optional<device_vector<J>>    Cc;
        std::optional<device_vector<T>>    Cv;
        std::optional<device_vector<char>> tmp;
        ASSERT_EQ((f.prepare<I, J, T>(Cp, Cc, Cv, tmp)), rocsparse_status_success);
        const std::vector<T> sentinel(Cv->n, scalar<T>(-7.f));
        ASSERT_EQ(hipMemcpy(Cv->ptr, sentinel.data(), sizeof(T) * Cv->n, hipMemcpyHostToDevice),
                  hipSuccess);
        const std::vector<I> ptr_before = to_host(Cp);
        const std::vector<J> col_before = to_host(*Cc);
        for(auto s : {rocsparse_spgemm_stage_nnz,
                      rocsparse_spgemm_stage_compute,
                      rocsparse_spgemm_stage_symbolic,
                      rocsparse_spgemm_stage_numeric})
            EXPECT_EQ(stage_t(s, tmp->ptr), rocsparse_status_not_implemented) << "stage " << s;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(to_host(Cp), ptr_before) << "C row pointer was written";
        EXPECT_EQ(to_host(*Cc), col_before) << "C column indices were written";
        const std::vector<T> got = to_host(*Cv);
        for(size_t k = 0; k < got.size(); ++k)
            EXPECT_TRUE(extra_ut::close<T>(got[k], extra_ut::to_c<T>(sentinel[k])))
                << "C value " << k << " was written";
    }

    class ExtraGemmGeam : public HandleTest
    {
    };
}

TEST_F(ExtraGemmGeam, spgemm_type_index_transpose)
{
    const rocsparse_operation N = rocsparse_operation_none;
    const rocsparse_operation T = rocsparse_operation_transpose;
    const rocsparse_operation C = rocsparse_operation_conjugate_transpose;

    run_pair_spgemm<int32_t, int32_t, float>(handle, true);

    // Transposed operands are documented as not implemented.
    run_transposed_spgemm<int32_t, int32_t, float>(handle, T, N);
    run_transposed_spgemm<int32_t, int32_t, float>(handle, N, T);
    run_transposed_spgemm<int32_t, int32_t, float>(handle, T, T);
    run_transposed_spgemm<int32_t, int32_t, float>(handle, C, N);
    run_transposed_spgemm<int32_t, int32_t, rocsparse_float_complex>(handle, C, C);
    run_transposed_spgemm<int64_t, int64_t, double>(handle, T, C);
}

TEST_F(ExtraGemmGeam, spgemm_rejected_types_formats_and_bsr)
{
    constexpr auto I32 = rocsparse_indextype_i32;
    constexpr auto ZB  = rocsparse_index_base_zero;
    constexpr auto F32 = rocsparse_datatype_f32_r;
    constexpr auto N   = rocsparse_operation_none;
    constexpr auto DEF = rocsparse_spgemm_alg_default;
    constexpr auto BUF = rocsparse_spgemm_stage_buffer_size;

    // The four operands of C = A * B + D: a 2x2 A, B and C, and an empty D.
    struct Operands
    {
        SpmatH A, B, D, C;
    };

    // BSR is implemented for buffer/nnz/compute; symbolic/numeric are CSR only.
    // C = 2 * A * B with blocks of distinct values, so for bd = 2 the block
    // direction changes the result (bd = 1 takes the csrgemm path).
    auto bsr_case = [&](auto* tag, rocsparse_datatype dt, rocsparse_direction dir, int64_t bd) {
        using V = std::remove_pointer_t<decltype(tag)>;
        SCOPED_TRACE("dir=" + std::to_string(dir) + " bd=" + std::to_string(bd));
        Bsr hA, hB;
        hA.mb = hA.nb = hB.mb = hB.nb = 2;
        hA.bd = hB.bd = bd;
        hA.ptr        = {0, 2, 3};
        hA.col        = {0, 1, 1};
        hB.ptr        = {0, 1, 3};
        hB.col        = {0, 0, 1};
        for(int64_t e = 0; e < 3 * bd * bd; ++e)
        {
            hA.val.push_back(static_cast<double>((e * 5) % 7 - 3));
            hB.val.push_back(static_cast<double>((e * 3 + 1) % 7 - 3));
        }
        const Bsr want = extra_ut::bsr_gemm(
            true, 2.0, hA, hB, false, 0.0, extra_ut::empty_bsr(2, 2, bd), 2, 2);
        const V                alpha_v(2.f), zero_v(0.f);
        auto                   Ap = extra_ut::up_idx<int32_t>(hA.ptr);
        auto                   Ac = extra_ut::up_idx<int32_t>(hA.col);
        auto                   Bp = extra_ut::up_idx<int32_t>(hB.ptr);
        auto                   Bc = extra_ut::up_idx<int32_t>(hB.col);
        DevBuf<V>              Av(extra_ut::bsr_pack<V>(hA, dir));
        DevBuf<V>              Bv(extra_ut::bsr_pack<V>(hB, dir));
        device_vector<int32_t> cptr{size_t{3}};
        ASSERT_TRUE(Ap.ptr && Ac.ptr && Bp.ptr && Bc.ptr && Av.ptr && Bv.ptr && cptr.ptr);
        Operands o;
        auto     mk = [&](SpmatH& m, const DevBuf<int32_t>& p, const DevBuf<int32_t>& c, V* v) {
            return rocsparse_create_bsr_descr(
                &m.p, 2, 2, 3, dir, bd, p.ptr, c.ptr, v, I32, I32, ZB, dt);
        };
        ASSERT_EQ(mk(o.A, Ap, Ac, Av.ptr), rocsparse_status_success);
        ASSERT_EQ(mk(o.B, Bp, Bc, Bv.ptr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(
                      &o.D.p, 0, 0, 0, dir, bd, nullptr, nullptr, nullptr, I32, I32, ZB, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(
                      &o.C.p, 2, 2, 0, dir, bd, cptr.ptr, nullptr, nullptr, I32, I32, ZB, dt),
                  rocsparse_status_success);

        size_t bsz  = 0;
        auto   call = [&](rocsparse_spgemm_stage stage, void* buf) {
            return rocsparse_spgemm(handle,
                                    N,
                                    N,
                                    &alpha_v,
                                    o.A.p,
                                    o.B.p,
                                    &zero_v,
                                    o.D.p,
                                    o.C.p,
                                    dt,
                                    DEF,
                                    stage,
                                    &bsz,
                                    buf);
        };
        ASSERT_EQ(call(BUF, nullptr), rocsparse_status_success);
        device_vector<char> tmp{bsz ? bsz : size_t{1}};
        ASSERT_TRUE(tmp.ptr);
        ASSERT_EQ(call(rocsparse_spgemm_stage_nnz, tmp.ptr), rocsparse_status_success);
        EXPECT_TRUE(status_is(call(rocsparse_spgemm_stage_symbolic, tmp.ptr), any_error));
        int64_t brows = 0, bcols = 0, bnnz = 0;
        ASSERT_EQ(rocsparse_spmat_get_size(o.C.p, &brows, &bcols, &bnnz), rocsparse_status_success);
        ASSERT_EQ(bnnz, want.nnzb());
        device_vector<int32_t> ccol{static_cast<size_t>(bnnz)};
        device_vector<V>       cv{static_cast<size_t>(bnnz * bd * bd)};
        ASSERT_TRUE(ccol.ptr && cv.ptr);
        ASSERT_EQ(rocsparse_bsr_set_pointers(o.C.p, cptr.ptr, ccol.ptr, cv.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(call(rocsparse_spgemm_stage_compute, tmp.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        extra_ut::expect_bsr_equal<V, int32_t, int32_t>(
            to_host(cptr), to_host(ccol), to_host(cv), want, dir, 0);

        // A transposed A is documented as not implemented, and C is not written.
        const std::vector<V> sentinel(cv.n, V(-7.f));
        ASSERT_EQ(hipMemcpy(cv.ptr, sentinel.data(), sizeof(V) * cv.n, hipMemcpyHostToDevice),
                  hipSuccess);
        auto call_t = [&](rocsparse_spgemm_stage stage, void* buf) {
            return rocsparse_spgemm(handle,
                                    rocsparse_operation_transpose,
                                    N,
                                    &alpha_v,
                                    o.A.p,
                                    o.B.p,
                                    &zero_v,
                                    o.D.p,
                                    o.C.p,
                                    dt,
                                    DEF,
                                    stage,
                                    &bsz,
                                    buf);
        };
        EXPECT_EQ(call_t(BUF, nullptr), rocsparse_status_not_implemented);
        for(auto stage : {rocsparse_spgemm_stage_nnz, rocsparse_spgemm_stage_compute})
            EXPECT_EQ(call_t(stage, tmp.ptr), rocsparse_status_not_implemented)
                << "stage " << stage;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const std::vector<V> got = to_host(cv);
        for(size_t k = 0; k < got.size(); ++k)
            EXPECT_TRUE(extra_ut::close<V>(got[k], extra_ut::to_c<V>(sentinel[k])))
                << "C value " << k << " was written";
    };
    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
        for(int64_t bd : {1, 2})
        {
            bsr_case(static_cast<float*>(nullptr), F32, dir, bd);
            bsr_case(static_cast<_Float16*>(nullptr), rocsparse_datatype_f16_r, dir, bd);
            bsr_case(static_cast<rocsparse_bfloat16*>(nullptr), rocsparse_datatype_bf16_r, dir, bd);
        }
}
