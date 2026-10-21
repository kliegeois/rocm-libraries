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
// Device unit tests for library/src/extra/rocsparse_spgeam.cpp.
//
// The generic rocsparse_spgeam API is driven through the combined
// analysis/compute flow and the split symbolic/numeric flow, with host and
// device scalars whose datatype differs from the compute datatype. Every flow
// compares the whole C (row pointer, column indices and values) against a host
// reference C = alpha * A + beta * B.
//
#include "unit_test_extra_common.hpp"

#include <vector>

using namespace extra_ut;

namespace rocsparse_ut
{
    template <>
    inline rocsparse_datatype dt_of<int32_t>()
    {
        return rocsparse_datatype_i32_r;
    }
}

namespace
{
    // C = alpha * A + beta * B with the matrices in the compute type T and alpha / beta
    // in the scalar type S (S != T). The values have small integer parts, exact in both
    // types.
    template <typename T, typename S>
    void mixed_scalar_geam(rocsparse_handle handle, bool device, bool split)
    {
        SCOPED_TRACE(testing::Message()
                     << "compute " << rocsparse_ut::dt_of<T>() << " scalar "
                     << rocsparse_ut::dt_of<S>() << " device " << device << " split " << split);
        using I = int32_t;
        Rng       rng(311);
        const Csr A = random_csr<T>(9, 7, 4, rng);
        const Csr B = random_csr<T>(9, 7, 4, rng);
        ASSERT_GT(A.nnz(), 0);
        ASSERT_GT(B.nnz(), 0);
        const S   alpha_h = S(2), beta_h = S(-1);
        const Bsr want = bsr_geam(to_c(alpha_h), bsr_from_csr(A), to_c(beta_h), bsr_from_csr(B));

        DevBuf<I> pA, pB, cA, cB;
        DevBuf<T> vA, vB;
        SpmatH    sA, sB, sC;
        ASSERT_NO_FATAL_FAILURE((make_csr_descr<T, I, I>(sA, A, pA, cA, vA)));
        ASSERT_NO_FATAL_FAILURE((make_csr_descr<T, I, I>(sB, B, pB, cB, vB)));
        DevBuf<S> d_alpha(std::vector<S>{alpha_h}), d_beta(std::vector<S>{beta_h});
        ASSERT_TRUE(d_alpha.ptr && d_beta.ptr);

        PointerModeGuard mode(handle, mode_of(device));
        SpgeamDescrH     h;
        ASSERT_NE(h.d, nullptr);
        ASSERT_NO_FATAL_FAILURE(spgeam_set_inputs(
            handle,
            h.d,
            rocsparse_ut::dt_of<S>(),
            rocsparse_ut::dt_of<T>(),
            device ? static_cast<const void*>(d_alpha.ptr) : static_cast<const void*>(&alpha_h),
            device ? static_cast<const void*>(d_beta.ptr) : static_cast<const void*>(&beta_h)));

        auto run = [&](rocsparse_spgeam_stage stage, rocsparse_spmat_descr C) {
            size_t bs = 0;
            EXPECT_EQ(rocsparse_spgeam_buffer_size(handle, h.d, sA.p, sB.p, C, stage, &bs, nullptr),
                      rocsparse_status_success);
            DevBuf<char> tmp(bs);
            return rocsparse_spgeam(handle, h.d, sA.p, sB.p, C, stage, bs, tmp.ptr, nullptr);
        };

        ASSERT_EQ(
            run(split ? rocsparse_spgeam_stage_symbolic_analysis : rocsparse_spgeam_stage_analysis,
                nullptr),
            rocsparse_status_success);
        int64_t nnz_C = -1;
        ASSERT_EQ(rocsparse_spgeam_get_output(
                      handle, h.d, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(nnz_C), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(nnz_C, want.nnzb());

        // C's values sit between two guard pads of `pad` elements.
        constexpr size_t pad   = 64;
        const T          guard = T(-12345);
        DevBuf<I>        pC(static_cast<size_t>(A.m) + 1), cC(static_cast<size_t>(nnz_C));
        DevBuf<T>        vC_padded(std::vector<T>(static_cast<size_t>(nnz_C) + 2 * pad, guard));
        ASSERT_TRUE(pC.ptr && cC.ptr && vC_padded.ptr);
        ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                             A.m,
                                             A.n,
                                             nnz_C,
                                             pC.ptr,
                                             cC.ptr,
                                             vC_padded.ptr + pad,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        if(split)
        {
            for(auto stage : {rocsparse_spgeam_stage_symbolic_compute,
                              rocsparse_spgeam_stage_numeric_analysis,
                              rocsparse_spgeam_stage_numeric_compute})
                ASSERT_EQ(run(stage, sC.p), rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(run(rocsparse_spgeam_stage_compute, sC.p), rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<T> all = vC_padded.host();
        const size_t         nnz = static_cast<size_t>(nnz_C);
        for(size_t k = 0; k < pad; ++k)
        {
            EXPECT_EQ(all[k], guard) << "front pad " << k;
            EXPECT_EQ(all[pad + nnz + k], guard) << "back pad " << k;
        }
        std::vector<I> hp = pC.host(), hc = cC.host();
        std::vector<T> hv(all.begin() + pad, all.begin() + pad + nnz);
        sort_rows(hp, 0, hc, hv);
        expect_idx_equal<I>(hp, want.ptr, 0, "C row_ptr");
        expect_idx_equal<I>(hc, want.col, 0, "C col_ind");
        expect_vec_equal<T>(hv, want.val, "C val");
    }

    struct ExtraSpgeam : public rocsparse_ut::HandleTest
    {
    };
}

// The scalar datatype differs from the compute datatype, including real or integer
// scalars with a complex compute type: alpha and beta are converted to the compute
// type, C equals the host reference C = alpha * A + beta * B, and nothing is written
// outside C.
TEST_F(ExtraSpgeam, scalar_type_differs_from_compute_type)
{
    for(const bool device : {false, true})
        for(const bool split : {false, true})
        {
            ASSERT_NO_FATAL_FAILURE((mixed_scalar_geam<float, double>(handle, device, split)));
            ASSERT_NO_FATAL_FAILURE((mixed_scalar_geam<double, float>(handle, device, split)));
            ASSERT_NO_FATAL_FAILURE(
                (mixed_scalar_geam<rocsparse_float_complex, float>(handle, device, split)));
            ASSERT_NO_FATAL_FAILURE(
                (mixed_scalar_geam<rocsparse_float_complex, double>(handle, device, split)));
            ASSERT_NO_FATAL_FAILURE(
                (mixed_scalar_geam<rocsparse_double_complex, float>(handle, device, split)));
            ASSERT_NO_FATAL_FAILURE(
                (mixed_scalar_geam<rocsparse_double_complex, double>(handle, device, split)));
            ASSERT_NO_FATAL_FAILURE((mixed_scalar_geam<float, int32_t>(handle, device, split)));
            ASSERT_NO_FATAL_FAILURE(
                (mixed_scalar_geam<rocsparse_double_complex, int32_t>(handle, device, split)));
        }
}
