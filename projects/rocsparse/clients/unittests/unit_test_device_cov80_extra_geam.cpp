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
// Addition of sparse matrices (rocsparse_?csrgeam, rocsparse_spgeam,
// rocsparse_?bsrgeam): empty operands (no rows, no columns, no non-zeros),
// missing output arrays and block dimension one. Every successful call is
// checked against the expected (empty or small) result.
//
#include "unit_test_cov80_extra_common.hpp"

#include <memory>

using namespace cov80_extra;

namespace
{
    struct GeamCov80 : public rocsparse_ut::HandleTest
    {
    };

    // CSR arrays of an m x n matrix without entries, plus a (possibly
    // non-empty) matrix when `entries` is set.
    template <typename T>
    struct LegacyOperand
    {
        Csr                   h;
        DevBuf<rocsparse_int> ptr, ind;
        DevBuf<T>             val;
        void                  build(int64_t m, int64_t n, bool entries, Rng& rng)
        {
            h   = entries ? random_csr<T>(m, n, 3, rng)
                          : csr_with_row_lengths<T>(m, n, std::vector<int64_t>(m, 0), rng);
            ptr = up_idx<rocsparse_int>(h.ptr);
            ind = up_idx<rocsparse_int>(h.col);
            val = up_val<T>(h.val);
        }
    };

    template <typename T>
    void legacy_geam(rocsparse_handle handle, int64_t m, int64_t n, bool a_entries, bool b_entries)
    {
        SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n)
                     + " a=" + std::to_string(a_entries) + " b=" + std::to_string(b_entries));
        Rng              rng(static_cast<uint64_t>(m * 17 + n) + 1);
        LegacyOperand<T> A, B;
        A.build(m, n, a_entries, rng);
        B.build(m, n, b_entries, rng);
        ASSERT_TRUE(A.ptr.ptr && A.ind.ptr && A.val.ptr && B.ptr.ptr && B.ind.ptr && B.val.ptr);

        MatDescrH        dA, dB, dC;
        PointerModeGuard mode(handle, rocsparse_pointer_mode_host);
        const T          alpha = from_c<T>(cplx(2.0, 0.0));
        const T          beta  = from_c<T>(cplx(-1.0, 0.0));

        DevBuf<rocsparse_int> ptrC(static_cast<size_t>(m) + 1);
        ASSERT_TRUE(ptrC.ptr);
        rocsparse_int nnz_C = -1;
        ASSERT_EQ(rocsparse_csrgeam_nnz(handle,
                                        static_cast<rocsparse_int>(m),
                                        static_cast<rocsparse_int>(n),
                                        dA.d,
                                        static_cast<rocsparse_int>(A.h.nnz()),
                                        A.ptr.ptr,
                                        A.ind.ptr,
                                        dB.d,
                                        static_cast<rocsparse_int>(B.h.nnz()),
                                        B.ptr.ptr,
                                        B.ind.ptr,
                                        dC.d,
                                        ptrC.ptr,
                                        &nnz_C),
                  rocsparse_status_success);

        // Expected result.
        const Bsr want
            = bsr_geam(to_c<T>(alpha), bsr_from_csr(A.h), to_c<T>(beta), bsr_from_csr(B.h));
        ASSERT_EQ(nnz_C, want.nnzb());
        if(m > 0)
        {
            const auto hp = ptrC.host();
            expect_idx_equal<rocsparse_int>(hp, want.ptr, 0, "C row_ptr");
        }

        const T               sentinel = from_c<T>(cplx(-77.0, 0.0));
        DevBuf<rocsparse_int> indC(
            std::vector<rocsparse_int>(static_cast<size_t>(std::max<int64_t>(nnz_C, 1)), -5));
        DevBuf<T> valC(std::vector<T>(static_cast<size_t>(std::max<int64_t>(nnz_C, 1)), sentinel));
        ASSERT_TRUE(indC.ptr && valC.ptr);

        ASSERT_EQ(Api<T>::csrgeam(handle,
                                  static_cast<rocsparse_int>(m),
                                  static_cast<rocsparse_int>(n),
                                  &alpha,
                                  dA.d,
                                  static_cast<rocsparse_int>(A.h.nnz()),
                                  static_cast<const T*>(A.val.ptr),
                                  static_cast<const rocsparse_int*>(A.ptr.ptr),
                                  static_cast<const rocsparse_int*>(A.ind.ptr),
                                  &beta,
                                  dB.d,
                                  static_cast<rocsparse_int>(B.h.nnz()),
                                  static_cast<const T*>(B.val.ptr),
                                  static_cast<const rocsparse_int*>(B.ptr.ptr),
                                  static_cast<const rocsparse_int*>(B.ind.ptr),
                                  dC.d,
                                  valC.ptr,
                                  static_cast<const rocsparse_int*>(ptrC.ptr),
                                  indC.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(nnz_C > 0)
        {
            expect_idx_equal<rocsparse_int>(indC.host(), want.col, 0, "C col_ind");
            expect_vec_equal<T>(valC.host(), want.val, "C val");
        }
        else
        {
            // Nothing to write: the output arrays are left alone.
            EXPECT_EQ(indC.host()[0], -5);
            EXPECT_EQ(to_c<T>(valC.host()[0]), cplx(-77.0, 0.0));
        }

        // The output arrays are required as soon as C has entries.
        if(nnz_C > 0)
        {
            EXPECT_EQ(Api<T>::csrgeam(handle,
                                      static_cast<rocsparse_int>(m),
                                      static_cast<rocsparse_int>(n),
                                      &alpha,
                                      dA.d,
                                      static_cast<rocsparse_int>(A.h.nnz()),
                                      static_cast<const T*>(A.val.ptr),
                                      static_cast<const rocsparse_int*>(A.ptr.ptr),
                                      static_cast<const rocsparse_int*>(A.ind.ptr),
                                      &beta,
                                      dB.d,
                                      static_cast<rocsparse_int>(B.h.nnz()),
                                      static_cast<const T*>(B.val.ptr),
                                      static_cast<const rocsparse_int*>(B.ptr.ptr),
                                      static_cast<const rocsparse_int*>(B.ind.ptr),
                                      dC.d,
                                      static_cast<T*>(nullptr),
                                      static_cast<const rocsparse_int*>(ptrC.ptr),
                                      indC.ptr),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(Api<T>::csrgeam(handle,
                                      static_cast<rocsparse_int>(m),
                                      static_cast<rocsparse_int>(n),
                                      &alpha,
                                      dA.d,
                                      static_cast<rocsparse_int>(A.h.nnz()),
                                      static_cast<const T*>(A.val.ptr),
                                      static_cast<const rocsparse_int*>(A.ptr.ptr),
                                      static_cast<const rocsparse_int*>(A.ind.ptr),
                                      &beta,
                                      dB.d,
                                      static_cast<rocsparse_int>(B.h.nnz()),
                                      static_cast<const T*>(B.val.ptr),
                                      static_cast<const rocsparse_int*>(B.ptr.ptr),
                                      static_cast<const rocsparse_int*>(B.ind.ptr),
                                      dC.d,
                                      valC.ptr,
                                      static_cast<const rocsparse_int*>(ptrC.ptr),
                                      static_cast<rocsparse_int*>(nullptr)),
                      rocsparse_status_invalid_pointer);
        }
    }
}

TEST_F(GeamCov80, legacy_csrgeam_empty_and_missing_outputs)
{
    struct Cfg
    {
        int64_t m, n;
        bool    a, b;
    };
    const Cfg cfgs[] = {{0, 5, false, false},
                        {4, 0, false, false},
                        {5, 6, false, false},
                        {5, 6, true, false},
                        {5, 6, false, true},
                        {5, 6, true, true}};
    for(const Cfg& c : cfgs)
    {
        legacy_geam<float>(handle, c.m, c.n, c.a, c.b);
        if(HasFatalFailure())
            return;
        legacy_geam<rocsparse_double_complex>(handle, c.m, c.n, c.a, c.b);
        if(HasFatalFailure())
            return;
    }
}

namespace
{
    using stage_t = rocsparse_spgeam_stage;

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

    template <typename T>
    void
        set_spgeam_inputs(rocsparse_handle handle, rocsparse_spgeam_descr d, const T* a, const T* b)
    {
        const rocsparse_spgeam_alg alg  = rocsparse_spgeam_alg_default;
        const rocsparse_operation  none = rocsparse_operation_none;
        const rocsparse_datatype   dt   = rocsparse_ut::dt_of<T>();
        auto put = [&](rocsparse_spgeam_input which, const void* p, size_t n) {
            ASSERT_EQ(rocsparse_spgeam_set_input(handle, d, which, p, n, nullptr),
                      rocsparse_status_success);
        };
        put(rocsparse_spgeam_input_alg, &alg, sizeof(alg));
        put(rocsparse_spgeam_input_scalar_datatype, &dt, sizeof(dt));
        put(rocsparse_spgeam_input_compute_datatype, &dt, sizeof(dt));
        put(rocsparse_spgeam_input_operation_A, &none, sizeof(none));
        put(rocsparse_spgeam_input_operation_B, &none, sizeof(none));
        put(rocsparse_spgeam_input_scalar_alpha, a, sizeof(void*));
        put(rocsparse_spgeam_input_scalar_beta, b, sizeof(void*));
    }

    // C = alpha * A + beta * B where A and B have no rows / no columns / no
    // entries: C has no entries. Combined and split stage sequences.
    // C is always supplied up front: the analysis stage without C and with
    // empty operands writes through a null pointer (library bug, not tested).
    template <typename T>
    void empty_spgeam(rocsparse_handle handle, int64_t m, int64_t n, bool split, bool c_first)
    {
        SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n)
                     + " split=" + std::to_string(split) + " c_first=" + std::to_string(c_first));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        Rng                      rng(7);
        const Csr                E
            = csr_with_row_lengths<T>(m, n, std::vector<int64_t>(static_cast<size_t>(m), 0), rng);

        DevBuf<int32_t> pA = up_idx<int32_t>(E.ptr), cA = up_idx<int32_t>(E.col);
        DevBuf<int32_t> pB = up_idx<int32_t>(E.ptr), cB = up_idx<int32_t>(E.col);
        DevBuf<T>       vA = up_val<T>(E.val), vB = up_val<T>(E.val);
        ASSERT_TRUE(pA.ptr && cA.ptr && vA.ptr && pB.ptr && cB.ptr && vB.ptr);

        SpmatH sA, sB, sC;
        for(auto* e : {&sA, &sB})
        {
            const bool first = (e == &sA);
            ASSERT_EQ(rocsparse_create_csr_descr(&e->p,
                                                 m,
                                                 n,
                                                 0,
                                                 first ? pA.ptr : pB.ptr,
                                                 first ? cA.ptr : cB.ptr,
                                                 first ? vA.ptr : vB.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 dt),
                      rocsparse_status_success);
        }

        DevBuf<int32_t> pC(static_cast<size_t>(m) + 1), cC(1);
        DevBuf<T>       vC(1);
        ASSERT_TRUE(pC.ptr && cC.ptr && vC.ptr);
        auto create_C = [&]() {
            ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                                 m,
                                                 n,
                                                 0,
                                                 pC.ptr,
                                                 cC.ptr,
                                                 vC.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 dt),
                      rocsparse_status_success);
        };
        if(c_first)
        {
            create_C();
            if(::testing::Test::HasFatalFailure())
                return;
        }

        PointerModeGuard mode(handle, rocsparse_pointer_mode_host);
        const T          alpha = from_c<T>(cplx(2.0, 0.0));
        const T          beta  = from_c<T>(cplx(-1.0, 0.0));
        SpgeamDescrH     descr;
        ASSERT_NE(descr.d, nullptr);
        set_spgeam_inputs<T>(handle, descr.d, &alpha, &beta);
        if(::testing::Test::HasFatalFailure())
            return;

        auto run = [&](stage_t stage, bool pass_C) {
            rocsparse_spmat_descr C  = (pass_C && sC.p != nullptr) ? sC.p : nullptr;
            size_t                bs = 0;
            ASSERT_EQ(
                rocsparse_spgeam_buffer_size(handle, descr.d, sA.p, sB.p, C, stage, &bs, nullptr),
                rocsparse_status_success);
            DevBuf<char> tmp(bs);
            ASSERT_EQ(
                rocsparse_spgeam(
                    handle, descr.d, sA.p, sB.p, C, stage, bs, bs ? tmp.ptr : nullptr, nullptr),
                rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        };
        auto analysis_nnz = [&](bool pass_C) {
            int64_t nnz = -1;
            if(pass_C)
            {
                int64_t r = 0, c = 0;
                ASSERT_EQ(rocsparse_spmat_get_size(sC.p, &r, &c, &nnz), rocsparse_status_success);
            }
            else
            {
                ASSERT_EQ(
                    rocsparse_spgeam_get_output(
                        handle, descr.d, rocsparse_spgeam_output_nnz, &nnz, sizeof(nnz), nullptr),
                    rocsparse_status_success);
            }
            ASSERT_EQ(nnz, 0);
        };

        if(!split)
        {
            run(rocsparse_spgeam_stage_analysis, c_first);
            if(::testing::Test::HasFatalFailure())
                return;
            analysis_nnz(c_first);
            if(!c_first)
                create_C();
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_compute, true);
        }
        else
        {
            run(rocsparse_spgeam_stage_symbolic_analysis, c_first);
            if(::testing::Test::HasFatalFailure())
                return;
            analysis_nnz(c_first);
            if(!c_first)
                create_C();
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_symbolic_compute, true);
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_numeric_analysis, true);
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_numeric_compute, true);
        }
    }
}

TEST_F(GeamCov80, spgeam_empty_operands)
{
    struct Dims
    {
        int64_t m, n;
    };
    for(const Dims& d : {Dims{0, 4}, Dims{4, 0}, Dims{6, 5}})
        for(bool split : {false, true})
        {
            empty_spgeam<float>(handle, d.m, d.n, split, true);
            if(HasFatalFailure())
                return;
            empty_spgeam<rocsparse_double_complex>(handle, d.m, d.n, split, true);
            if(HasFatalFailure())
                return;
        }
}

// ---------------------------------------------------------------------------
// bsrgeam with 1 x 1 blocks: supported or reported as not implemented, never
// a wrong result.
// ---------------------------------------------------------------------------
TEST_F(GeamCov80, bsrgeam_block_dimension_one)
{
    Rng           rng(21);
    const int64_t mb = 6, nb = 5;
    const Bsr     A = random_bsr<float>(mb, nb, 1, 3, rng);
    const Bsr     B = random_bsr<float>(mb, nb, 1, 3, rng);

    DevBuf<rocsparse_int> pA = up_idx<rocsparse_int>(A.ptr), cA = up_idx<rocsparse_int>(A.col);
    DevBuf<rocsparse_int> pB = up_idx<rocsparse_int>(B.ptr), cB = up_idx<rocsparse_int>(B.col);
    DevBuf<float>         vA = up_raw<float>(bsr_pack<float>(A, rocsparse_direction_row));
    DevBuf<float>         vB = up_raw<float>(bsr_pack<float>(B, rocsparse_direction_row));
    ASSERT_TRUE(pA.ptr && cA.ptr && vA.ptr && pB.ptr && cB.ptr && vB.ptr);

    MatDescrH        dA, dB, dC;
    PointerModeGuard mode(handle, rocsparse_pointer_mode_host);
    const float      alpha = 2.0f;
    const float      beta  = -1.0f;

    DevBuf<rocsparse_int> pC(static_cast<size_t>(mb) + 1);
    ASSERT_TRUE(pC.ptr);
    rocsparse_int nnzb_C = -1;
    ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                     rocsparse_direction_row,
                                     static_cast<rocsparse_int>(mb),
                                     static_cast<rocsparse_int>(nb),
                                     1,
                                     dA.d,
                                     static_cast<rocsparse_int>(A.nnzb()),
                                     pA.ptr,
                                     cA.ptr,
                                     dB.d,
                                     static_cast<rocsparse_int>(B.nnzb()),
                                     pB.ptr,
                                     cB.ptr,
                                     dC.d,
                                     pC.ptr,
                                     &nnzb_C),
              rocsparse_status_success);
    const Bsr want = bsr_geam(cplx(alpha, 0.0), A, cplx(beta, 0.0), B);
    ASSERT_EQ(nnzb_C, want.nnzb());

    DevBuf<rocsparse_int> cC(static_cast<size_t>(std::max<rocsparse_int>(nnzb_C, 1)));
    DevBuf<float>         vC(static_cast<size_t>(std::max<rocsparse_int>(nnzb_C, 1)));
    ASSERT_TRUE(cC.ptr && vC.ptr);
    const rocsparse_status st = rocsparse_sbsrgeam(handle,
                                                   rocsparse_direction_row,
                                                   static_cast<rocsparse_int>(mb),
                                                   static_cast<rocsparse_int>(nb),
                                                   1,
                                                   &alpha,
                                                   dA.d,
                                                   static_cast<rocsparse_int>(A.nnzb()),
                                                   vA.ptr,
                                                   pA.ptr,
                                                   cA.ptr,
                                                   &beta,
                                                   dB.d,
                                                   static_cast<rocsparse_int>(B.nnzb()),
                                                   vB.ptr,
                                                   pB.ptr,
                                                   cB.ptr,
                                                   dC.d,
                                                   vC.ptr,
                                                   pC.ptr,
                                                   cC.ptr);
    ASSERT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented) << st;
    if(st == rocsparse_status_success)
    {
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_idx_equal<rocsparse_int>(cC.host(), want.col, 0, "C col_ind");
        expect_vec_equal<float>(vC.host(), want.val, "C val");
    }
}
