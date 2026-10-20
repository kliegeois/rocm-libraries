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
// Generic rocsparse_spgemm on CSR and BSR matrices:
//
//   C = alpha * A * B + beta * D     (alpha and beta given)
//   C = alpha * A * B                (beta == nullptr)
//   C = beta * D                     (alpha == nullptr)
//
// through the stages buffer_size / nnz / compute, and for CSR also through
// buffer_size / nnz / symbolic / numeric (the numeric stage is repeated with
// different scalars). Index types (i32/i32, i64/i32, i64/i64), value types,
// index bases of every operand, host and device scalars, empty operands and
// empty dimensions are covered. The whole of C (row pointer, column indices,
// values) is compared with a host reference.
//
#include "unit_test_cov80_extra_common.hpp"

using namespace cov80_extra;

namespace
{
    enum class Mode
    {
        multadd,
        mult,
        add
    };

    struct Case
    {
        bool                bsr  = false;
        int64_t             bd   = 1;
        rocsparse_direction dir  = rocsparse_direction_row;
        Mode                mode = Mode::multadd;
        int64_t             mb = 7, nb = 9, kb = 6;
        int64_t             max_row = 3;
        bool                device  = false;
        // Index bases of A, B, C, D (independent of each other).
        rocsparse_index_base bA = rocsparse_index_base_zero, bB = rocsparse_index_base_zero,
                             bC = rocsparse_index_base_zero, bD = rocsparse_index_base_zero;
        bool     split   = false; // CSR only: symbolic + numeric stages
        bool     empty_A = false, empty_B = false, empty_D = false;
        uint64_t seed = 1;
    };

    std::string describe(const Case& c)
    {
        return std::string(c.bsr ? "bsr" : "csr") + " mode=" + std::to_string(int(c.mode))
               + " mb=" + std::to_string(c.mb) + " nb=" + std::to_string(c.nb)
               + " kb=" + std::to_string(c.kb) + " bd=" + std::to_string(c.bd)
               + " dir=" + std::to_string(int(c.dir)) + " dev=" + std::to_string(c.device)
               + " split=" + std::to_string(c.split) + " bases=" + std::to_string(int(c.bA))
               + std::to_string(int(c.bB)) + std::to_string(int(c.bC)) + std::to_string(int(c.bD));
    }

    Bsr empty_bsr(int64_t mb, int64_t nb, int64_t bd)
    {
        Bsr E;
        E.mb  = mb;
        E.nb  = nb;
        E.bd  = bd;
        E.ptr = std::vector<int64_t>(static_cast<size_t>(mb) + 1, 0);
        return E;
    }

    // An operand: host matrix, device arrays and the generic descriptor.
    template <typename T, typename I, typename J>
    struct Operand
    {
        Bsr       h;
        DevBuf<I> p;
        DevBuf<J> c;
        DevBuf<T> v;
        SpmatH    d;

        void create(const Bsr& host, bool bsr, rocsparse_direction dir, rocsparse_index_base base)
        {
            h = host;
            p = up_idx<I>(h.ptr, static_cast<int64_t>(base));
            c = up_idx<J>(h.col, static_cast<int64_t>(base));
            v = up_raw<T>(bsr_pack<T>(h, dir));
            ASSERT_TRUE(p.ptr && c.ptr && v.ptr);
            if(bsr)
            {
                ASSERT_EQ(rocsparse_create_bsr_descr(&d.p,
                                                     h.mb,
                                                     h.nb,
                                                     h.nnzb(),
                                                     dir,
                                                     h.bd,
                                                     p.ptr,
                                                     c.ptr,
                                                     v.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     rocsparse_ut::it_of<J>(),
                                                     base,
                                                     rocsparse_ut::dt_of<T>()),
                          rocsparse_status_success);
            }
            else
            {
                ASSERT_EQ(rocsparse_create_csr_descr(&d.p,
                                                     h.mb,
                                                     h.nb,
                                                     h.nnzb(),
                                                     p.ptr,
                                                     c.ptr,
                                                     v.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     rocsparse_ut::it_of<J>(),
                                                     base,
                                                     rocsparse_ut::dt_of<T>()),
                          rocsparse_status_success);
            }
        }
    };

    // Compares a computed matrix with the reference. The blocks of every row
    // are matched by column index, the order they are emitted in is not
    // checked (the generic documentation does not specify it).
    template <typename T, typename I, typename J>
    void expect_bsr_equal(const std::vector<I>& ptr,
                          const std::vector<J>& col,
                          const std::vector<T>& val_dev,
                          const Bsr&            want,
                          rocsparse_direction   dir,
                          int64_t               base)
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

    // The generic call with a given stage.
    template <typename T>
    rocsparse_status spgemm_call(rocsparse_handle            handle,
                                 const void*                 alpha,
                                 rocsparse_const_spmat_descr A,
                                 rocsparse_const_spmat_descr B,
                                 const void*                 beta,
                                 rocsparse_const_spmat_descr D,
                                 rocsparse_spmat_descr       C,
                                 rocsparse_spgemm_stage      stage,
                                 size_t*                     bs,
                                 void*                       tmp)
    {
        return rocsparse_spgemm(handle,
                                rocsparse_operation_none,
                                rocsparse_operation_none,
                                alpha,
                                A,
                                B,
                                beta,
                                D,
                                C,
                                rocsparse_ut::dt_of<T>(),
                                rocsparse_spgemm_alg_default,
                                stage,
                                bs,
                                tmp);
    }

    template <typename T, typename I, typename J>
    void run(rocsparse_handle handle, const Case& c)
    {
        SCOPED_TRACE(describe(c));
        const bool    mul = (c.mode != Mode::add);
        const bool    add = (c.mode != Mode::mult);
        const int64_t bd  = c.bsr ? c.bd : 1;

        Rng rng(c.seed * 977 + static_cast<uint64_t>(c.mb * 13 + c.nb * 7 + c.kb + bd));

        Bsr A = random_bsr<T>(c.mb, c.kb, bd, c.max_row, rng);
        Bsr B = random_bsr<T>(c.kb, c.nb, bd, c.max_row, rng);
        Bsr D = random_bsr<T>(c.mb, c.nb, bd, c.max_row, rng);
        if(c.empty_A)
            A = empty_bsr(c.mb, c.kb, bd);
        if(c.empty_B)
            B = empty_bsr(c.kb, c.nb, bd);
        if(c.empty_D)
            D = empty_bsr(c.mb, c.nb, bd);
        // Operands that do not take part in the operation are empty.
        const Bsr Ause = mul ? A : empty_bsr(c.mb, c.kb, bd);
        const Bsr Buse = mul ? B : empty_bsr(c.kb, c.nb, bd);
        const Bsr Duse = add ? D : empty_bsr(c.mb, c.nb, bd);

        Operand<T, I, J> oA, oB, oD;
        ASSERT_NO_FATAL_FAILURE(oA.create(Ause, c.bsr, c.dir, c.bA));
        ASSERT_NO_FATAL_FAILURE(oB.create(Buse, c.bsr, c.dir, c.bB));
        ASSERT_NO_FATAL_FAILURE(oD.create(Duse, c.bsr, c.dir, c.bD));

        DevBuf<I> pC(static_cast<size_t>(c.mb) + 1);
        ASSERT_TRUE(pC.ptr);
        SpmatH C;
        if(c.bsr)
        {
            ASSERT_EQ(rocsparse_create_bsr_descr(&C.p,
                                                 c.mb,
                                                 c.nb,
                                                 0,
                                                 c.dir,
                                                 bd,
                                                 pC.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 c.bC,
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&C.p,
                                                 c.mb,
                                                 c.nb,
                                                 0,
                                                 pC.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 c.bC,
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }

        PointerModeGuard mode(handle, mode_of(c.device));

        auto scalars = [&](cplx a, cplx b) {
            return std::make_pair(std::make_unique<Scalar<T>>(a, c.device),
                                  std::make_unique<Scalar<T>>(b, c.device));
        };
        auto s1 = scalars(cplx(2.0, 1.0), cplx(-1.0, 0.5));

        auto call = [&](const Scalar<T>&       a,
                        const Scalar<T>&       b,
                        rocsparse_spgemm_stage stage,
                        size_t*                bs,
                        void*                  tmp) {
            return spgemm_call<T>(handle,
                                  mul ? a.get() : nullptr,
                                  oA.d.p,
                                  oB.d.p,
                                  add ? b.get() : nullptr,
                                  oD.d.p,
                                  C.p,
                                  stage,
                                  bs,
                                  tmp);
        };

        size_t bsz = 0;
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_buffer_size, &bsz, nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_nnz, &bsz, buffer.ptr),
                  rocsparse_status_success);

        const Bsr want1 = bsr_gemm(
            mul, to_c<T>(s1.first->h), Ause, Buse, add, to_c<T>(s1.second->h), Duse, c.mb, c.nb);

        int64_t rows = 0, cols = 0, nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz), rocsparse_status_success);
        ASSERT_EQ(nnz, want1.nnzb());

        DevBuf<J> cC(static_cast<size_t>(nnz));
        DevBuf<T> vC(static_cast<size_t>(nnz * bd * bd));
        ASSERT_TRUE(cC.ptr && vC.ptr);
        if(c.bsr)
            ASSERT_EQ(rocsparse_bsr_set_pointers(C.p, pC.ptr, cC.ptr, vC.ptr),
                      rocsparse_status_success);
        else
            ASSERT_EQ(rocsparse_csr_set_pointers(C.p, pC.ptr, cC.ptr, vC.ptr),
                      rocsparse_status_success);

        auto check = [&](const Bsr& want) {
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            expect_bsr_equal<T, I, J>(
                pC.host(), cC.host(), vC.host(), want, c.dir, static_cast<int64_t>(c.bC));
        };

        if(!c.split)
        {
            ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_compute, &bsz, buffer.ptr),
                      rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(check(want1));
            return;
        }

        // Symbolic stage: row pointer and column indices of C only.
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_symbolic, &bsz, buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        {
            // The pattern is already final, the values are not required.
            std::vector<int64_t> cols_want = want1.col;
            std::vector<J>       got       = cC.host();
            std::vector<I>       gp        = pC.host();
            ASSERT_GE(gp.size(), want1.ptr.size());
            for(int64_t i = 0; i < c.mb; ++i)
            {
                std::set<int64_t> g, w;
                for(int64_t q = static_cast<int64_t>(gp[static_cast<size_t>(i)] - c.bC);
                    q < static_cast<int64_t>(gp[static_cast<size_t>(i) + 1] - c.bC);
                    ++q)
                    g.insert(static_cast<int64_t>(got[static_cast<size_t>(q)]) - c.bC);
                for(int64_t q = want1.ptr[static_cast<size_t>(i)];
                    q < want1.ptr[static_cast<size_t>(i) + 1];
                    ++q)
                    w.insert(want1.col[static_cast<size_t>(q)]);
                ASSERT_EQ(g, w) << "symbolic pattern of row " << i;
            }
        }

        // Numeric stage, repeated with different scalars.
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_numeric, &bsz, buffer.ptr),
                  rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(check(want1));

        auto s2 = scalars(cplx(-3.0, 0.0), cplx(2.0, -1.0));
        ASSERT_EQ(call(*s2.first, *s2.second, rocsparse_spgemm_stage_numeric, &bsz, buffer.ptr),
                  rocsparse_status_success);
        const Bsr want2 = bsr_gemm(
            mul, to_c<T>(s2.first->h), Ause, Buse, add, to_c<T>(s2.second->h), Duse, c.mb, c.nb);
        ASSERT_NO_FATAL_FAILURE(check(want2));
    }

    const std::vector<Mode> kModes = {Mode::multadd, Mode::mult, Mode::add};

    // Every mode, with and without the split symbolic / numeric flow.
    template <typename T, typename I, typename J>
    void sweep_csr(rocsparse_handle handle)
    {
        for(Mode m : kModes)
            for(bool split : {false, true})
            {
                Case c;
                c.mode  = m;
                c.split = split;
                run<T, I, J>(handle, c);
                if(::testing::Test::HasFatalFailure())
                    return;
            }
    }

    template <typename T, typename I, typename J>
    void sweep_bsr(rocsparse_handle handle, const std::vector<int64_t>& dims)
    {
        for(int64_t bd : dims)
            for(Mode m : kModes)
                for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
                {
                    Case c;
                    c.bsr  = true;
                    c.bd   = bd;
                    c.mode = m;
                    c.dir  = dir;
                    run<T, I, J>(handle, c);
                    if(::testing::Test::HasFatalFailure())
                        return;
                }
    }

    template <typename T, typename I, typename J>
    void sweep_options(rocsparse_handle handle, bool bsr)
    {
        const rocsparse_index_base Z = rocsparse_index_base_zero;
        const rocsparse_index_base O = rocsparse_index_base_one;
        for(Mode m : kModes)
        {
            Case c;
            c.bsr  = bsr;
            c.bd   = bsr ? 3 : 1;
            c.mode = m;
            for(bool split : {false, true})
            {
                c.split  = split && !bsr;
                c.device = true;
                run<T, I, J>(handle, c);
                c.device = false;
                c.bA = c.bB = c.bC = c.bD = O;
                run<T, I, J>(handle, c);
                c.device = true;
                run<T, I, J>(handle, c);
                c.device = false;
                c.bA     = O;
                c.bB     = Z;
                c.bC     = O;
                c.bD     = Z;
                run<T, I, J>(handle, c);
                c.bA = Z;
                c.bB = O;
                c.bC = Z;
                c.bD = O;
                run<T, I, J>(handle, c);
                c.bA = c.bB = c.bC = c.bD = Z;
                if(::testing::Test::HasFatalFailure())
                    return;
            }
        }
    }

    template <typename T, typename I, typename J>
    void sweep_degenerate(rocsparse_handle handle, bool bsr)
    {
        for(Mode m : kModes)
            for(int which = 0; which < 7; ++which)
            {
                Case c;
                c.bsr  = bsr;
                c.bd   = bsr ? 2 : 1;
                c.mode = m;
                switch(which)
                {
                case 0:
                    c.empty_A = true;
                    break;
                case 1:
                    c.empty_B = true;
                    break;
                case 2:
                    c.empty_D = true;
                    break;
                case 3:
                    c.kb = 0;
                    break;
                case 4:
                    c.mb = 0;
                    break;
                case 5:
                    c.nb = 0;
                    break;
                case 6:
                    c.empty_A = c.empty_D = true;
                    break;
                }
                for(bool dev : {false, true})
                    for(bool split : {false, true})
                    {
                        if(split && bsr)
                            continue;
                        c.device = dev;
                        c.split  = split;
                        run<T, I, J>(handle, c);
                        if(::testing::Test::HasFatalFailure())
                            return;
                    }
            }
    }

    struct SpgemmCov80 : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(SpgemmCov80, csr_index_types_float)
{
    sweep_csr<float, int32_t, int32_t>(handle);
    sweep_csr<float, int64_t, int32_t>(handle);
    sweep_csr<float, int64_t, int64_t>(handle);
}

TEST_F(SpgemmCov80, csr_value_types)
{
    sweep_csr<double, int32_t, int32_t>(handle);
    sweep_csr<rocsparse_float_complex, int32_t, int32_t>(handle);
    sweep_csr<rocsparse_double_complex, int64_t, int32_t>(handle);
    sweep_csr<rocsparse_double_complex, int64_t, int64_t>(handle);
}

TEST_F(SpgemmCov80, csr_options)
{
    sweep_options<float, int32_t, int32_t>(handle, false);
    sweep_options<double, int64_t, int64_t>(handle, false);
}

TEST_F(SpgemmCov80, csr_degenerate)
{
    sweep_degenerate<float, int32_t, int32_t>(handle, false);
    sweep_degenerate<double, int64_t, int32_t>(handle, false);
}

TEST_F(SpgemmCov80, bsr_block_dims_float)
{
    sweep_bsr<float, int32_t, int32_t>(handle, {2, 3, 4, 5, 8, 9, 16, 17});
}

TEST_F(SpgemmCov80, bsr_index_and_value_types)
{
    sweep_bsr<float, int64_t, int32_t>(handle, {2, 5});
    sweep_bsr<double, int64_t, int64_t>(handle, {3, 8});
    sweep_bsr<rocsparse_float_complex, int32_t, int32_t>(handle, {2, 4});
    sweep_bsr<rocsparse_double_complex, int32_t, int32_t>(handle, {3});
}

TEST_F(SpgemmCov80, bsr_options)
{
    sweep_options<float, int32_t, int32_t>(handle, true);
    sweep_options<double, int64_t, int32_t>(handle, true);
}

TEST_F(SpgemmCov80, bsr_degenerate)
{
    sweep_degenerate<float, int32_t, int32_t>(handle, true);
    sweep_degenerate<rocsparse_double_complex, int64_t, int64_t>(handle, true);
}

// BSR has no separate symbolic / numeric stage.
TEST_F(SpgemmCov80, bsr_symbolic_numeric_not_available)
{
    Case c;
    c.bsr   = true;
    c.bd    = 2;
    c.split = true;
    Rng                              rng(5);
    const Bsr                        A = random_bsr<float>(c.mb, c.kb, 2, 3, rng);
    const Bsr                        B = random_bsr<float>(c.kb, c.nb, 2, 3, rng);
    const Bsr                        D = random_bsr<float>(c.mb, c.nb, 2, 3, rng);
    Operand<float, int32_t, int32_t> oA, oB, oD;
    ASSERT_NO_FATAL_FAILURE(oA.create(A, true, c.dir, c.bA));
    ASSERT_NO_FATAL_FAILURE(oB.create(B, true, c.dir, c.bB));
    ASSERT_NO_FATAL_FAILURE(oD.create(D, true, c.dir, c.bD));
    DevBuf<int32_t> pC(static_cast<size_t>(c.mb) + 1);
    SpmatH          C;
    ASSERT_EQ(rocsparse_create_bsr_descr(&C.p,
                                         c.mb,
                                         c.nb,
                                         0,
                                         c.dir,
                                         2,
                                         pC.ptr,
                                         nullptr,
                                         nullptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         c.bC,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    const float alpha = 1.0f, beta = 1.0f;
    size_t      bsz = 0;
    ASSERT_EQ(spgemm_call<float>(handle,
                                 &alpha,
                                 oA.d.p,
                                 oB.d.p,
                                 &beta,
                                 oD.d.p,
                                 C.p,
                                 rocsparse_spgemm_stage_buffer_size,
                                 &bsz,
                                 nullptr),
              rocsparse_status_success);
    DevBuf<char> buffer(bsz);
    ASSERT_TRUE(buffer.ptr);
    for(rocsparse_spgemm_stage stage :
        {rocsparse_spgemm_stage_symbolic, rocsparse_spgemm_stage_numeric})
    {
        EXPECT_NE(spgemm_call<float>(
                      handle, &alpha, oA.d.p, oB.d.p, &beta, oD.d.p, C.p, stage, &bsz, buffer.ptr),
                  rocsparse_status_success);
    }
}

// Rows with many intermediate products go through the larger hashing groups
// and, beyond 65536 products, the bitmap algorithm.
namespace
{
    template <typename T>
    void long_rows(rocsparse_handle handle, Mode mode, bool sorted_B)
    {
        const bool    mul = (mode != Mode::add);
        const bool    add = (mode != Mode::mult);
        const int64_t m = 5, k = 300, n = 3000;
        Rng           rng(11);

        // A: first row touches every row of B, the others a few; B rows of 260
        // entries: row 0 of C has 78000 intermediate products.
        std::vector<int64_t> la = {300, 150, 40, 3, 0};
        std::vector<int64_t> lb(static_cast<size_t>(k), 260);
        std::vector<int64_t> ld = {2000, 5, 0, 100, 40};
        const Csr            A  = csr_with_row_lengths<T>(m, k, la, rng);
        const Csr            B  = csr_with_row_lengths<T>(k, n, lb, rng);
        const Csr            D  = csr_with_row_lengths<T>(m, n, ld, rng);
        Csr                  E0, E1, E2;
        E0.m          = m;
        E0.n          = k;
        E0.ptr        = std::vector<int64_t>(static_cast<size_t>(m) + 1, 0);
        E1.m          = k;
        E1.n          = n;
        E1.ptr        = std::vector<int64_t>(static_cast<size_t>(k) + 1, 0);
        E2.m          = m;
        E2.n          = n;
        E2.ptr        = std::vector<int64_t>(static_cast<size_t>(m) + 1, 0);
        const Csr& Au = mul ? A : E0;
        const Csr& Bu = mul ? B : E1;
        const Csr& Du = add ? D : E2;

        const Bsr want = bsr_gemm(mul,
                                  cplx(2.0, 0.0),
                                  bsr_from_csr(Au),
                                  bsr_from_csr(Bu),
                                  add,
                                  cplx(-1.0, 0.0),
                                  bsr_from_csr(Du),
                                  m,
                                  n);

        Operand<T, int32_t, int32_t> oA, oB, oD;
        ASSERT_NO_FATAL_FAILURE(
            oA.create(bsr_from_csr(Au), false, rocsparse_direction_row, rocsparse_index_base_zero));
        ASSERT_NO_FATAL_FAILURE(
            oB.create(bsr_from_csr(Bu), false, rocsparse_direction_row, rocsparse_index_base_zero));
        ASSERT_NO_FATAL_FAILURE(
            oD.create(bsr_from_csr(Du), false, rocsparse_direction_row, rocsparse_index_base_zero));
        if(!sorted_B)
        {
            const rocsparse_storage_mode unsorted = rocsparse_storage_mode_unsorted;
            ASSERT_EQ(rocsparse_spmat_set_attribute(
                          oB.d.p, rocsparse_spmat_storage_mode, &unsorted, sizeof(unsorted)),
                      rocsparse_status_success);
        }

        DevBuf<int32_t> pC(static_cast<size_t>(m) + 1);
        SpmatH          C;
        ASSERT_EQ(rocsparse_create_csr_descr(&C.p,
                                             m,
                                             n,
                                             0,
                                             pC.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);

        const Scalar<T> alpha(cplx(2.0, 0.0), false), beta(cplx(-1.0, 0.0), false);
        const void*     a   = mul ? alpha.get() : nullptr;
        const void*     b   = add ? beta.get() : nullptr;
        size_t          bsz = 0;
        ASSERT_EQ(spgemm_call<T>(handle,
                                 a,
                                 oA.d.p,
                                 oB.d.p,
                                 b,
                                 oD.d.p,
                                 C.p,
                                 rocsparse_spgemm_stage_buffer_size,
                                 &bsz,
                                 nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        const rocsparse_status st = spgemm_call<T>(handle,
                                                   a,
                                                   oA.d.p,
                                                   oB.d.p,
                                                   b,
                                                   oD.d.p,
                                                   C.p,
                                                   rocsparse_spgemm_stage_nnz,
                                                   &bsz,
                                                   buffer.ptr);
        if(!sorted_B && mul)
        {
            // A row of C needs the bitmap algorithm, which requires sorted B.
            ASSERT_EQ(st, rocsparse_status_requires_sorted_storage);
            return;
        }
        ASSERT_EQ(st, rocsparse_status_success);
        int64_t rows = 0, cols = 0, nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz), rocsparse_status_success);
        ASSERT_EQ(nnz, want.nnzb());
        DevBuf<int32_t> cC(static_cast<size_t>(nnz));
        DevBuf<T>       vC(static_cast<size_t>(nnz));
        ASSERT_TRUE(cC.ptr && vC.ptr);
        ASSERT_EQ(rocsparse_csr_set_pointers(C.p, pC.ptr, cC.ptr, vC.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(spgemm_call<T>(handle,
                                 a,
                                 oA.d.p,
                                 oB.d.p,
                                 b,
                                 oD.d.p,
                                 C.p,
                                 rocsparse_spgemm_stage_compute,
                                 &bsz,
                                 buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_bsr_equal<T, int32_t, int32_t>(
            pC.host(), cC.host(), vC.host(), want, rocsparse_direction_row, 0);
    }
}

TEST_F(SpgemmCov80, long_rows_sorted)
{
    for(Mode m : kModes)
    {
        long_rows<float>(handle, m, true);
        if(HasFatalFailure())
            return;
    }
}

TEST_F(SpgemmCov80, long_rows_unsorted_B)
{
    for(Mode m : kModes)
    {
        long_rows<float>(handle, m, false);
        if(HasFatalFailure())
            return;
    }
}

// Legacy entry points with an empty A in the product-only mode.
TEST_F(SpgemmCov80, legacy_nnz_empty_product)
{
    const rocsparse_int   m = 4, n = 5, k = 3;
    MatDescrH             dA, dB, dC, dD;
    MatInfoH              info;
    DevBuf<rocsparse_int> pA(static_cast<size_t>(m) + 1), iA(size_t{1}),
        pB(static_cast<size_t>(k) + 1), iB(size_t{1}), pD(static_cast<size_t>(m) + 1),
        iD(size_t{1}), pC(static_cast<size_t>(m) + 1);
    ASSERT_TRUE(pA.ptr && pB.ptr && pC.ptr && pD.ptr);
    const float alpha = 1.0f;
    size_t      bsz   = 0;
    ASSERT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                             rocsparse_operation_none,
                                             rocsparse_operation_none,
                                             m,
                                             n,
                                             k,
                                             &alpha,
                                             dA.d,
                                             0,
                                             pA.ptr,
                                             iA.ptr,
                                             dB.d,
                                             0,
                                             pB.ptr,
                                             iB.ptr,
                                             nullptr,
                                             dD.d,
                                             0,
                                             pD.ptr,
                                             iD.ptr,
                                             info.i,
                                             &bsz),
              rocsparse_status_success);
    DevBuf<char>  buffer(bsz);
    rocsparse_int nnz = -1;
    ASSERT_EQ(rocsparse_csrgemm_nnz(handle,
                                    rocsparse_operation_none,
                                    rocsparse_operation_none,
                                    m,
                                    n,
                                    k,
                                    dA.d,
                                    0,
                                    pA.ptr,
                                    iA.ptr,
                                    dB.d,
                                    0,
                                    pB.ptr,
                                    iB.ptr,
                                    dD.d,
                                    0,
                                    pD.ptr,
                                    iD.ptr,
                                    dC.d,
                                    pC.ptr,
                                    &nnz,
                                    info.i,
                                    buffer.ptr),
              rocsparse_status_success);
    EXPECT_EQ(nnz, 0);
}
