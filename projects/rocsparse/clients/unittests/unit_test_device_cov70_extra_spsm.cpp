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
// Device unit tests for library/src/level3/refactored/rocsparse_spsm.cpp.
//
// The generic rocsparse_spsm API solves op(A) * C = alpha * op(B) for a
// triangular sparse A. The three stages (buffer size, preprocess, compute)
// are run for CSR, CSC and COO matrices, both fill modes, unit and non-unit
// diagonals, every operation on A and B, row and column major B and C, host
// and device alpha and several numbers of right-hand sides. The complete
// solution C is compared with a host forward / backward substitution and the
// input B and the padding of the leading dimensions are checked to be
// untouched. The argument checks and the unsupported formats are exercised
// separately.
//
#include "unit_test_cov70_extra_common.hpp"

#include <memory>

using namespace cov70;

namespace
{
    enum class Fmt
    {
        csr,
        csc,
        coo
    };

    struct SpsmCase
    {
        Fmt                 fmt     = Fmt::csr;
        rocsparse_operation opA     = rocsparse_operation_none;
        rocsparse_operation opB     = rocsparse_operation_none;
        rocsparse_order     order_B = rocsparse_order_column;
        rocsparse_order     order_C = rocsparse_order_column;
        rocsparse_fill_mode fill    = rocsparse_fill_mode_lower;
        rocsparse_diag_type diag    = rocsparse_diag_type_non_unit;
        int64_t             m       = 11;
        int64_t             nrhs    = 5;
        bool                device  = false;
        cplx                alpha   = cplx(2.0, 0.0);
        int64_t             ld_pad  = 2;
        uint64_t            seed    = 1;
    };

    std::string describe(const SpsmCase& c)
    {
        return "fmt=" + std::to_string(static_cast<int>(c.fmt))
               + " opA=" + std::to_string(static_cast<int>(c.opA))
               + " opB=" + std::to_string(static_cast<int>(c.opB))
               + " order_B=" + std::to_string(static_cast<int>(c.order_B))
               + " order_C=" + std::to_string(static_cast<int>(c.order_C))
               + " fill=" + std::to_string(static_cast<int>(c.fill))
               + " diag=" + std::to_string(static_cast<int>(c.diag)) + " m=" + std::to_string(c.m)
               + " nrhs=" + std::to_string(c.nrhs) + " device=" + std::to_string(c.device);
    }

    // Well conditioned triangular matrix: up to three off-diagonal entries of
    // magnitude one and a diagonal of 4 or 8.
    Csr triangular(int64_t m, bool lower, Rng& rng)
    {
        Csr A;
        A.m = A.n = m;
        A.ptr.assign(1, 0);
        for(int64_t i = 0; i < m; ++i)
        {
            const int64_t        room = lower ? i : (m - 1 - i);
            std::vector<int64_t> cols;
            const int64_t        k = std::min<int64_t>(room, rng.range(0, 3));
            while(static_cast<int64_t>(cols.size()) < k)
            {
                const int64_t j = lower ? rng.range(0, i - 1) : rng.range(i + 1, m - 1);
                if(std::find(cols.begin(), cols.end(), j) == cols.end())
                    cols.push_back(j);
            }
            cols.push_back(i);
            std::sort(cols.begin(), cols.end());
            for(int64_t j : cols)
            {
                A.col.push_back(j);
                if(j == i)
                    A.val.push_back(cplx(rng.range(0, 1) == 0 ? 4.0 : 8.0, 0.0));
                else
                    A.val.push_back(cplx(rng.range(0, 1) == 0 ? -1.0 : 1.0, 0.0));
            }
            A.ptr.push_back(static_cast<int64_t>(A.col.size()));
        }
        return A;
    }

    // Host reference: solve M X = R for a dense triangular M.
    Dense solve_triangular(const Dense& M, bool lower, const Dense& R)
    {
        Dense         X(R.rows, R.cols);
        const int64_t m = M.rows;
        for(int64_t col = 0; col < R.cols; ++col)
            for(int64_t s = 0; s < m; ++s)
            {
                const int64_t i   = lower ? s : (m - 1 - s);
                cplx          sum = R.at(i, col);
                for(int64_t j = 0; j < m; ++j)
                    if(j != i && (lower ? j < i : j > i))
                        sum -= M.at(i, j) * X.at(j, col);
                X.at(i, col) = sum / M.at(i, i);
            }
        return X;
    }

    template <typename T, typename I, typename J>
    struct Problem
    {
        rocsparse_handle handle = nullptr;
        SpsmCase         c;
        Csr              A;
        Dense            B_logical, B_stored, want;
        int64_t          ldb = 0, ldc = 0;
        DevBuf<I>        ptr;
        DevBuf<J>        ind;
        DevBuf<I>        coo_col;
        DevBuf<T>        val, dB, dC;
        SpmatH           sA;
        DnmatH           mB, mC;
        DevBuf<char>     buffer;
        size_t           bsz = 0;

        std::unique_ptr<Scalar<T>>        alpha;
        std::unique_ptr<PointerModeGuard> mode;

        void init(rocsparse_handle h, const SpsmCase& cs)
        {
            handle = h;
            c      = cs;
            Rng rng(c.seed * 977 + static_cast<uint64_t>(c.m * 13 + c.nrhs));
            A = triangular(c.m, c.fill == rocsparse_fill_mode_lower, rng);

            B_logical = random_dense<T>(c.m, c.nrhs, rng);
            B_stored  = apply_op(B_logical, c.opB);
            const Dense C0(c.m, c.nrhs);
            ldb = min_ld(B_stored, c.order_B) + c.ld_pad;
            ldc = min_ld(C0, c.order_C) + c.ld_pad;
            dB  = up_raw<T>(store_dense<T>(B_stored, c.order_B, ldb));
            dC  = up_raw<T>(store_dense<T>(C0, c.order_C, ldc));
            ASSERT_TRUE(dB.ptr && dC.ptr);

            // Reference.
            Dense M = dense_of(A);
            if(c.diag == rocsparse_diag_type_unit)
                for(int64_t i = 0; i < c.m; ++i)
                    M.at(i, i) = cplx(1.0, 0.0);
            M = apply_op(M, c.opA);
            const bool lower
                = (c.fill == rocsparse_fill_mode_lower) == (c.opA == rocsparse_operation_none);
            Dense R = B_logical;
            alpha.reset(new Scalar<T>(c.alpha, c.device));
            for(auto& x : R.v)
                x *= to_c<T>(alpha->h);
            want = solve_triangular(M, lower, R);

            mode = std::make_unique<PointerModeGuard>(handle, mode_of(c.device));

            ASSERT_NO_FATAL_FAILURE(create_matrix());
            ASSERT_EQ(rocsparse_spmat_set_attribute(
                          sA.p, rocsparse_spmat_fill_mode, &c.fill, sizeof(c.fill)),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spmat_set_attribute(
                          sA.p, rocsparse_spmat_diag_type, &c.diag, sizeof(c.diag)),
                      rocsparse_status_success);

            const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
            ASSERT_EQ(rocsparse_create_dnmat_descr(
                          &mB.p, B_stored.rows, B_stored.cols, ldb, dB.ptr, dt, c.order_B),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_dnmat_descr(&mC.p, c.m, c.nrhs, ldc, dC.ptr, dt, c.order_C),
                      rocsparse_status_success);
        }

        void create_matrix()
        {
            const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
            switch(c.fmt)
            {
            case Fmt::csr:
            {
                ptr = up_idx<I>(A.ptr);
                ind = up_idx<J>(A.col);
                val = up_val<T>(A.val);
                ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr);
                ASSERT_EQ(rocsparse_create_csr_descr(&sA.p,
                                                     A.m,
                                                     A.n,
                                                     A.nnz(),
                                                     ptr.ptr,
                                                     ind.ptr,
                                                     val.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     rocsparse_ut::it_of<J>(),
                                                     rocsparse_index_base_zero,
                                                     dt),
                          rocsparse_status_success);
                return;
            }
            case Fmt::csc:
            {
                const Csr At = transpose(A);
                ptr          = up_idx<I>(At.ptr);
                ind          = up_idx<J>(At.col);
                val          = up_val<T>(At.val);
                ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr);
                ASSERT_EQ(rocsparse_create_csc_descr(&sA.p,
                                                     A.m,
                                                     A.n,
                                                     A.nnz(),
                                                     ptr.ptr,
                                                     ind.ptr,
                                                     val.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     rocsparse_ut::it_of<J>(),
                                                     rocsparse_index_base_zero,
                                                     dt),
                          rocsparse_status_success);
                return;
            }
            case Fmt::coo:
            {
                std::vector<int64_t> rows;
                for(int64_t i = 0; i < A.m; ++i)
                    for(int64_t p = A.ptr[static_cast<size_t>(i)];
                        p < A.ptr[static_cast<size_t>(i) + 1];
                        ++p)
                        rows.push_back(i);
                ptr     = up_idx<I>(rows);
                coo_col = up_idx<I>(A.col);
                val     = up_val<T>(A.val);
                ASSERT_TRUE(ptr.ptr && coo_col.ptr && val.ptr);
                ASSERT_EQ(rocsparse_create_coo_descr(&sA.p,
                                                     A.m,
                                                     A.n,
                                                     A.nnz(),
                                                     ptr.ptr,
                                                     coo_col.ptr,
                                                     val.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     rocsparse_index_base_zero,
                                                     dt),
                          rocsparse_status_success);
                return;
            }
            }
        }

        rocsparse_status stage(rocsparse_spsm_stage s)
        {
            return rocsparse_spsm(handle,
                                  c.opA,
                                  c.opB,
                                  alpha->get(),
                                  sA.p,
                                  mB.p,
                                  mC.p,
                                  rocsparse_ut::dt_of<T>(),
                                  rocsparse_spsm_alg_default,
                                  s,
                                  &bsz,
                                  buffer.ptr);
        }

        void run_all()
        {
            ASSERT_EQ(stage(rocsparse_spsm_stage_buffer_size), rocsparse_status_success);
            buffer = DevBuf<char>(bsz);
            ASSERT_TRUE(buffer.ptr);
            ASSERT_EQ(stage(rocsparse_spsm_stage_preprocess), rocsparse_status_success);
            ASSERT_EQ(stage(rocsparse_spsm_stage_compute), rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        void check()
        {
            expect_dense_equal<T>(dC.host(), want, c.order_C, ldc, "C");
            // B is an input only.
            expect_dense_equal<T>(dB.host(), B_stored, c.order_B, ldb, "B");
        }
    };

    template <typename T, typename I, typename J>
    void run_spsm(rocsparse_handle handle, const SpsmCase& c)
    {
        SCOPED_TRACE(describe(c));
        Problem<T, I, J> p;
        ASSERT_NO_FATAL_FAILURE(p.init(handle, c));
        ASSERT_NO_FATAL_FAILURE(p.run_all());
        ASSERT_NO_FATAL_FAILURE(p.check());
    }

    template <typename T, typename I, typename J>
    void spsm_grid(rocsparse_handle                        handle,
                   const std::vector<Fmt>&                 fmts,
                   const std::vector<rocsparse_operation>& opAs,
                   const std::vector<rocsparse_operation>& opBs,
                   bool                                    device,
                   uint64_t                                seed)
    {
        const rocsparse_order orders[] = {rocsparse_order_column, rocsparse_order_row};
        for(Fmt f : fmts)
            for(rocsparse_operation opA : opAs)
                for(rocsparse_operation opB : opBs)
                    for(rocsparse_order oB : orders)
                        for(rocsparse_order oC : orders)
                            for(rocsparse_fill_mode fill :
                                {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                                for(rocsparse_diag_type diag :
                                    {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                                {
                                    SpsmCase c;
                                    c.fmt     = f;
                                    c.opA     = opA;
                                    c.opB     = opB;
                                    c.order_B = oB;
                                    c.order_C = oC;
                                    c.fill    = fill;
                                    c.diag    = diag;
                                    c.device  = device;
                                    c.alpha   = device ? cplx(-0.5, 0.0) : cplx(2.0, 0.0);
                                    c.seed    = seed++;
                                    run_spsm<T, I, J>(handle, c);
                                    if(::testing::Test::HasFatalFailure())
                                        return;
                                }
    }

    const std::vector<Fmt>                 all_fmts = {Fmt::csr, Fmt::csc, Fmt::coo};
    const std::vector<rocsparse_operation> real_ops
        = {rocsparse_operation_none, rocsparse_operation_transpose};
    const std::vector<rocsparse_operation> complex_ops = {rocsparse_operation_none,
                                                          rocsparse_operation_transpose,
                                                          rocsparse_operation_conjugate_transpose};

    struct SpsmCov70 : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(SpsmCov70, solve_float)
{
    spsm_grid<float, int32_t, int32_t>(handle, all_fmts, real_ops, real_ops, false, 1);
}

TEST_F(SpsmCov70, solve_double_device_alpha)
{
    spsm_grid<double, int32_t, int32_t>(handle, all_fmts, real_ops, real_ops, true, 1001);
}

TEST_F(SpsmCov70, solve_complex)
{
    spsm_grid<rocsparse_float_complex, int32_t, int32_t>(
        handle, all_fmts, complex_ops, complex_ops, false, 2001);
    spsm_grid<rocsparse_double_complex, int32_t, int32_t>(
        handle,
        {Fmt::csr, Fmt::coo},
        {rocsparse_operation_none, rocsparse_operation_conjugate_transpose},
        {rocsparse_operation_none, rocsparse_operation_conjugate_transpose},
        true,
        3001);
}

TEST_F(SpsmCov70, index_types)
{
    for(Fmt f : all_fmts)
        for(rocsparse_operation opA : real_ops)
        {
            SpsmCase c;
            c.fmt = f;
            c.opA = opA;
            run_spsm<float, int64_t, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
            run_spsm<double, int64_t, int64_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}

TEST_F(SpsmCov70, right_hand_side_counts_and_sizes)
{
    const int64_t nrhs_list[] = {1, 2, 3, 8, 31, 64};
    const int64_t m_list[]    = {1, 2, 17, 40};
    uint64_t      seed        = 5000;
    for(Fmt f : all_fmts)
        for(int64_t nrhs : nrhs_list)
            for(int64_t m : m_list)
            {
                SpsmCase c;
                c.fmt  = f;
                c.m    = m;
                c.nrhs = nrhs;
                c.seed = seed++;
                c.opB  = (nrhs % 2 == 0) ? rocsparse_operation_transpose : rocsparse_operation_none;
                c.order_C = (m % 2 == 0) ? rocsparse_order_row : rocsparse_order_column;
                c.order_B = (nrhs > 3) ? rocsparse_order_row : rocsparse_order_column;
                run_spsm<float, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(SpsmCov70, repeated_compute_reuses_the_preprocess)
{
    SpsmCase c;
    c.seed = 77;
    for(Fmt f : all_fmts)
    {
        c.fmt = f;
        SCOPED_TRACE(describe(c));
        Problem<double, int32_t, int32_t> p;
        ASSERT_NO_FATAL_FAILURE(p.init(handle, c));
        ASSERT_NO_FATAL_FAILURE(p.run_all());
        ASSERT_NO_FATAL_FAILURE(p.check());
        // A second preprocess is a no-op and a second compute gives the same result.
        ASSERT_EQ(p.stage(rocsparse_spsm_stage_preprocess), rocsparse_status_success);
        // Reset C (including the padding sentinel).
        const Dense C0(c.m, c.nrhs);
        ASSERT_EQ(hipMemcpy(p.dC.ptr,
                            store_dense<double>(C0, c.order_C, p.ldc).data(),
                            p.dC.n * sizeof(double),
                            hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(p.stage(rocsparse_spsm_stage_compute), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_NO_FATAL_FAILURE(p.check());
    }
}

TEST_F(SpsmCov70, argument_errors_and_unsupported_formats)
{
    using P = Problem<float, int32_t, int32_t>;
    SpsmCase c;
    c.order_B = rocsparse_order_row;
    c.order_C = rocsparse_order_row;
    P p;
    ASSERT_NO_FATAL_FAILURE(p.init(handle, c));
    ASSERT_EQ(p.stage(rocsparse_spsm_stage_buffer_size), rocsparse_status_success);
    p.buffer = DevBuf<char>(p.bsz);
    ASSERT_EQ(p.stage(rocsparse_spsm_stage_preprocess), rocsparse_status_success);

    const rocsparse_datatype dt    = rocsparse_datatype_f32_r;
    const float              alpha = 1.0f;
    size_t                   bsz   = p.bsz;
    const auto               N     = rocsparse_operation_none;
    auto                     call  = [&](rocsparse_handle            h,
                    rocsparse_operation         opA,
                    rocsparse_operation         opB,
                    const void*                 a,
                    rocsparse_const_spmat_descr A,
                    rocsparse_const_dnmat_descr B,
                    rocsparse_dnmat_descr       C,
                    rocsparse_datatype          type,
                    rocsparse_spsm_alg          alg,
                    rocsparse_spsm_stage        stage,
                    size_t*                     size) {
        return rocsparse_spsm(h, opA, opB, a, A, B, C, type, alg, stage, size, p.buffer.ptr);
    };
    const auto S  = rocsparse_spsm_stage_compute;
    const auto AL = rocsparse_spsm_alg_default;
    EXPECT_EQ(call(nullptr, N, N, &alpha, p.sA.p, p.mB.p, p.mC.p, dt, AL, S, &bsz),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(
        call(handle, (rocsparse_operation)99, N, &alpha, p.sA.p, p.mB.p, p.mC.p, dt, AL, S, &bsz),
        rocsparse_status_invalid_value);
    EXPECT_EQ(
        call(handle, N, (rocsparse_operation)99, &alpha, p.sA.p, p.mB.p, p.mC.p, dt, AL, S, &bsz),
        rocsparse_status_invalid_value);
    EXPECT_EQ(call(handle, N, N, nullptr, p.sA.p, p.mB.p, p.mC.p, dt, AL, S, &bsz),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle, N, N, &alpha, nullptr, p.mB.p, p.mC.p, dt, AL, S, &bsz),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle, N, N, &alpha, p.sA.p, nullptr, p.mC.p, dt, AL, S, &bsz),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle, N, N, &alpha, p.sA.p, p.mB.p, nullptr, dt, AL, S, &bsz),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        call(handle, N, N, &alpha, p.sA.p, p.mB.p, p.mC.p, (rocsparse_datatype)12345, AL, S, &bsz),
        rocsparse_status_invalid_value);
    // compute type different from the type of the matrices
    EXPECT_EQ(
        call(handle, N, N, &alpha, p.sA.p, p.mB.p, p.mC.p, rocsparse_datatype_f64_r, AL, S, &bsz),
        rocsparse_status_not_implemented);
    EXPECT_EQ(
        call(handle, N, N, &alpha, p.sA.p, p.mB.p, p.mC.p, dt, (rocsparse_spsm_alg)99, S, &bsz),
        rocsparse_status_invalid_value);
    EXPECT_EQ(
        call(handle, N, N, &alpha, p.sA.p, p.mB.p, p.mC.p, dt, AL, (rocsparse_spsm_stage)99, &bsz),
        rocsparse_status_invalid_value);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   &alpha,
                   p.sA.p,
                   p.mB.p,
                   p.mC.p,
                   dt,
                   AL,
                   rocsparse_spsm_stage_buffer_size,
                   nullptr),
              rocsparse_status_invalid_pointer);

    // The failed calls did not disturb the descriptors: a valid compute still works.
    ASSERT_EQ(p.stage(rocsparse_spsm_stage_compute), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_NO_FATAL_FAILURE(p.check());

    // Formats without an SpSM implementation.
    {
        std::vector<int64_t> rows;
        for(int64_t i = 0; i < p.A.m; ++i)
            for(int64_t q = p.A.ptr[static_cast<size_t>(i)];
                q < p.A.ptr[static_cast<size_t>(i) + 1];
                ++q)
            {
                rows.push_back(i);
                rows.push_back(p.A.col[static_cast<size_t>(q)]);
            }
        DevBuf<int32_t> aos = up_idx<int32_t>(rows);
        SpmatH          coo_aos;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&coo_aos.p,
                                                 p.A.m,
                                                 p.A.n,
                                                 p.A.nnz(),
                                                 aos.ptr,
                                                 p.val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 dt),
                  rocsparse_status_success);

        // ELL with the width of the longest row (columns / values are never read).
        const int64_t   width = 1;
        DevBuf<int32_t> ell_col(size_t(p.A.m * width));
        DevBuf<float>   ell_val(size_t(p.A.m * width));
        SpmatH          ell;
        ASSERT_EQ(rocsparse_create_ell_descr(&ell.p,
                                             p.A.m,
                                             p.A.n,
                                             ell_col.ptr,
                                             ell_val.ptr,
                                             width,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);

        for(rocsparse_spmat_descr unsupported : {coo_aos.p, ell.p})
            for(rocsparse_spsm_stage st : {rocsparse_spsm_stage_buffer_size,
                                           rocsparse_spsm_stage_preprocess,
                                           rocsparse_spsm_stage_compute})
                EXPECT_EQ(call(handle, N, N, &alpha, unsupported, p.mB.p, p.mC.p, dt, AL, st, &bsz),
                          rocsparse_status_not_implemented)
                    << "stage " << st;
    }
}
