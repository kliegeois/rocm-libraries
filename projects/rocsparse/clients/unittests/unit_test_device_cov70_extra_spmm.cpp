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
// Device unit tests for the sparse-times-dense multiplication paths:
//   library/src/level3/rocsparse_spmm.cpp          (CSR / CSC / COO dispatch)
//   library/src/level3/rocsparse_cscmm_impl.cpp    (CSC -> CSR operation flip)
//   library/src/level3/rocsparse_csrmm_impl.cpp    (algorithm / transpose dispatch,
//                                                   quick return, argument checks)
//   library/src/level3/rocsparse_csrmm_template_row_split.cpp
//                                                  (row-split kernels for every
//                                                   dense column count)
//
// All tests go through the public API (rocsparse_spmm, rocsparse_?csrmm). The
// complete C matrix is compared with a host reference computed in double
// complex precision, including the untouched leading-dimension padding.
//
#include "unit_test_cov70_extra_common.hpp"

#include <limits>

using namespace cov70;

namespace
{
    enum class SFmt
    {
        csr,
        csc,
        coo
    };

    struct SpmmCase
    {
        SFmt                 fmt        = SFmt::csr;
        rocsparse_spmm_alg   alg        = rocsparse_spmm_alg_default;
        rocsparse_operation  opA        = rocsparse_operation_none;
        rocsparse_operation  opB        = rocsparse_operation_none;
        rocsparse_order      order_B    = rocsparse_order_column;
        rocsparse_order      order_C    = rocsparse_order_column;
        int64_t              m          = 9; // rows of the stored A
        int64_t              k          = 7; // columns of the stored A
        int64_t              n          = 5; // columns of op(B) / C
        int64_t              max_nnz    = 4; // max non-zeros per row
        bool                 device     = false;
        cplx                 alpha      = cplx(2.0, 0.0);
        cplx                 beta       = cplx(-1.0, 0.0);
        bool                 preprocess = true;
        bool                 long_rows  = false; // one row touching all columns, one empty row
        int64_t              ld_pad     = 3;
        rocsparse_index_base base       = rocsparse_index_base_zero;
        // C holds NaN on input; only meaningful with beta == 0, where C must not be read.
        bool nan_C = false;
    };

    std::string describe(const SpmmCase& c)
    {
        return "fmt=" + std::to_string(static_cast<int>(c.fmt))
               + " alg=" + std::to_string(static_cast<int>(c.alg))
               + " opA=" + std::to_string(static_cast<int>(c.opA))
               + " opB=" + std::to_string(static_cast<int>(c.opB))
               + " oB=" + std::to_string(static_cast<int>(c.order_B))
               + " oC=" + std::to_string(static_cast<int>(c.order_C)) + " m=" + std::to_string(c.m)
               + " k=" + std::to_string(c.k) + " n=" + std::to_string(c.n) + " dev="
               + std::to_string(c.device) + " base=" + std::to_string(static_cast<int>(c.base))
               + " nan_C=" + std::to_string(c.nan_C);
    }

    // Device storage + descriptor of the sparse operand.
    template <typename T, typename I, typename J>
    struct DevSparse
    {
        DevBuf<I> ptr; // CSR row ptr / CSC col ptr / COO row ind
        DevBuf<J> ind; // CSR col ind / CSC row ind
        DevBuf<I> coo_col;
        DevBuf<T> val;
        SpmatH    h;
    };

    template <typename T, typename I, typename J>
    void make_sparse(DevSparse<T, I, J>&  d,
                     SFmt                 fmt,
                     const Csr&           A,
                     rocsparse_index_base base = rocsparse_index_base_zero)
    {
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        const int64_t            b  = static_cast<int64_t>(base);
        switch(fmt)
        {
        case SFmt::csr:
        {
            make_csr_descr<T, I, J>(d.h, A, d.ptr, d.ind, d.val, base);
            return;
        }
        case SFmt::csc:
        {
            const Csr At = transpose(A);
            d.ptr        = up_idx<I>(At.ptr, b);
            d.ind        = up_idx<J>(At.col, b);
            d.val        = up_val<T>(At.val);
            ASSERT_TRUE(d.ptr.ptr && d.ind.ptr && d.val.ptr);
            ASSERT_EQ(rocsparse_create_csc_descr(&d.h.p,
                                                 A.m,
                                                 A.n,
                                                 A.nnz(),
                                                 d.ptr.ptr,
                                                 d.ind.ptr,
                                                 d.val.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 base,
                                                 dt),
                      rocsparse_status_success);
            return;
        }
        case SFmt::coo:
        {
            std::vector<int64_t> rows;
            for(int64_t i = 0; i < A.m; ++i)
                for(int64_t p = A.ptr[static_cast<size_t>(i)];
                    p < A.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                    rows.push_back(i);
            d.ptr     = up_idx<I>(rows, b);
            d.coo_col = up_idx<I>(A.col, b);
            d.val     = up_val<T>(A.val);
            ASSERT_TRUE(d.ptr.ptr && d.coo_col.ptr && d.val.ptr);
            ASSERT_EQ(rocsparse_create_coo_descr(&d.h.p,
                                                 A.m,
                                                 A.n,
                                                 A.nnz(),
                                                 d.ptr.ptr,
                                                 d.coo_col.ptr,
                                                 d.val.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 base,
                                                 dt),
                      rocsparse_status_success);
            return;
        }
        }
    }

    // Runs the three stages of rocsparse_spmm and compares the whole C.
    template <typename T, typename I, typename J>
    void run_spmm(rocsparse_handle handle, const SpmmCase& c)
    {
        SCOPED_TRACE(describe(c));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();

        Rng rng(static_cast<uint64_t>(c.m * 131 + c.k * 17 + c.n) + 5);

        // Host operands.
        std::vector<int64_t> len(static_cast<size_t>(c.m));
        for(auto& l : len)
            l = rng.range(0, c.max_nnz);
        if(c.long_rows && c.m > 2)
        {
            len[1] = c.k;
            len[2] = 0;
        }
        const Csr A = csr_with_row_lengths<T>(c.m, c.k, len, rng);

        const bool    nonT = (c.opA == rocsparse_operation_none);
        const int64_t M    = nonT ? c.m : c.k; // rows of op(A) and of C
        const int64_t K    = nonT ? c.k : c.m; // columns of op(A), rows of op(B)

        const Dense opB_logical = random_dense<T>(K, c.n, rng);
        // apply_op is an involution, so applying it gives the stored operand.
        const Dense B_stored = apply_op(opB_logical, c.opB);
        Dense       C0       = random_dense<T>(M, c.n, rng);
        if(c.nan_C)
        {
            // beta == 0: the reference does not read C.
            ASSERT_EQ(c.beta, cplx(0.0, 0.0));
        }
        const Dense C_ref = c.nan_C ? Dense(M, c.n) : C0;
        if(c.nan_C)
            for(auto& x : C0.v)
                x = cplx(std::numeric_limits<double>::quiet_NaN(),
                         is_complex_t<T>::value ? std::numeric_limits<double>::quiet_NaN() : 0.0);

        const int64_t ldb = min_ld(B_stored, c.order_B) + c.ld_pad;
        const int64_t ldc = min_ld(C0, c.order_C) + c.ld_pad;

        DevBuf<T> dB = up_raw<T>(store_dense<T>(B_stored, c.order_B, ldb));
        DevBuf<T> dC = up_raw<T>(store_dense<T>(C0, c.order_C, ldc));
        ASSERT_TRUE(dB.ptr && dC.ptr);

        DevSparse<T, I, J> sp;
        make_sparse<T, I, J>(sp, c.fmt, A, c.base);
        if(::testing::Test::HasFatalFailure())
            return;

        DnmatH mB, mC;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &mB.p, B_stored.rows, B_stored.cols, ldb, dB.ptr, dt, c.order_B),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mC.p, M, c.n, ldc, dC.ptr, dt, c.order_C),
                  rocsparse_status_success);

        Scalar<T>        alpha(c.alpha, c.device);
        Scalar<T>        beta(c.beta, c.device);
        PointerModeGuard mode(handle, mode_of(c.device));

        size_t bsz = 0;
        ASSERT_EQ(rocsparse_spmm(handle,
                                 c.opA,
                                 c.opB,
                                 alpha.get(),
                                 sp.h.p,
                                 mB.p,
                                 beta.get(),
                                 mC.p,
                                 dt,
                                 c.alg,
                                 rocsparse_spmm_stage_buffer_size,
                                 &bsz,
                                 nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        if(c.preprocess)
        {
            ASSERT_EQ(rocsparse_spmm(handle,
                                     c.opA,
                                     c.opB,
                                     alpha.get(),
                                     sp.h.p,
                                     mB.p,
                                     beta.get(),
                                     mC.p,
                                     dt,
                                     c.alg,
                                     rocsparse_spmm_stage_preprocess,
                                     &bsz,
                                     buffer.ptr),
                      rocsparse_status_success);
        }
        ASSERT_EQ(rocsparse_spmm(handle,
                                 c.opA,
                                 c.opB,
                                 alpha.get(),
                                 sp.h.p,
                                 mB.p,
                                 beta.get(),
                                 mC.p,
                                 dt,
                                 c.alg,
                                 rocsparse_spmm_stage_compute,
                                 &bsz,
                                 buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Dense want = axpby(to_c<T>(alpha.h),
                                 matmul(apply_op(dense_of(A), c.opA), opB_logical),
                                 to_c<T>(beta.h),
                                 C_ref);
        expect_dense_equal<T>(dC.host(), want, c.order_C, ldc, "C");
        // B is an input only.
        expect_dense_equal<T>(dB.host(), B_stored, c.order_B, ldb, "B");
    }

    const rocsparse_operation all_ops[]    = {rocsparse_operation_none,
                                           rocsparse_operation_transpose,
                                           rocsparse_operation_conjugate_transpose};
    const rocsparse_order     all_orders[] = {rocsparse_order_column, rocsparse_order_row};

    const rocsparse_spmm_alg csr_algs[] = {rocsparse_spmm_alg_default,
                                           rocsparse_spmm_alg_csr,
                                           rocsparse_spmm_alg_csr_row_split,
                                           rocsparse_spmm_alg_csr_nnz_split,
                                           rocsparse_spmm_alg_csr_merge_path};

    // CSR or CSC with a grid of algorithms, operations, orders.
    template <typename T, typename I, typename J>
    void grid(rocsparse_handle                        handle,
              SFmt                                    fmt,
              const std::vector<rocsparse_spmm_alg>&  algs,
              const std::vector<rocsparse_operation>& opAs,
              const std::vector<rocsparse_operation>& opBs,
              bool                                    both_orders,
              bool                                    device)
    {
        for(rocsparse_spmm_alg alg : algs)
            for(rocsparse_operation opA : opAs)
                for(rocsparse_operation opB : opBs)
                    for(rocsparse_order oB : all_orders)
                        for(rocsparse_order oC : all_orders)
                        {
                            if(!both_orders && oB != oC)
                                continue;
                            SpmmCase c;
                            c.fmt       = fmt;
                            c.alg       = alg;
                            c.opA       = opA;
                            c.opB       = opB;
                            c.order_B   = oB;
                            c.order_C   = oC;
                            c.device    = device;
                            c.long_rows = true;
                            run_spmm<T, I, J>(handle, c);
                            if(::testing::Test::HasFatalFailure())
                                return;
                        }
    }

    const std::vector<rocsparse_spmm_alg>  csr_alg_list(std::begin(csr_algs), std::end(csr_algs));
    const std::vector<rocsparse_operation> op_list(std::begin(all_ops), std::end(all_ops));

    struct SpmmCov70 : public rocsparse_ut::HandleTest
    {
    };
}

// ---------------------------------------------------------------------------
// CSR: every algorithm, A / B operation and storage order.
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, csr_grid_float)
{
    grid<float, int32_t, int32_t>(handle, SFmt::csr, csr_alg_list, op_list, op_list, true, false);
}

TEST_F(SpmmCov70, csr_grid_double_device_scalars)
{
    grid<double, int32_t, int32_t>(handle,
                                   SFmt::csr,
                                   csr_alg_list,
                                   op_list,
                                   {rocsparse_operation_none, rocsparse_operation_transpose},
                                   false,
                                   true);
}

TEST_F(SpmmCov70, csr_grid_complex)
{
    const std::vector<rocsparse_spmm_alg> algs = {rocsparse_spmm_alg_default,
                                                  rocsparse_spmm_alg_csr_row_split,
                                                  rocsparse_spmm_alg_csr_merge_path};
    grid<rocsparse_float_complex, int32_t, int32_t>(
        handle, SFmt::csr, algs, op_list, op_list, false, false);
    grid<rocsparse_double_complex, int32_t, int32_t>(
        handle, SFmt::csr, algs, op_list, op_list, false, true);
}

// ---------------------------------------------------------------------------
// CSC: goes through rocsparse_cscmm_impl.cpp (operation flip).
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, csc_grid_float)
{
    grid<float, int32_t, int32_t>(handle, SFmt::csc, csr_alg_list, op_list, op_list, true, false);
}

TEST_F(SpmmCov70, csc_grid_double_device_scalars)
{
    grid<double, int32_t, int32_t>(handle,
                                   SFmt::csc,
                                   csr_alg_list,
                                   op_list,
                                   {rocsparse_operation_none, rocsparse_operation_transpose},
                                   false,
                                   true);
}

TEST_F(SpmmCov70, csc_grid_complex)
{
    const std::vector<rocsparse_spmm_alg> algs = {rocsparse_spmm_alg_default,
                                                  rocsparse_spmm_alg_csr_row_split,
                                                  rocsparse_spmm_alg_csr_nnz_split};
    grid<rocsparse_float_complex, int32_t, int32_t>(
        handle, SFmt::csc, algs, op_list, op_list, false, false);
    grid<rocsparse_double_complex, int32_t, int32_t>(
        handle, SFmt::csc, algs, op_list, op_list, false, true);
}

TEST_F(SpmmCov70, index_types)
{
    const std::vector<rocsparse_spmm_alg> algs
        = {rocsparse_spmm_alg_default, rocsparse_spmm_alg_csr_row_split};
    const std::vector<rocsparse_operation> ops
        = {rocsparse_operation_none, rocsparse_operation_transpose};
    for(SFmt f : {SFmt::csr, SFmt::csc})
    {
        grid<float, int64_t, int32_t>(
            handle, f, algs, ops, {rocsparse_operation_none}, false, false);
        grid<double, int64_t, int64_t>(
            handle, f, algs, ops, {rocsparse_operation_none}, false, false);
        grid<double, int32_t, int32_t>(
            handle, f, algs, ops, {rocsparse_operation_none}, false, false);
        grid<rocsparse_float_complex, int64_t, int64_t>(
            handle, f, algs, ops, {rocsparse_operation_none}, false, false);
    }
}

TEST_F(SpmmCov70, coo_algorithms)
{
    const std::vector<rocsparse_spmm_alg> algs = {rocsparse_spmm_alg_default,
                                                  rocsparse_spmm_alg_coo_segmented,
                                                  rocsparse_spmm_alg_coo_atomic,
                                                  rocsparse_spmm_alg_coo_segmented_atomic};
    grid<float, int32_t, int32_t>(handle,
                                  SFmt::coo,
                                  algs,
                                  {rocsparse_operation_none, rocsparse_operation_transpose},
                                  {rocsparse_operation_none, rocsparse_operation_transpose},
                                  false,
                                  false);
}

// ---------------------------------------------------------------------------
// Scalars: alpha / beta combinations (host and device), including the
// alpha == 0 && beta == 1 host quick return.
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, scalar_combinations)
{
    const cplx alphas[] = {cplx(0.0, 0.0), cplx(1.0, 0.0), cplx(-2.0, 0.0), cplx(0.5, 0.0)};
    const cplx betas[]  = {cplx(0.0, 0.0), cplx(1.0, 0.0), cplx(3.0, 0.0)};
    for(SFmt f : {SFmt::csr, SFmt::csc})
        for(bool device : {false, true})
            for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
                for(const cplx& a : alphas)
                    for(const cplx& b : betas)
                    {
                        SpmmCase c;
                        c.fmt    = f;
                        c.opA    = opA;
                        c.alpha  = a;
                        c.beta   = b;
                        c.device = device;
                        run_spmm<float, int32_t, int32_t>(handle, c);
                        if(HasFatalFailure())
                            return;
                    }
}

// ---------------------------------------------------------------------------
// beta == 0: C is not read, NaN values in C must not reach the result (for
// every format, algorithm and kernel family).
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, beta_zero_ignores_nan_in_C)
{
    struct FmtAlgs
    {
        SFmt                            fmt;
        std::vector<rocsparse_spmm_alg> algs;
    };
    const std::vector<FmtAlgs> sets
        = {{SFmt::csr, csr_alg_list},
           {SFmt::csc, {rocsparse_spmm_alg_default, rocsparse_spmm_alg_csr_row_split}},
           {SFmt::coo,
            {rocsparse_spmm_alg_default,
             rocsparse_spmm_alg_coo_segmented,
             rocsparse_spmm_alg_coo_atomic,
             rocsparse_spmm_alg_coo_segmented_atomic}}};
    for(const FmtAlgs& fa : sets)
        for(rocsparse_spmm_alg alg : fa.algs)
            for(bool device : {false, true})
                for(rocsparse_operation opA :
                    {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    SpmmCase c;
                    c.fmt       = fa.fmt;
                    c.alg       = alg;
                    c.opA       = opA;
                    c.device    = device;
                    c.beta      = cplx(0.0, 0.0);
                    c.nan_C     = true;
                    c.long_rows = true;
                    run_spmm<float, int32_t, int32_t>(handle, c);
                    if(HasFatalFailure())
                        return;
                    run_spmm<rocsparse_double_complex, int32_t, int32_t>(handle, c);
                    if(HasFatalFailure())
                        return;
                }
}

// ---------------------------------------------------------------------------
// Index base one.
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, index_base_one)
{
    for(SFmt f : {SFmt::csr, SFmt::csc, SFmt::coo})
        for(bool device : {false, true})
            for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                SpmmCase c;
                c.fmt       = f;
                c.opA       = opA;
                c.device    = device;
                c.base      = rocsparse_index_base_one;
                c.long_rows = true;
                run_spmm<float, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                run_spmm<double, int64_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                c.alg = (f == SFmt::coo) ? rocsparse_spmm_alg_coo_atomic
                                         : rocsparse_spmm_alg_csr_row_split;
                run_spmm<rocsparse_float_complex, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

// ---------------------------------------------------------------------------
// The preprocess stage is optional; a second preprocess / re-compute reuses the
// cached row profile.
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, compute_without_preprocess)
{
    for(SFmt f : {SFmt::csr, SFmt::csc})
        for(rocsparse_spmm_alg alg :
            {rocsparse_spmm_alg_default, rocsparse_spmm_alg_csr, rocsparse_spmm_alg_csr_row_split})
            for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                SpmmCase c;
                c.fmt        = f;
                c.alg        = alg;
                c.opA        = opA;
                c.preprocess = false;
                c.long_rows  = true;
                run_spmm<float, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(SpmmCov70, repeated_preprocess_and_compute)
{
    const rocsparse_datatype dt = rocsparse_datatype_f32_r;
    Rng                      rng(11);
    const Csr                A  = random_csr<float>(12, 10, 6, rng);
    const Dense              B  = random_dense<float>(10, 4, rng);
    const Dense              C0 = random_dense<float>(12, 4, rng);

    for(SFmt f : {SFmt::csr, SFmt::csc})
    {
        DevSparse<float, int32_t, int32_t> sp;
        make_sparse<float, int32_t, int32_t>(sp, f, A);
        ASSERT_FALSE(HasFatalFailure());
        DevBuf<float> dB = up_raw<float>(store_dense<float>(B, rocsparse_order_column, 10));
        DevBuf<float> dC = up_raw<float>(store_dense<float>(C0, rocsparse_order_column, 12));
        DnmatH        mB, mC;
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mB.p, 10, 4, 10, dB.ptr, dt, rocsparse_order_column),
            rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mC.p, 12, 4, 12, dC.ptr, dt, rocsparse_order_column),
            rocsparse_status_success);
        const float alpha = 1.f, beta = 0.f;
        size_t      bsz  = 0;
        auto        call = [&](rocsparse_spmm_stage stage, void* buf) {
            return rocsparse_spmm(handle,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  &alpha,
                                  sp.h.p,
                                  mB.p,
                                  &beta,
                                  mC.p,
                                  dt,
                                  rocsparse_spmm_alg_default,
                                  stage,
                                  &bsz,
                                  buf);
        };
        ASSERT_EQ(call(rocsparse_spmm_stage_buffer_size, nullptr), rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        for(int rep = 0; rep < 2; ++rep)
        {
            ASSERT_EQ(call(rocsparse_spmm_stage_preprocess, buffer.ptr), rocsparse_status_success);
            ASSERT_EQ(call(rocsparse_spmm_stage_compute, buffer.ptr), rocsparse_status_success);
            ASSERT_EQ(call(rocsparse_spmm_stage_compute, buffer.ptr), rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const Dense want = matmul(dense_of(A), B);
        expect_dense_equal<float>(dC.host(), want, rocsparse_order_column, 12, "C");
    }
}

// ---------------------------------------------------------------------------
// Degenerate shapes take the quick return in csrmm (C is only scaled by beta).
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, degenerate_shapes)
{
    struct Shape
    {
        int64_t m, k, n;
    };
    const Shape shapes[] = {{0, 5, 3}, {6, 0, 3}, {6, 5, 0}, {0, 0, 2}, {1, 1, 1}};
    for(SFmt f : {SFmt::csr, SFmt::csc})
        for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
            for(const Shape& s : shapes)
            {
                SpmmCase c;
                c.fmt = f;
                c.opA = opA;
                c.m   = s.m;
                c.k   = s.k;
                c.n   = s.n;
                run_spmm<float, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                c.device = true;
                run_spmm<double, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

// ---------------------------------------------------------------------------
// Row-split csrmm for every dense column count n: the kernel selection chain
// in csrmm_template_row_split.cpp depends on n (loop width, remainder, ...).
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, row_split_column_sweep)
{
    std::vector<int64_t> ns;
    for(int64_t n = 1; n <= 72; ++n)
        ns.push_back(n);
    for(int64_t n : {80, 96, 100, 127, 128, 129})
        ns.push_back(n);
    for(int64_t n : ns)
        for(rocsparse_order oB : all_orders)
        {
            SpmmCase c;
            c.alg     = rocsparse_spmm_alg_csr_row_split;
            c.n       = n;
            c.m       = 6;
            c.k       = 5;
            c.order_B = oB;
            c.order_C = oB;
            c.ld_pad  = 1;
            run_spmm<float, int32_t, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}

TEST_F(SpmmCov70, row_split_column_sweep_transposed_and_types)
{
    for(int64_t n : {1, 7, 8, 9, 17, 33, 64, 65})
        for(rocsparse_operation opB : all_ops)
        {
            SpmmCase c;
            c.alg = rocsparse_spmm_alg_csr_row_split;
            c.n   = n;
            c.m   = 6;
            c.k   = 5;
            c.opB = opB;
            run_spmm<double, int32_t, int32_t>(handle, c);
            run_spmm<rocsparse_float_complex, int32_t, int32_t>(handle, c);
            c.opA     = rocsparse_operation_transpose;
            c.order_B = rocsparse_order_row;
            c.order_C = rocsparse_order_row;
            run_spmm<rocsparse_double_complex, int32_t, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}

// ---------------------------------------------------------------------------
// Legacy csrmm entry point (column major): numerical check for the operation
// combinations plus the argument checks of csrmm_impl.cpp.
// ---------------------------------------------------------------------------
TEST_F(SpmmCov70, legacy_csrmm_values_and_errors)
{
    Rng                   rng(21);
    const int64_t         m = 8, k = 6, n = 4;
    const Csr             A = random_csr<float>(m, k, 4, rng);
    MatDescrH             descr;
    DevBuf<rocsparse_int> dptr = up_idx<rocsparse_int>(A.ptr);
    DevBuf<rocsparse_int> dcol = up_idx<rocsparse_int>(A.col);
    DevBuf<float>         dval = up_val<float>(A.val);
    // The same matrix with index base one.
    MatDescrH             descr1(rocsparse_index_base_one);
    DevBuf<rocsparse_int> dptr1 = up_idx<rocsparse_int>(A.ptr, 1);
    DevBuf<rocsparse_int> dcol1 = up_idx<rocsparse_int>(A.col, 1);

    for(int one_based = 0; one_based < 2; ++one_based)
        for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
            for(rocsparse_operation opB : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                const int64_t M    = (opA == rocsparse_operation_none) ? m : k;
                const int64_t K    = (opA == rocsparse_operation_none) ? k : m;
                const Dense   opBl = random_dense<float>(K, n, rng);
                const Dense   Bs   = apply_op(opBl, opB);
                const Dense   C0   = random_dense<float>(M, n, rng);
                const int64_t ldb  = min_ld(Bs, rocsparse_order_column) + 2;
                const int64_t ldc  = min_ld(C0, rocsparse_order_column) + 2;
                DevBuf<float> dB
                    = up_raw<float>(store_dense<float>(Bs, rocsparse_order_column, ldb));
                DevBuf<float> dC
                    = up_raw<float>(store_dense<float>(C0, rocsparse_order_column, ldc));
                const float alpha = 1.5f, beta = 0.5f;
                ASSERT_EQ(rocsparse_scsrmm(handle,
                                           opA,
                                           opB,
                                           static_cast<rocsparse_int>(m),
                                           static_cast<rocsparse_int>(n),
                                           static_cast<rocsparse_int>(k),
                                           static_cast<rocsparse_int>(A.nnz()),
                                           &alpha,
                                           one_based ? descr1.d : descr.d,
                                           dval.ptr,
                                           one_based ? dptr1.ptr : dptr.ptr,
                                           one_based ? dcol1.ptr : dcol.ptr,
                                           dB.ptr,
                                           static_cast<rocsparse_int>(ldb),
                                           &beta,
                                           dC.ptr,
                                           static_cast<rocsparse_int>(ldc)),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                const Dense want = axpby(
                    cplx(alpha, 0), matmul(apply_op(dense_of(A), opA), opBl), cplx(beta, 0), C0);
                expect_dense_equal<float>(dC.host(), want, rocsparse_order_column, ldc, "C");
            }

    // argument checks
    DevBuf<float> dB(size_t{256}), dC(size_t{256});
    const float   one = 1.f, zero = 0.f;
    auto          call = [&](rocsparse_handle     h,
                    rocsparse_operation  opA,
                    rocsparse_operation  opB,
                    rocsparse_int        mm,
                    rocsparse_int        nn,
                    rocsparse_int        kk,
                    rocsparse_int        nnz,
                    const float*         a,
                    const float*         val,
                    const rocsparse_int* ptr,
                    const rocsparse_int* col,
                    const float*         B,
                    rocsparse_int        ldb,
                    const float*         b,
                    float*               C,
                    rocsparse_int        ldc) {
        return rocsparse_scsrmm(
            h, opA, opB, mm, nn, kk, nnz, a, descr.d, val, ptr, col, B, ldb, b, C, ldc);
    };
    const auto          N   = rocsparse_operation_none;
    const auto          T   = rocsparse_operation_transpose;
    const rocsparse_int nnz = static_cast<rocsparse_int>(A.nnz());
    EXPECT_EQ(call(nullptr,
                   N,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   nullptr,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   nullptr,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   nullptr,
                   8,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   &zero,
                   nullptr,
                   8),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   -1,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   -1,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   4,
                   -1,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_size);
    // ldb / ldc too small for each operation combination
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   5,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call(handle,
                   N,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   6,
                   &zero,
                   dC.ptr,
                   7),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call(handle,
                   N,
                   T,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   3,
                   &zero,
                   dC.ptr,
                   8),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call(handle,
                   T,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   7,
                   &zero,
                   dC.ptr,
                   6),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call(handle,
                   T,
                   N,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   8,
                   &zero,
                   dC.ptr,
                   5),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call(handle,
                   T,
                   T,
                   8,
                   4,
                   6,
                   nnz,
                   &one,
                   dval.ptr,
                   dptr.ptr,
                   dcol.ptr,
                   dB.ptr,
                   3,
                   &zero,
                   dC.ptr,
                   6),
              rocsparse_status_invalid_size);
}
