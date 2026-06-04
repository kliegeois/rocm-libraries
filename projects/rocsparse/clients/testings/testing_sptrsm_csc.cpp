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

#include "testing.hpp"

//
// The SpTrSM path (rocsparse_sptrsm / rocsparse_sptrsm_buffer_size) only
// supports CSR and COO sparse matrices. A CSC matrix is rejected with
// rocsparse_status_not_implemented in the internal format dispatch (see
// library/src/level3/rocsparse_sptrsm.cpp). This routine provides positive
// coverage of that format guard. It deliberately stops at the buffer-size
// query of each stage, which is where CSC is rejected, and never allocates a
// buffer or runs analysis/compute, since the unsupported CSC path is not safe
// to execute any further.
//

template <typename I, typename J, typename T>
void testing_sptrsm_csc_bad_arg(const Arguments& arg)
{
    // CSC is an unsupported format for the SpTrSM path; the not_implemented
    // guard is exercised directly in testing_sptrsm_csc below.
}

template <typename I, typename J, typename T>
void testing_sptrsm_csc(const Arguments& arg)
{
    J                    M       = arg.M;
    J                    N       = arg.N;
    J                    K       = arg.K;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_operation  trans_B = arg.transB;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_fill_mode  uplo    = arg.uplo;
    rocsparse_diag_type  diag    = arg.diag;

    // Use a valid, square sparse matrix and consistent dense operands so that
    // the SpTrSM call passes its generic argument validation and reaches the
    // internal format dispatch, which is where CSC is rejected.
    if(M <= 0)
    {
        M = 8;
    }
    N = M;
    if(K <= 0)
    {
        K = 4;
    }

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    rocsparse_local_handle handle(arg);

    const I nnz = M;

    // Build a trivial diagonal matrix in CSC storage. The actual values are
    // irrelevant: the guard returns before they are ever read.
    const T halpha = arg.get_alpha<T>();

    host_vector<I> hcsc_col_ptr(N + 1);
    host_vector<J> hcsc_row_ind(nnz);
    host_vector<T> hcsc_val(nnz);
    for(J j = 0; j <= N; ++j)
    {
        hcsc_col_ptr[j] = j + base;
    }
    for(J i = 0; i < M; ++i)
    {
        hcsc_row_ind[i] = i + base;
        hcsc_val[i]     = halpha;
    }

    device_vector<I> dcsc_col_ptr(N + 1);
    device_vector<J> dcsc_row_ind(nnz);
    device_vector<T> dcsc_val(nnz);
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_col_ptr, hcsc_col_ptr.data(), sizeof(I) * (N + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_row_ind, hcsc_row_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsc_val, hcsc_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Dense matrices B (X) and C (Y), column-major, leading dimension M.
    const int64_t    ldb = M;
    const int64_t    ldc = M;
    device_vector<T> dB(int64_t(ldb) * K);
    device_vector<T> dC(int64_t(ldc) * K);

    // CSC sparse descriptor (csc_format = true).
    rocsparse_local_spmat A(
        M, N, nnz, dcsc_col_ptr, dcsc_row_ind, dcsc_val, itype, jtype, base, ttype, true);

    rocsparse_local_dnmat B(M, K, ldb, dB, ttype, rocsparse_order_column);
    rocsparse_local_dnmat C(M, K, ldc, dC, ttype, rocsparse_order_column);

    rocsparse_error* p_error = nullptr;

    // Make the sparse descriptor a fully valid triangular operand so the only
    // thing left to reject is the unsupported CSC format.
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    // Create and fully configure the SpTrSM descriptor so that argument
    // validation succeeds and the call reaches the format dispatch.
    rocsparse_sptrsm_descr sptrsm_descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsm_descr(&sptrsm_descr));

    {
        const rocsparse_sptrsm_alg alg = rocsparse_sptrsm_alg_default;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(
            handle, sptrsm_descr, rocsparse_sptrsm_input_alg, &alg, sizeof(alg), p_error));
    }
    {
        const rocsparse_operation operation_A = trans_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_operation_A,
                                                         &operation_A,
                                                         sizeof(operation_A),
                                                         p_error));
    }
    {
        const rocsparse_operation operation_X = trans_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_operation_X,
                                                         &operation_X,
                                                         sizeof(operation_X),
                                                         p_error));
    }
    {
        const rocsparse_datatype scalar_datatype = ttype;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_datatype,
                                                         &scalar_datatype,
                                                         sizeof(scalar_datatype),
                                                         p_error));
    }
    {
        const rocsparse_datatype compute_datatype = ttype;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_compute_datatype,
                                                         &compute_datatype,
                                                         sizeof(compute_datatype),
                                                         p_error));
    }
    {
        const rocsparse_analysis_policy apol = rocsparse_analysis_policy_reuse;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_analysis_policy,
                                                         &apol,
                                                         sizeof(apol),
                                                         p_error));
    }

    // Stage 1 (analysis): the buffer-size query must reject CSC. A non-null
    // buffer-size pointer is required by argument validation; no buffer is
    // allocated from the (unwritten) result.
    size_t buffer_size = 0;
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_sptrsm_buffer_size(
            handle, sptrsm_descr, A, B, C, rocsparse_sptrsm_stage_analysis, &buffer_size, p_error),
        rocsparse_status_not_implemented);

    // Stage 2 (compute): the buffer-size query must also reject CSC. The guard
    // returns before any buffer is dereferenced or analysis/compute is run.
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_sptrsm_buffer_size(
            handle, sptrsm_descr, A, B, C, rocsparse_sptrsm_stage_compute, &buffer_size, p_error),
        rocsparse_status_not_implemented);

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsm_descr(sptrsm_descr));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_sptrsm_csc_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsm_csc<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

void testing_sptrsm_csc_extra(const Arguments& arg) {}
