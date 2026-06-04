/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
// rocsparse_spsm() supports only CSR and COO sparse matrices (see the public
// header: "The sparse matrix formats currently supported are:
// rocsparse_format_coo and rocsparse_format_csr."). Passing a CSC matrix must
// therefore be rejected with rocsparse_status_not_implemented. This routine
// provides positive coverage of that format guard. It deliberately stops at the
// first rejecting stage of each call and never allocates a buffer or runs the
// solve, since the unsupported CSC path is not safe to execute any further.
//

template <typename I, typename J, typename T>
void testing_spsm_csc_bad_arg(const Arguments& arg)
{
    // CSC is an unsupported format for rocsparse_spsm(); the not_implemented
    // guard is exercised directly in testing_spsm_csc below.
}

template <typename I, typename J, typename T>
void testing_spsm_csc(const Arguments& arg)
{
    J                   M       = arg.M;
    J                   N       = arg.N;
    J                   K       = arg.K;
    rocsparse_operation trans_A = arg.transA;
    rocsparse_operation trans_B = arg.transB;
    rocsparse_index_base base   = arg.baseA;
    rocsparse_spsm_alg   alg    = arg.spsm_alg;

    // Use a valid, square sparse matrix and consistent dense operands so that
    // rocsparse_spsm() passes its generic argument validation and reaches the
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

    // Dense matrices B and C (column-major, leading dimension M).
    const int64_t    ldb = M;
    const int64_t    ldc = M;
    device_vector<T> dB(int64_t(ldb) * K);
    device_vector<T> dC(int64_t(ldc) * K);

    // CSC sparse descriptor (csc_format = true).
    rocsparse_local_spmat A(
        M, N, nnz, dcsc_col_ptr, dcsc_row_ind, dcsc_val, itype, jtype, base, ttype, true);

    rocsparse_local_dnmat B(M, K, ldb, dB, ttype, rocsparse_order_column);
    rocsparse_local_dnmat C(M, K, ldc, dC, ttype, rocsparse_order_column);

    // Stage 1: buffer_size must reject CSC. A non-null buffer_size pointer is
    // required by argument validation; no buffer is allocated from it.
    size_t buffer_size;
    EXPECT_ROCSPARSE_STATUS(rocsparse_spsm(handle,
                                           trans_A,
                                           trans_B,
                                           &halpha,
                                           A,
                                           B,
                                           C,
                                           ttype,
                                           alg,
                                           rocsparse_spsm_stage_buffer_size,
                                           &buffer_size,
                                           nullptr),
                            rocsparse_status_not_implemented);

    // Stage 2: preprocess must also reject CSC. The guard returns before the
    // (null) temp_buffer is ever dereferenced.
    EXPECT_ROCSPARSE_STATUS(rocsparse_spsm(handle,
                                           trans_A,
                                           trans_B,
                                           &halpha,
                                           A,
                                           B,
                                           C,
                                           ttype,
                                           alg,
                                           rocsparse_spsm_stage_preprocess,
                                           nullptr,
                                           nullptr),
                            rocsparse_status_not_implemented);
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                   template void testing_spsm_csc_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg);     template void testing_spsm_csc<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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

void testing_spsm_csc_extra(const Arguments& arg) {}
