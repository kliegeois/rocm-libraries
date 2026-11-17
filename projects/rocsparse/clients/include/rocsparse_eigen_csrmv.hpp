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

#pragma once

#include "rocsparse.h"

// Matrices larger than this skip the Eigen check. The quick and pre_checkin
// csrmv cases, including the SuiteSparse files used there, sit under it.
static constexpr rocsparse_int rocsparse_eigen_csrmv_nnz_limit = 5000000;

/*! \brief y_out = alpha * op(A) * x + beta * y_in, using Eigen as the reference.
 *
 *  \p A is the stored CSR matrix. General, Hermitian, and triangular descriptors
 *  use those stored entries only, which matches what csrmv computes for them.
 *  Symmetric csrmv also mirrors the off-diagonal entries, so this returns false
 *  and leaves \p y_out untouched. It also returns false above
 *  \p rocsparse_eigen_csrmv_nnz_limit.
 */
template <typename T>
bool rocsparse_eigen_csrmv(rocsparse_operation   trans,
                           rocsparse_int         m,
                           rocsparse_int         n,
                           rocsparse_int         nnz,
                           T                     alpha,
                           const rocsparse_int*  csr_row_ptr,
                           const rocsparse_int*  csr_col_ind,
                           const T*              csr_val,
                           const T*              x,
                           T                     beta,
                           const T*              y_in,
                           T*                    y_out,
                           rocsparse_index_base  base,
                           rocsparse_matrix_type matrix_type);
