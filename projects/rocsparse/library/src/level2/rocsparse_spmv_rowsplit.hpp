/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    rocsparse_status spmv_buffer_size_rowsplit(rocsparse_handle             handle,
                                               rocsparse_operation          trans,
                                               int64_t                      m,
                                               int64_t                      n,
                                               int64_t                      nnz,
                                               rocsparse_datatype           alpha_datatype,
                                               const void*                  alpha,
                                               const rocsparse_mat_descr    descr,
                                               rocsparse_datatype           csr_val_datatype,
                                               const void*                  csr_val,
                                               rocsparse_indextype          csr_row_ptr_indextype,
                                               const void*                  csr_row_ptr,
                                               rocsparse_indextype          csr_col_ind_indextype,
                                               const void*                  csr_col_ind,
                                               rocsparse_datatype           x_datatype,
                                               const void*                  x,
                                               rocsparse_datatype           beta_datatype,
                                               const void*                  beta,
                                               rocsparse_datatype           y_datatype,
                                               const void*                  y,
                                               rocsparse_spmv_alg           alg,
                                               rocsparse_spmv_stage         stage,
                                               size_t*                      buffer_size,
                                               void*                        temp_buffer);

    rocsparse_status spmv_preprocess_rowsplit(rocsparse_handle             handle,
                                              rocsparse_operation          trans,
                                              int64_t                      m,
                                              int64_t                      n,
                                              int64_t                      nnz,
                                              rocsparse_datatype           alpha_datatype,
                                              const void*                  alpha,
                                              const rocsparse_mat_descr    descr,
                                              rocsparse_datatype           csr_val_datatype,
                                              const void*                  csr_val,
                                              rocsparse_indextype          csr_row_ptr_indextype,
                                              const void*                  csr_row_ptr,
                                              rocsparse_indextype          csr_col_ind_indextype,
                                              const void*                  csr_col_ind,
                                              rocsparse_datatype           x_datatype,
                                              const void*                  x,
                                              rocsparse_datatype           beta_datatype,
                                              const void*                  beta,
                                              rocsparse_datatype           y_datatype,
                                              const void*                  y,
                                              rocsparse_spmv_alg           alg,
                                              rocsparse_spmv_stage         stage,
                                              void*                        temp_buffer);

    rocsparse_status spmv_compute_rowsplit(rocsparse_handle             handle,
                                           rocsparse_operation          trans,
                                           int64_t                      m,
                                           int64_t                      n,
                                           int64_t                      nnz,
                                           rocsparse_datatype           alpha_datatype,
                                           const void*                  alpha,
                                           const rocsparse_mat_descr    descr,
                                           rocsparse_datatype           csr_val_datatype,
                                           const void*                  csr_val,
                                           rocsparse_indextype          csr_row_ptr_indextype,
                                           const void*                  csr_row_ptr,
                                           rocsparse_indextype          csr_col_ind_indextype,
                                           const void*                  csr_col_ind,
                                           rocsparse_datatype           x_datatype,
                                           const void*                  x,
                                           rocsparse_datatype           beta_datatype,
                                           const void*                  beta,
                                           rocsparse_datatype           y_datatype,
                                           void*                        y,
                                           rocsparse_spmv_alg           alg,
                                           rocsparse_spmv_stage         stage,
                                           void*                        temp_buffer);
}
