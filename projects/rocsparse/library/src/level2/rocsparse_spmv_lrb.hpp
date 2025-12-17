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
    /*! \brief Computes buffer size required for LRB (Load-Balanced Reduction) algorithm
     *
     * \details
     * This function computes the buffer size required for the LRB SpMV algorithm.
     * LRB is optimized for matrices with skewed row length distributions and performs
     * load-balanced reduction across thread blocks.
     *
     * @param[in]
     * handle          rocsparse library context
     * @param[in]
     * trans           sparse matrix operation
     * @param[in]
     * m               number of rows in matrix
     * @param[in]
     * n               number of columns in matrix
     * @param[in]
     * nnz             number of non-zero entries in matrix
     * @param[in]
     * alpha_datatype  datatype of alpha scalar
     * @param[in]
     * alpha           scalar alpha
     * @param[in]
     * descr           matrix descriptor
     * @param[in]
     * csr_val_datatype datatype of matrix values
     * @param[in]
     * csr_val         matrix values
     * @param[in]
     * csr_row_ptr_indextype index type of row pointers
     * @param[in]
     * csr_row_ptr     CSR row pointers
     * @param[in]
     * csr_col_ind_indextype index type of column indices
     * @param[in]
     * csr_col_ind     CSR column indices
     * @param[in]
     * x_datatype      datatype of input vector x
     * @param[in]
     * x               input vector x
     * @param[in]
     * beta_datatype   datatype of beta scalar
     * @param[in]
     * beta            scalar beta
     * @param[in]
     * y_datatype      datatype of output vector y
     * @param[in]
     * y               output vector y
     * @param[in]
     * alg             sparse matrix algorithm (must be rocsparse_spmv_alg_csr_lrb)
     * @param[in]
     * stage           sparse matrix stage
     * @param[out]
     * buffer_size     buffer size required for computation
     * @param[in]
     * temp_buffer     temporary buffer (not used in this stage)
     *
     * \retval     rocsparse_status_success the operation completed successfully.
     * \retval     rocsparse_status_invalid_handle the library context was not initialized.
     */
    rocsparse_status spmv_buffer_size_lrb(rocsparse_handle             handle,
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

    /*! \brief LRB algorithm preprocessing stage
     *
     * \details
     * This function performs preprocessing for the LRB algorithm including
     * matrix analysis and kernel selection based on load characteristics.
     */
    rocsparse_status spmv_preprocess_lrb(rocsparse_handle             handle,
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

    /*! \brief LRB algorithm computation stage
     *
     * \details
     * This function performs the actual SpMV computation using the LRB algorithm.
     */
    rocsparse_status spmv_compute_lrb(rocsparse_handle             handle,
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
