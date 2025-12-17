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

#include "rocsparse_spmv_dispatch.hpp"
#include "rocsparse_spmv_rowsplit.hpp"
#include "rocsparse_spmv_nnzsplit.hpp"
#include "rocsparse_spmv_adaptive.hpp"
#include "rocsparse_spmv_lrb.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::spmv_dispatch(rocsparse_handle             handle,
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
                                          size_t*                      buffer_size,
                                          void*                        temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Route to algorithm-specific dispatcher
    switch(alg)
    {
    case rocsparse_spmv_alg_csr_rowsplit:
    {
        if(stage == rocsparse_spmv_stage_buffer_size)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_buffer_size_rowsplit(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, buffer_size,
                temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_preprocess)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_preprocess_rowsplit(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_compute)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_compute_rowsplit(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        break;
    }
    case rocsparse_spmv_alg_csr_nnzsplit:
    {
        if(stage == rocsparse_spmv_stage_buffer_size)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_buffer_size_nnzsplit(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, buffer_size,
                temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_preprocess)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_preprocess_nnzsplit(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_compute)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_compute_nnzsplit(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        break;
    }
    case rocsparse_spmv_alg_csr_adaptive:
    {
        if(stage == rocsparse_spmv_stage_buffer_size)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_buffer_size_adaptive(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, buffer_size,
                temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_preprocess)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_preprocess_adaptive(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_compute)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_compute_adaptive(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        break;
    }
    case rocsparse_spmv_alg_csr_lrb:
    {
        if(stage == rocsparse_spmv_stage_buffer_size)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_buffer_size_lrb(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, buffer_size,
                temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_preprocess)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_preprocess_lrb(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        else if(stage == rocsparse_spmv_stage_compute)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_compute_lrb(
                handle, trans, m, n, nnz, alpha_datatype, alpha, descr, csr_val_datatype,
                csr_val, csr_row_ptr_indextype, csr_row_ptr, csr_col_ind_indextype, csr_col_ind,
                x_datatype, x, beta_datatype, beta, y_datatype, y, alg, stage, temp_buffer));
        }
        break;
    }
    default:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    }

    return rocsparse_status_success;
}
