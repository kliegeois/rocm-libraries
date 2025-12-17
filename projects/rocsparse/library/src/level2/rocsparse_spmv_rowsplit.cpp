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

#include "rocsparse_spmv_rowsplit.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include <map>
#include <sstream>

namespace rocsparse
{
    typedef rocsparse_status (*spmv_rowsplit_buffer_size_t)(rocsparse_handle,
                                                            rocsparse_operation,
                                                            int64_t,
                                                            int64_t,
                                                            int64_t,
                                                            rocsparse_datatype,
                                                            const void*,
                                                            const rocsparse_mat_descr,
                                                            rocsparse_datatype,
                                                            const void*,
                                                            rocsparse_indextype,
                                                            const void*,
                                                            rocsparse_indextype,
                                                            const void*,
                                                            rocsparse_datatype,
                                                            const void*,
                                                            rocsparse_datatype,
                                                            const void*,
                                                            rocsparse_datatype,
                                                            const void*,
                                                            rocsparse_spmv_alg,
                                                            rocsparse_spmv_stage,
                                                            size_t*,
                                                            void*);

    typedef rocsparse_status (*spmv_rowsplit_preprocess_t)(rocsparse_handle,
                                                           rocsparse_operation,
                                                           int64_t,
                                                           int64_t,
                                                           int64_t,
                                                           rocsparse_datatype,
                                                           const void*,
                                                           const rocsparse_mat_descr,
                                                           rocsparse_datatype,
                                                           const void*,
                                                           rocsparse_indextype,
                                                           const void*,
                                                           rocsparse_indextype,
                                                           const void*,
                                                           rocsparse_datatype,
                                                           const void*,
                                                           rocsparse_datatype,
                                                           const void*,
                                                           rocsparse_datatype,
                                                           const void*,
                                                           rocsparse_spmv_alg,
                                                           rocsparse_spmv_stage,
                                                           void*);

    typedef rocsparse_status (*spmv_rowsplit_compute_t)(rocsparse_handle,
                                                        rocsparse_operation,
                                                        int64_t,
                                                        int64_t,
                                                        int64_t,
                                                        rocsparse_datatype,
                                                        const void*,
                                                        const rocsparse_mat_descr,
                                                        rocsparse_datatype,
                                                        const void*,
                                                        rocsparse_indextype,
                                                        const void*,
                                                        rocsparse_indextype,
                                                        const void*,
                                                        rocsparse_datatype,
                                                        const void*,
                                                        rocsparse_datatype,
                                                        const void*,
                                                        rocsparse_datatype,
                                                        void*,
                                                        rocsparse_spmv_alg,
                                                        rocsparse_spmv_stage,
                                                        void*);

    using spmv_rowsplit_tuple = std::tuple<rocsparse_datatype,
                                           rocsparse_indextype,
                                           rocsparse_indextype,
                                           rocsparse_datatype,
                                           rocsparse_datatype,
                                           rocsparse_datatype>;

    // Rowsplit algorithm supports different precision combinations than other algorithms
    // This enables f16/bf16 support for algorithms that don't require atomicAdd
    static const std::map<spmv_rowsplit_tuple, spmv_rowsplit_buffer_size_t>
        s_spmv_rowsplit_buffer_size_dispatch{
            // Uniform precisions
            {spmv_rowsplit_tuple(rocsparse_datatype_f32_r,
                                 rocsparse_indextype_i32,
                                 rocsparse_indextype_i32,
                                 rocsparse_datatype_f32_r,
                                 rocsparse_datatype_f32_r,
                                 rocsparse_datatype_f32_r),
             nullptr}, // Placeholder
            // Mixed precisions - f16 support
            {spmv_rowsplit_tuple(rocsparse_datatype_f32_r,
                                 rocsparse_indextype_i32,
                                 rocsparse_indextype_i32,
                                 rocsparse_datatype_f16_r,
                                 rocsparse_datatype_f16_r,
                                 rocsparse_datatype_f32_r),
             nullptr}, // Placeholder
            // Mixed precisions - bf16 support
            {spmv_rowsplit_tuple(rocsparse_datatype_f32_r,
                                 rocsparse_indextype_i32,
                                 rocsparse_indextype_i32,
                                 rocsparse_datatype_bf16_r,
                                 rocsparse_datatype_bf16_r,
                                 rocsparse_datatype_f32_r),
             nullptr}, // Placeholder
        };
}

rocsparse_status rocsparse::spmv_buffer_size_rowsplit(rocsparse_handle             handle,
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
                                                      void*                        temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Rowsplit algorithm doesn't require additional buffer space
    *buffer_size = 0;

    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmv_preprocess_rowsplit(rocsparse_handle             handle,
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
                                                     void*                        temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Rowsplit preprocess - no analysis needed
    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmv_compute_rowsplit(rocsparse_handle             handle,
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
                                                  void*                        temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Delegate to csrmv rowsplit algorithm
    rocsparse::csrmv_alg csrmv_alg = rocsparse::csrmv_alg_rowsplit;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv(handle, trans, csrmv_alg, m, n, nnz, alpha_datatype,
                                               alpha, descr, csr_val_datatype, csr_val,
                                               csr_row_ptr_indextype, csr_row_ptr,
                                               csr_row_ptr_indextype,
                                               reinterpret_cast<const char*>(csr_row_ptr)
                                                   + rocsparse::indextype_sizeof(csr_row_ptr_indextype),
                                               csr_col_ind_indextype, csr_col_ind, nullptr,
                                               x_datatype, x, beta_datatype, beta, y_datatype, y,
                                               0, nullptr, nullptr, false));

    return rocsparse_status_success;
}
