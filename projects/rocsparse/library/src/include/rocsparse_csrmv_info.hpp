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

#include "rocsparse_adaptive_info.hpp"
#include "rocsparse_lrb_info.hpp"
#include "rocsparse_mat_descr.hpp"
#include "rocsparse_nnzsplit_info.hpp"

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis.
 *******************************************************************************/
typedef struct _rocsparse_csrmv_info
{
    _rocsparse_adaptive_info adaptive{};
    _rocsparse_lrb_info      lrb{};
    _rocsparse_nnzsplit_info nnzsplit{};

    // some data to verify correct execution
    rocsparse_operation         trans = rocsparse_operation_none;
    int64_t                     m{};
    int64_t                     n{};
    int64_t                     nnz{};
    int64_t                     max_rows{};
    const _rocsparse_mat_descr* descr{};
    const void*                 csr_row_ptr{};
    const void*                 csr_col_ind{};

    rocsparse_indextype index_type_I = rocsparse_indextype_u16;
    rocsparse_indextype index_type_J = rocsparse_indextype_u16;

    // Residual computation parameters
    const void*        gamma_ptr;
    const void*        z_ptr;
    rocsparse_datatype gamma_datatype;
    rocsparse_datatype z_datatype;
    bool               residual_enabled;

    _rocsparse_csrmv_info()
        : gamma_ptr(nullptr)
        , z_ptr(nullptr)
        , gamma_datatype(rocsparse_datatype_f32_r)
        , z_datatype(rocsparse_datatype_f32_r)
        , residual_enabled(false)
    {
    }

    ~_rocsparse_csrmv_info()
    {
        this->clear();
    }

    void clear()
    {
        this->adaptive.clear();
        this->lrb.clear();
        this->trans        = rocsparse_operation_none;
        this->m            = 0;
        this->n            = 0;
        this->nnz          = 0;
        this->max_rows     = 0;
        this->descr        = nullptr;
        this->csr_row_ptr  = nullptr;
        this->csr_col_ind  = nullptr;
        this->index_type_I = rocsparse_indextype_u16;
        this->index_type_J = rocsparse_indextype_u16;
        this->clear_residual_params();
    }

    void set_residual_params(const void*        gamma,
                             rocsparse_datatype gamma_type,
                             const void*        z,
                             rocsparse_datatype z_type)
    {
        gamma_ptr        = gamma;
        z_ptr            = z;
        gamma_datatype   = gamma_type;
        z_datatype       = z_type;
        residual_enabled = true;
    }

    void get_residual_params(const void**        gamma,
                             rocsparse_datatype* gamma_type,
                             const void**        z,
                             rocsparse_datatype* z_type) const
    {
        if(gamma)
            *gamma = gamma_ptr;
        if(gamma_type)
            *gamma_type = gamma_datatype;
        if(z)
            *z = z_ptr;
        if(z_type)
            *z_type = z_datatype;
    }

    void clear_residual_params()
    {
        gamma_ptr        = nullptr;
        z_ptr            = nullptr;
        residual_enabled = false;
    }

    bool is_residual_enabled() const
    {
        return residual_enabled;
    }

} * rocsparse_csrmv_info, *rocsparse_cscmv_info;

namespace rocsparse
{
    /********************************************************************************
   * \brief Copy csrmv info.
   *******************************************************************************/
    rocsparse_status copy_csrmv_info(rocsparse_csrmv_info dest, const rocsparse_csrmv_info src);
}
