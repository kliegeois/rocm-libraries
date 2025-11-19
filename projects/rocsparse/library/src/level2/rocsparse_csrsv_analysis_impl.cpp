/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_csrsv.hpp"

#include "../conversion/rocsparse_coo2csr.hpp"
#include "../conversion/rocsparse_csr2coo.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "csrsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse::trm_analysis(rocsparse_handle          handle,
                                         rocsparse_operation       trans,
                                         J                         m,
                                         I                         nnz,
                                         const rocsparse_mat_descr descr,
                                         const T*                  csr_val,
                                         const I*                  csr_row_ptr,
                                         const J*                  csr_col_ind,
                                         rocsparse::trm_info_t*    trm_info,
                                         rocsparse::pivot_info_t*  pivot_info,
                                         void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": trm_analysis start\n";
    // stream
    hipStream_t stream = handle->stream;
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": got stream\n";

    // If analyzing transposed, allocate some info memory to hold the transposed matrix
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        hipDeviceSynchronize();
        std::cout << "trm_analysis:" << __LINE__ << ": transposed case\n";

        // Check if analysis was already done: buffers exist and dimensions already stored
        bool analysis_already_done = (trm_info->get_transposed_perm() != nullptr
                                      && trm_info->get_transposed_row_ptr() != nullptr
                                      && trm_info->get_transposed_col_ind() != nullptr
                                      && trm_info->get_m() == m && trm_info->get_nnz() == nnz);

        if(analysis_already_done)
        {
            std::cout << "trm_analysis:" << __LINE__
                      << ": reusing existing transposed arrays (m=" << m << ", nnz=" << nnz
                      << "), skipping transpose computation\n";
            // Buffers already contain the transposed matrix from previous call
            // Skip all transpose logic below and return
            return rocsparse_status_success;
        }

        // Check if dimensions changed and we need to free old buffers
        bool buffers_exist = (trm_info->get_transposed_perm() != nullptr
                              || trm_info->get_transposed_row_ptr() != nullptr
                              || trm_info->get_transposed_col_ind() != nullptr);

        if(buffers_exist && (trm_info->get_m() != m || trm_info->get_nnz() != nnz))
        {
            std::cout << "trm_analysis:" << __LINE__
                      << ": dimensions changed (m: " << trm_info->get_m() << " -> " << m
                      << ", nnz: " << trm_info->get_nnz() << " -> " << nnz
                      << "), freeing old buffers\n";

            // Free existing buffers
            void** ref_transposed_perm = trm_info->get_ref_transposed_perm();
            if(*ref_transposed_perm != nullptr)
            {
                std::cout << "trm_analysis:" << __LINE__ << ": freeing existing transposed_perm\n";
                RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(*ref_transposed_perm, stream));
                *ref_transposed_perm = nullptr;
            }

            void** ref_transposed_row_ptr = trm_info->get_ref_transposed_row_ptr();
            if(*ref_transposed_row_ptr != nullptr)
            {
                std::cout << "trm_analysis:" << __LINE__
                          << ": freeing existing transposed_row_ptr\n";
                RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(*ref_transposed_row_ptr, stream));
                *ref_transposed_row_ptr = nullptr;
            }

            void** ref_transposed_col_ind = trm_info->get_ref_transposed_col_ind();
            if(*ref_transposed_col_ind != nullptr)
            {
                std::cout << "trm_analysis:" << __LINE__
                          << ": freeing existing transposed_col_ind\n";
                RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(*ref_transposed_col_ind, stream));
                *ref_transposed_col_ind = nullptr;
            }

            // CRITICAL: Must synchronize after freeing before reallocating
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            std::cout << "trm_analysis:" << __LINE__
                      << ": synchronized after freeing old buffers\n";
        }

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);

        // work1 buffer
        J* tmp_work1 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

        // work2 buffer
        I* tmp_work2 = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * nnz - 1) / 256 + 1) * 256;

        // rocprim buffer
        void* rocprim_buffer = reinterpret_cast<void*>(ptr);
        hipDeviceSynchronize();
        std::cout << "trm_analysis:" << __LINE__ << ": setup transpose buffers\n";

        // Load CSR column indices into work1 buffer
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            tmp_work1, csr_col_ind, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        hipDeviceSynchronize();
        std::cout << "trm_analysis:" << __LINE__ << ": copied col_ind\n";

        // Allocate transposed arrays if they don't exist
        if(trm_info->get_transposed_row_ptr() == nullptr)
        {
            std::cout << "trm_analysis:" << __LINE__ << ": allocating transposed_row_ptr\n";
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                trm_info->get_ref_transposed_row_ptr(), sizeof(I) * (m + 1), stream));
        }

        if(nnz > 0)
        {
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": nnz > 0\n";

            if(trm_info->get_transposed_perm() == nullptr)
            {
                std::cout << "trm_analysis:" << __LINE__
                          << ": allocating transposed_perm and transposed_col_ind\n";
                RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                    trm_info->get_ref_transposed_perm(), sizeof(I) * nnz, stream));
                RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                    trm_info->get_ref_transposed_col_ind(), sizeof(J) * nnz, stream));
            }

            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

            I* transposed_perm = (I*)trm_info->get_transposed_perm();
            std::cout << "trm_analysis:" << __LINE__
                      << ": transposed_perm pointer=" << transposed_perm << "\n";

            // Create identity permutation in BOTH buffers
            // This is critical because radix_sort_pairs may swap current/alternate
            // and we need both buffers to have valid data initially
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::create_identity_permutation_template(handle, nnz, transposed_perm));
            std::cout << "trm_analysis:" << __LINE__
                      << ": created identity permutation in transposed_perm\n";

            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::create_identity_permutation_template(handle, nnz, tmp_work2));
            std::cout << "trm_analysis:" << __LINE__
                      << ": created identity permutation in tmp_work2\n";

            // CRITICAL: Synchronize to ensure identity permutations are written
            // before radix_sort_pairs reads them
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            std::cout << "trm_analysis:" << __LINE__
                      << ": synchronized after identity permutation creation\n";

            // Stable sort COO by columns
            J* transposed_col_ind = (J*)trm_info->get_transposed_col_ind();
            std::cout << "trm_analysis:" << __LINE__
                      << ": transposed_col_ind pointer=" << transposed_col_ind << "\n";
            std::cout << "trm_analysis:" << __LINE__ << ": tmp_work2 pointer=" << tmp_work2 << "\n";
            rocsparse::primitives::double_buffer<J> keys(tmp_work1, transposed_col_ind);
            rocsparse::primitives::double_buffer<I> vals(transposed_perm, tmp_work2);
            std::cout << "trm_analysis:" << __LINE__
                      << ": created double buffers, vals.current=" << vals.current()
                      << " vals.alternate=" << vals.alternate() << "\n";

            uint32_t startbit = 0;
            uint32_t endbit   = rocsparse::clz(m);

            size_t rocprim_size;
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__
                      << ": calling radix_sort_pairs_buffer_size for transpose\n";
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, I>(
                handle, nnz, startbit, endbit, &rocprim_size)));
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": rocprim_size=" << rocprim_size
                      << ", calling radix_sort_pairs\n";
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": radix_sort_pairs args: handle=" << handle
                      << " keys.current=" << keys.current()
                      << " keys.alternate=" << keys.alternate()
                      << " vals.current=" << vals.current()
                      << " vals.alternate=" << vals.alternate() << " nnz=" << nnz
                      << " startbit=" << startbit << " endbit=" << endbit
                      << " rocprim_size=" << rocprim_size << " rocprim_buffer=" << rocprim_buffer
                      << "\n";
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
                handle, keys, vals, nnz, startbit, endbit, rocprim_size, rocprim_buffer));
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__
                      << ": radix_sort_pairs completed for transpose\n";
            std::cout << "trm_analysis:" << __LINE__
                      << ": after sort: vals.current=" << vals.current()
                      << " (transposed_perm=" << transposed_perm << ")\n";
            std::cout << "trm_analysis:" << __LINE__
                      << ": after sort: keys.current=" << keys.current()
                      << " (tmp_work1=" << tmp_work1 << ")\n";

            // Copy permutation vector, if not already available
            if(vals.current() != transposed_perm)
            {
                hipDeviceSynchronize();
                std::cout << "trm_analysis:" << __LINE__ << ": copying vals to transposed_perm\n";
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(transposed_perm,
                                                   vals.current(),
                                                   sizeof(I) * nnz,
                                                   hipMemcpyDeviceToDevice,
                                                   stream));
                hipDeviceSynchronize();
                std::cout << "trm_analysis:" << __LINE__ << ": copied vals to transposed_perm\n";
            }

            I* transposed_row_ptr = (I*)trm_info->get_transposed_row_ptr();
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__
                      << ": got transposed_row_ptr, calling coo2csr\n";
            // Create column pointers
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coo2csr_template(
                handle, keys.current(), nnz, m, transposed_row_ptr, descr->base));
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": coo2csr completed\n";

            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": calling csr2coo for row indices\n";
            // Create row indices
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csr2coo_template(handle, csr_row_ptr, nnz, m, tmp_work1, descr->base));
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": csr2coo completed\n";

            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__
                      << ": validating transposed_perm before gthr\n";

            // Validate and print transposed_perm
            std::vector<I> h_transposed_perm(nnz);
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(h_transposed_perm.data(),
                                               transposed_perm,
                                               sizeof(I) * nnz,
                                               hipMemcpyDeviceToHost,
                                               stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

            std::cout << "trm_analysis:" << __LINE__ << ": transposed_perm values: ";
            //for(I i = 0; i < nnz; ++i)
            //{
            //    std::cout << h_transposed_perm[i] << " ";
            //}
            std::cout << "\n";

            // Check validity: each index should be in [0, nnz)
            bool valid = true;
            for(I i = 0; i < nnz; ++i)
            {
                I idx = h_transposed_perm[i] - rocsparse_index_base_zero;
                if(idx < 0 || idx >= nnz)
                {
                    if(valid)
                    {
                        std::cout << "trm_analysis:" << __LINE__ << ": ERROR: transposed_perm[" << i
                                  << "] = " << h_transposed_perm[i] << " is out of bounds [0, "
                                  << nnz << ")\n";
                    }
                    valid = false;
                }
            }

            if(!valid)
            {
                std::cout << "trm_analysis:" << __LINE__ << ": transposed_perm validation FAILED\n";
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            std::cout << "trm_analysis:" << __LINE__ << ": transposed_perm validation PASSED\n";

            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__
                      << ": calling gthr to permute column indices\n";
            // Permute column indices
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::gthr_template<I, J>(handle,
                                                                      nnz,
                                                                      tmp_work1,
                                                                      transposed_col_ind,
                                                                      transposed_perm,
                                                                      rocsparse_index_base_zero)));
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": gthr completed\n";
        }
        else
        {
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": nnz == 0, calling valset\n";
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            I* transposed_row_ptr = (I*)trm_info->get_transposed_row_ptr();
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::valset(handle, m + 1, static_cast<I>(descr->base), transposed_row_ptr));
            hipDeviceSynchronize();
            std::cout << "trm_analysis:" << __LINE__ << ": valset completed\n";
        }
    }

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": setting up main buffers\n";

    // Initialize temporary buffer with 0
    size_t buffer_size = 256 + ((sizeof(int) * m - 1) / 256 + 1) * 256;
    RETURN_IF_HIP_ERROR(hipMemsetAsync(ptr, 0, sizeof(char) * buffer_size, stream));
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": memset done\n";

    // max_nnz
    I* d_max_nnz = reinterpret_cast<I*>(ptr);
    ptr += 256;

    // done array
    int* done_array = reinterpret_cast<int*>(ptr);
    ptr += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    // workspace
    J* workspace = reinterpret_cast<J*>(ptr);
    ptr += ((sizeof(J) * m - 1) / 256 + 1) * 256;

    // workspace2
    int* workspace2 = reinterpret_cast<int*>(ptr);
    ptr += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    // rocprim buffer
    void* rocprim_buffer = reinterpret_cast<void*>(ptr);
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": rocprim_buffer setup\n";

    // Check if we need to reallocate diag_ind and row_map
    // These should always match the current dimension m
    bool need_realloc_main = false;
    if(trm_info->get_diag_ind() == nullptr || trm_info->get_row_map() == nullptr)
    {
        need_realloc_main = true;
        std::cout << "trm_analysis:" << __LINE__ << ": main arrays not allocated, will allocate\n";
    }
    else if(trm_info->get_m() != m)
    {
        need_realloc_main = true;
        std::cout << "trm_analysis:" << __LINE__ << ": dimension m changed, will reallocate\n";

        void** ref_diag_ind = trm_info->get_ref_diag_ind();
        if(*ref_diag_ind != nullptr)
        {
            std::cout << "trm_analysis:" << __LINE__ << ": freeing existing diag_ind\n";
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(*ref_diag_ind, stream));
            *ref_diag_ind = nullptr;
        }

        void** ref_row_map = trm_info->get_ref_row_map();
        if(*ref_row_map != nullptr)
        {
            std::cout << "trm_analysis:" << __LINE__ << ": freeing existing row_map\n";
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(*ref_row_map, stream));
            *ref_row_map = nullptr;
        }

        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }
    else
    {
        std::cout << "trm_analysis:" << __LINE__ << ": reusing existing main arrays (m=" << m
                  << ")\n";
    }

    // Allocate buffers only if needed
    if(need_realloc_main)
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(trm_info->get_ref_diag_ind(), sizeof(I) * m, stream));
        hipDeviceSynchronize();
        std::cout << "trm_analysis:" << __LINE__ << ": allocated diag_ind\n";

        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(trm_info->get_ref_row_map(), sizeof(J) * m, stream));
        hipDeviceSynchronize();
        std::cout << "trm_analysis:" << __LINE__ << ": allocated row_map\n";
    }

    // Allocate buffer to hold zero pivot (always reallocate as it's small and simple)
    pivot_info->create_zero_pivot_async(rocsparse::get_indextype<J>(), stream);
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": created zero_pivot\n";

    //
    // Synchronization needed.
    //
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": synchronized before analysis kernels\n";

    J* row_map  = (J*)trm_info->get_row_map();
    I* diag_ind = (I*)trm_info->get_diag_ind();
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": got row_map and diag_ind pointers\n";

    // Determine archid and ASIC revision
    const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
    const int         asicRev       = handle->asic_rev;

    // Run analysis
#define CSRSV_DIM 1024
    dim3 csrsv_blocks(((int64_t)handle->wavefront_size * m - 1) / CSRSV_DIM + 1);
    dim3 csrsv_threads(CSRSV_DIM);
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": setup kernel dims\n";

    void* zero_pivot = pivot_info->get_zero_pivot();
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": about to launch analysis kernels\n";

    if(trans == rocsparse_operation_none)
    {
        if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
        {
            // LCOV_EXCL_START
            if(descr->fill_mode == rocsparse_fill_mode_upper)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrsv_analysis_upper_kernel<CSRSV_DIM, 64, true>),
                    csrsv_blocks,
                    csrsv_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    diag_ind,
                    done_array,
                    d_max_nnz,
                    (J*)zero_pivot,
                    descr->base,
                    descr->diag_type);
            }
            else if(descr->fill_mode == rocsparse_fill_mode_lower)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrsv_analysis_lower_kernel<CSRSV_DIM, 64, true>),
                    csrsv_blocks,
                    csrsv_threads,
                    0,
                    stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    diag_ind,
                    done_array,
                    d_max_nnz,
                    (J*)zero_pivot,
                    descr->base,
                    descr->diag_type);
            }
            // LCOV_EXCL_STOP
        }
        else
        {
            if(handle->wavefront_size == 32)
            {
                // LCOV_EXCL_START
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_upper_kernel<CSRSV_DIM, 32, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_lower_kernel<CSRSV_DIM, 32, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
                // LCOV_EXCL_STOP
            }
            else
            {
                rocsparse_host_assert(handle->wavefront_size == 64,
                                      "Wrong wavefront size dispatch.");
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_upper_kernel<CSRSV_DIM, 64, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_lower_kernel<CSRSV_DIM, 64, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
            }
        }
    }
    else if(trans == rocsparse_operation_transpose
            || trans == rocsparse_operation_conjugate_transpose)
    {
        if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
        {
            // LCOV_EXCL_START
            if(descr->fill_mode == rocsparse_fill_mode_upper)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrsv_analysis_lower_kernel<CSRSV_DIM, 64, true>),
                    csrsv_blocks,
                    csrsv_threads,
                    0,
                    stream,
                    m,
                    (const I*)trm_info->get_transposed_row_ptr(),
                    (const J*)trm_info->get_transposed_col_ind(),
                    diag_ind,
                    done_array,
                    d_max_nnz,
                    (J*)zero_pivot,
                    descr->base,
                    descr->diag_type);
            }
            else if(descr->fill_mode == rocsparse_fill_mode_lower)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrsv_analysis_upper_kernel<CSRSV_DIM, 64, true>),
                    csrsv_blocks,
                    csrsv_threads,
                    0,
                    stream,
                    m,
                    (const I*)trm_info->get_transposed_row_ptr(),
                    (const J*)trm_info->get_transposed_col_ind(),
                    diag_ind,
                    done_array,
                    d_max_nnz,
                    (J*)zero_pivot,
                    descr->base,
                    descr->diag_type);
            }
            // LCOV_EXCL_STOP
        }
        else
        {
            if(handle->wavefront_size == 32)
            {
                // LCOV_EXCL_START
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_lower_kernel<CSRSV_DIM, 32, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        (const I*)trm_info->get_transposed_row_ptr(),
                        (const J*)trm_info->get_transposed_col_ind(),
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_upper_kernel<CSRSV_DIM, 32, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        (const I*)trm_info->get_transposed_row_ptr(),
                        (const J*)trm_info->get_transposed_col_ind(),
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
                // LCOV_EXCL_STOP
            }
            else
            {
                rocsparse_host_assert(handle->wavefront_size == 64,
                                      "Wrong wavefront size dispatch.");
                if(descr->fill_mode == rocsparse_fill_mode_upper)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_lower_kernel<CSRSV_DIM, 64, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        (const I*)trm_info->get_transposed_row_ptr(),
                        (const J*)trm_info->get_transposed_col_ind(),
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
                else if(descr->fill_mode == rocsparse_fill_mode_lower)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsv_analysis_upper_kernel<CSRSV_DIM, 64, false>),
                        csrsv_blocks,
                        csrsv_threads,
                        0,
                        stream,
                        m,
                        (const I*)trm_info->get_transposed_row_ptr(),
                        (const J*)trm_info->get_transposed_col_ind(),
                        diag_ind,
                        done_array,
                        d_max_nnz,
                        (J*)zero_pivot,
                        descr->base,
                        descr->diag_type);
                }
            }
        }
    }
    else
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        // LCOV_EXCL_STOP
    }
#undef CSRSV_DIM
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": analysis kernels completed\n";

    // Post processing
    I max_nnz;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&max_nnz, d_max_nnz, sizeof(I), hipMemcpyDeviceToHost, stream));
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": copied max_nnz to host\n";
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    trm_info->set_max_nnz(max_nnz);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::create_identity_permutation_template(handle, m, workspace));
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": created identity permutation\n";

    size_t rocprim_size;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(m);
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": preparing for final radix sort, m=" << m << "\n";

    rocsparse::primitives::double_buffer<int> keys(done_array, workspace2);
    rocsparse::primitives::double_buffer<J>   vals(workspace, row_map);
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": setup double buffers for final sort\n";

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int, J>(
        handle, m, startbit, endbit, &rocprim_size)));
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": final rocprim_size=" << rocprim_size
              << ", calling radix_sort_pairs\n";
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
        handle, keys, vals, m, startbit, endbit, rocprim_size, rocprim_buffer));
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": final radix_sort_pairs completed\n";

    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": synchronized after final sort\n";

    if(vals.current() != row_map)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            row_map, vals.current(), sizeof(J) * m, hipMemcpyDeviceToDevice, stream));
        hipDeviceSynchronize();
        std::cout << "trm_analysis:" << __LINE__ << ": copied vals to row_map\n";
    }

    // Store some pointers to verify correct execution

    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": storing info in trm_info\n";

    trm_info->set_m(m);
    trm_info->set_nnz(nnz);
    trm_info->set_descr(descr);

    trm_info->set_row_ptr((trans == rocsparse_operation_none) ? csr_row_ptr
                                                              : trm_info->get_transposed_row_ptr());
    trm_info->set_col_ind((trans == rocsparse_operation_none) ? csr_col_ind
                                                              : trm_info->get_transposed_col_ind());

    trm_info->set_offset_indextype(
        (sizeof(I) == sizeof(uint16_t))
            ? rocsparse_indextype_u16
            : ((sizeof(I) == sizeof(int32_t)) ? rocsparse_indextype_i32 : rocsparse_indextype_i64));

    trm_info->set_index_indextype(
        (sizeof(J) == sizeof(uint16_t))
            ? rocsparse_indextype_u16
            : ((sizeof(J) == sizeof(int32_t)) ? rocsparse_indextype_i32 : rocsparse_indextype_i64));

    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": trm_analysis completed successfully\n";
    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    int64_t                   m,
                                                    int64_t                   nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               csr_val_,
                                                    const void*               csr_row_ptr_,
                                                    const void*               csr_col_ind_,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_analysis_policy analysis,
                                                    rocsparse_solve_policy    solve,
                                                    rocsparse_csrsv_info*     p_csrsv_info,
                                                    void*                     temp_buffer)
{
    const T* csr_val     = reinterpret_cast<const T*>(csr_val_);
    const I* csr_row_ptr = reinterpret_cast<const I*>(csr_row_ptr_);
    const J* csr_col_ind = reinterpret_cast<const J*>(csr_col_ind_);

    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_POINTER(8, info);
    ROCSPARSE_CHECKARG_POINTER(11, p_csrsv_info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsrsv_analysis"),
                         trans,
                         m,
                         nnz,
                         descr,
                         csr_val,
                         csr_row_ptr,
                         csr_col_ind,
                         info,
                         solve,
                         analysis,
                         p_csrsv_info,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(9, analysis);
    ROCSPARSE_CHECKARG_ENUM(10, solve);

    // Check matrix type
    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);

    // Check matrix sorting mode

    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(11, temp_buffer);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);

    auto csrsv_info = p_csrsv_info[0];

    // Differentiate the analysis policies
    if(analysis == rocsparse_analysis_policy_reuse)
    {
        rocsparse::trm_info_t* p = nullptr;
        p = (p != nullptr) ? p : info->get_csrsv_info(trans, descr->fill_mode);

        if((descr->fill_mode == rocsparse_fill_mode_lower) && (trans == rocsparse_operation_none))
        {
            p = (p != nullptr) ? p : info->get_csrilu0_info(trans, descr->fill_mode);
            p = (p != nullptr) ? p : info->get_csric0_info(trans, descr->fill_mode);
        }

        p = (p != nullptr) ? p : info->get_csrsm_info(trans, descr->fill_mode);
        if(p != nullptr)
        {
            info->set_csrsv_info(trans, descr->fill_mode, p);
            return rocsparse_status_success;
        }
    }

    if(csrsv_info == nullptr)
    {
        csrsv_info      = new _rocsparse_csrsv_info();
        p_csrsv_info[0] = csrsv_info;
    }

    // Perform analysis

    hipDeviceSynchronize();
    std::cout << "trm_analysis:" << __LINE__ << ": csrsv_analysis_template\n";

    RETURN_IF_ROCSPARSE_ERROR(csrsv_info->recreate(handle,
                                                   trans,
                                                   static_cast<J>(m),
                                                   static_cast<I>(nnz),
                                                   descr,
                                                   csr_val,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   temp_buffer));

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                                                 \
    template rocsparse_status rocsparse::trm_analysis(rocsparse_handle          handle,      \
                                                      rocsparse_operation       trans,       \
                                                      J                         m,           \
                                                      I                         nnz,         \
                                                      const rocsparse_mat_descr descr,       \
                                                      const T*                  csr_val,     \
                                                      const I*                  csr_row_ptr, \
                                                      const J*                  csr_col_ind, \
                                                      rocsparse::trm_info_t*    info,        \
                                                      rocsparse::pivot_info_t*  pivot_info,  \
                                                      void*                     temp_buffer);

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

INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE

#define INSTANTIATE(I, J, T)                                               \
    template rocsparse_status rocsparse::csrsv_analysis_template<I, J, T>( \
        rocsparse_handle          handle,                                  \
        rocsparse_operation       trans,                                   \
        int64_t                   m,                                       \
        int64_t                   nnz,                                     \
        const rocsparse_mat_descr descr,                                   \
        const void*               csr_val,                                 \
        const void*               csr_row_ptr,                             \
        const void*               csr_col_ind,                             \
        rocsparse_mat_info        info,                                    \
        rocsparse_analysis_policy analysis,                                \
        rocsparse_solve_policy    solve,                                   \
        rocsparse_csrsv_info*     p_csrsv_info,                            \
        void*                     temp_buffer)

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

INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, T)                                                                          \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                           \
                                     rocsparse_operation       trans,                            \
                                     rocsparse_int             m,                                \
                                     rocsparse_int             nnz,                              \
                                     const rocsparse_mat_descr descr,                            \
                                     const T*                  csr_val,                          \
                                     const rocsparse_int*      csr_row_ptr,                      \
                                     const rocsparse_int*      csr_col_ind,                      \
                                     rocsparse_mat_info        info,                             \
                                     rocsparse_analysis_policy analysis,                         \
                                     rocsparse_solve_policy    solve,                            \
                                     void*                     temp_buffer)                      \
    try                                                                                          \
    {                                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                                 \
        rocsparse_csrsv_info csrsv_info = (info != nullptr) ? info->get_csrsv_info() : nullptr;  \
        RETURN_IF_ROCSPARSE_ERROR(                                                               \
            (rocsparse::csrsv_analysis_template<rocsparse_int, rocsparse_int, T>(handle,         \
                                                                                 trans,          \
                                                                                 m,              \
                                                                                 nnz,            \
                                                                                 descr,          \
                                                                                 csr_val,        \
                                                                                 csr_row_ptr,    \
                                                                                 csr_col_ind,    \
                                                                                 info,           \
                                                                                 analysis,       \
                                                                                 solve,          \
                                                                                 &csrsv_info,    \
                                                                                 temp_buffer))); \
        return rocsparse_status_success;                                                         \
    }                                                                                            \
    catch(...)                                                                                   \
    {                                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                                            \
    }

C_IMPL(rocsparse_scsrsv_analysis, float);
C_IMPL(rocsparse_dcsrsv_analysis, double);
C_IMPL(rocsparse_ccsrsv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsv_analysis, rocsparse_double_complex);

#undef C_IMPL
