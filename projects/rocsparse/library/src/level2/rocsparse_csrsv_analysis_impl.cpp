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

inline uint32_t rocsparse_array_hash(const void* data, size_t size)
{
    void* buffer                   = malloc(size);
    std::ignore                    = hipMemcpy(buffer, data, size, hipMemcpyDefault);
    const unsigned char* byte_data = static_cast<const unsigned char*>(buffer);
    uint32_t             hash      = 0x811C9DC5; // FNV_prime_32
    for(size_t i = 0; i < size; ++i)
    {
        hash ^= byte_data[i];
        hash *= 0x01000193; // FNV_offset_basis_32
    }
    free(buffer);
    return hash;
}

template <typename T>
bool validate_array(const T* data, size_t size, T min_val, T max_val)
{
    void* buffer       = malloc(size * sizeof(T));
    std::ignore        = hipMemcpy(buffer, data, size * sizeof(T), hipMemcpyDefault);
    const T* host_data = static_cast<const T*>(buffer);
    bool     is_valid  = true;
    for(size_t i = 0; i < size; ++i)
    {
        if(host_data[i] < min_val || host_data[i] > max_val)
        {
            std::cout << "Validation failed at index " << i << ": " << host_data[i] << " not in ["
                      << min_val << ", " << max_val << "]\n";
            is_valid = false;
            break;
        }
    }
    free(buffer);
    return is_valid;
}

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

    const bool call_permute = true;
    const bool call_hash    = false;

    const bool debug_log = false;

#define SYNC_PRINT()                             \
    RETURN_IF_HIP_ERROR(hipDeviceSynchronize()); \
    if(debug_log)                                \
        std::cout << "------ " << __FILE__ << ":" << __LINE__ << "\n";

    // stream
    hipStream_t stream = handle->stream;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(temp_buffer, 0, 85826048, stream));

    // If analyzing transposed, allocate some info memory to hold the transposed matrix
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        // TODO: this need to be changed.
        // LCOV_EXCL_START
        if(trm_info->get_transposed_perm() != nullptr
           || trm_info->get_transposed_row_ptr() != nullptr
           || trm_info->get_transposed_col_ind() != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }
        // LCOV_EXCL_STOP

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

        // Load CSR column indices into work1 buffer
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            tmp_work1, csr_col_ind, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));

        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            trm_info->get_ref_transposed_row_ptr(), sizeof(I) * (m + 1), stream));

        if(nnz > 0)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                trm_info->get_ref_transposed_perm(), sizeof(I) * nnz, stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                trm_info->get_ref_transposed_col_ind(), sizeof(J) * nnz, stream));

            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

            I* transposed_perm = (I*)trm_info->get_transposed_perm();
            // Create identity permutation
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::create_identity_permutation_template(handle, nnz, transposed_perm));

            if(!validate_array(transposed_perm, nnz, static_cast<I>(0), static_cast<I>(nnz - 1)))
            {
                if(debug_log)
                {
                    SYNC_PRINT();
                    std::cout << "--------- Validation failed for transposed_perm\n";
                }
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            // Stable sort COO by columns
            J* transposed_col_ind = (J*)trm_info->get_transposed_col_ind();
            rocsparse::primitives::double_buffer<J> keys(tmp_work1, transposed_col_ind);
            rocsparse::primitives::double_buffer<I> vals(transposed_perm, tmp_work2);

            uint32_t startbit = 0;
            uint32_t endbit   = rocsparse::clz(m);

            size_t rocprim_size;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, I>(
                handle, nnz, startbit, endbit, &rocprim_size)));

            SYNC_PRINT();
            // Verify that rocprim_size + offset equals 87523
            size_t rocprim_buffer_offset
                = reinterpret_cast<char*>(rocprim_buffer) - reinterpret_cast<char*>(temp_buffer);
            rocsparse_host_assert(rocprim_size + rocprim_buffer_offset * sizeof(char) == 85826048,
                                  "rocprim_size + rocprim_buffer offset verification failed");

            SYNC_PRINT();

            if(call_hash)
            {
                // Print hash, size, and address for tmp_work1
                {
                    size_t   keys_size = sizeof(J) * nnz;
                    uint32_t hash      = rocsparse_array_hash(tmp_work1, keys_size);

                    if(debug_log)
                    {
                        std::cout << "--------- tmp_work1 address=" << tmp_work1
                                  << " size=" << keys_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for transposed_col_ind
                {
                    size_t   transposed_col_ind_size = sizeof(J) * nnz;
                    uint32_t hash
                        = rocsparse_array_hash(transposed_col_ind, transposed_col_ind_size);

                    if(debug_log)
                    {
                        std::cout << "--------- transposed_col_ind address=" << transposed_col_ind
                                  << " size=" << transposed_col_ind_size << " hash=" << hash
                                  << "\n";
                    }
                }
                // Print hash, size, and address for transposed_perm
                {
                    size_t   transposed_perm_size = sizeof(I) * nnz;
                    uint32_t hash = rocsparse_array_hash(transposed_perm, transposed_perm_size);

                    if(debug_log)
                    {
                        std::cout << "--------- transposed_perm address=" << transposed_perm
                                  << " size=" << transposed_perm_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for tmp_work2
                {
                    size_t   tmp_work2_size = sizeof(I) * nnz;
                    uint32_t hash           = rocsparse_array_hash(tmp_work2, tmp_work2_size);

                    if(debug_log)
                    {
                        std::cout << "--------- tmp_work2 address=" << tmp_work2
                                  << " size=" << tmp_work2_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for keys.current()
                {
                    size_t   keys_size = sizeof(J) * nnz;
                    uint32_t hash      = rocsparse_array_hash(keys.current(), keys_size);

                    if(debug_log)
                    {
                        std::cout << "--------- keys.current() address=" << keys.current()
                                  << " size=" << keys_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for keys.alternate()
                {
                    size_t   keys_size = sizeof(J) * nnz;
                    uint32_t hash      = rocsparse_array_hash(keys.alternate(), keys_size);

                    if(debug_log)
                    {
                        std::cout << "--------- keys.alternate() address=" << keys.alternate()
                                  << " size=" << keys_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for vals.current()
                {
                    size_t   vals_size = sizeof(I) * nnz;
                    uint32_t hash      = rocsparse_array_hash(vals.current(), vals_size);
                    if(debug_log)
                    {
                        std::cout << "--------- vals.current() address=" << vals.current()
                                  << " size=" << vals_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for vals.alternate()
                {
                    size_t   vals_size = sizeof(I) * nnz;
                    uint32_t hash      = rocsparse_array_hash(vals.alternate(), vals_size);

                    if(debug_log)
                    {
                        std::cout << "--------- vals.alternate() address=" << vals.alternate()
                                  << " size=" << vals_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for rocprim_buffer
                {
                    uint32_t hash = rocsparse_array_hash(rocprim_buffer, rocprim_size);

                    if(debug_log)
                    {
                        std::cout << "--------- rocprim_buffer address=" << rocprim_buffer
                                  << " size=" << rocprim_size << " hash=" << hash << "\n";
                    }
                }
            }

            // Validate tmp_work1
            if(!validate_array(tmp_work1, nnz, static_cast<J>(0), static_cast<J>(m - 1)))
            {
                if(debug_log)
                {
                    SYNC_PRINT();
                    std::cout << "--------- Validation failed for tmp_work1\n";
                }
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            // Validating transposed_col_ind is not necessary here

            // Validate transposed_perm
            if(!validate_array(transposed_perm, nnz, static_cast<I>(0), static_cast<I>(nnz - 1)))
            {
                if(debug_log)
                {
                    SYNC_PRINT();
                    std::cout << "--------- Validation failed for transposed_perm\n";
                }
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            SYNC_PRINT();

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
                handle, keys, vals, nnz, startbit, endbit, rocprim_size, rocprim_buffer));
            SYNC_PRINT();

            // Validate tmp_work1
            if(!validate_array(tmp_work1, nnz, static_cast<J>(0), static_cast<J>(m - 1)))
            {
                if(debug_log)
                {
                    SYNC_PRINT();
                    std::cout << "--------- Validation failed for tmp_work1\n";
                }
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            // Validate transposed_col_ind
            if(!validate_array(transposed_col_ind, nnz, static_cast<J>(0), static_cast<J>(m - 1)))
            {
                if(debug_log)
                {
                    SYNC_PRINT();
                    std::cout << "--------- Validation failed for transposed_col_ind\n";
                }
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            // Validate transposed_perm
            if(!validate_array(transposed_perm, nnz, static_cast<I>(0), static_cast<I>(nnz - 1)))
            {
                if(debug_log)
                {
                    SYNC_PRINT();
                    std::cout << "--------- Validation failed for transposed_perm\n";
                }
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            if(call_hash)
            {
                // Print hash, size, and address for keys.current()
                {
                    size_t   keys_size = sizeof(J) * nnz;
                    uint32_t hash      = rocsparse_array_hash(keys.current(), keys_size);

                    if(debug_log)
                    {
                        std::cout << "--------- AFTER keys.current() address=" << keys.current()
                                  << " size=" << keys_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for keys.alternate()
                {
                    size_t   keys_size = sizeof(J) * nnz;
                    uint32_t hash      = rocsparse_array_hash(keys.alternate(), keys_size);

                    if(debug_log)
                    {
                        std::cout << "--------- AFTER keys.alternate() address=" << keys.alternate()
                                  << " size=" << keys_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for vals.current()
                {
                    size_t   vals_size = sizeof(I) * nnz;
                    uint32_t hash      = rocsparse_array_hash(vals.current(), vals_size);
                    if(debug_log)
                    {
                        std::cout << "--------- AFTER vals.current() address=" << vals.current()
                                  << " size=" << vals_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for vals.alternate()
                {
                    size_t   vals_size = sizeof(I) * nnz;
                    uint32_t hash      = rocsparse_array_hash(vals.alternate(), vals_size);

                    if(debug_log)
                    {
                        std::cout << "--------- AFTER vals.alternate() address=" << vals.alternate()
                                  << " size=" << vals_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for rocprim_buffer
                {
                    uint32_t hash = rocsparse_array_hash(rocprim_buffer, rocprim_size);

                    if(debug_log)
                    {
                        std::cout << "--------- AFTER rocprim_buffer address=" << rocprim_buffer
                                  << " size=" << rocprim_size << " hash=" << hash << "\n";
                    }
                }
            }

            // Copy permutation vector, if not already available
            if(vals.current() != transposed_perm)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(transposed_perm,
                                                   vals.current(),
                                                   sizeof(I) * nnz,
                                                   hipMemcpyDeviceToDevice,
                                                   stream));
            }

            I* transposed_row_ptr = (I*)trm_info->get_transposed_row_ptr();
            // Create column pointers
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coo2csr_template(
                handle, keys.current(), nnz, m, transposed_row_ptr, descr->base));

            // Create row indices
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csr2coo_template(handle, csr_row_ptr, nnz, m, tmp_work1, descr->base));

            if(call_hash)
            {
                // Print hash, size, and address for tmp_work1
                {
                    size_t   tmp_work1_size = sizeof(J) * nnz;
                    uint32_t hash           = rocsparse_array_hash(tmp_work1, tmp_work1_size);

                    if(debug_log)
                    {
                        std::cout << "--------- tmp_work1 (row indices) address=" << tmp_work1
                                  << " size=" << tmp_work1_size << " hash=" << hash << "\n";
                    }
                }

                // Print hash, size, and address for transposed_col_ind
                {
                    size_t   transposed_col_ind_size = sizeof(J) * nnz;
                    uint32_t hash
                        = rocsparse_array_hash(transposed_col_ind, transposed_col_ind_size);

                    if(debug_log)
                    {
                        std::cout << "--------- transposed_col_ind address=" << transposed_col_ind
                                  << " size=" << transposed_col_ind_size << " hash=" << hash
                                  << "\n";
                    }
                }

                // Print hash, size, and address for transposed_perm
                {
                    size_t   transposed_perm_size = sizeof(I) * nnz;
                    uint32_t hash = rocsparse_array_hash(transposed_perm, transposed_perm_size);

                    if(debug_log)
                    {
                        std::cout << "--------- transposed_perm address=" << transposed_perm
                                  << " size=" << transposed_perm_size << " hash=" << hash << "\n";
                    }
                }
            }

            SYNC_PRINT();
            if(call_permute)
            {
                // Validate tmp_work1
                if(!validate_array(tmp_work1, nnz, static_cast<J>(0), static_cast<J>(m - 1)))
                {
                    if(debug_log)
                    {
                        SYNC_PRINT();
                        std::cout << "--------- Validation failed for tmp_work1\n";
                    }
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
                }

                // Validate transposed_col_ind
                if(!validate_array(
                       transposed_col_ind, nnz, static_cast<J>(0), static_cast<J>(m - 1)))
                {
                    if(debug_log)
                    {
                        SYNC_PRINT();
                        std::cout << "--------- Validation failed for transposed_col_ind\n";
                    }
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
                }

                // Validate transposed_perm
                if(!validate_array(
                       transposed_perm, nnz, static_cast<I>(0), static_cast<I>(nnz - 1)))
                {
                    if(debug_log)
                    {
                        SYNC_PRINT();
                        std::cout << "--------- Validation failed for transposed_perm\n";
                    }
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
                }

                // Permute column indices
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::gthr_template<I, J>(handle,
                                                    nnz,
                                                    tmp_work1,
                                                    transposed_col_ind,
                                                    transposed_perm,
                                                    rocsparse_index_base_zero)));
            }
            SYNC_PRINT();
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            I* transposed_row_ptr = (I*)trm_info->get_transposed_row_ptr();
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::valset(handle, m + 1, static_cast<I>(descr->base), transposed_row_ptr));
        }
    }

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // Initialize temporary buffer with 0
    size_t buffer_size = 256 + ((sizeof(int) * m - 1) / 256 + 1) * 256;
    RETURN_IF_HIP_ERROR(hipMemsetAsync(ptr, 0, sizeof(char) * buffer_size, stream));

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

    // Allocate buffer to hold diagonal entry point
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(trm_info->get_ref_diag_ind(), sizeof(I) * m, stream));

    // Allocate buffer to hold zero pivot
    pivot_info->create_zero_pivot_async(rocsparse::get_indextype<J>(), stream);

    // Allocate buffer to hold row map
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(trm_info->get_ref_row_map(), sizeof(J) * m, stream));

    //
    // Synchronization needed.
    //
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    J* row_map  = (J*)trm_info->get_row_map();
    I* diag_ind = (I*)trm_info->get_diag_ind();

    // Determine archid and ASIC revision
    const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
    const int         asicRev       = handle->asic_rev;

    // Run analysis
#define CSRSV_DIM 1024
    dim3 csrsv_blocks(((int64_t)handle->wavefront_size * m - 1) / CSRSV_DIM + 1);
    dim3 csrsv_threads(CSRSV_DIM);

    void* zero_pivot = pivot_info->get_zero_pivot();

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

    // Post processing
    I max_nnz;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&max_nnz, d_max_nnz, sizeof(I), hipMemcpyDeviceToHost, stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    trm_info->set_max_nnz(max_nnz);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::create_identity_permutation_template(handle, m, workspace));

    size_t rocprim_size;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(m);

    rocsparse::primitives::double_buffer<int> keys(done_array, workspace2);
    rocsparse::primitives::double_buffer<J>   vals(workspace, row_map);

    SYNC_PRINT();
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int, J>(
        handle, m, startbit, endbit, &rocprim_size)));
    SYNC_PRINT();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
        handle, keys, vals, m, startbit, endbit, rocprim_size, rocprim_buffer));
    SYNC_PRINT();

    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    if(vals.current() != row_map)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            row_map, vals.current(), sizeof(J) * m, hipMemcpyDeviceToDevice, stream));
    }

    // Store some pointers to verify correct execution

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
