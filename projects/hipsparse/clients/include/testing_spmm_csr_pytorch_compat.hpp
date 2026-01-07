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

/**
 * @file testing_spmm_csr_pytorch_compat.hpp
 * @brief Tests for SpMM CSR operations with float16/bfloat16 to verify
 *        compatibility with PyTorch's test_addmm_all_sparse_csr test.
 *
 * This test reproduces the behavior of PyTorch's test_addmm_all_sparse_csr test:
 * - M = randn(10, 25) - output/bias matrix
 * - m1 = randn(10, 50) - first sparse matrix
 * - m2 = randn(50, 25) - second sparse matrix (dense in this test)
 * - Computes: alpha * (m1 @ m2) + beta * M
 * - Uses alpha=1.2, beta=0.8 (default values from PyTorch test)
 * - Tolerance: atol=0.1, rtol=0.001 for float16 (precisionOverride from PyTorch)
 */

#pragma once
#ifndef TESTING_SPMM_CSR_PYTORCH_COMPAT_HPP
#define TESTING_SPMM_CSR_PYTORCH_COMPAT_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <hipsparse_test_unique_ptr.hpp>
#include <iostream>
#include <random>
#include <vector>

namespace hipsparse_test
{
    // Custom half-precision type wrappers for host code
    // These wrap the raw bits and provide conversion to/from float
    struct hipsparse_spmm_bf16_t
    {
        uint16_t data;

        hipsparse_spmm_bf16_t()
            : data(0)
        {
        }
        explicit hipsparse_spmm_bf16_t(float f)
        {
            // Convert float to bfloat16
            uint32_t bits;
            memcpy(&bits, &f, sizeof(float));
            // Round to nearest even
            uint32_t rounding_bias = (bits >> 16) & 1;
            bits += 0x7FFF + rounding_bias;
            data = static_cast<uint16_t>(bits >> 16);
        }
        explicit operator float() const
        {
            // Convert bfloat16 to float
            uint32_t bits = static_cast<uint32_t>(data) << 16;
            float    f;
            memcpy(&f, &bits, sizeof(float));
            return f;
        }
    };

    struct hipsparse_spmm_f16_t
    {
        uint16_t data;

        hipsparse_spmm_f16_t()
            : data(0)
        {
        }
        explicit hipsparse_spmm_f16_t(float f)
        {
            // Convert float to IEEE 754 half-precision
            uint32_t bits;
            memcpy(&bits, &f, sizeof(float));

            uint32_t sign     = (bits >> 31) & 0x1;
            int32_t  exponent = ((bits >> 23) & 0xFF) - 127;
            uint32_t mantissa = bits & 0x7FFFFF;

            if(exponent > 15)
            {
                // Overflow to infinity
                data = static_cast<uint16_t>((sign << 15) | 0x7C00);
            }
            else if(exponent < -14)
            {
                // Underflow to zero or denormal
                if(exponent < -24)
                {
                    data = static_cast<uint16_t>(sign << 15);
                }
                else
                {
                    mantissa |= 0x800000;
                    uint32_t shift = -exponent - 14 + 13;
                    data           = static_cast<uint16_t>((sign << 15) | (mantissa >> shift));
                }
            }
            else
            {
                data = static_cast<uint16_t>((sign << 15) | ((exponent + 15) << 10)
                                             | (mantissa >> 13));
            }
        }
        explicit operator float() const
        {
            // Convert IEEE 754 half-precision to float
            uint32_t sign     = (data >> 15) & 0x1;
            uint32_t exponent = (data >> 10) & 0x1F;
            uint32_t mantissa = data & 0x3FF;

            float result;
            if(exponent == 0)
            {
                if(mantissa == 0)
                {
                    // Zero
                    uint32_t bits = sign << 31;
                    memcpy(&result, &bits, sizeof(float));
                }
                else
                {
                    // Denormal
                    result = std::ldexp(static_cast<float>(mantissa), -24);
                    if(sign)
                        result = -result;
                }
            }
            else if(exponent == 31)
            {
                // Inf or NaN
                uint32_t bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
                memcpy(&result, &bits, sizeof(float));
            }
            else
            {
                // Normal
                uint32_t bits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
                memcpy(&result, &bits, sizeof(float));
            }
            return result;
        }
    };

    // Generate random dense matrix with values from N(0,1)
    template <typename T>
    void generate_random_dense_matrix_pytorch(std::vector<T>& data,
                                              int64_t         rows,
                                              int64_t         cols,
                                              unsigned int    seed)
    {
        std::mt19937                    gen(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        data.resize(rows * cols);
        for(int64_t i = 0; i < rows * cols; i++)
        {
            data[i] = T(normal(gen));
        }
    }

    // Generate random CSR matrix from dense matrix (sparsify by converting to CSR)
    template <typename I, typename J, typename T>
    void dense_to_csr_pytorch(const std::vector<T>& dense,
                              J                     rows,
                              J                     cols,
                              std::vector<I>&       row_ptr,
                              std::vector<J>&       col_ind,
                              std::vector<T>&       values,
                              hipsparseIndexBase_t  base = HIPSPARSE_INDEX_BASE_ZERO)
    {
        row_ptr.resize(rows + 1);
        col_ind.clear();
        values.clear();

        I nnz    = 0;
        I offset = (base == HIPSPARSE_INDEX_BASE_ONE) ? 1 : 0;

        for(J i = 0; i < rows; i++)
        {
            row_ptr[i] = nnz + offset;
            for(J j = 0; j < cols; j++)
            {
                T val = dense[i * cols + j];
                // Include all non-zero values
                float fval = static_cast<float>(val);
                if(fval != 0.0f)
                {
                    col_ind.push_back(j + offset);
                    values.push_back(val);
                    nnz++;
                }
            }
        }
        row_ptr[rows] = nnz + offset;
    }

    // Host SpMM reference: C = alpha * A @ B + beta * C
    // A is sparse CSR (M x K), B is dense (K x N), C is dense (M x N)
    template <typename I, typename J, typename T>
    void host_spmm_csr_pytorch(J                     M,
                               J                     N,
                               J                     K,
                               float                 alpha,
                               const std::vector<I>& row_ptr,
                               const std::vector<J>& col_ind,
                               const std::vector<T>& values,
                               const std::vector<T>& B,
                               float                 beta,
                               std::vector<T>&       C,
                               hipsparseIndexBase_t  base = HIPSPARSE_INDEX_BASE_ZERO)
    {
        I offset = (base == HIPSPARSE_INDEX_BASE_ONE) ? 1 : 0;

        for(J i = 0; i < M; i++)
        {
            for(J j = 0; j < N; j++)
            {
                float sum = 0.0f;
                for(I k = row_ptr[i] - offset; k < row_ptr[i + 1] - offset; k++)
                {
                    J     col   = col_ind[k] - offset;
                    float a_val = static_cast<float>(values[k]);
                    float b_val = static_cast<float>(B[col * N + j]);
                    sum += a_val * b_val;
                }
                float c_val  = static_cast<float>(C[i * N + j]);
                C[i * N + j] = T(alpha * sum + beta * c_val);
            }
        }
    }

    // Compare results with tolerance
    template <typename T>
    bool compare_results_spmm_pytorch(const std::vector<T>& result,
                                      const std::vector<T>& expected,
                                      float                 atol,
                                      float                 rtol,
                                      bool                  verbose = true)
    {
        if(result.size() != expected.size())
        {
            if(verbose)
                std::cerr << "Size mismatch: " << result.size() << " vs " << expected.size()
                          << std::endl;
            return false;
        }

        size_t mismatches   = 0;
        float  max_abs_diff = 0.0f;
        float  max_rel_diff = 0.0f;

        for(size_t i = 0; i < result.size(); i++)
        {
            float r        = static_cast<float>(result[i]);
            float e        = static_cast<float>(expected[i]);
            float abs_diff = std::abs(r - e);
            float rel_diff = (e != 0.0f) ? std::abs((r - e) / e) : 0.0f;

            max_abs_diff = std::max(max_abs_diff, abs_diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if(abs_diff > atol && rel_diff > rtol)
            {
                mismatches++;
                if(verbose && mismatches <= 5)
                {
                    std::cerr << "Mismatch at index " << i << ": got " << r << ", expected " << e
                              << " (abs_diff=" << abs_diff << ", rel_diff=" << rel_diff << ")"
                              << std::endl;
                }
            }
        }

        if(mismatches > 0 && verbose)
        {
            std::cerr << "Total mismatches: " << mismatches << "/" << result.size() << std::endl;
            std::cerr << "Max absolute difference: " << max_abs_diff << std::endl;
            std::cerr << "Max relative difference: " << max_rel_diff << std::endl;
        }

        return mismatches == 0;
    }

    // Test function for SpMM CSR with float16
    // Reproduces PyTorch's test_addmm_all_sparse_csr with float16
    template <typename I = int32_t, typename J = int32_t>
    void testing_spmm_csr_float16_pytorch_compat(
        J M = 10, J N = 25, J K = 50, float alpha = 1.2f, float beta = 0.8f)
    {
        using T = hipsparse_spmm_f16_t;

        std::cout << "Running SpMM CSR Float16 PyTorch Compatibility Test" << std::endl;
        std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;

        // Initialize HIP
        int deviceCount;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        {
            std::cerr << "No HIP devices found, skipping test" << std::endl;
            return;
        }

        // Create hipsparse handle
        hipsparseHandle_t handle;
        if(hipsparseCreate(&handle) != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "Failed to create hipsparse handle" << std::endl;
            return;
        }

        // Generate random matrices (matching PyTorch test dimensions)
        std::vector<T> h_A_dense, h_B, h_C;
        generate_random_dense_matrix_pytorch(h_A_dense, M, K, 12345);
        generate_random_dense_matrix_pytorch(h_B, K, N, 12346);
        generate_random_dense_matrix_pytorch(h_C, M, N, 12347);

        // Convert A to CSR
        std::vector<I> h_row_ptr;
        std::vector<J> h_col_ind;
        std::vector<T> h_values;
        dense_to_csr_pytorch(h_A_dense, M, K, h_row_ptr, h_col_ind, h_values);

        I nnz_A = h_col_ind.size();
        std::cout << "Sparse matrix A: " << M << "x" << K << ", nnz=" << nnz_A << std::endl;

        // Make a copy of C for GPU result comparison
        std::vector<T> h_C_gpu = h_C;
        std::vector<T> h_C_ref = h_C;

        // Allocate device memory
        I* d_row_ptr;
        J* d_col_ind;
        T* d_values;
        T* d_B;
        T* d_C;

        hipMalloc(&d_row_ptr, sizeof(I) * (M + 1));
        hipMalloc(&d_col_ind, sizeof(J) * nnz_A);
        hipMalloc(&d_values, sizeof(T) * nnz_A);
        hipMalloc(&d_B, sizeof(T) * K * N);
        hipMalloc(&d_C, sizeof(T) * M * N);

        // Copy to device
        hipMemcpy(d_row_ptr, h_row_ptr.data(), sizeof(I) * (M + 1), hipMemcpyHostToDevice);
        hipMemcpy(d_col_ind, h_col_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice);
        hipMemcpy(d_values, h_values.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), sizeof(T) * K * N, hipMemcpyHostToDevice);
        hipMemcpy(d_C, h_C_gpu.data(), sizeof(T) * M * N, hipMemcpyHostToDevice);

        // Create sparse matrix descriptor
        hipsparseSpMatDescr_t matA;
        hipsparseIndexType_t  row_type
            = (sizeof(I) == 4) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
        hipsparseIndexType_t col_type
            = (sizeof(J) == 4) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;

        hipsparseCreateCsr(&matA,
                           M,
                           K,
                           nnz_A,
                           d_row_ptr,
                           d_col_ind,
                           d_values,
                           row_type,
                           col_type,
                           HIPSPARSE_INDEX_BASE_ZERO,
                           HIP_R_16F);

        // Create dense matrix descriptors
        hipsparseDnMatDescr_t matB, matC;
        hipsparseCreateDnMat(&matB, K, N, N, d_B, HIP_R_16F, HIPSPARSE_ORDER_ROW);
        hipsparseCreateDnMat(&matC, M, N, N, d_C, HIP_R_16F, HIPSPARSE_ORDER_ROW);

        // Get buffer size
        size_t bufferSize = 0;
        hipsparseSpMM_bufferSize(handle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 matA,
                                 matB,
                                 &beta,
                                 matC,
                                 HIP_R_32F, // Compute type
                                 HIPSPARSE_SPMM_ALG_DEFAULT,
                                 &bufferSize);

        void* buffer = nullptr;
        if(bufferSize > 0)
        {
            hipMalloc(&buffer, bufferSize);
        }

        // Perform SpMM
        hipsparseStatus_t status = hipsparseSpMM(handle,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 matA,
                                                 matB,
                                                 &beta,
                                                 matC,
                                                 HIP_R_32F,
                                                 HIPSPARSE_SPMM_ALG_DEFAULT,
                                                 buffer);

        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "hipsparseSpMM failed with status: " << status << std::endl;
            // Cleanup
            hipsparseDestroySpMat(matA);
            hipsparseDestroyDnMat(matB);
            hipsparseDestroyDnMat(matC);
            if(buffer)
                hipFree(buffer);
            hipFree(d_row_ptr);
            hipFree(d_col_ind);
            hipFree(d_values);
            hipFree(d_B);
            hipFree(d_C);
            hipsparseDestroy(handle);
            throw std::runtime_error("hipsparseSpMM failed");
        }

        // Copy result back
        hipMemcpy(h_C_gpu.data(), d_C, sizeof(T) * M * N, hipMemcpyDeviceToHost);

        // Compute reference on host
        host_spmm_csr_pytorch(M, N, K, alpha, h_row_ptr, h_col_ind, h_values, h_B, beta, h_C_ref);

        // Compare results
        // PyTorch uses atol=0.1, rtol=0.001 for float16 in test_addmm_all_sparse_csr
        float atol   = 0.1f;
        float rtol   = 0.001f;
        bool  passed = compare_results_spmm_pytorch(h_C_gpu, h_C_ref, atol, rtol);

        // Cleanup
        hipsparseDestroySpMat(matA);
        hipsparseDestroyDnMat(matB);
        hipsparseDestroyDnMat(matC);
        if(buffer)
            hipFree(buffer);
        hipFree(d_row_ptr);
        hipFree(d_col_ind);
        hipFree(d_values);
        hipFree(d_B);
        hipFree(d_C);
        hipsparseDestroy(handle);

        if(!passed)
        {
            throw std::runtime_error("SpMM CSR Float16 PyTorch compatibility test FAILED");
        }
        std::cout << "SpMM CSR Float16 PyTorch compatibility test PASSED" << std::endl;
    }

    // Test function for SpMM CSR with bfloat16
    template <typename I = int32_t, typename J = int32_t>
    void testing_spmm_csr_bfloat16_pytorch_compat(
        J M = 10, J N = 25, J K = 50, float alpha = 1.2f, float beta = 0.8f)
    {
        using T = hipsparse_spmm_bf16_t;

        std::cout << "Running SpMM CSR BFloat16 PyTorch Compatibility Test" << std::endl;
        std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;

        // Initialize HIP
        int deviceCount;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        {
            std::cerr << "No HIP devices found, skipping test" << std::endl;
            return;
        }

        // Create hipsparse handle
        hipsparseHandle_t handle;
        if(hipsparseCreate(&handle) != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "Failed to create hipsparse handle" << std::endl;
            return;
        }

        // Generate random matrices (matching PyTorch test dimensions)
        std::vector<T> h_A_dense, h_B, h_C;
        generate_random_dense_matrix_pytorch(h_A_dense, M, K, 12345);
        generate_random_dense_matrix_pytorch(h_B, K, N, 12346);
        generate_random_dense_matrix_pytorch(h_C, M, N, 12347);

        // Convert A to CSR
        std::vector<I> h_row_ptr;
        std::vector<J> h_col_ind;
        std::vector<T> h_values;
        dense_to_csr_pytorch(h_A_dense, M, K, h_row_ptr, h_col_ind, h_values);

        I nnz_A = h_col_ind.size();
        std::cout << "Sparse matrix A: " << M << "x" << K << ", nnz=" << nnz_A << std::endl;

        // Make a copy of C for GPU result comparison
        std::vector<T> h_C_gpu = h_C;
        std::vector<T> h_C_ref = h_C;

        // Allocate device memory
        I* d_row_ptr;
        J* d_col_ind;
        T* d_values;
        T* d_B;
        T* d_C;

        hipMalloc(&d_row_ptr, sizeof(I) * (M + 1));
        hipMalloc(&d_col_ind, sizeof(J) * nnz_A);
        hipMalloc(&d_values, sizeof(T) * nnz_A);
        hipMalloc(&d_B, sizeof(T) * K * N);
        hipMalloc(&d_C, sizeof(T) * M * N);

        // Copy to device
        hipMemcpy(d_row_ptr, h_row_ptr.data(), sizeof(I) * (M + 1), hipMemcpyHostToDevice);
        hipMemcpy(d_col_ind, h_col_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice);
        hipMemcpy(d_values, h_values.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), sizeof(T) * K * N, hipMemcpyHostToDevice);
        hipMemcpy(d_C, h_C_gpu.data(), sizeof(T) * M * N, hipMemcpyHostToDevice);

        // Create sparse matrix descriptor
        hipsparseSpMatDescr_t matA;
        hipsparseIndexType_t  row_type
            = (sizeof(I) == 4) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
        hipsparseIndexType_t col_type
            = (sizeof(J) == 4) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;

        hipsparseCreateCsr(&matA,
                           M,
                           K,
                           nnz_A,
                           d_row_ptr,
                           d_col_ind,
                           d_values,
                           row_type,
                           col_type,
                           HIPSPARSE_INDEX_BASE_ZERO,
                           HIP_R_16BF);

        // Create dense matrix descriptors
        hipsparseDnMatDescr_t matB, matC;
        hipsparseCreateDnMat(&matB, K, N, N, d_B, HIP_R_16BF, HIPSPARSE_ORDER_ROW);
        hipsparseCreateDnMat(&matC, M, N, N, d_C, HIP_R_16BF, HIPSPARSE_ORDER_ROW);

        // Get buffer size
        size_t bufferSize = 0;
        hipsparseSpMM_bufferSize(handle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 matA,
                                 matB,
                                 &beta,
                                 matC,
                                 HIP_R_32F, // Compute type
                                 HIPSPARSE_SPMM_ALG_DEFAULT,
                                 &bufferSize);

        void* buffer = nullptr;
        if(bufferSize > 0)
        {
            hipMalloc(&buffer, bufferSize);
        }

        // Perform SpMM
        hipsparseStatus_t status = hipsparseSpMM(handle,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 matA,
                                                 matB,
                                                 &beta,
                                                 matC,
                                                 HIP_R_32F,
                                                 HIPSPARSE_SPMM_ALG_DEFAULT,
                                                 buffer);

        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "hipsparseSpMM failed with status: " << status << std::endl;
            // Cleanup
            hipsparseDestroySpMat(matA);
            hipsparseDestroyDnMat(matB);
            hipsparseDestroyDnMat(matC);
            if(buffer)
                hipFree(buffer);
            hipFree(d_row_ptr);
            hipFree(d_col_ind);
            hipFree(d_values);
            hipFree(d_B);
            hipFree(d_C);
            hipsparseDestroy(handle);
            throw std::runtime_error("hipsparseSpMM failed");
        }

        // Copy result back
        hipMemcpy(h_C_gpu.data(), d_C, sizeof(T) * M * N, hipMemcpyDeviceToHost);

        // Compute reference on host
        host_spmm_csr_pytorch(M, N, K, alpha, h_row_ptr, h_col_ind, h_values, h_B, beta, h_C_ref);

        // Compare results
        // PyTorch uses atol=0.6 for bfloat16 in test_addmm_all_sparse_csr (precisionOverride)
        float atol   = 0.6f;
        float rtol   = 0.01f;
        bool  passed = compare_results_spmm_pytorch(h_C_gpu, h_C_ref, atol, rtol);

        // Cleanup
        hipsparseDestroySpMat(matA);
        hipsparseDestroyDnMat(matB);
        hipsparseDestroyDnMat(matC);
        if(buffer)
            hipFree(buffer);
        hipFree(d_row_ptr);
        hipFree(d_col_ind);
        hipFree(d_values);
        hipFree(d_B);
        hipFree(d_C);
        hipsparseDestroy(handle);

        if(!passed)
        {
            throw std::runtime_error("SpMM CSR BFloat16 PyTorch compatibility test FAILED");
        }
        std::cout << "SpMM CSR BFloat16 PyTorch compatibility test PASSED" << std::endl;
    }

} // namespace hipsparse_test

#endif // TESTING_SPMM_CSR_PYTORCH_COMPAT_HPP
