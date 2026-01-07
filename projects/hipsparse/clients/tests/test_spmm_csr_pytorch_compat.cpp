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
 * @file test_spmm_csr_pytorch_compat.cpp
 * @brief GTest tests for SpMM CSR operations with float16/bfloat16
 *        to verify compatibility with PyTorch's test_addmm_all_sparse_csr test.
 *
 * These tests reproduce the behavior of PyTorch's test:
 * PYTORCH_TEST_WITH_ROCM=1 python test/test_sparse_csr.py TestSparseCSRCUDA.test_addmm_all_sparse_csr_SparseCSR_cuda_float16
 * PYTORCH_TEST_WITH_ROCM=1 python test/test_sparse_csr.py TestSparseCSRCUDA.test_addmm_all_sparse_csr_SparseCSR_cuda_bfloat16
 */

#include "testing_spmm_csr_pytorch_compat.hpp"
#include <gtest/gtest.h>

using namespace hipsparse_test;

// Test fixture for SpMM CSR Float16 PyTorch compatibility tests
class SpmmCsrFloat16PytorchCompat : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if HIP device is available
        int deviceCount = 0;
        (void)hipGetDeviceCount(&deviceCount);
        if(deviceCount == 0)
        {
            GTEST_SKIP() << "No HIP devices available";
        }
    }
};

// Test fixture for SpMM CSR BFloat16 PyTorch compatibility tests
class SpmmCsrBfloat16PytorchCompat : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if HIP device is available
        int deviceCount = 0;
        (void)hipGetDeviceCount(&deviceCount);
        if(deviceCount == 0)
        {
            GTEST_SKIP() << "No HIP devices available";
        }
    }
};

//
// Float16 Tests - Reproducing PyTorch's test_addmm_all_sparse_csr_SparseCSR_cuda_float16
//

// Test matching PyTorch default dimensions: M=10, N=25, K=50
TEST_F(SpmmCsrFloat16PytorchCompat, DefaultDimensions_Int32)
{
    testing_spmm_csr_float16_pytorch_compat<int32_t, int32_t>(10, 25, 50);
}

TEST_F(SpmmCsrFloat16PytorchCompat, DefaultDimensions_Int64)
{
    testing_spmm_csr_float16_pytorch_compat<int64_t, int64_t>(10, 25, 50);
}

// Test with alpha=1, beta=0 (simple matrix multiply without add)
TEST_F(SpmmCsrFloat16PytorchCompat, SimpleMatmul)
{
    testing_spmm_csr_float16_pytorch_compat<int32_t, int32_t>(10, 25, 50, 1.0f, 0.0f);
}

// Test with larger matrices
TEST_F(SpmmCsrFloat16PytorchCompat, LargerMatrices)
{
    testing_spmm_csr_float16_pytorch_compat<int32_t, int32_t>(64, 64, 64);
}

// Test with non-square matrices
TEST_F(SpmmCsrFloat16PytorchCompat, TallMatrix)
{
    testing_spmm_csr_float16_pytorch_compat<int32_t, int32_t>(100, 20, 30);
}

TEST_F(SpmmCsrFloat16PytorchCompat, WideMatrix)
{
    testing_spmm_csr_float16_pytorch_compat<int32_t, int32_t>(20, 100, 30);
}

//
// BFloat16 Tests - Reproducing PyTorch's test_addmm_all_sparse_csr_SparseCSR_cuda_bfloat16
//

// Test matching PyTorch default dimensions: M=10, N=25, K=50
TEST_F(SpmmCsrBfloat16PytorchCompat, DefaultDimensions_Int32)
{
    testing_spmm_csr_bfloat16_pytorch_compat<int32_t, int32_t>(10, 25, 50);
}

TEST_F(SpmmCsrBfloat16PytorchCompat, DefaultDimensions_Int64)
{
    testing_spmm_csr_bfloat16_pytorch_compat<int64_t, int64_t>(10, 25, 50);
}

// Test with alpha=1, beta=0 (simple matrix multiply without add)
TEST_F(SpmmCsrBfloat16PytorchCompat, SimpleMatmul)
{
    testing_spmm_csr_bfloat16_pytorch_compat<int32_t, int32_t>(10, 25, 50, 1.0f, 0.0f);
}

// Test with larger matrices
TEST_F(SpmmCsrBfloat16PytorchCompat, LargerMatrices)
{
    testing_spmm_csr_bfloat16_pytorch_compat<int32_t, int32_t>(64, 64, 64);
}

// Test with non-square matrices
TEST_F(SpmmCsrBfloat16PytorchCompat, TallMatrix)
{
    testing_spmm_csr_bfloat16_pytorch_compat<int32_t, int32_t>(100, 20, 30);
}

TEST_F(SpmmCsrBfloat16PytorchCompat, WideMatrix)
{
    testing_spmm_csr_bfloat16_pytorch_compat<int32_t, int32_t>(20, 100, 30);
}
