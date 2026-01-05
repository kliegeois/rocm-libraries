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
 * @file test_spmv_csr_pytorch_compat.cpp
 * @brief GTest-based tests that reproduce PyTorch's test_csr_matvec for half precision types.
 *
 * These tests replicate the PyTorch tests:
 *   PYTORCH_TEST_WITH_ROCM=1 python test/test_sparse_csr.py \
 *       TestSparseCSRCUDA.test_csr_matvec_cuda_bfloat16
 *   PYTORCH_TEST_WITH_ROCM=1 python test/test_sparse_csr.py \
 *       TestSparseCSRCUDA.test_csr_matvec_cuda_float16
 */

#include "testing_spmv_csr_pytorch_compat.hpp"
#include <gtest/gtest.h>

// ============================================================================
// BFloat16 Tests - Reproducing PyTorch test_csr_matvec_cuda_bfloat16
// ============================================================================

class SpmvCsrBfloat16PytorchCompat : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if device is available
        int deviceCount = 0;
        hipGetDeviceCount(&deviceCount);
        if(deviceCount == 0)
        {
            GTEST_SKIP() << "No HIP devices available";
        }
    }
};

// Main test: 100x100 matrix with 1000 nnz, int32 indices (matches PyTorch default)
TEST_F(SpmvCsrBfloat16PytorchCompat, Matrix100x100_NNZ1000_Int32)
{
    testing_spmv_csr_bfloat16_pytorch_compat(100, 100, 1000, false);
}

// Same with int64 indices (PyTorch also tests with torch.int64)
TEST_F(SpmvCsrBfloat16PytorchCompat, Matrix100x100_NNZ1000_Int64)
{
    testing_spmv_csr_bfloat16_pytorch_compat(100, 100, 1000, true);
}

// Small matrix edge case
TEST_F(SpmvCsrBfloat16PytorchCompat, SmallMatrix10x10)
{
    testing_spmv_csr_bfloat16_pytorch_compat(10, 10, 30, false);
}

// Non-square matrix
TEST_F(SpmvCsrBfloat16PytorchCompat, NonSquareMatrix50x100)
{
    testing_spmv_csr_bfloat16_pytorch_compat(50, 100, 500, false);
}

// Very sparse matrix
TEST_F(SpmvCsrBfloat16PytorchCompat, SparseMatrix100x100_NNZ50)
{
    testing_spmv_csr_bfloat16_pytorch_compat(100, 100, 50, false);
}

// ============================================================================
// Float16 Tests - Reproducing PyTorch test_csr_matvec_cuda_float16
// ============================================================================

class SpmvCsrFloat16PytorchCompat : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if device is available
        int deviceCount = 0;
        hipGetDeviceCount(&deviceCount);
        if(deviceCount == 0)
        {
            GTEST_SKIP() << "No HIP devices available";
        }
    }
};

// Main test: 100x100 matrix with 1000 nnz, int32 indices (matches PyTorch default)
TEST_F(SpmvCsrFloat16PytorchCompat, Matrix100x100_NNZ1000_Int32)
{
    testing_spmv_csr_float16_pytorch_compat(100, 100, 1000, false);
}

// Same with int64 indices (PyTorch also tests with torch.int64)
TEST_F(SpmvCsrFloat16PytorchCompat, Matrix100x100_NNZ1000_Int64)
{
    testing_spmv_csr_float16_pytorch_compat(100, 100, 1000, true);
}

// Small matrix edge case
TEST_F(SpmvCsrFloat16PytorchCompat, SmallMatrix10x10)
{
    testing_spmv_csr_float16_pytorch_compat(10, 10, 30, false);
}

// Non-square matrix
TEST_F(SpmvCsrFloat16PytorchCompat, NonSquareMatrix50x100)
{
    testing_spmv_csr_float16_pytorch_compat(50, 100, 500, false);
}

// Very sparse matrix
TEST_F(SpmvCsrFloat16PytorchCompat, SparseMatrix100x100_NNZ50)
{
    testing_spmv_csr_float16_pytorch_compat(100, 100, 50, false);
}
