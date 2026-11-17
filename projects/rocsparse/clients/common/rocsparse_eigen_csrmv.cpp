/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_eigen_csrmv.hpp"

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Sparse>

#include <complex>
#include <cstdint>
#include <vector>

template <typename T>
struct rocsparse_eigen_scalar;

template <>
struct rocsparse_eigen_scalar<float>
{
    using type = float;
    static type to(float value)
    {
        return value;
    }
    static float from(type value)
    {
        return value;
    }
};

template <>
struct rocsparse_eigen_scalar<double>
{
    using type = double;
    static type to(double value)
    {
        return value;
    }
    static double from(type value)
    {
        return value;
    }
};

template <>
struct rocsparse_eigen_scalar<rocsparse_float_complex>
{
    using type = std::complex<float>;
    static type to(rocsparse_float_complex value)
    {
        return static_cast<std::complex<float>>(value);
    }
    static rocsparse_float_complex from(type value)
    {
        return rocsparse_float_complex(value.real(), value.imag());
    }
};

template <>
struct rocsparse_eigen_scalar<rocsparse_double_complex>
{
    using type = std::complex<double>;
    static type to(rocsparse_double_complex value)
    {
        return static_cast<std::complex<double>>(value);
    }
    static rocsparse_double_complex from(type value)
    {
        return rocsparse_double_complex(value.real(), value.imag());
    }
};

template <typename T>
bool rocsparse_eigen_csrmv(rocsparse_operation   trans,
                           rocsparse_int         m,
                           rocsparse_int         n,
                           rocsparse_int         nnz,
                           T                     alpha,
                           const rocsparse_int*  csr_row_ptr,
                           const rocsparse_int*  csr_col_ind,
                           const T*              csr_val,
                           const T*              x,
                           T                     beta,
                           const T*              y_in,
                           T*                    y_out,
                           rocsparse_index_base  base,
                           rocsparse_matrix_type matrix_type)
{
    if(matrix_type == rocsparse_matrix_type_symmetric)
    {
        return false;
    }
    if(static_cast<int64_t>(nnz) > static_cast<int64_t>(rocsparse_eigen_csrmv_nnz_limit))
    {
        return false;
    }

    using scalar                    = typename rocsparse_eigen_scalar<T>::type;
    using triplet                   = Eigen::Triplet<scalar, rocsparse_int>;
    const scalar        alpha_eigen = rocsparse_eigen_scalar<T>::to(alpha);
    const scalar        beta_eigen  = rocsparse_eigen_scalar<T>::to(beta);
    const rocsparse_int x_len       = (trans == rocsparse_operation_none) ? n : m;
    const rocsparse_int y_len       = (trans == rocsparse_operation_none) ? m : n;
    const rocsparse_int index_base  = (base == rocsparse_index_base_one)
                                          ? static_cast<rocsparse_int>(1)
                                          : static_cast<rocsparse_int>(0);

    std::vector<triplet> triplets;
    triplets.reserve(static_cast<size_t>(nnz));
    if(m > 0)
    {
        for(rocsparse_int row = 0; row < m; ++row)
        {
            const rocsparse_int begin = csr_row_ptr[row] - index_base;
            const rocsparse_int end   = csr_row_ptr[row + 1] - index_base;
            for(rocsparse_int k = begin; k < end; ++k)
            {
                triplets.emplace_back(
                    row, csr_col_ind[k] - index_base, rocsparse_eigen_scalar<T>::to(csr_val[k]));
            }
        }
    }

    Eigen::SparseMatrix<scalar, Eigen::RowMajor, rocsparse_int> a(m, n);
    a.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::Matrix<scalar, Eigen::Dynamic, 1> x_eigen(x_len);
    Eigen::Matrix<scalar, Eigen::Dynamic, 1> y_eigen(y_len);
    for(rocsparse_int i = 0; i < x_len; ++i)
    {
        x_eigen(i) = rocsparse_eigen_scalar<T>::to(x[i]);
    }
    for(rocsparse_int i = 0; i < y_len; ++i)
    {
        y_eigen(i) = rocsparse_eigen_scalar<T>::to(y_in[i]);
    }

    Eigen::Matrix<scalar, Eigen::Dynamic, 1> y_ref;
    if(trans == rocsparse_operation_none)
    {
        y_ref = alpha_eigen * a * x_eigen + beta_eigen * y_eigen;
    }
    else if(trans == rocsparse_operation_transpose)
    {
        y_ref = alpha_eigen * a.transpose() * x_eigen + beta_eigen * y_eigen;
    }
    else
    {
        y_ref = alpha_eigen * a.adjoint() * x_eigen + beta_eigen * y_eigen;
    }

    for(rocsparse_int i = 0; i < y_len; ++i)
    {
        y_out[i] = rocsparse_eigen_scalar<T>::from(y_ref(i));
    }
    return true;
}

template bool rocsparse_eigen_csrmv<float>(rocsparse_operation,
                                           rocsparse_int,
                                           rocsparse_int,
                                           rocsparse_int,
                                           float,
                                           const rocsparse_int*,
                                           const rocsparse_int*,
                                           const float*,
                                           const float*,
                                           float,
                                           const float*,
                                           float*,
                                           rocsparse_index_base,
                                           rocsparse_matrix_type);

template bool rocsparse_eigen_csrmv<double>(rocsparse_operation,
                                            rocsparse_int,
                                            rocsparse_int,
                                            rocsparse_int,
                                            double,
                                            const rocsparse_int*,
                                            const rocsparse_int*,
                                            const double*,
                                            const double*,
                                            double,
                                            const double*,
                                            double*,
                                            rocsparse_index_base,
                                            rocsparse_matrix_type);

template bool rocsparse_eigen_csrmv<rocsparse_float_complex>(rocsparse_operation,
                                                             rocsparse_int,
                                                             rocsparse_int,
                                                             rocsparse_int,
                                                             rocsparse_float_complex,
                                                             const rocsparse_int*,
                                                             const rocsparse_int*,
                                                             const rocsparse_float_complex*,
                                                             const rocsparse_float_complex*,
                                                             rocsparse_float_complex,
                                                             const rocsparse_float_complex*,
                                                             rocsparse_float_complex*,
                                                             rocsparse_index_base,
                                                             rocsparse_matrix_type);

template bool rocsparse_eigen_csrmv<rocsparse_double_complex>(rocsparse_operation,
                                                              rocsparse_int,
                                                              rocsparse_int,
                                                              rocsparse_int,
                                                              rocsparse_double_complex,
                                                              const rocsparse_int*,
                                                              const rocsparse_int*,
                                                              const rocsparse_double_complex*,
                                                              const rocsparse_double_complex*,
                                                              rocsparse_double_complex,
                                                              const rocsparse_double_complex*,
                                                              rocsparse_double_complex*,
                                                              rocsparse_index_base,
                                                              rocsparse_matrix_type);
