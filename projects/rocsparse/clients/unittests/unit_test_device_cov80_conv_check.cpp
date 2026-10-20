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

// check_matrix_csr / check_matrix_gebsr / check_matrix_coo: rows of different lengths
// (the CSR checker picks a lane-group width from the average row length), unsorted block
// columns, and every value type of the COO checker.

#include "unit_test_cov80_conv_common.hpp"

#include <limits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    using cov80conv::DevBuf;

    rocsparse_index_base to_base(int base)
    {
        return static_cast<rocsparse_index_base>(base);
    }

    // ------------------------------------------------------------------------------
    // CSR
    // ------------------------------------------------------------------------------
    struct Csr
    {
        rocsparse_int              m = 0, n = 0;
        std::vector<rocsparse_int> ptr, ind;
        std::vector<float>         val;
    };

    // `m` rows with `k` consecutive columns each, row r starting at 3 r.
    Csr make_csr(rocsparse_int m, rocsparse_int n, rocsparse_int k, int base)
    {
        Csr a;
        a.m = m;
        a.n = n;
        a.ptr.push_back(base);
        for(rocsparse_int r = 0; r < m; ++r)
        {
            for(rocsparse_int j = 0; j < k; ++j)
            {
                a.ind.push_back(base + 3 * r + j);
                a.val.push_back(1.0f + j);
            }
            a.ptr.push_back(base + static_cast<rocsparse_int>(a.ind.size()));
        }
        return a;
    }

    rocsparse_data_status check_csr(rocsparse_handle       handle,
                                    const Csr&             a,
                                    int                    base,
                                    rocsparse_matrix_type  type,
                                    rocsparse_storage_mode storage,
                                    bool                   expect_success = true)
    {
        const rocsparse_int   nnz = static_cast<rocsparse_int>(a.ind.size());
        DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind);
        DevBuf<float>         d_val(a.val);
        size_t                size = 0;
        if(!expect_success)
        {
            // A non-general matrix type requires a square matrix.
            EXPECT_NE(rocsparse_scheck_matrix_csr_buffer_size(handle,
                                                              a.m,
                                                              a.n,
                                                              nnz,
                                                              d_val,
                                                              d_ptr,
                                                              d_ind,
                                                              to_base(base),
                                                              type,
                                                              rocsparse_fill_mode_lower,
                                                              storage,
                                                              &size),
                      rocsparse_status_success);
            return static_cast<rocsparse_data_status>(-1);
        }
        EXPECT_EQ(rocsparse_scheck_matrix_csr_buffer_size(handle,
                                                          a.m,
                                                          a.n,
                                                          nnz,
                                                          d_val,
                                                          d_ptr,
                                                          d_ind,
                                                          to_base(base),
                                                          type,
                                                          rocsparse_fill_mode_lower,
                                                          storage,
                                                          &size),
                  rocsparse_status_success);
        DevBuf<char>          d_buffer(std::vector<char>{}, size);
        rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
        EXPECT_EQ(rocsparse_scheck_matrix_csr(handle,
                                              a.m,
                                              a.n,
                                              nnz,
                                              d_val,
                                              d_ptr,
                                              d_ind,
                                              to_base(base),
                                              type,
                                              rocsparse_fill_mode_lower,
                                              storage,
                                              &ds,
                                              d_buffer),
                  rocsparse_status_success);
        return ds;
    }

    TEST_F(HandleTest, Cov80ConvCheckCsrRowLengths)
    {
        const auto general  = rocsparse_matrix_type_general;
        const auto sorted   = rocsparse_storage_mode_sorted;
        const auto unsorted = rocsparse_storage_mode_unsorted;

        // The average row length is 6, 12, 50 and 100 entries.
        for(rocsparse_int k : {6, 12, 50, 100})
        {
            for(int base : {0, 1})
            {
                SCOPED_TRACE("row length " + std::to_string(k) + " base " + std::to_string(base));
                const rocsparse_int m = 5, n = 200;
                const Csr           good         = make_csr(m, n, k, base);
                const rocsparse_int last_of_row2 = good.ptr[3] - base - 1;
                const rocsparse_int last_of_row1 = good.ptr[2] - base - 1;
                const rocsparse_int last_of_row3 = good.ptr[4] - base - 1;

                EXPECT_EQ(check_csr(handle, good, base, general, sorted),
                          rocsparse_data_status_success);
                EXPECT_EQ(check_csr(handle, good, base, general, unsorted),
                          rocsparse_data_status_success);

                {
                    Csr a               = good;
                    a.val[last_of_row2] = std::numeric_limits<float>::quiet_NaN();
                    EXPECT_EQ(check_csr(handle, a, base, general, sorted),
                              rocsparse_data_status_nan);
                }
                {
                    Csr a               = good;
                    a.val[last_of_row2] = std::numeric_limits<float>::infinity();
                    EXPECT_EQ(check_csr(handle, a, base, general, sorted),
                              rocsparse_data_status_inf);
                }
                {
                    // A column index past the last column, at the end of a row.
                    Csr a               = good;
                    a.ind[last_of_row1] = base + n;
                    EXPECT_EQ(check_csr(handle, a, base, general, sorted),
                              rocsparse_data_status_invalid_index);
                }
                {
                    // The last two columns of a row swapped.
                    Csr a = good;
                    std::swap(a.ind[last_of_row3], a.ind[last_of_row3 - 1]);
                    EXPECT_EQ(check_csr(handle, a, base, general, sorted),
                              rocsparse_data_status_invalid_sorting);
                    EXPECT_EQ(check_csr(handle, a, base, general, unsorted),
                              rocsparse_data_status_success);
                }
            }
        }
    }

    // A non-general matrix type needs a square matrix.
    TEST_F(HandleTest, Cov80ConvCheckCsrNonSquareTriangular)
    {
        const Csr a = make_csr(5, 200, 6, 0);
        check_csr(
            handle, a, 0, rocsparse_matrix_type_symmetric, rocsparse_storage_mode_sorted, false);
    }

    // ------------------------------------------------------------------------------
    // GEBSR
    // ------------------------------------------------------------------------------
    struct Gebsr
    {
        rocsparse_int              mb = 0, nb = 0, rdim = 0, cdim = 0;
        std::vector<rocsparse_int> ptr, ind;
        std::vector<float>         val;
    };

    // 3 x 4 blocks of 2 x 3. Block row r has the block columns 0..r+1, in ascending order.
    Gebsr make_gebsr(int base)
    {
        Gebsr a;
        a.mb   = 3;
        a.nb   = 4;
        a.rdim = 2;
        a.cdim = 3;
        a.ptr.push_back(base);
        for(rocsparse_int r = 0; r < a.mb; ++r)
        {
            for(rocsparse_int c = 0; c <= r + 1; ++c)
            {
                a.ind.push_back(base + c);
            }
            a.ptr.push_back(base + static_cast<rocsparse_int>(a.ind.size()));
        }
        a.val.assign(a.ind.size() * a.rdim * a.cdim, 1.0f);
        return a;
    }

    rocsparse_data_status check_gebsr(rocsparse_handle       handle,
                                      const Gebsr&           a,
                                      int                    base,
                                      rocsparse_direction    dir,
                                      rocsparse_matrix_type  type,
                                      rocsparse_storage_mode storage,
                                      bool                   expect_success = true)
    {
        const rocsparse_int   nnzb = static_cast<rocsparse_int>(a.ind.size());
        DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind);
        DevBuf<float>         d_val(a.val);
        size_t                size = 0;
        if(!expect_success)
        {
            // The buffer size query may or may not reject the arguments; the check must.
            rocsparse_scheck_matrix_gebsr_buffer_size(handle,
                                                      dir,
                                                      a.mb,
                                                      a.nb,
                                                      nnzb,
                                                      a.rdim,
                                                      a.cdim,
                                                      d_val,
                                                      d_ptr,
                                                      d_ind,
                                                      to_base(base),
                                                      type,
                                                      rocsparse_fill_mode_lower,
                                                      storage,
                                                      &size);
            DevBuf<char>          d_buffer(std::vector<char>{}, size);
            rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
            EXPECT_NE(rocsparse_scheck_matrix_gebsr(handle,
                                                    dir,
                                                    a.mb,
                                                    a.nb,
                                                    nnzb,
                                                    a.rdim,
                                                    a.cdim,
                                                    d_val,
                                                    d_ptr,
                                                    d_ind,
                                                    to_base(base),
                                                    type,
                                                    rocsparse_fill_mode_lower,
                                                    storage,
                                                    &ds,
                                                    d_buffer),
                      rocsparse_status_success);
            return ds;
        }
        EXPECT_EQ(rocsparse_scheck_matrix_gebsr_buffer_size(handle,
                                                            dir,
                                                            a.mb,
                                                            a.nb,
                                                            nnzb,
                                                            a.rdim,
                                                            a.cdim,
                                                            d_val,
                                                            d_ptr,
                                                            d_ind,
                                                            to_base(base),
                                                            type,
                                                            rocsparse_fill_mode_lower,
                                                            storage,
                                                            &size),
                  rocsparse_status_success);
        DevBuf<char>          d_buffer(std::vector<char>{}, size);
        rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
        EXPECT_EQ(rocsparse_scheck_matrix_gebsr(handle,
                                                dir,
                                                a.mb,
                                                a.nb,
                                                nnzb,
                                                a.rdim,
                                                a.cdim,
                                                d_val,
                                                d_ptr,
                                                d_ind,
                                                to_base(base),
                                                type,
                                                rocsparse_fill_mode_lower,
                                                storage,
                                                &ds,
                                                d_buffer),
                  rocsparse_status_success);
        return ds;
    }

    TEST_F(HandleTest, Cov80ConvCheckGebsr)
    {
        const auto general  = rocsparse_matrix_type_general;
        const auto sorted   = rocsparse_storage_mode_sorted;
        const auto unsorted = rocsparse_storage_mode_unsorted;
        for(int base : {0, 1})
        {
            for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " column dir "
                             + std::to_string(dir == rocsparse_direction_column));
                const Gebsr good = make_gebsr(base);
                EXPECT_EQ(check_gebsr(handle, good, base, dir, general, sorted),
                          rocsparse_data_status_success);
                EXPECT_EQ(check_gebsr(handle, good, base, dir, general, unsorted),
                          rocsparse_data_status_success);

                {
                    // Unsorted block columns of the third block row.
                    Gebsr a = good;
                    std::swap(a.ind[a.ptr[2] - base], a.ind[a.ptr[2] - base + 2]);
                    EXPECT_EQ(check_gebsr(handle, a, base, dir, general, sorted),
                              rocsparse_data_status_invalid_sorting);
                    EXPECT_EQ(check_gebsr(handle, a, base, dir, general, unsorted),
                              rocsparse_data_status_success);
                }
                {
                    Gebsr a  = good;
                    a.val[7] = std::numeric_limits<float>::quiet_NaN();
                    EXPECT_EQ(check_gebsr(handle, a, base, dir, general, unsorted),
                              rocsparse_data_status_nan);
                }
                {
                    Gebsr a  = good;
                    a.ind[1] = base + a.nb;
                    EXPECT_EQ(check_gebsr(handle, a, base, dir, general, sorted),
                              rocsparse_data_status_invalid_index);
                }
            }
        }
    }

    // A non-general matrix needs equal block dimensions and a square block matrix.
    TEST_F(HandleTest, Cov80ConvCheckGebsrNonSquareSymmetric)
    {
        const Gebsr a = make_gebsr(0);
        check_gebsr(handle,
                    a,
                    0,
                    rocsparse_direction_row,
                    rocsparse_matrix_type_symmetric,
                    rocsparse_storage_mode_sorted,
                    false);
    }

    // ------------------------------------------------------------------------------
    // COO, every value type
    // ------------------------------------------------------------------------------
    template <typename T>
    struct CooApi;

#define COV80_COO_API(T, PREFIX)                                                          \
    template <>                                                                           \
    struct CooApi<T>                                                                      \
    {                                                                                     \
        static rocsparse_status buffer_size(rocsparse_handle       handle,                \
                                            rocsparse_int          m,                     \
                                            rocsparse_int          n,                     \
                                            rocsparse_int          nnz,                   \
                                            const T*               val,                   \
                                            const rocsparse_int*   row,                   \
                                            const rocsparse_int*   col,                   \
                                            rocsparse_index_base   base,                  \
                                            rocsparse_matrix_type  type,                  \
                                            rocsparse_fill_mode    uplo,                  \
                                            rocsparse_storage_mode storage,               \
                                            size_t*                size)                  \
        {                                                                                 \
            return rocsparse_##PREFIX##check_matrix_coo_buffer_size(                      \
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, size);       \
        }                                                                                 \
        static rocsparse_status check(rocsparse_handle       handle,                      \
                                      rocsparse_int          m,                           \
                                      rocsparse_int          n,                           \
                                      rocsparse_int          nnz,                         \
                                      const T*               val,                         \
                                      const rocsparse_int*   row,                         \
                                      const rocsparse_int*   col,                         \
                                      rocsparse_index_base   base,                        \
                                      rocsparse_matrix_type  type,                        \
                                      rocsparse_fill_mode    uplo,                        \
                                      rocsparse_storage_mode storage,                     \
                                      rocsparse_data_status* ds,                          \
                                      void*                  buffer)                      \
        {                                                                                 \
            return rocsparse_##PREFIX##check_matrix_coo(                                  \
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, ds, buffer); \
        }                                                                                 \
    }

    COV80_COO_API(float, s);
    COV80_COO_API(double, d);
    COV80_COO_API(rocsparse_float_complex, c);
    COV80_COO_API(rocsparse_double_complex, z);
#undef COV80_COO_API

    template <typename T>
    T coo_value(int k)
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return static_cast<T>(1 + k);
        }
        else
        {
            using R = std::conditional_t<std::is_same_v<T, rocsparse_float_complex>, float, double>;
            return T(static_cast<R>(1 + k), static_cast<R>(-k));
        }
    }

    template <typename T>
    T coo_nan()
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return std::numeric_limits<T>::quiet_NaN();
        }
        else
        {
            using R = std::conditional_t<std::is_same_v<T, rocsparse_float_complex>, float, double>;
            return T(std::numeric_limits<R>::quiet_NaN(), static_cast<R>(1));
        }
    }

    template <typename T>
    rocsparse_data_status check_coo(rocsparse_handle           handle,
                                    rocsparse_int              m,
                                    rocsparse_int              n,
                                    const std::vector<T>&      val,
                                    std::vector<rocsparse_int> row,
                                    std::vector<rocsparse_int> col,
                                    int                        base,
                                    rocsparse_matrix_type      type,
                                    rocsparse_storage_mode     storage,
                                    bool                       expect_success = true)
    {
        const rocsparse_int   nnz = static_cast<rocsparse_int>(val.size());
        DevBuf<rocsparse_int> d_row(row), d_col(col);
        DevBuf<T>             d_val(val);
        size_t                size = 0;
        if(!expect_success)
        {
            EXPECT_NE(CooApi<T>::buffer_size(handle,
                                             m,
                                             n,
                                             nnz,
                                             d_val,
                                             d_row,
                                             d_col,
                                             to_base(base),
                                             type,
                                             rocsparse_fill_mode_lower,
                                             storage,
                                             &size),
                      rocsparse_status_success);
            return static_cast<rocsparse_data_status>(-1);
        }
        EXPECT_EQ(CooApi<T>::buffer_size(handle,
                                         m,
                                         n,
                                         nnz,
                                         d_val,
                                         d_row,
                                         d_col,
                                         to_base(base),
                                         type,
                                         rocsparse_fill_mode_lower,
                                         storage,
                                         &size),
                  rocsparse_status_success);
        DevBuf<char>          d_buffer(std::vector<char>{}, size);
        rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
        EXPECT_EQ(CooApi<T>::check(handle,
                                   m,
                                   n,
                                   nnz,
                                   d_val,
                                   d_row,
                                   d_col,
                                   to_base(base),
                                   type,
                                   rocsparse_fill_mode_lower,
                                   storage,
                                   &ds,
                                   d_buffer),
                  rocsparse_status_success);
        return ds;
    }

    template <typename T>
    void run_coo(rocsparse_handle handle)
    {
        const rocsparse_int m = 4, n = 5;
        for(int base : {0, 1})
        {
            SCOPED_TRACE("base " + std::to_string(base));
            // Row-major sorted entries.
            const std::vector<rocsparse_int> row
                = {base, base, base + 1, base + 2, base + 2, base + 3};
            const std::vector<rocsparse_int> col
                = {base, base + 3, base + 1, base, base + 4, base + 2};
            std::vector<T> val;
            for(int k = 0; k < 6; ++k)
                val.push_back(coo_value<T>(k));

            const auto general  = rocsparse_matrix_type_general;
            const auto sorted   = rocsparse_storage_mode_sorted;
            const auto unsorted = rocsparse_storage_mode_unsorted;

            EXPECT_EQ(check_coo<T>(handle, m, n, val, row, col, base, general, sorted),
                      rocsparse_data_status_success);
            {
                std::vector<T> v = val;
                v[4]             = coo_nan<T>();
                EXPECT_EQ(check_coo<T>(handle, m, n, v, row, col, base, general, sorted),
                          rocsparse_data_status_nan);
            }
            {
                std::vector<rocsparse_int> c = col;
                c[5]                         = base + n;
                EXPECT_EQ(check_coo<T>(handle, m, n, val, row, c, base, general, sorted),
                          rocsparse_data_status_invalid_index);
            }
            {
                // The entries of the first row swapped.
                std::vector<rocsparse_int> c = col;
                std::swap(c[0], c[1]);
                EXPECT_EQ(check_coo<T>(handle, m, n, val, row, c, base, general, sorted),
                          rocsparse_data_status_invalid_sorting);
                EXPECT_EQ(check_coo<T>(handle, m, n, val, row, c, base, general, unsorted),
                          rocsparse_data_status_success);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvCheckCooFloat)
    {
        run_coo<float>(handle);
    }
    TEST_F(HandleTest, Cov80ConvCheckCooDouble)
    {
        run_coo<double>(handle);
    }
    TEST_F(HandleTest, Cov80ConvCheckCooFloatComplex)
    {
        run_coo<rocsparse_float_complex>(handle);
    }
    TEST_F(HandleTest, Cov80ConvCheckCooDoubleComplex)
    {
        run_coo<rocsparse_double_complex>(handle);
    }

    TEST_F(HandleTest, Cov80ConvCheckCooNonSquareSymmetric)
    {
        const std::vector<float>         val = {1.0f, 2.0f};
        const std::vector<rocsparse_int> row = {0, 1}, col = {0, 3};
        check_coo<float>(handle,
                         3,
                         5,
                         val,
                         row,
                         col,
                         0,
                         rocsparse_matrix_type_symmetric,
                         rocsparse_storage_mode_sorted,
                         false);
    }
}
