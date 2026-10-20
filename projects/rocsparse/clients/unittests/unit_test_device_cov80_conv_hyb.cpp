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

// HYB matrices: csr2hyb (every partition type, both index bases, the four value
// types, conversion into an already filled HYB matrix), hyb2csr, and
// rocsparse_check_matrix_hyb / _buffer_size on valid and defective matrices.
//
// A HYB matrix is an ELL part (the first ell_width entries of every row) and a COO
// part (the remaining entries). The defects of the check tests are placed in the COO
// part: rocsparse_check_matrix_hyb checks the ELL part, then the COO part, and the
// second check overwrites the data status of the first one, so a defect in the ELL
// part is not reported (library behavior that is not pinned here). Only general
// matrices are checked: the ELL checker rejects the other matrix types with
// rocsparse_status_invalid_value although the HYB documentation lists them.
// Duplicate entries are not reported by the COO checker either and are not tested.

#include "unit_test_cov80_conv_common.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

using namespace rocsparse_ut;
using namespace cov80conv;

namespace
{
    // ------------------------------------------------------------------
    // Typed access to the s/d/c/z entry points.
    // ------------------------------------------------------------------

    template <typename T>
    struct Api;

#define COV80_HYB_API(T__, P__)                                                 \
    template <>                                                                 \
    struct Api<T__>                                                             \
    {                                                                           \
        static rocsparse_status csr2hyb(rocsparse_handle          h,            \
                                        rocsparse_int             m,            \
                                        rocsparse_int             n,            \
                                        const rocsparse_mat_descr d,            \
                                        const T__*                v,            \
                                        const rocsparse_int*      p,            \
                                        const rocsparse_int*      c,            \
                                        rocsparse_hyb_mat         hyb,          \
                                        rocsparse_int             w,            \
                                        rocsparse_hyb_partition   part)         \
        {                                                                       \
            return rocsparse_##P__##csr2hyb(h, m, n, d, v, p, c, hyb, w, part); \
        }                                                                       \
        static rocsparse_status hyb2csr(rocsparse_handle          h,            \
                                        const rocsparse_mat_descr d,            \
                                        const rocsparse_hyb_mat   hyb,          \
                                        T__*                      v,            \
                                        rocsparse_int*            p,            \
                                        rocsparse_int*            c,            \
                                        void*                     buf)          \
        {                                                                       \
            return rocsparse_##P__##hyb2csr(h, d, hyb, v, p, c, buf);           \
        }                                                                       \
    }

    COV80_HYB_API(float, s);
    COV80_HYB_API(double, d);
    COV80_HYB_API(rocsparse_float_complex, c);
    COV80_HYB_API(rocsparse_double_complex, z);
#undef COV80_HYB_API

    template <typename T>
    T make_value(int k)
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return static_cast<T>(1.0 + 0.5 * k);
        }
        else
        {
            using R = std::conditional_t<std::is_same_v<T, rocsparse_float_complex>, float, double>;
            return T(static_cast<R>(1.0 + 0.5 * k), static_cast<R>(-(k % 3)));
        }
    }

    template <typename T>
    bool is_finite(T v)
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return std::isfinite(v);
        }
        else
        {
            return std::isfinite(std::real(v)) && std::isfinite(std::imag(v));
        }
    }

    // ------------------------------------------------------------------
    // Host CSR matrix, index base 0 or 1.
    // ------------------------------------------------------------------

    template <typename T>
    struct HostCsr
    {
        int                        m = 0, n = 0, base = 0;
        std::vector<rocsparse_int> ptr;
        std::vector<rocsparse_int> ind;
        std::vector<T>             val;

        int nnz() const
        {
            return static_cast<int>(ind.size());
        }
    };

    // Row r has 1 + (3 r) % 5 entries at the columns (r + 2 k) % n (sorted, unique).
    template <typename T>
    HostCsr<T> make_csr(int m, int n, int base)
    {
        HostCsr<T> a;
        a.m    = m;
        a.n    = n;
        a.base = base;
        a.ptr.push_back(base);
        int k = 0;
        for(int r = 0; r < m; ++r)
        {
            const int        len = std::min(n, 1 + (3 * r) % 5);
            std::vector<int> cols;
            for(int e = 0; e < len; ++e)
            {
                cols.push_back((r + 2 * e) % n);
            }
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            for(int c : cols)
            {
                a.ind.push_back(c + base);
                a.val.push_back(make_value<T>(k++));
            }
            a.ptr.push_back(base + static_cast<int>(a.ind.size()));
        }
        return a;
    }

    template <typename T>
    std::map<std::pair<int, int>, T> to_map(const std::vector<rocsparse_int>& ptr,
                                            const std::vector<rocsparse_int>& ind,
                                            const std::vector<T>&             val,
                                            int                               m,
                                            int                               base)
    {
        std::map<std::pair<int, int>, T> out;
        for(int r = 0; r < m; ++r)
        {
            for(rocsparse_int k = ptr[r] - base; k < ptr[r + 1] - base; ++k)
            {
                out[{r, ind[k] - base}] = val[k];
            }
        }
        return out;
    }

    rocsparse_index_base to_base(int base)
    {
        return base == 0 ? rocsparse_index_base_zero : rocsparse_index_base_one;
    }

    struct MatDescrHolder
    {
        rocsparse_mat_descr d = nullptr;
        MatDescrGuard       guard{d};
        explicit MatDescrHolder(int base)
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&d), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_index_base(d, to_base(base)), rocsparse_status_success);
        }
    };

    struct HybHolder
    {
        rocsparse_hyb_mat h = nullptr;
        HybGuard          guard{h};
        HybHolder()
        {
            EXPECT_EQ(rocsparse_create_hyb_mat(&h), rocsparse_status_success);
        }
    };

    const std::pair<rocsparse_hyb_partition, rocsparse_int> s_partitions[]
        = {{rocsparse_hyb_partition_auto, 0},
           {rocsparse_hyb_partition_user, 0},
           {rocsparse_hyb_partition_user, 2},
           {rocsparse_hyb_partition_user, 5},
           {rocsparse_hyb_partition_max, 0}};

    // ------------------------------------------------------------------
    // csr2hyb -> hyb2csr round trip, and the check of the valid HYB matrix.
    // ------------------------------------------------------------------

    // Converts `a` into `hyb` and verifies the result: hyb2csr gives the matrix back
    // and rocsparse_check_matrix_hyb accepts it.
    template <typename T>
    void convert_and_verify(rocsparse_handle          handle,
                            const HostCsr<T>&         a,
                            const rocsparse_mat_descr descr,
                            const rocsparse_int*      d_ptr,
                            const rocsparse_int*      d_ind,
                            const T*                  d_val,
                            rocsparse_hyb_mat         hyb,
                            rocsparse_hyb_partition   partition,
                            rocsparse_int             ell_width)
    {
        const int base = a.base;
        SCOPED_TRACE("base " + std::to_string(base) + " partition "
                     + std::to_string(static_cast<int>(partition)) + " width "
                     + std::to_string(ell_width));
        ASSERT_EQ(Api<T>::csr2hyb(
                      handle, a.m, a.n, descr, d_val, d_ptr, d_ind, hyb, ell_width, partition),
                  rocsparse_status_success);

        // hyb2csr gives back the matrix.
        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, d_ptr, &buffer_size),
                  rocsparse_status_success);
        DevBuf<char>          d_buffer(std::vector<char>{}, buffer_size);
        DevBuf<rocsparse_int> o_ptr(std::vector<rocsparse_int>{}, a.m + 1);
        DevBuf<rocsparse_int> o_ind(std::vector<rocsparse_int>{}, a.nnz());
        DevBuf<T>             o_val(std::vector<T>{}, a.nnz());
        ASSERT_EQ(Api<T>::hyb2csr(handle, descr, hyb, o_val, o_ptr, o_ind, d_buffer),
                  rocsparse_status_success);
        std::vector<rocsparse_int> h_ptr(a.m + 1), h_ind(a.nnz());
        std::vector<T>             h_val(a.nnz());
        ASSERT_EQ(hipMemcpy(h_ptr.data(),
                            o_ptr.get(),
                            h_ptr.size() * sizeof(rocsparse_int),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(h_ptr[a.m] - base, a.nnz());
        ASSERT_EQ(hipMemcpy(h_ind.data(),
                            o_ind.get(),
                            h_ind.size() * sizeof(rocsparse_int),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpy(h_val.data(), o_val.get(), h_val.size() * sizeof(T), hipMemcpyDeviceToHost),
            hipSuccess);
        EXPECT_EQ(to_map<T>(h_ptr, h_ind, h_val, a.m, base),
                  to_map<T>(a.ptr, a.ind, a.val, a.m, base));

        // The check accepts it for both storage modes.
        for(rocsparse_storage_mode storage :
            {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted})
        {
            size_t check_size = 0;
            ASSERT_EQ(rocsparse_check_matrix_hyb_buffer_size(handle,
                                                             hyb,
                                                             to_base(base),
                                                             rocsparse_matrix_type_general,
                                                             rocsparse_fill_mode_lower,
                                                             storage,
                                                             &check_size),
                      rocsparse_status_success);
            DevBuf<char>          d_check(std::vector<char>{}, check_size);
            rocsparse_data_status ds = rocsparse_data_status_invalid_fill;
            ASSERT_EQ(rocsparse_check_matrix_hyb(handle,
                                                 hyb,
                                                 to_base(base),
                                                 rocsparse_matrix_type_general,
                                                 rocsparse_fill_mode_lower,
                                                 storage,
                                                 &ds,
                                                 d_check),
                      rocsparse_status_success);
            EXPECT_EQ(ds, rocsparse_data_status_success);
        }
    }

    template <typename T>
    void run_roundtrip(rocsparse_handle handle)
    {
        for(int base : {0, 1})
        {
            const HostCsr<T>      a = make_csr<T>(8, 8, base);
            DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind);
            DevBuf<T>             d_val(a.val);
            MatDescrHolder        descr(base);

            // Every partition into a new HYB matrix.
            for(const auto& part : s_partitions)
            {
                HybHolder hyb;
                convert_and_verify<T>(
                    handle, a, descr.d, d_ptr, d_ind, d_val, hyb.h, part.first, part.second);
            }

            // Converting into a HYB matrix that already holds a matrix replaces it.
            // Both partitions keep a non-empty ELL and COO part (the old parts are freed
            // and the new ones allocated).
            HybHolder hyb;
            for(const auto& part : {s_partitions[0], s_partitions[2], s_partitions[0]})
            {
                convert_and_verify<T>(
                    handle, a, descr.d, d_ptr, d_ind, d_val, hyb.h, part.first, part.second);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvHybRoundTripFloat)
    {
        run_roundtrip<float>(handle);
    }
    TEST_F(HandleTest, Cov80ConvHybRoundTripDouble)
    {
        run_roundtrip<double>(handle);
    }
    TEST_F(HandleTest, Cov80ConvHybRoundTripFloatComplex)
    {
        run_roundtrip<rocsparse_float_complex>(handle);
    }
    TEST_F(HandleTest, Cov80ConvHybRoundTripDoubleComplex)
    {
        run_roundtrip<rocsparse_double_complex>(handle);
    }

    // ------------------------------------------------------------------
    // Defective matrices. The defect is always in the COO part.
    // ------------------------------------------------------------------

    template <typename T>
    rocsparse_data_status check_defect(rocsparse_handle        handle,
                                       const HostCsr<T>&       a,
                                       rocsparse_hyb_partition partition,
                                       rocsparse_int           ell_width,
                                       rocsparse_matrix_type   type,
                                       rocsparse_fill_mode     uplo,
                                       rocsparse_storage_mode  storage)
    {
        DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind);
        DevBuf<T>             d_val(a.val);
        MatDescrHolder        descr(a.base);
        HybHolder             hyb;
        EXPECT_EQ(Api<T>::csr2hyb(
                      handle, a.m, a.n, descr.d, d_val, d_ptr, d_ind, hyb.h, ell_width, partition),
                  rocsparse_status_success);
        size_t size = 0;
        EXPECT_EQ(rocsparse_check_matrix_hyb_buffer_size(
                      handle, hyb.h, to_base(a.base), type, uplo, storage, &size),
                  rocsparse_status_success);
        DevBuf<char>          d_check(std::vector<char>{}, size);
        rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
        EXPECT_EQ(rocsparse_check_matrix_hyb(
                      handle, hyb.h, to_base(a.base), type, uplo, storage, &ds, d_check),
                  rocsparse_status_success);
        return ds;
    }

    // 3 x 4: row 0 has one entry, row 1 three, row 2 one. With an ELL width of one the
    // COO part holds the entries 2 and 3 (row 1, the columns 1 and 2).
    template <typename T>
    HostCsr<T> defect_matrix(int base)
    {
        HostCsr<T> a;
        a.m    = 3;
        a.n    = 4;
        a.base = base;
        a.ptr  = {base, base + 1, base + 4, base + 5};
        a.ind  = {base + 0, base + 0, base + 1, base + 2, base + 3};
        for(int k = 0; k < 5; ++k)
            a.val.push_back(make_value<T>(k));
        return a;
    }

    template <typename T>
    T make_nonfinite(double v)
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return static_cast<T>(v);
        }
        else
        {
            using R = std::conditional_t<std::is_same_v<T, rocsparse_float_complex>, float, double>;
            return T(static_cast<R>(v), static_cast<R>(1));
        }
    }

    template <typename T>
    void run_defects(rocsparse_handle handle)
    {
        const auto general = rocsparse_matrix_type_general;
        const auto lower   = rocsparse_fill_mode_lower;
        const auto sorted  = rocsparse_storage_mode_sorted;
        for(int base : {0, 1})
        {
            SCOPED_TRACE("base " + std::to_string(base));
            const auto check = [&](const HostCsr<T>& a, rocsparse_storage_mode storage) {
                return check_defect<T>(
                    handle, a, rocsparse_hyb_partition_user, 1, general, lower, storage);
            };

            EXPECT_EQ(check(defect_matrix<T>(base), sorted), rocsparse_data_status_success);

            {
                HostCsr<T> a = defect_matrix<T>(base);
                a.val[2]     = make_nonfinite<T>(std::numeric_limits<double>::quiet_NaN());
                EXPECT_EQ(check(a, sorted), rocsparse_data_status_nan);
            }
            {
                HostCsr<T> a = defect_matrix<T>(base);
                a.val[3]     = make_nonfinite<T>(std::numeric_limits<double>::infinity());
                EXPECT_EQ(check(a, sorted), rocsparse_data_status_inf);
            }
            {
                // A column index past the last column.
                HostCsr<T> a = defect_matrix<T>(base);
                a.ind[3]     = base + a.n;
                EXPECT_EQ(check(a, sorted), rocsparse_data_status_invalid_index);
            }
            {
                // Unsorted columns are an error for sorted storage only.
                HostCsr<T> a = defect_matrix<T>(base);
                std::swap(a.ind[2], a.ind[3]);
                EXPECT_EQ(check(a, sorted), rocsparse_data_status_invalid_sorting);
                EXPECT_EQ(check(a, rocsparse_storage_mode_unsorted), rocsparse_data_status_success);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvHybDefectsFloat)
    {
        run_defects<float>(handle);
    }
    TEST_F(HandleTest, Cov80ConvHybDefectsDouble)
    {
        run_defects<double>(handle);
    }
    TEST_F(HandleTest, Cov80ConvHybDefectsFloatComplex)
    {
        run_defects<rocsparse_float_complex>(handle);
    }
    TEST_F(HandleTest, Cov80ConvHybDefectsDoubleComplex)
    {
        run_defects<rocsparse_double_complex>(handle);
    }
}
