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

// rocsparse_check_spmat on every supported format (coo, csr, csc, ell, bsr), every
// supported value type and index type combination. A well formed matrix must report
// rocsparse_data_status_success; each malformed matrix has exactly one defect and
// must report the matching data status. Unsupported combinations must return
// rocsparse_status_not_implemented.

#include "unit_test_device_cov70_conv_spmat.hpp"

#include <cmath>
#include <limits>
#include <string>

using namespace rocsparse_ut;
using namespace cov70conv;

namespace
{
    template <typename T>
    T make_special(bool nan)
    {
        using R = std::conditional_t<
            std::is_same_v<T, float> || std::is_same_v<T, rocsparse_float_complex>,
            float,
            double>;
        const R v = nan ? std::numeric_limits<R>::quiet_NaN() : std::numeric_limits<R>::infinity();
        if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        {
            return rocsparse_float_complex(v, 1.0f);
        }
        else if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_double_complex(v, 1.0);
        }
        else
        {
            return v;
        }
    }

    struct Attributes
    {
        rocsparse_matrix_type  type    = rocsparse_matrix_type_general;
        rocsparse_fill_mode    fill    = rocsparse_fill_mode_lower;
        rocsparse_storage_mode storage = rocsparse_storage_mode_sorted;
    };

    // Runs both stages of rocsparse_check_spmat on the descriptor and returns the
    // status of the compute stage and the reported data status.
    rocsparse_status run_check(rocsparse_handle       handle,
                               rocsparse_spmat_descr  descr,
                               const Attributes&      attributes,
                               rocsparse_data_status* data_status)
    {
        rocsparse_status st = rocsparse_spmat_set_attribute(
            descr, rocsparse_spmat_matrix_type, &attributes.type, sizeof(attributes.type));
        EXPECT_EQ(st, rocsparse_status_success);
        st = rocsparse_spmat_set_attribute(
            descr, rocsparse_spmat_fill_mode, &attributes.fill, sizeof(attributes.fill));
        EXPECT_EQ(st, rocsparse_status_success);
        st = rocsparse_spmat_set_attribute(
            descr, rocsparse_spmat_storage_mode, &attributes.storage, sizeof(attributes.storage));
        EXPECT_EQ(st, rocsparse_status_success);

        size_t buffer_size = 0;
        st                 = rocsparse_check_spmat(handle,
                                   descr,
                                   data_status,
                                   rocsparse_check_spmat_stage_buffer_size,
                                   &buffer_size,
                                   nullptr);
        if(st != rocsparse_status_success)
        {
            return st;
        }
        device_vector<char> temp(std::max(buffer_size, size_t(1)));
        EXPECT_TRUE(temp.ptr != nullptr);
        st = rocsparse_check_spmat(handle,
                                   descr,
                                   data_status,
                                   rocsparse_check_spmat_stage_compute,
                                   &buffer_size,
                                   temp.ptr);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return st;
    }

    template <typename T, typename I>
    rocsparse_data_status check_matrix(rocsparse_handle     handle,
                                       const HostMat<T, I>& h,
                                       const Attributes&    attributes = Attributes{})
    {
        DevMat<T, I> d(h);
        SpDescr      descr;
        EXPECT_EQ(create_descr(h, d, descr), rocsparse_status_success);
        // Start from a value that no check reports.
        rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
        EXPECT_EQ(run_check(handle, descr.d, attributes, &ds), rocsparse_status_success);
        return ds;
    }

    // Every format of a given type, base and index type must be accepted.
    template <typename T, typename I>
    void check_valid_formats(rocsparse_handle handle)
    {
        const std::vector<T> dense = make_dense<T>();
        for(rocsparse_format f : {rocsparse_format_coo,
                                  rocsparse_format_csr,
                                  rocsparse_format_csc,
                                  rocsparse_format_ell,
                                  rocsparse_format_bsr})
        {
            for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            {
                for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
                {
                    if(f != rocsparse_format_bsr && dir == rocsparse_direction_column)
                    {
                        continue;
                    }
                    SCOPED_TRACE(std::string(format_name(f))
                                 + " base=" + std::to_string(static_cast<int>(base))
                                 + " dir=" + std::to_string(static_cast<int>(dir)));
                    const HostMat<T, I> h = build<T, I>(f, base, 3, dir, dense);
                    EXPECT_EQ((check_matrix<T, I>(handle, h)), rocsparse_data_status_success);
                }
            }
        }
    }

    class Cov70ConvCheckSpmat : public HandleTest
    {
    };

    // Applies one defect (named as in check_defects) to a matrix.
    template <typename T, typename I>
    void apply_defect(HostMat<T, I>& h, const std::string& defect)
    {
        const I b   = static_cast<I>(h.base);
        auto&   idx = h.ind;
        if(defect == "nan" || defect == "inf")
        {
            h.val[3] = make_special<T>(defect == "nan");
        }
        else if(defect == "index_out_of_range")
        {
            switch(h.format)
            {
            case rocsparse_format_coo:
                h.row[0] = static_cast<I>(M) + b;
                break;
            case rocsparse_format_csr:
            case rocsparse_format_ell:
                idx[0] = static_cast<I>(N) + b;
                break;
            case rocsparse_format_csc:
                idx[0] = static_cast<I>(M) + b;
                break;
            case rocsparse_format_bsr:
                idx[0] = static_cast<I>(h.cols) + b;
                break;
            default:
                break;
            }
        }
        else if(defect == "offset_not_monotone")
        {
            // The pointer of the third row/column drops below the previous one.
            h.ptr[3] = h.ptr[2] - 1;
        }
        else if(defect == "duplicate")
        {
            // First two entries of the first row (column) become identical.
            if(h.format == rocsparse_format_coo)
            {
                h.ind[1] = h.ind[0];
            }
            else
            {
                idx[1] = idx[0];
            }
        }
        else if(defect == "unsorted")
        {
            if(h.format == rocsparse_format_coo)
            {
                std::swap(h.ind[0], h.ind[1]);
            }
            else
            {
                std::swap(idx[0], idx[1]);
            }
        }
    }

    template <typename T, typename I>
    void check_defects(rocsparse_handle handle, rocsparse_index_base base)
    {
        const std::vector<T> dense = make_dense<T>();

        struct Case
        {
            rocsparse_format      format;
            const char*           defect;
            rocsparse_data_status expected;
        };
        const std::vector<Case> cases{
            {rocsparse_format_csr, "nan", rocsparse_data_status_nan},
            {rocsparse_format_csr, "inf", rocsparse_data_status_inf},
            {rocsparse_format_csr, "index_out_of_range", rocsparse_data_status_invalid_index},
            {rocsparse_format_csr, "offset_not_monotone", rocsparse_data_status_invalid_offset_ptr},
            {rocsparse_format_csr, "duplicate", rocsparse_data_status_duplicate_entry},
            {rocsparse_format_csr, "unsorted", rocsparse_data_status_invalid_sorting},
            {rocsparse_format_csc, "nan", rocsparse_data_status_nan},
            {rocsparse_format_csc, "inf", rocsparse_data_status_inf},
            {rocsparse_format_csc, "index_out_of_range", rocsparse_data_status_invalid_index},
            {rocsparse_format_csc, "offset_not_monotone", rocsparse_data_status_invalid_offset_ptr},
            {rocsparse_format_csc, "duplicate", rocsparse_data_status_duplicate_entry},
            {rocsparse_format_csc, "unsorted", rocsparse_data_status_invalid_sorting},
            {rocsparse_format_coo, "nan", rocsparse_data_status_nan},
            {rocsparse_format_coo, "inf", rocsparse_data_status_inf},
            {rocsparse_format_coo, "index_out_of_range", rocsparse_data_status_invalid_index},
            {rocsparse_format_coo, "unsorted", rocsparse_data_status_invalid_sorting},
            {rocsparse_format_ell, "nan", rocsparse_data_status_nan},
            {rocsparse_format_ell, "inf", rocsparse_data_status_inf},
            {rocsparse_format_ell, "index_out_of_range", rocsparse_data_status_invalid_index},
            {rocsparse_format_bsr, "nan", rocsparse_data_status_nan},
            {rocsparse_format_bsr, "inf", rocsparse_data_status_inf},
            {rocsparse_format_bsr, "index_out_of_range", rocsparse_data_status_invalid_index},
            {rocsparse_format_bsr, "offset_not_monotone", rocsparse_data_status_invalid_offset_ptr},
        };

        for(const Case& c : cases)
        {
            SCOPED_TRACE(std::string(format_name(c.format)) + " defect=" + c.defect
                         + " base=" + std::to_string(static_cast<int>(base)));
            HostMat<T, I> h = build<T, I>(c.format, base, 3, rocsparse_direction_row, dense);
            apply_defect(h, c.defect);
            EXPECT_EQ((check_matrix<T, I>(handle, h)), c.expected);
        }
    }
}

TEST_F(Cov70ConvCheckSpmat, ValidFloatInt32)
{
    check_valid_formats<float, int32_t>(handle);
}

TEST_F(Cov70ConvCheckSpmat, ValidDoubleInt64)
{
    check_valid_formats<double, int64_t>(handle);
}

TEST_F(Cov70ConvCheckSpmat, ValidFloatComplexInt32)
{
    check_valid_formats<rocsparse_float_complex, int32_t>(handle);
}

TEST_F(Cov70ConvCheckSpmat, ValidDoubleComplexInt64)
{
    check_valid_formats<rocsparse_double_complex, int64_t>(handle);
}

TEST_F(Cov70ConvCheckSpmat, ValidDoubleInt32)
{
    check_valid_formats<double, int32_t>(handle);
}

TEST_F(Cov70ConvCheckSpmat, DefectsFloatInt32)
{
    check_defects<float, int32_t>(handle, rocsparse_index_base_zero);
    check_defects<float, int32_t>(handle, rocsparse_index_base_one);
}

TEST_F(Cov70ConvCheckSpmat, DefectsDoubleComplexInt64)
{
    check_defects<rocsparse_double_complex, int64_t>(handle, rocsparse_index_base_one);
}

// Invalid fill mode of a triangular / symmetric matrix: the dense matrix has
// entries on both sides of the diagonal.
TEST_F(Cov70ConvCheckSpmat, InvalidFill)
{
    const std::vector<double> dense = make_dense<double>();
    for(rocsparse_format f : {rocsparse_format_csr, rocsparse_format_csc, rocsparse_format_coo})
    {
        for(rocsparse_matrix_type type :
            {rocsparse_matrix_type_symmetric, rocsparse_matrix_type_triangular})
        {
            for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
            {
                SCOPED_TRACE(std::string(format_name(f))
                             + " type=" + std::to_string(static_cast<int>(type))
                             + " fill=" + std::to_string(static_cast<int>(fill)));
                const HostMat<double, int32_t> h = build<double, int32_t>(
                    f, rocsparse_index_base_zero, 3, rocsparse_direction_row, dense);
                Attributes attributes;
                attributes.type = type;
                attributes.fill = fill;
                EXPECT_EQ((check_matrix<double, int32_t>(handle, h, attributes)),
                          rocsparse_data_status_invalid_fill);
            }
        }
    }
}

// A matrix that declares unsorted storage may hold unsorted indices.
TEST_F(Cov70ConvCheckSpmat, UnsortedStorage)
{
    const std::vector<float> dense = make_dense<float>();
    for(rocsparse_format f : {rocsparse_format_csr, rocsparse_format_csc})
    {
        SCOPED_TRACE(format_name(f));
        HostMat<float, int32_t> h = build<float, int32_t>(
            f, rocsparse_index_base_zero, 3, rocsparse_direction_row, dense);
        apply_defect(h, "unsorted");
        Attributes attributes;
        attributes.storage = rocsparse_storage_mode_unsorted;
        EXPECT_EQ((check_matrix<float, int32_t>(handle, h, attributes)),
                  rocsparse_data_status_success);
    }
}

// Row/column pointers in 64 bit with 32 bit indices are supported, the opposite
// is not.
TEST_F(Cov70ConvCheckSpmat, MixedIndexTypes)
{
    const std::vector<float> dense = make_dense<float>();

    for(rocsparse_format f : {rocsparse_format_csr, rocsparse_format_csc, rocsparse_format_bsr})
    {
        SCOPED_TRACE(format_name(f));
        const HostMat<float, int64_t> h64 = build<float, int64_t>(
            f, rocsparse_index_base_zero, 3, rocsparse_direction_row, dense);
        const HostMat<float, int32_t> h32 = build<float, int32_t>(
            f, rocsparse_index_base_zero, 3, rocsparse_direction_row, dense);
        device_vector<int64_t> d_ptr64(h64.ptr);
        device_vector<int32_t> d_ptr32(h32.ptr);
        device_vector<int64_t> d_ind64(h64.ind);
        device_vector<int32_t> d_ind32(h32.ind);
        device_vector<float>   d_val(h32.val);
        ASSERT_TRUE(d_ptr64.ptr && d_ptr32.ptr && d_ind64.ptr && d_ind32.ptr && d_val.ptr);

        auto create = [&](SpDescr&            out,
                          void*               ptr,
                          void*               ind,
                          rocsparse_indextype pt,
                          rocsparse_indextype it) {
            if(f == rocsparse_format_csr)
            {
                return rocsparse_create_csr_descr(&out.d,
                                                  h32.rows,
                                                  h32.cols,
                                                  h32.nnz,
                                                  ptr,
                                                  ind,
                                                  d_val.ptr,
                                                  pt,
                                                  it,
                                                  h32.base,
                                                  rocsparse_datatype_f32_r);
            }
            if(f == rocsparse_format_csc)
            {
                return rocsparse_create_csc_descr(&out.d,
                                                  h32.rows,
                                                  h32.cols,
                                                  h32.nnz,
                                                  ptr,
                                                  ind,
                                                  d_val.ptr,
                                                  pt,
                                                  it,
                                                  h32.base,
                                                  rocsparse_datatype_f32_r);
            }
            return rocsparse_create_bsr_descr(&out.d,
                                              h32.rows,
                                              h32.cols,
                                              h32.nnz,
                                              h32.dir,
                                              h32.bd,
                                              ptr,
                                              ind,
                                              d_val.ptr,
                                              pt,
                                              it,
                                              h32.base,
                                              rocsparse_datatype_f32_r);
        };

        {
            SpDescr descr;
            ASSERT_EQ(create(descr,
                             d_ptr64.ptr,
                             d_ind32.ptr,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32),
                      rocsparse_status_success);
            rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
            EXPECT_EQ(run_check(handle, descr.d, Attributes{}, &ds), rocsparse_status_success);
            EXPECT_EQ(ds, rocsparse_data_status_success);
        }
        {
            SpDescr descr;
            ASSERT_EQ(create(descr,
                             d_ptr32.ptr,
                             d_ind64.ptr,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i64),
                      rocsparse_status_success);
            rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
            EXPECT_EQ(run_check(handle, descr.d, Attributes{}, &ds),
                      rocsparse_status_not_implemented);
        }
        {
            SpDescr descr;
            ASSERT_EQ(create(descr,
                             d_ptr64.ptr,
                             d_ind64.ptr,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64),
                      rocsparse_status_success);
            rocsparse_data_status ds = static_cast<rocsparse_data_status>(-1);
            EXPECT_EQ(run_check(handle, descr.d, Attributes{}, &ds), rocsparse_status_success);
            EXPECT_EQ(ds, rocsparse_data_status_success);
        }
    }
}

// Formats and value types the routine does not support, and invalid arguments.
TEST_F(Cov70ConvCheckSpmat, UnsupportedAndArguments)
{
    const std::vector<float>      dense = make_dense<float>();
    const HostMat<float, int32_t> csr   = build<float, int32_t>(
        rocsparse_format_csr, rocsparse_index_base_zero, 3, rocsparse_direction_row, dense);
    DevMat<float, int32_t> d_csr(csr);
    SpDescr                descr;
    ASSERT_EQ(create_descr(csr, d_csr, descr), rocsparse_status_success);

    // Value types that are not supported.
    for(rocsparse_datatype type : {rocsparse_datatype_i8_r,
                                   rocsparse_datatype_u8_r,
                                   rocsparse_datatype_i32_r,
                                   rocsparse_datatype_u32_r})
    {
        SCOPED_TRACE(static_cast<int>(type));
        SpDescr other;
        ASSERT_EQ(rocsparse_create_csr_descr(&other.d,
                                             csr.rows,
                                             csr.cols,
                                             csr.nnz,
                                             d_csr.ptr.ptr,
                                             d_csr.ind.ptr,
                                             d_csr.val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             csr.base,
                                             type),
                  rocsparse_status_success);
        rocsparse_data_status ds = rocsparse_data_status_success;
        EXPECT_EQ(run_check(handle, other.d, Attributes{}, &ds), rocsparse_status_not_implemented);
    }

    // Coordinate (AoS) matrices are not supported.
    {
        const HostMat<float, int32_t> aos = build<float, int32_t>(
            rocsparse_format_coo_aos, rocsparse_index_base_zero, 3, rocsparse_direction_row, dense);
        DevMat<float, int32_t> d_aos(aos);
        SpDescr                other;
        ASSERT_EQ(create_descr(aos, d_aos, other), rocsparse_status_success);
        rocsparse_data_status ds = rocsparse_data_status_success;
        EXPECT_EQ(run_check(handle, other.d, Attributes{}, &ds), rocsparse_status_not_implemented);
    }

    // Blocked ELL matrices are not supported.
    {
        device_vector<int32_t> ind(static_cast<size_t>(M * N));
        device_vector<float>   val(static_cast<size_t>(M * N));
        ASSERT_TRUE(ind.ptr && val.ptr);
        SpDescr other;
        ASSERT_EQ(rocsparse_create_bell_descr(&other.d,
                                              M,
                                              N,
                                              rocsparse_direction_row,
                                              2,
                                              2,
                                              ind.ptr,
                                              val.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_data_status ds = rocsparse_data_status_success;
        EXPECT_EQ(run_check(handle, other.d, Attributes{}, &ds), rocsparse_status_not_implemented);
    }

    // Invalid arguments.
    rocsparse_data_status ds          = rocsparse_data_status_success;
    size_t                buffer_size = 0;
    EXPECT_EQ(
        rocsparse_check_spmat(
            nullptr, descr.d, &ds, rocsparse_check_spmat_stage_buffer_size, &buffer_size, nullptr),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_check_spmat(
            handle, nullptr, &ds, rocsparse_check_spmat_stage_buffer_size, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_check_spmat(handle,
                                    descr.d,
                                    nullptr,
                                    rocsparse_check_spmat_stage_buffer_size,
                                    &buffer_size,
                                    nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_check_spmat(handle,
                                    descr.d,
                                    &ds,
                                    static_cast<rocsparse_check_spmat_stage>(99),
                                    &buffer_size,
                                    nullptr),
              rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_check_spmat(
                  handle, descr.d, &ds, rocsparse_check_spmat_stage_compute, nullptr, nullptr),
              rocsparse_status_invalid_pointer);

    // Batched matrices are not supported.
    SpDescr batched;
    ASSERT_EQ(create_descr(csr, d_csr, batched), rocsparse_status_success);
    ASSERT_EQ(rocsparse_csr_set_strided_batch(batched.d, 2, csr.rows + 1, csr.nnz),
              rocsparse_status_success);
    EXPECT_EQ(
        rocsparse_check_spmat(
            handle, batched.d, &ds, rocsparse_check_spmat_stage_buffer_size, &buffer_size, nullptr),
        rocsparse_status_not_implemented);
}
