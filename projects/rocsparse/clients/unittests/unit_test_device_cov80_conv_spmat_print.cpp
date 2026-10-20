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

// rocsparse::internal_spmat_print writes a description of a sparse matrix
// descriptor (header fields, then optionally the index arrays and the values) to
// a std::ostream. Nothing in the library calls it and its symbol is hidden in
// librocsparse, so the defining translation unit is compiled into the device unit
// test binary (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES); it is not reachable
// through the public API.
//
// The output is not a documented format. The tests therefore check the content
// that follows from the descriptor (matrix sizes, nnz, the printed arrays) and do
// not depend on the spacing of the header fields.

#include "unit_test_cov80_conv_common.hpp"

#include "rocsparse_internal_spmat_print.hpp"

#include <algorithm>
#include <cctype>
#include <complex>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cov80conv;

namespace
{
    struct Printed
    {
        rocsparse_status                   status = rocsparse_status_success;
        std::map<std::string, std::string> fields; // "- key : value" lines
        std::vector<std::string>           body; // every other line
    };

    std::string trim(const std::string& s)
    {
        size_t b = 0, e = s.size();
        while(b < e && std::isspace(static_cast<unsigned char>(s[b])))
            ++b;
        while(e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
            --e;
        return s.substr(b, e - b);
    }

    Printed print(rocsparse_const_spmat_descr d, bool symbolic, bool numeric)
    {
        Printed            p;
        std::ostringstream os;
        p.status = rocsparse::internal_spmat_print(os, d, symbolic, numeric);
        std::istringstream is(os.str());
        std::string        line;
        while(std::getline(is, line))
        {
            if(line.compare(0, 2, "- ") == 0)
            {
                const size_t colon = line.find(':');
                if(colon == std::string::npos)
                {
                    ADD_FAILURE() << "header line without a colon: " << line;
                    continue;
                }
                p.fields[trim(line.substr(2, colon - 2))] = trim(line.substr(colon + 1));
            }
            else
            {
                p.body.push_back(line);
            }
        }
        return p;
    }

    // "[i] = v" lines of one printed array.
    template <typename T>
    std::vector<std::string> array_lines(const std::vector<T>& v)
    {
        std::vector<std::string> out;
        for(size_t i = 0; i < v.size(); ++i)
        {
            std::ostringstream os;
            os << "[" << i << "] = " << v[i];
            out.push_back(os.str());
        }
        return out;
    }

    // Rows of a printed column-major m x w matrix with leading dimension m.
    template <typename T>
    std::vector<std::string> matrix_rows(const std::vector<T>& v, int m, int w)
    {
        std::vector<std::string> out;
        for(int i = 0; i < m; ++i)
        {
            std::ostringstream os;
            for(int j = 0; j < w; ++j)
            {
                os << " " << v[j * m + i];
            }
            out.push_back(os.str());
        }
        return out;
    }

    void append(std::vector<std::string>& dst, const std::vector<std::string>& src)
    {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    bool contains(const std::string& s, const char* what)
    {
        std::string a = s, b = what;
        std::transform(
            a.begin(), a.end(), a.begin(), [](unsigned char c) { return std::tolower(c); });
        std::transform(
            b.begin(), b.end(), b.begin(), [](unsigned char c) { return std::tolower(c); });
        return a.find(b) != std::string::npos;
    }

    const std::pair<bool, bool> s_combos[]
        = {{false, false}, {true, false}, {false, true}, {true, true}};

    // The 3 x 4 matrix used by csr, csc, coo and coo_aos:
    //   [1 0 0 2]
    //   [0 3 0 0]
    //   [4 0 5 0]
    const std::vector<float> s_val = {1, 2, 3, 4, 5};

    template <typename T>
    std::vector<T> as(const std::vector<int>& v)
    {
        return std::vector<T>(v.begin(), v.end());
    }

    // Shifts the indices by the index base.
    std::vector<int> shifted(std::vector<int> v, int base)
    {
        for(int& x : v)
            x += base;
        return v;
    }

    TEST(Cov80ConvSpmatPrint, Csr)
    {
        for(int base : {0, 1})
        {
            const std::vector<int32_t> ptr = as<int32_t>(shifted({0, 2, 3, 5}, base));
            const std::vector<int32_t> ind = as<int32_t>(shifted({0, 3, 1, 0, 2}, base));
            DevBuf<int32_t>            d_ptr(ptr), d_ind(ind);
            DevBuf<float>              d_val(s_val);

            rocsparse_spmat_descr descr = nullptr;
            SpmatGuard            guard(descr);
            ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                                 3,
                                                 4,
                                                 5,
                                                 d_ptr,
                                                 d_ind,
                                                 d_val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 base == 0 ? rocsparse_index_base_zero
                                                           : rocsparse_index_base_one,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);

            for(const auto& c : s_combos)
            {
                SCOPED_TRACE("base " + std::to_string(base) + " symbolic " + std::to_string(c.first)
                             + " numeric " + std::to_string(c.second));
                const Printed p = print(descr, c.first, c.second);
                ASSERT_EQ(p.status, rocsparse_status_success);
                EXPECT_TRUE(contains(p.fields.at("format"), "csr"));
                EXPECT_EQ(p.fields.at("m"), "3");
                EXPECT_EQ(p.fields.at("n"), "4");
                EXPECT_EQ(p.fields.at("nnz"), "5");
                EXPECT_EQ(p.fields.at("base"), std::to_string(base));
                EXPECT_TRUE(contains(p.fields.at("ptr_type"), "i32"));
                EXPECT_TRUE(contains(p.fields.at("ind_type"), "i32"));
                EXPECT_TRUE(contains(p.fields.at("data_type"), "f32"));

                std::vector<std::string> expected;
                if(c.first)
                {
                    append(expected, array_lines(ptr));
                    append(expected, array_lines(ind));
                }
                if(c.second)
                {
                    append(expected, array_lines(s_val));
                }
                EXPECT_EQ(p.body, expected);
            }
        }
    }

    TEST(Cov80ConvSpmatPrint, Csc)
    {
        // The same matrix stored by columns: col_ptr has n + 1 entries.
        const std::vector<int32_t> ptr = {1, 3, 4, 5, 6};
        const std::vector<int32_t> ind = {1, 3, 2, 3, 1};
        const std::vector<float>   val = {1, 4, 3, 5, 2};
        DevBuf<int32_t>            d_ptr(ptr), d_ind(ind);
        DevBuf<float>              d_val(val);

        rocsparse_spmat_descr descr = nullptr;
        SpmatGuard            guard(descr);
        ASSERT_EQ(rocsparse_create_csc_descr(&descr,
                                             3,
                                             4,
                                             5,
                                             d_ptr,
                                             d_ind,
                                             d_val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_one,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        for(const auto& c : s_combos)
        {
            SCOPED_TRACE("symbolic " + std::to_string(c.first) + " numeric "
                         + std::to_string(c.second));
            const Printed p = print(descr, c.first, c.second);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_TRUE(contains(p.fields.at("format"), "csc"));
            EXPECT_EQ(p.fields.at("m"), "3");
            EXPECT_EQ(p.fields.at("n"), "4");
            EXPECT_EQ(p.fields.at("nnz"), "5");
            EXPECT_EQ(p.fields.at("base"), "1");

            std::vector<std::string> expected;
            if(c.first)
            {
                append(expected, array_lines(ptr));
                append(expected, array_lines(ind));
            }
            if(c.second)
            {
                append(expected, array_lines(val));
            }
            EXPECT_EQ(p.body, expected);
        }
    }

    TEST(Cov80ConvSpmatPrint, Coo)
    {
        const std::vector<int32_t> row = {0, 0, 1, 2, 2};
        const std::vector<int32_t> col = {0, 3, 1, 0, 2};
        DevBuf<int32_t>            d_row(row), d_col(col);
        DevBuf<float>              d_val(s_val);

        rocsparse_spmat_descr descr = nullptr;
        SpmatGuard            guard(descr);
        ASSERT_EQ(rocsparse_create_coo_descr(&descr,
                                             3,
                                             4,
                                             5,
                                             d_row,
                                             d_col,
                                             d_val,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        for(const auto& c : s_combos)
        {
            SCOPED_TRACE("symbolic " + std::to_string(c.first) + " numeric "
                         + std::to_string(c.second));
            const Printed p = print(descr, c.first, c.second);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_TRUE(contains(p.fields.at("format"), "coo"));
            EXPECT_EQ(p.fields.at("m"), "3");
            EXPECT_EQ(p.fields.at("n"), "4");
            EXPECT_EQ(p.fields.at("nnz"), "5");
            EXPECT_EQ(p.fields.at("base"), "0");
            EXPECT_TRUE(contains(p.fields.at("row_type"), "i32"));
            EXPECT_TRUE(contains(p.fields.at("col_type"), "i32"));

            std::vector<std::string> expected;
            if(c.first)
            {
                append(expected, array_lines(row));
                append(expected, array_lines(col));
            }
            if(c.second)
            {
                append(expected, array_lines(s_val));
            }
            EXPECT_EQ(p.body, expected);
        }
    }

    TEST(Cov80ConvSpmatPrint, CooAos)
    {
        // (row, col) pairs.
        const std::vector<int32_t> ind = {0, 0, 0, 3, 1, 1, 2, 0, 2, 2};
        DevBuf<int32_t>            d_ind(ind);
        DevBuf<float>              d_val(s_val);

        rocsparse_spmat_descr descr = nullptr;
        SpmatGuard            guard(descr);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                                 3,
                                                 4,
                                                 5,
                                                 d_ind,
                                                 d_val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        for(const auto& c : s_combos)
        {
            SCOPED_TRACE("symbolic " + std::to_string(c.first) + " numeric "
                         + std::to_string(c.second));
            const Printed p = print(descr, c.first, c.second);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_EQ(p.fields.at("m"), "3");
            EXPECT_EQ(p.fields.at("n"), "4");
            EXPECT_EQ(p.fields.at("nnz"), "5");

            std::vector<std::string> expected;
            if(c.first)
            {
                append(expected, array_lines(ind));
            }
            if(c.second)
            {
                append(expected, array_lines(s_val));
            }
            EXPECT_EQ(p.body, expected);
        }
    }

    TEST(Cov80ConvSpmatPrint, Ell)
    {
        // Column-major: entry (i, j) of the 3 x 2 arrays is at j * 3 + i. A padding
        // entry has the column index -1.
        const std::vector<int32_t> ind = {0, 1, 0, 3, -1, 2};
        const std::vector<float>   val = {1, 3, 4, 2, 0, 5};
        DevBuf<int32_t>            d_ind(ind);
        DevBuf<float>              d_val(val);

        rocsparse_spmat_descr descr = nullptr;
        SpmatGuard            guard(descr);
        ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                             3,
                                             4,
                                             d_ind,
                                             d_val,
                                             2,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        for(const auto& c : s_combos)
        {
            SCOPED_TRACE("symbolic " + std::to_string(c.first) + " numeric "
                         + std::to_string(c.second));
            const Printed p = print(descr, c.first, c.second);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_TRUE(contains(p.fields.at("format"), "ell"));
            EXPECT_EQ(p.fields.at("m"), "3");
            EXPECT_EQ(p.fields.at("n"), "4");
            EXPECT_EQ(p.fields.at("ell width"), "2");

            std::vector<std::string> expected;
            if(c.first)
            {
                append(expected, matrix_rows(ind, 3, 2));
            }
            if(c.second)
            {
                append(expected, matrix_rows(val, 3, 2));
            }
            EXPECT_EQ(p.body, expected);
        }
    }

    TEST(Cov80ConvSpmatPrint, Bsr)
    {
        // 2 x 2 blocks of size 2, three of them stored.
        const std::vector<int32_t> ptr = {0, 2, 3};
        const std::vector<int32_t> ind = {0, 1, 1};
        std::vector<float>         val(12);
        for(size_t i = 0; i < val.size(); ++i)
            val[i] = static_cast<float>(i + 1);
        DevBuf<int32_t> d_ptr(ptr), d_ind(ind);
        DevBuf<float>   d_val(val);

        rocsparse_spmat_descr descr = nullptr;
        SpmatGuard            guard(descr);
        ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                             2,
                                             2,
                                             3,
                                             rocsparse_direction_column,
                                             2,
                                             d_ptr,
                                             d_ind,
                                             d_val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        for(const auto& c : s_combos)
        {
            SCOPED_TRACE("symbolic " + std::to_string(c.first) + " numeric "
                         + std::to_string(c.second));
            const Printed p = print(descr, c.first, c.second);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_TRUE(contains(p.fields.at("format"), "bsr"));
            EXPECT_EQ(p.fields.at("mb"), "2");
            EXPECT_EQ(p.fields.at("nb"), "2");
            EXPECT_EQ(p.fields.at("nnzb"), "3");
            EXPECT_EQ(p.fields.at("dimb"), "2");
            EXPECT_EQ(p.fields.at("m"), "4");
            EXPECT_EQ(p.fields.at("n"), "4");
            EXPECT_EQ(p.fields.at("nnz"), "12");
            EXPECT_TRUE(contains(p.fields.at("dirb"), "column"));

            std::vector<std::string> expected;
            if(c.first)
            {
                append(expected, array_lines(ptr));
                append(expected, array_lines(ind));
            }
            if(c.second)
            {
                append(expected, array_lines(val));
            }
            EXPECT_EQ(p.body, expected);
        }
    }

    TEST(Cov80ConvSpmatPrint, BlockedEll)
    {
        // The printer reads rows * width index entries and width * rows * dim^2 values
        // for a blocked ELL matrix, which is more than the arrays of the matrix hold
        // for a block size above one. The buffers are therefore over-allocated here and
        // only the part that follows from the descriptor is checked: the header fields
        // and the leading values.
        constexpr int        rows = 4, cols = 6, dim = 2, width = 4;
        std::vector<int32_t> ind = {0, 2, 1, 2};
        std::vector<float>   val(rows * width);
        for(size_t i = 0; i < val.size(); ++i)
            val[i] = static_cast<float>(i + 1);
        DevBuf<int32_t> d_ind(ind, rows * width);
        DevBuf<float>   d_val(val, rows * width * dim * dim);

        rocsparse_spmat_descr descr = nullptr;
        SpmatGuard            guard(descr);
        ASSERT_EQ(rocsparse_create_bell_descr(&descr,
                                              rows,
                                              cols,
                                              rocsparse_direction_row,
                                              dim,
                                              width,
                                              d_ind,
                                              d_val,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        for(const auto& c : s_combos)
        {
            SCOPED_TRACE("symbolic " + std::to_string(c.first) + " numeric "
                         + std::to_string(c.second));
            const Printed p = print(descr, c.first, c.second);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_TRUE(contains(p.fields.at("format"), "bell"));
            EXPECT_EQ(p.fields.at("m"), std::to_string(rows));
            EXPECT_EQ(p.fields.at("n"), std::to_string(cols));
            EXPECT_EQ(p.fields.at("width"), std::to_string(width));
            EXPECT_EQ(p.fields.at("dimb"), std::to_string(dim));
            EXPECT_TRUE(contains(p.fields.at("dirb"), "row"));

            if(!c.first && !c.second)
            {
                EXPECT_TRUE(p.body.empty());
            }
            if(c.first)
            {
                // One printed row per matrix row.
                EXPECT_GE(p.body.size(), size_t(rows));
            }
            if(c.second && !c.first)
            {
                // The value array holds rows * width entries.
                ASSERT_GE(p.body.size(), val.size());
                const std::vector<std::string> head = array_lines(val);
                EXPECT_TRUE(std::equal(head.begin(), head.end(), p.body.begin()));
            }
        }
    }

    TEST(Cov80ConvSpmatPrint, Sell)
    {
        // Only the header is printed for a SELL matrix, whatever the flags.
        const std::vector<int32_t> offsets = {0, 4, 8};
        const std::vector<int32_t> ind(8, 0);
        const std::vector<float>   val(8, 1.0f);
        DevBuf<int32_t>            d_off(offsets), d_ind(ind);
        DevBuf<float>              d_val(val);

        rocsparse_spmat_descr descr = nullptr;
        SpmatGuard            guard(descr);
        ASSERT_EQ(rocsparse_create_sell_descr(&descr,
                                              4,
                                              4,
                                              6,
                                              2,
                                              8,
                                              d_off,
                                              d_ind,
                                              d_val,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        for(const auto& c : s_combos)
        {
            SCOPED_TRACE("symbolic " + std::to_string(c.first) + " numeric "
                         + std::to_string(c.second));
            const Printed p = print(descr, c.first, c.second);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_TRUE(contains(p.fields.at("format"), "sell"));
            EXPECT_EQ(p.fields.at("m"), "4");
            EXPECT_EQ(p.fields.at("n"), "4");
            EXPECT_EQ(p.fields.at("nnz"), "6");
            EXPECT_EQ(p.fields.at("sell_slice_size"), "2");
            EXPECT_EQ(p.fields.at("sell_colval_size"), "8");
            EXPECT_TRUE(p.body.empty());
        }
    }

    // ------------------------------------------------------------------
    // Value types. The printed value of the small integers 1..4 is the integer
    // itself for every type; complex values print as the host stream does.
    // ------------------------------------------------------------------

    template <typename T>
    std::string text(T v)
    {
        std::ostringstream os;
        os << v;
        return os.str();
    }

    struct TypedValues
    {
        rocsparse_datatype       type;
        std::vector<uint8_t>     raw; // the four values, in the device layout
        std::vector<std::string> printed; // their printed form
        bool                     check_text; // false for 8-bit types (printed as characters)
    };

    template <typename T>
    void push_value(TypedValues& tv, T v)
    {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        tv.raw.insert(tv.raw.end(), b, b + sizeof(T));
        tv.printed.push_back(text(v));
    }

    void push_bits16(TypedValues& tv, uint16_t bits, int printed)
    {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&bits);
        tv.raw.insert(tv.raw.end(), b, b + sizeof(bits));
        tv.printed.push_back(std::to_string(printed));
    }

    std::vector<TypedValues> typed_values()
    {
        std::vector<TypedValues> all;
        {
            // IEEE half: 1, 2, 3, 4.
            TypedValues    tv{rocsparse_datatype_f16_r, {}, {}, true};
            const uint16_t bits[4] = {0x3C00, 0x4000, 0x4200, 0x4400};
            for(int i = 0; i < 4; ++i)
                push_bits16(tv, bits[i], i + 1);
            all.push_back(tv);
        }
        {
            // bfloat16: 1, 2, 3, 4.
            TypedValues    tv{rocsparse_datatype_bf16_r, {}, {}, true};
            const uint16_t bits[4] = {0x3F80, 0x4000, 0x4040, 0x4080};
            for(int i = 0; i < 4; ++i)
                push_bits16(tv, bits[i], i + 1);
            all.push_back(tv);
        }
        {
            TypedValues tv{rocsparse_datatype_f32_r, {}, {}, true};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, static_cast<float>(i));
            all.push_back(tv);
        }
        {
            TypedValues tv{rocsparse_datatype_f64_r, {}, {}, true};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, static_cast<double>(i));
            all.push_back(tv);
        }
        {
            TypedValues tv{rocsparse_datatype_f32_c, {}, {}, true};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, rocsparse_float_complex(static_cast<float>(i), -1.0f));
            all.push_back(tv);
        }
        {
            TypedValues tv{rocsparse_datatype_f64_c, {}, {}, true};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, rocsparse_double_complex(static_cast<double>(i), -1.0));
            all.push_back(tv);
        }
        {
            TypedValues tv{rocsparse_datatype_i32_r, {}, {}, true};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, static_cast<int32_t>(i));
            all.push_back(tv);
        }
        {
            TypedValues tv{rocsparse_datatype_u32_r, {}, {}, true};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, static_cast<uint32_t>(i));
            all.push_back(tv);
        }
        {
            // 8-bit values are streamed as characters; only the layout is checked.
            TypedValues tv{rocsparse_datatype_i8_r, {}, {}, false};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, static_cast<int8_t>(64 + i));
            all.push_back(tv);
        }
        {
            TypedValues tv{rocsparse_datatype_u8_r, {}, {}, false};
            for(int i = 1; i <= 4; ++i)
                push_value(tv, static_cast<uint8_t>(64 + i));
            all.push_back(tv);
        }
        return all;
    }

    // Lines "[i] = v" for the printed form of the four values.
    std::vector<std::string> typed_lines(const TypedValues& tv)
    {
        std::vector<std::string> out;
        for(size_t i = 0; i < tv.printed.size(); ++i)
            out.push_back("[" + std::to_string(i) + "] = " + tv.printed[i]);
        return out;
    }

    TEST(Cov80ConvSpmatPrint, ValueTypesCsr)
    {
        // 1 x 4 row of ones at the columns 0..3; the values are the four numbers.
        const std::vector<int32_t> ptr = {0, 4};
        const std::vector<int32_t> ind = {0, 1, 2, 3};
        DevBuf<int32_t>            d_ptr(ptr), d_ind(ind);

        for(const TypedValues& tv : typed_values())
        {
            SCOPED_TRACE("data type " + std::to_string(static_cast<int>(tv.type)));
            DevBuf<uint8_t> d_val(tv.raw);

            rocsparse_spmat_descr descr = nullptr;
            SpmatGuard            guard(descr);
            ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                                 1,
                                                 4,
                                                 4,
                                                 d_ptr,
                                                 d_ind,
                                                 d_val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 tv.type),
                      rocsparse_status_success);
            const Printed p = print(descr, false, true);
            ASSERT_EQ(p.status, rocsparse_status_success);
            ASSERT_EQ(p.body.size(), size_t(4));
            if(tv.check_text)
            {
                EXPECT_EQ(p.body, typed_lines(tv));
            }
            else
            {
                for(size_t i = 0; i < 4; ++i)
                {
                    EXPECT_EQ(p.body[i].rfind("[" + std::to_string(i) + "] = ", 0), size_t(0));
                }
            }
        }
    }

    TEST(Cov80ConvSpmatPrint, ValueTypesEll)
    {
        // 2 x 2 ELL matrix, width 2: the four values are the entries.
        const std::vector<int32_t> ind = {0, 1, 1, 0};
        DevBuf<int32_t>            d_ind(ind);

        for(const TypedValues& tv : typed_values())
        {
            SCOPED_TRACE("data type " + std::to_string(static_cast<int>(tv.type)));
            DevBuf<uint8_t> d_val(tv.raw);

            rocsparse_spmat_descr descr = nullptr;
            SpmatGuard            guard(descr);
            ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                                 2,
                                                 2,
                                                 d_ind,
                                                 d_val,
                                                 2,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 tv.type),
                      rocsparse_status_success);
            const Printed p = print(descr, false, true);
            ASSERT_EQ(p.status, rocsparse_status_success);
            ASSERT_EQ(p.body.size(), size_t(2));
            if(tv.check_text)
            {
                // Entry (i, j) is value number j * 2 + i.
                EXPECT_EQ(p.body[0], " " + tv.printed[0] + " " + tv.printed[2]);
                EXPECT_EQ(p.body[1], " " + tv.printed[1] + " " + tv.printed[3]);
            }
        }
    }

    // ------------------------------------------------------------------
    // Index types.
    // ------------------------------------------------------------------

    TEST(Cov80ConvSpmatPrint, IndexTypesCsr)
    {
        const std::vector<int> ptr = {0, 2, 3, 5};
        const std::vector<int> ind = {0, 3, 1, 0, 2};
        for(rocsparse_indextype pt : {rocsparse_indextype_i32, rocsparse_indextype_i64})
        {
            for(rocsparse_indextype it : {rocsparse_indextype_i32, rocsparse_indextype_i64})
            {
                SCOPED_TRACE("ptr type " + std::to_string(static_cast<int>(pt)) + " ind type "
                             + std::to_string(static_cast<int>(it)));
                const auto make = [](rocsparse_indextype t, const std::vector<int>& v) {
                    std::vector<uint8_t> raw;
                    for(int x : v)
                    {
                        if(t == rocsparse_indextype_i32)
                        {
                            const int32_t y = x;
                            raw.insert(raw.end(),
                                       reinterpret_cast<const uint8_t*>(&y),
                                       reinterpret_cast<const uint8_t*>(&y) + sizeof(y));
                        }
                        else
                        {
                            const int64_t y = x;
                            raw.insert(raw.end(),
                                       reinterpret_cast<const uint8_t*>(&y),
                                       reinterpret_cast<const uint8_t*>(&y) + sizeof(y));
                        }
                    }
                    return raw;
                };
                DevBuf<uint8_t> d_ptr(make(pt, ptr)), d_ind(make(it, ind));
                DevBuf<float>   d_val(s_val);

                rocsparse_spmat_descr descr = nullptr;
                SpmatGuard            guard(descr);
                ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                                     3,
                                                     4,
                                                     5,
                                                     d_ptr,
                                                     d_ind,
                                                     d_val,
                                                     pt,
                                                     it,
                                                     rocsparse_index_base_zero,
                                                     rocsparse_datatype_f32_r),
                          rocsparse_status_success);
                const Printed p = print(descr, true, true);
                ASSERT_EQ(p.status, rocsparse_status_success);
                std::vector<std::string> expected = array_lines(ptr);
                append(expected, array_lines(ind));
                append(expected, array_lines(s_val));
                EXPECT_EQ(p.body, expected);
                EXPECT_TRUE(contains(p.fields.at("ptr_type"),
                                     pt == rocsparse_indextype_i32 ? "i32" : "i64"));
                EXPECT_TRUE(contains(p.fields.at("ind_type"),
                                     it == rocsparse_indextype_i32 ? "i32" : "i64"));
            }
        }
    }

    TEST(Cov80ConvSpmatPrint, IndexType64CooEllBsr)
    {
        const std::vector<int64_t> row = {0, 0, 1, 2, 2};
        const std::vector<int64_t> col = {0, 3, 1, 0, 2};
        {
            DevBuf<int64_t>       d_row(row), d_col(col);
            DevBuf<float>         d_val(s_val);
            rocsparse_spmat_descr descr = nullptr;
            SpmatGuard            guard(descr);
            ASSERT_EQ(rocsparse_create_coo_descr(&descr,
                                                 3,
                                                 4,
                                                 5,
                                                 d_row,
                                                 d_col,
                                                 d_val,
                                                 rocsparse_indextype_i64,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            const Printed p = print(descr, true, false);
            ASSERT_EQ(p.status, rocsparse_status_success);
            std::vector<std::string> expected = array_lines(row);
            append(expected, array_lines(col));
            EXPECT_EQ(p.body, expected);
        }
        {
            const std::vector<int64_t> ell_ind = {0, 1, 0, 3, -1, 2};
            DevBuf<int64_t>            d_ind(ell_ind);
            DevBuf<float>              d_val(std::vector<float>(6, 1.0f));
            rocsparse_spmat_descr      descr = nullptr;
            SpmatGuard                 guard(descr);
            ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                                 3,
                                                 4,
                                                 d_ind,
                                                 d_val,
                                                 2,
                                                 rocsparse_indextype_i64,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            const Printed p = print(descr, true, false);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_EQ(p.body, matrix_rows(ell_ind, 3, 2));
        }
        {
            const std::vector<int64_t> ptr = {0, 1, 2};
            const std::vector<int64_t> ind = {0, 1};
            DevBuf<int64_t>            d_ptr(ptr), d_ind(ind);
            DevBuf<float>              d_val(std::vector<float>(2, 1.0f));
            rocsparse_spmat_descr      descr = nullptr;
            SpmatGuard                 guard(descr);
            ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                                 2,
                                                 2,
                                                 2,
                                                 rocsparse_direction_row,
                                                 1,
                                                 d_ptr,
                                                 d_ind,
                                                 d_val,
                                                 rocsparse_indextype_i64,
                                                 rocsparse_indextype_i64,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            const Printed p = print(descr, true, true);
            ASSERT_EQ(p.status, rocsparse_status_success);
            std::vector<std::string> expected = array_lines(ptr);
            append(expected, array_lines(ind));
            append(expected, array_lines(std::vector<float>(2, 1.0f)));
            EXPECT_EQ(p.body, expected);
        }
    }

    // ------------------------------------------------------------------
    // Empty arrays and invalid input.
    // ------------------------------------------------------------------

    TEST(Cov80ConvSpmatPrint, EmptyArrays)
    {
        // A CSR matrix without entries: only the row pointer is printed.
        const std::vector<int32_t> ptr(4, 0);
        DevBuf<int32_t>            d_ptr(ptr), d_ind(std::vector<int32_t>{});
        DevBuf<float>              d_val(std::vector<float>{});
        {
            rocsparse_spmat_descr descr = nullptr;
            SpmatGuard            guard(descr);
            ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                                 3,
                                                 4,
                                                 0,
                                                 d_ptr,
                                                 d_ind,
                                                 d_val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            const Printed p = print(descr, true, true);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_EQ(p.fields.at("nnz"), "0");
            EXPECT_EQ(p.body, array_lines(ptr));
        }
        {
            // An ELL matrix of width zero prints no matrix.
            rocsparse_spmat_descr descr = nullptr;
            SpmatGuard            guard(descr);
            ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                                 3,
                                                 4,
                                                 d_ind,
                                                 d_val,
                                                 0,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            const Printed p = print(descr, true, true);
            ASSERT_EQ(p.status, rocsparse_status_success);
            EXPECT_EQ(p.fields.at("ell width"), "0");
            EXPECT_TRUE(p.body.empty());
        }
    }

    TEST(Cov80ConvSpmatPrint, NullDescriptor)
    {
        std::ostringstream os;
        COV80_EXPECT_REJECTED(rocsparse::internal_spmat_print(os, nullptr, true, true));
        EXPECT_TRUE(os.str().empty());
    }
}
