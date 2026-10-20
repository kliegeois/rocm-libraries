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

// Shared helpers for the unit_test_device_cov70_conv_*.cpp tests: RAII wrappers
// for the rocSPARSE descriptors and small host reference conversions.

#pragma once

#include "unit_test_utils.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace cov70conv
{
    // Value with the given real and imaginary parts (the imaginary part is ignored
    // for real types).
    template <typename T>
    T make_val(int re, int im)
    {
        if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        {
            return rocsparse_float_complex(static_cast<float>(re), static_cast<float>(im));
        }
        else if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_double_complex(static_cast<double>(re), static_cast<double>(im));
        }
        else
        {
            return static_cast<T>(re);
        }
    }

    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        explicit MatDescr(rocsparse_index_base base = rocsparse_index_base_zero)
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&d), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_index_base(d, base), rocsparse_status_success);
        }
        ~MatDescr()
        {
            if(d)
            {
                EXPECT_EQ(rocsparse_destroy_mat_descr(d), rocsparse_status_success);
            }
        }
        MatDescr(const MatDescr&) = delete;
        MatDescr& operator=(const MatDescr&) = delete;
                  operator rocsparse_mat_descr() const
        {
            return d;
        }
    };

    struct MatInfo
    {
        rocsparse_mat_info i = nullptr;
        MatInfo()
        {
            EXPECT_EQ(rocsparse_create_mat_info(&i), rocsparse_status_success);
        }
        ~MatInfo()
        {
            if(i)
            {
                EXPECT_EQ(rocsparse_destroy_mat_info(i), rocsparse_status_success);
            }
        }
        MatInfo(const MatInfo&) = delete;
        MatInfo& operator=(const MatInfo&) = delete;
    };

    // Dense host matrix (column major) -> block compressed rows (gebsr).
    template <typename T>
    struct HostGebsr
    {
        rocsparse_direction dir;
        int                 mb   = 0;
        int                 nb   = 0;
        int                 rbd  = 0;
        int                 cbd  = 0;
        int                 base = 0;
        std::vector<int>    row_ptr;
        std::vector<int>    col_ind;
        std::vector<T>      val;
    };

    // Host reference gebsr -> csr (any block direction, any base). Rows are
    // emitted with ascending column indices.
    template <typename T>
    void host_gebsr2csr(const HostGebsr<T>& g,
                        int                 csr_base,
                        std::vector<int>&   row_ptr,
                        std::vector<int>&   col_ind,
                        std::vector<T>&     val)
    {
        const int m = g.mb * g.rbd;
        row_ptr.assign(static_cast<size_t>(m) + 1, 0);
        col_ind.clear();
        val.clear();
        for(int i = 0; i < m; ++i)
        {
            const int ib                    = i / g.rbd;
            const int ii                    = i % g.rbd;
            row_ptr[static_cast<size_t>(i)] = static_cast<int>(col_ind.size()) + csr_base;
            for(int k = g.row_ptr[static_cast<size_t>(ib)] - g.base;
                k < g.row_ptr[static_cast<size_t>(ib) + 1] - g.base;
                ++k)
            {
                const int jb = g.col_ind[static_cast<size_t>(k)] - g.base;
                for(int jj = 0; jj < g.cbd; ++jj)
                {
                    const size_t off = (g.dir == rocsparse_direction_row)
                                           ? static_cast<size_t>(k) * g.rbd * g.cbd
                                                 + static_cast<size_t>(ii) * g.cbd + jj
                                           : static_cast<size_t>(k) * g.rbd * g.cbd
                                                 + static_cast<size_t>(jj) * g.rbd + ii;
                    col_ind.push_back(jb * g.cbd + jj + csr_base);
                    val.push_back(g.val[off]);
                }
            }
        }
        row_ptr[static_cast<size_t>(m)] = static_cast<int>(col_ind.size()) + csr_base;
    }

    // Deterministic, nonzero and (almost always) distinct values.
    inline float pattern_value(size_t i)
    {
        return 1.0f + static_cast<float>((i * 7) % 61) * 0.25f;
    }
} // namespace cov70conv
