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
#include <functional>
#include <type_traits>
#include <vector>

namespace cov70conv
{
    // Destroys a descriptor owned by a local variable when the scope is left, also
    // when an ASSERT_* returns early. Declare it right after the variable:
    //     rocsparse_spmat_descr mat = nullptr;
    //     SpmatGuard            mat_guard(mat);
    // The destroy status is checked. A guarded variable that is null (never created
    // or explicitly destroyed and reset) is skipped.
    template <auto DESTROY>
    class Guard
    {
        std::function<void()> m_cleanup;

    public:
        template <typename V>
        explicit Guard(V& ref)
            : m_cleanup([&ref]() {
                if(ref != nullptr)
                {
                    EXPECT_EQ(DESTROY(ref), rocsparse_status_success);
                    ref = nullptr;
                }
            })
        {
        }
        ~Guard()
        {
            m_cleanup();
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };

    inline rocsparse_status destroy_spmat(rocsparse_const_spmat_descr d)
    {
        return rocsparse_destroy_spmat_descr(const_cast<rocsparse_spmat_descr>(d));
    }
    inline rocsparse_status destroy_spvec(rocsparse_const_spvec_descr d)
    {
        return rocsparse_destroy_spvec_descr(const_cast<rocsparse_spvec_descr>(d));
    }
    inline rocsparse_status destroy_dnmat(rocsparse_const_dnmat_descr d)
    {
        return rocsparse_destroy_dnmat_descr(const_cast<rocsparse_dnmat_descr>(d));
    }
    inline rocsparse_status destroy_dnvec(rocsparse_const_dnvec_descr d)
    {
        return rocsparse_destroy_dnvec_descr(const_cast<rocsparse_dnvec_descr>(d));
    }
    inline rocsparse_status destroy_color(rocsparse_color_info d)
    {
        return rocsparse_destroy_color_info(d);
    }
    inline rocsparse_status destroy_s2s(rocsparse_sparse_to_sparse_descr d)
    {
        return rocsparse_destroy_sparse_to_sparse_descr(d);
    }

    using SpmatGuard = Guard<destroy_spmat>;
    using SpvecGuard = Guard<destroy_spvec>;
    using DnmatGuard = Guard<destroy_dnmat>;
    using DnvecGuard = Guard<destroy_dnvec>;
    using ColorGuard = Guard<destroy_color>;
    using S2sGuard   = Guard<destroy_s2s>;

    // Owns a rocsparse_spsort_descr (its destruction needs the handle).
    struct SpsortDescr
    {
        rocsparse_handle       handle = nullptr;
        rocsparse_spsort_descr d      = nullptr;
        explicit SpsortDescr(rocsparse_handle h)
            : handle(h)
        {
        }
        ~SpsortDescr()
        {
            if(d != nullptr)
            {
                EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, d, nullptr),
                          rocsparse_status_success);
            }
        }
        SpsortDescr(const SpsortDescr&) = delete;
        SpsortDescr& operator=(const SpsortDescr&) = delete;
                     operator rocsparse_spsort_descr() const
        {
            return d;
        }
    };
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

    // Host copy of a general block compressed sparse row (gebsr) matrix: block row
    // pointer, block column indices and values. Each block holds rbd x cbd values,
    // stored row major for rocsparse_direction_row and column major otherwise.
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

    // Deterministic nonzero values that are exactly representable in float. The
    // sequence has period 61, i.e. only 61 distinct values (1.0 .. 16.0), so it is
    // not suitable to tell apart two positions that are a multiple of 61 apart.
    inline float pattern_value(size_t i)
    {
        return 1.0f + static_cast<float>((i * 7) % 61) * 0.25f;
    }
} // namespace cov70conv
