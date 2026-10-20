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

// Conversion routines: inverse permutation, nnz counting, pruning/compression, CSR2CSC
// symbolic, CSR2BSR/ELL-width edge cases, quick returns for empty matrices and device
// pointer mode. Only documented behavior is asserted; the matrices are small and the
// expectations come from host references that do not depend on the wavefront size.

#include "unit_test_cov80_conv_common.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    using cov80conv::DevBuf;

    // RAII for the matrix descriptor.
    class Descr
    {
        rocsparse_mat_descr m_d = nullptr;

    public:
        explicit Descr(rocsparse_index_base base = rocsparse_index_base_zero)
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&m_d), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_index_base(m_d, base), rocsparse_status_success);
        }
        ~Descr()
        {
            if(m_d != nullptr)
            {
                EXPECT_EQ(rocsparse_destroy_mat_descr(m_d), rocsparse_status_success);
            }
        }
        Descr(const Descr&) = delete;
        Descr& operator=(const Descr&) = delete;
               operator rocsparse_mat_descr() const
        {
            return m_d;
        }
    };

    // Sets the pointer mode of the handle, restores the previous one on exit.
    class PointerModeScope
    {
        rocsparse_handle       m_handle;
        rocsparse_pointer_mode m_old;

    public:
        PointerModeScope(rocsparse_handle handle, rocsparse_pointer_mode mode)
            : m_handle(handle)
        {
            EXPECT_EQ(rocsparse_get_pointer_mode(handle, &m_old), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_pointer_mode(handle, mode), rocsparse_status_success);
        }
        ~PointerModeScope()
        {
            EXPECT_EQ(rocsparse_set_pointer_mode(m_handle, m_old), rocsparse_status_success);
        }
        PointerModeScope(const PointerModeScope&) = delete;
        PointerModeScope& operator=(const PointerModeScope&) = delete;
    };

    constexpr rocsparse_pointer_mode s_modes[]
        = {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device};

    struct HostCsr
    {
        rocsparse_int              m = 0, n = 0;
        int                        base = 0;
        std::vector<rocsparse_int> ptr, ind;
        std::vector<float>         val;
    };

    // m rows, 2 + (r % 3) entries per row at the columns (r + k) % n, sorted. The values
    // are 0.5 or 2 + k, so that a tolerance of 1 splits them without ambiguity.
    HostCsr make_csr(rocsparse_int m, rocsparse_int n, int base)
    {
        HostCsr a;
        a.m    = m;
        a.n    = n;
        a.base = base;
        a.ptr.push_back(base);
        for(rocsparse_int r = 0; r < m; ++r)
        {
            std::vector<std::pair<rocsparse_int, float>> row;
            const int                                    cnt = std::min<int>(2 + (r % 3), n);
            for(int k = 0; k < cnt; ++k)
            {
                row.emplace_back((r + k) % n, ((r + k) % 2 == 0) ? 2.0f + k : 0.5f);
            }
            std::sort(row.begin(), row.end());
            for(const auto& e : row)
            {
                a.ind.push_back(e.first + base);
                a.val.push_back(e.second);
            }
            a.ptr.push_back(base + static_cast<rocsparse_int>(a.ind.size()));
        }
        return a;
    }

    // ------------------------------------------------------------------------------
    // rocsparse_inverse_permutation
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvInversePermutation)
    {
        // 701 is prime, so i -> (7 i + 3) % 701 is a permutation. More than one block.
        const rocsparse_int n = 701;
        for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        {
            const rocsparse_int        b = static_cast<rocsparse_int>(base);
            std::vector<rocsparse_int> p(n), expected(n);
            for(rocsparse_int i = 0; i < n; ++i)
            {
                p[i]               = (7 * i + 3) % n + b;
                expected[p[i] - b] = i + b;
            }
            DevBuf<rocsparse_int> d_p(p), d_q(std::vector<rocsparse_int>{}, n);
            ASSERT_EQ(rocsparse_inverse_permutation(handle, n, d_p, d_q, base),
                      rocsparse_status_success);
            EXPECT_EQ(d_q.read(n), expected) << "base " << b;
        }

        // Empty permutation.
        DevBuf<rocsparse_int> d_p(std::vector<rocsparse_int>{}), d_q(std::vector<rocsparse_int>{});
        EXPECT_EQ(rocsparse_inverse_permutation(handle, 0, d_p, d_q, rocsparse_index_base_zero),
                  rocsparse_status_success);

        // Invalid arguments.
        EXPECT_EQ(rocsparse_inverse_permutation(handle, -1, d_p, d_q, rocsparse_index_base_zero),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_inverse_permutation(handle, 1, nullptr, d_q, rocsparse_index_base_zero),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_inverse_permutation(handle, 1, d_p, nullptr, rocsparse_index_base_zero),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_inverse_permutation(
                      handle, 1, d_p, d_q, static_cast<rocsparse_index_base>(2)),
                  rocsparse_status_invalid_value);
    }

    // ------------------------------------------------------------------------------
    // rocsparse_snnz
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvNnzDense)
    {
        const rocsparse_int m = 6, n = 5, ld = 7;

        std::vector<float>         A(ld * n, 0.0f);
        std::vector<rocsparse_int> rows(m, 0), cols(n, 0);
        rocsparse_int              total = 0;
        for(rocsparse_int j = 0; j < n; ++j)
        {
            for(rocsparse_int i = 0; i < m; ++i)
            {
                if((i * 3 + j * 2) % 4 == 0)
                {
                    A[i + j * ld] = 1.0f + i + j;
                    ++rows[i];
                    ++cols[j];
                    ++total;
                }
            }
        }
        ASSERT_GT(total, 0);

        DevBuf<float> d_A(A);
        Descr         descr;

        for(rocsparse_pointer_mode mode : s_modes)
        {
            PointerModeScope scope(handle, mode);
            for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
            {
                SCOPED_TRACE("device mode " + std::to_string(mode == rocsparse_pointer_mode_device)
                             + " column " + std::to_string(dir == rocsparse_direction_column));
                const auto&           expected = (dir == rocsparse_direction_row) ? rows : cols;
                DevBuf<rocsparse_int> d_cnt(std::vector<rocsparse_int>{}, expected.size());
                DevBuf<rocsparse_int> d_total(std::vector<rocsparse_int>{}, 1);
                rocsparse_int         h_total = -1;
                rocsparse_int*        t
                    = (mode == rocsparse_pointer_mode_device) ? d_total.get() : &h_total;
                ASSERT_EQ(rocsparse_snnz(handle, dir, m, n, descr, d_A, ld, d_cnt, t),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                if(mode == rocsparse_pointer_mode_device)
                {
                    h_total = d_total.read(1)[0];
                }
                EXPECT_EQ(h_total, total);
                EXPECT_EQ(d_cnt.read(expected.size()), expected);
            }
        }

        // Invalid arguments.
        DevBuf<rocsparse_int> d_cnt(std::vector<rocsparse_int>{}, 8);
        DevBuf<rocsparse_int> d_total(std::vector<rocsparse_int>{}, 1);
        rocsparse_int         h_total = 0;
        EXPECT_EQ(rocsparse_snnz(handle,
                                 rocsparse_direction_row,
                                 m,
                                 n,
                                 descr,
                                 static_cast<const float*>(nullptr),
                                 ld,
                                 d_cnt,
                                 &h_total),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_snnz(handle,
                                 rocsparse_direction_row,
                                 m,
                                 n,
                                 descr,
                                 d_A,
                                 ld,
                                 d_cnt,
                                 static_cast<rocsparse_int*>(nullptr)),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_snnz(handle, rocsparse_direction_row, -1, n, descr, d_A, ld, d_cnt, &h_total),
            rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_snnz(
                      handle, rocsparse_direction_row, m, n, descr, d_A, m - 1, d_cnt, &h_total),
                  rocsparse_status_invalid_size);
    }

    TEST_F(HandleTest, Cov80ConvNnzEmpty)
    {
        Descr               descr;
        DevBuf<float>       d_A(std::vector<float>{}, 16);
        const rocsparse_int dims[][2] = {{0, 4}, {4, 0}, {0, 0}};
        for(rocsparse_pointer_mode mode : s_modes)
        {
            PointerModeScope scope(handle, mode);
            for(const auto& d : dims)
            {
                for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
                {
                    SCOPED_TRACE("device mode "
                                 + std::to_string(mode == rocsparse_pointer_mode_device) + " m "
                                 + std::to_string(d[0]) + " n " + std::to_string(d[1]) + " column "
                                 + std::to_string(dir == rocsparse_direction_column));
                    // Sized for either direction.
                    DevBuf<rocsparse_int> d_cnt(std::vector<rocsparse_int>(8, 77));
                    DevBuf<rocsparse_int> d_total(std::vector<rocsparse_int>(1, 77));
                    rocsparse_int         h_total = 77;
                    rocsparse_int*        t
                        = (mode == rocsparse_pointer_mode_device) ? d_total.get() : &h_total;
                    ASSERT_EQ(rocsparse_snnz(handle,
                                             dir,
                                             d[0],
                                             d[1],
                                             descr,
                                             d_A,
                                             std::max<rocsparse_int>(d[0], 1),
                                             d_cnt,
                                             t),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                    if(mode == rocsparse_pointer_mode_device)
                    {
                        h_total = d_total.read(1)[0];
                    }
                    EXPECT_EQ(h_total, 0);
                    const rocsparse_int count = (dir == rocsparse_direction_row) ? d[0] : d[1];
                    for(rocsparse_int v : d_cnt.read(count))
                    {
                        EXPECT_EQ(v, 0);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------------
    // rocsparse_sprune_dense2csr_nnz / rocsparse_sprune_dense2csr
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvPruneDense2Csr)
    {
        const rocsparse_int m = 9, n = 7, ld = 11;
        const float         threshold = 1.0f;
        for(int base : {0, 1})
        {
            // The entries are 0, 0.5, 0.25 (pruned) or 2..4 (kept).
            std::vector<float> A(ld * n, 0.0f);
            HostCsr            expected;
            expected.m    = m;
            expected.n    = n;
            expected.base = base;
            expected.ptr.push_back(base);
            for(rocsparse_int i = 0; i < m; ++i)
            {
                for(rocsparse_int j = 0; j < n; ++j)
                {
                    const int k = (i * 3 + j * 5) % 7;
                    float     v = 0.0f;
                    if(k == 1)
                        v = 0.5f;
                    else if(k == 2)
                        v = 0.25f;
                    else if(k > 2)
                        v = 2.0f + (i + j) % 3;
                    A[i + j * ld] = v;
                    if(v > threshold)
                    {
                        expected.ind.push_back(j + base);
                        expected.val.push_back(v);
                    }
                }
                expected.ptr.push_back(base + static_cast<rocsparse_int>(expected.ind.size()));
            }
            ASSERT_GT(expected.ind.size(), 0u);

            Descr               descr(static_cast<rocsparse_index_base>(base));
            DevBuf<float>       d_A(A);
            const rocsparse_int nnz = static_cast<rocsparse_int>(expected.ind.size());

            for(rocsparse_pointer_mode mode : s_modes)
            {
                SCOPED_TRACE("base " + std::to_string(base) + " device mode "
                             + std::to_string(mode == rocsparse_pointer_mode_device));
                PointerModeScope scope(handle, mode);

                DevBuf<float> d_threshold(std::vector<float>{threshold});
                const float*  thr
                    = (mode == rocsparse_pointer_mode_device) ? d_threshold.get() : &threshold;

                DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>{}, m + 1);
                size_t                buffer_size = 0;
                ASSERT_EQ(
                    rocsparse_sprune_dense2csr_buffer_size(
                        handle, m, n, d_A, ld, thr, descr, nullptr, d_ptr, nullptr, &buffer_size),
                    rocsparse_status_success);
                DevBuf<char>          d_buffer(std::vector<char>{}, buffer_size);
                DevBuf<rocsparse_int> d_total(std::vector<rocsparse_int>{}, 1);
                rocsparse_int         h_total = -1;
                rocsparse_int*        t
                    = (mode == rocsparse_pointer_mode_device) ? d_total.get() : &h_total;
                ASSERT_EQ(rocsparse_sprune_dense2csr_nnz(
                              handle, m, n, d_A, ld, thr, descr, d_ptr, t, d_buffer),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                if(mode == rocsparse_pointer_mode_device)
                {
                    h_total = d_total.read(1)[0];
                }
                ASSERT_EQ(h_total, nnz);
                EXPECT_EQ(d_ptr.read(m + 1), expected.ptr);

                DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{}, nnz);
                DevBuf<float>         d_val(std::vector<float>{}, nnz);
                ASSERT_EQ(rocsparse_sprune_dense2csr(
                              handle, m, n, d_A, ld, thr, descr, d_val, d_ptr, d_ind, d_buffer),
                          rocsparse_status_success);
                EXPECT_EQ(d_ind.read(nnz), expected.ind);
                EXPECT_EQ(d_val.read(nnz), expected.val);

                // The output arrays are required when there are entries.
                EXPECT_EQ(rocsparse_sprune_dense2csr(handle,
                                                     m,
                                                     n,
                                                     d_A,
                                                     ld,
                                                     thr,
                                                     descr,
                                                     static_cast<float*>(nullptr),
                                                     d_ptr,
                                                     d_ind,
                                                     d_buffer),
                          rocsparse_status_invalid_pointer);
                EXPECT_EQ(rocsparse_sprune_dense2csr(handle,
                                                     m,
                                                     n,
                                                     d_A,
                                                     ld,
                                                     thr,
                                                     descr,
                                                     d_val,
                                                     d_ptr,
                                                     static_cast<rocsparse_int*>(nullptr),
                                                     d_buffer),
                          rocsparse_status_invalid_pointer);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvPruneDense2CsrEmpty)
    {
        const rocsparse_int dims[][2] = {{0, 5}, {4, 0}};
        for(int base : {0, 1})
        {
            Descr         descr(static_cast<rocsparse_index_base>(base));
            DevBuf<float> d_A(std::vector<float>{}, 16);
            for(rocsparse_pointer_mode mode : s_modes)
            {
                PointerModeScope scope(handle, mode);
                for(const auto& d : dims)
                {
                    SCOPED_TRACE("base " + std::to_string(base) + " device mode "
                                 + std::to_string(mode == rocsparse_pointer_mode_device) + " m "
                                 + std::to_string(d[0]) + " n " + std::to_string(d[1]));
                    const float   threshold = 1.0f;
                    DevBuf<float> d_threshold(std::vector<float>{threshold});
                    const float*  thr
                        = (mode == rocsparse_pointer_mode_device) ? d_threshold.get() : &threshold;
                    DevBuf<char>          d_buffer(std::vector<char>{}, 64);
                    DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(8, 77));
                    DevBuf<rocsparse_int> d_total(std::vector<rocsparse_int>(1, 77));
                    rocsparse_int         h_total = 77;
                    rocsparse_int*        t
                        = (mode == rocsparse_pointer_mode_device) ? d_total.get() : &h_total;
                    ASSERT_EQ(rocsparse_sprune_dense2csr_nnz(handle,
                                                             d[0],
                                                             d[1],
                                                             d_A,
                                                             std::max<rocsparse_int>(d[0], 1),
                                                             thr,
                                                             descr,
                                                             d_ptr,
                                                             t,
                                                             d_buffer),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                    if(mode == rocsparse_pointer_mode_device)
                    {
                        h_total = d_total.read(1)[0];
                    }
                    EXPECT_EQ(h_total, 0);
                    for(rocsparse_int v : d_ptr.read(d[0] + 1))
                    {
                        EXPECT_EQ(v, base);
                    }

                    DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{});
                    DevBuf<float>         d_val(std::vector<float>{});
                    EXPECT_EQ(rocsparse_sprune_dense2csr(handle,
                                                         d[0],
                                                         d[1],
                                                         d_A,
                                                         std::max<rocsparse_int>(d[0], 1),
                                                         thr,
                                                         descr,
                                                         d_val,
                                                         d_ptr,
                                                         d_ind,
                                                         d_buffer),
                              rocsparse_status_success);
                }
            }
        }
    }

    // ------------------------------------------------------------------------------
    // rocsparse_snnz_compress / rocsparse_scsr2csr_compress
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvCsr2CsrCompress)
    {
        const float         tol = 1.0f;
        const rocsparse_int m = 13, n = 9;
        for(int base : {0, 1})
        {
            const HostCsr a = make_csr(m, n, base);

            // Entries with a magnitude above the tolerance are kept.
            HostCsr                    c;
            std::vector<rocsparse_int> per_row(m, 0);
            c.ptr.push_back(base);
            for(rocsparse_int r = 0; r < m; ++r)
            {
                for(rocsparse_int e = a.ptr[r] - base; e < a.ptr[r + 1] - base; ++e)
                {
                    if(std::abs(a.val[e]) > tol)
                    {
                        c.ind.push_back(a.ind[e]);
                        c.val.push_back(a.val[e]);
                        ++per_row[r];
                    }
                }
                c.ptr.push_back(base + static_cast<rocsparse_int>(c.ind.size()));
            }
            const rocsparse_int nnz_c = static_cast<rocsparse_int>(c.ind.size());
            ASSERT_GT(nnz_c, 0);
            ASSERT_LT(nnz_c, static_cast<rocsparse_int>(a.ind.size()));

            Descr                 descr(static_cast<rocsparse_index_base>(base));
            DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind);
            DevBuf<float>         d_val(a.val);

            for(rocsparse_pointer_mode mode : s_modes)
            {
                SCOPED_TRACE("base " + std::to_string(base) + " device mode "
                             + std::to_string(mode == rocsparse_pointer_mode_device));
                PointerModeScope scope(handle, mode);

                DevBuf<rocsparse_int> d_per_row(std::vector<rocsparse_int>{}, m);
                DevBuf<rocsparse_int> d_total(std::vector<rocsparse_int>{}, 1);
                rocsparse_int         h_total = -1;
                rocsparse_int*        t
                    = (mode == rocsparse_pointer_mode_device) ? d_total.get() : &h_total;
                ASSERT_EQ(
                    rocsparse_snnz_compress(handle, m, descr, d_val, d_ptr, d_per_row, t, tol),
                    rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                if(mode == rocsparse_pointer_mode_device)
                {
                    h_total = d_total.read(1)[0];
                }
                ASSERT_EQ(h_total, nnz_c);
                EXPECT_EQ(d_per_row.read(m), per_row);

                DevBuf<rocsparse_int> d_ptr_c(std::vector<rocsparse_int>{}, m + 1);
                DevBuf<rocsparse_int> d_ind_c(std::vector<rocsparse_int>{}, nnz_c);
                DevBuf<float>         d_val_c(std::vector<float>{}, nnz_c);
                ASSERT_EQ(rocsparse_scsr2csr_compress(handle,
                                                      m,
                                                      n,
                                                      descr,
                                                      d_val,
                                                      d_ptr,
                                                      d_ind,
                                                      static_cast<rocsparse_int>(a.ind.size()),
                                                      d_per_row,
                                                      d_val_c,
                                                      d_ptr_c,
                                                      d_ind_c,
                                                      tol),
                          rocsparse_status_success);
                EXPECT_EQ(d_ptr_c.read(m + 1), c.ptr);
                EXPECT_EQ(d_ind_c.read(nnz_c), c.ind);
                EXPECT_EQ(d_val_c.read(nnz_c), c.val);

                // The compressed arrays are required when entries are kept.
                EXPECT_EQ(rocsparse_scsr2csr_compress(handle,
                                                      m,
                                                      n,
                                                      descr,
                                                      d_val,
                                                      d_ptr,
                                                      d_ind,
                                                      static_cast<rocsparse_int>(a.ind.size()),
                                                      d_per_row,
                                                      static_cast<float*>(nullptr),
                                                      d_ptr_c,
                                                      d_ind_c,
                                                      tol),
                          rocsparse_status_invalid_pointer);
                EXPECT_EQ(rocsparse_scsr2csr_compress(handle,
                                                      m,
                                                      n,
                                                      descr,
                                                      d_val,
                                                      d_ptr,
                                                      d_ind,
                                                      static_cast<rocsparse_int>(a.ind.size()),
                                                      d_per_row,
                                                      d_val_c,
                                                      d_ptr_c,
                                                      static_cast<rocsparse_int*>(nullptr),
                                                      tol),
                          rocsparse_status_invalid_pointer);
            }

            // A tolerance above every magnitude keeps nothing.
            {
                const float           big = 100.0f;
                DevBuf<rocsparse_int> d_per_row(std::vector<rocsparse_int>{}, m);
                rocsparse_int         h_total = -1;
                ASSERT_EQ(rocsparse_snnz_compress(
                              handle, m, descr, d_val, d_ptr, d_per_row, &h_total, big),
                          rocsparse_status_success);
                EXPECT_EQ(h_total, 0);
                DevBuf<rocsparse_int> d_ptr_c(std::vector<rocsparse_int>{}, m + 1);
                EXPECT_EQ(rocsparse_scsr2csr_compress(handle,
                                                      m,
                                                      n,
                                                      descr,
                                                      d_val,
                                                      d_ptr,
                                                      d_ind,
                                                      static_cast<rocsparse_int>(a.ind.size()),
                                                      d_per_row,
                                                      static_cast<float*>(nullptr),
                                                      d_ptr_c,
                                                      static_cast<rocsparse_int*>(nullptr),
                                                      big),
                          rocsparse_status_success);
                for(rocsparse_int v : d_ptr_c.read(m + 1))
                {
                    EXPECT_EQ(v, base);
                }
            }
        }

        // A negative tolerance is invalid.
        Descr                 descr;
        const HostCsr         a = make_csr(m, n, 0);
        DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind),
            d_per_row(std::vector<rocsparse_int>{}, m);
        DevBuf<float> d_val(a.val);
        rocsparse_int h_total = 0;
        EXPECT_EQ(
            rocsparse_snnz_compress(handle, m, descr, d_val, d_ptr, d_per_row, &h_total, -1.0f),
            rocsparse_status_invalid_value);
    }

    TEST_F(HandleTest, Cov80ConvCsr2CsrCompressEmpty)
    {
        Descr                 descr;
        DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(8, 0)),
            d_ind(std::vector<rocsparse_int>{}, 8);
        DevBuf<rocsparse_int> d_per_row(std::vector<rocsparse_int>(8, 0));
        DevBuf<rocsparse_int> d_ptr_c(std::vector<rocsparse_int>(8, 0)),
            d_ind_c(std::vector<rocsparse_int>{}, 8);
        DevBuf<float>       d_val(std::vector<float>{}, 8), d_val_c(std::vector<float>{}, 8);
        const rocsparse_int dims[][2] = {{0, 4}, {4, 0}};
        for(const auto& d : dims)
        {
            SCOPED_TRACE("m " + std::to_string(d[0]) + " n " + std::to_string(d[1]));
            EXPECT_EQ(rocsparse_scsr2csr_compress(handle,
                                                  d[0],
                                                  d[1],
                                                  descr,
                                                  d_val,
                                                  d_ptr,
                                                  d_ind,
                                                  0,
                                                  d_per_row,
                                                  d_val_c,
                                                  d_ptr_c,
                                                  d_ind_c,
                                                  1.0f),
                      rocsparse_status_success);
        }
        rocsparse_int h_total = -1;
        EXPECT_EQ(
            rocsparse_snnz_compress(handle, 0, descr, d_val, d_ptr, d_per_row, &h_total, 1.0f),
            rocsparse_status_success);
        EXPECT_EQ(h_total, 0);
    }

    // ------------------------------------------------------------------------------
    // rocsparse_csr2ell_width
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvCsr2EllWidth)
    {
        const HostCsr a        = make_csr(11, 8, 0);
        rocsparse_int expected = 0;
        for(rocsparse_int r = 0; r < a.m; ++r)
        {
            expected = std::max(expected, a.ptr[r + 1] - a.ptr[r]);
        }
        Descr                 csr_descr, ell_descr;
        DevBuf<rocsparse_int> d_ptr(a.ptr);
        for(rocsparse_pointer_mode mode : s_modes)
        {
            SCOPED_TRACE("device mode " + std::to_string(mode == rocsparse_pointer_mode_device));
            PointerModeScope      scope(handle, mode);
            DevBuf<rocsparse_int> d_width(std::vector<rocsparse_int>(1, 77));
            rocsparse_int         h_width = 77;
            rocsparse_int* w = (mode == rocsparse_pointer_mode_device) ? d_width.get() : &h_width;
            ASSERT_EQ(rocsparse_csr2ell_width(handle, a.m, csr_descr, d_ptr, ell_descr, w),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            if(mode == rocsparse_pointer_mode_device)
            {
                h_width = d_width.read(1)[0];
            }
            EXPECT_EQ(h_width, expected);

            // No rows, no width.
            DevBuf<rocsparse_int> d_width0(std::vector<rocsparse_int>(1, 77));
            rocsparse_int         h_width0 = 77;
            rocsparse_int*        w0
                = (mode == rocsparse_pointer_mode_device) ? d_width0.get() : &h_width0;
            ASSERT_EQ(rocsparse_csr2ell_width(handle, 0, csr_descr, d_ptr, ell_descr, w0),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            if(mode == rocsparse_pointer_mode_device)
            {
                h_width0 = d_width0.read(1)[0];
            }
            EXPECT_EQ(h_width0, 0);
        }
    }

    // ------------------------------------------------------------------------------
    // rocsparse_csr2csc, symbolic action
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvCsr2CscSymbolic)
    {
        const rocsparse_int m = 17, n = 10;
        for(int base : {0, 1})
        {
            const HostCsr       a   = make_csr(m, n, base);
            const rocsparse_int nnz = static_cast<rocsparse_int>(a.ind.size());

            // Host transpose: column pointers and the row indices sorted by row.
            std::vector<rocsparse_int> cptr(n + 1, base), rind(nnz);
            for(rocsparse_int e = 0; e < nnz; ++e)
            {
                ++cptr[a.ind[e] - base + 1];
            }
            for(rocsparse_int c = 0; c < n; ++c)
            {
                cptr[c + 1] += cptr[c] - base;
            }
            std::vector<rocsparse_int> next(cptr.begin(), cptr.end() - 1);
            for(rocsparse_int r = 0; r < m; ++r)
            {
                for(rocsparse_int e = a.ptr[r] - base; e < a.ptr[r + 1] - base; ++e)
                {
                    rind[next[a.ind[e] - base]++ - base] = r + base;
                }
            }

            DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind);
            DevBuf<float>         d_val(a.val);
            for(rocsparse_action action : {rocsparse_action_symbolic, rocsparse_action_numeric})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " numeric "
                             + std::to_string(action == rocsparse_action_numeric));
                size_t size = 0;
                ASSERT_EQ(
                    rocsparse_csr2csc_buffer_size(handle, m, n, nnz, d_ptr, d_ind, action, &size),
                    rocsparse_status_success);
                DevBuf<char>          d_buffer(std::vector<char>{}, size);
                DevBuf<rocsparse_int> d_cptr(std::vector<rocsparse_int>{}, n + 1);
                DevBuf<rocsparse_int> d_rind(std::vector<rocsparse_int>{}, nnz);
                DevBuf<float>         d_cval(std::vector<float>{}, nnz);
                ASSERT_EQ(rocsparse_scsr2csc(handle,
                                             m,
                                             n,
                                             nnz,
                                             d_val,
                                             d_ptr,
                                             d_ind,
                                             d_cval,
                                             d_rind,
                                             d_cptr,
                                             action,
                                             static_cast<rocsparse_index_base>(base),
                                             d_buffer),
                          rocsparse_status_success);
                EXPECT_EQ(d_cptr.read(n + 1), cptr);
                EXPECT_EQ(d_rind.read(nnz), rind);

                // The temporary buffer is required.
                EXPECT_EQ(rocsparse_scsr2csc(handle,
                                             m,
                                             n,
                                             nnz,
                                             d_val,
                                             d_ptr,
                                             d_ind,
                                             d_cval,
                                             d_rind,
                                             d_cptr,
                                             action,
                                             static_cast<rocsparse_index_base>(base),
                                             nullptr),
                          rocsparse_status_invalid_pointer);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvCsr2CscEmpty)
    {
        for(int base : {0, 1})
        {
            for(rocsparse_action action : {rocsparse_action_symbolic, rocsparse_action_numeric})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " numeric "
                             + std::to_string(action == rocsparse_action_numeric));
                const rocsparse_int m = 4, n = 5;
                // No entries: the column pointers are all the index base.
                DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(m + 1, base));
                DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{});
                DevBuf<float>         d_val(std::vector<float>{});
                size_t                size = 123;
                ASSERT_EQ(
                    rocsparse_csr2csc_buffer_size(handle, m, n, 0, d_ptr, d_ind, action, &size),
                    rocsparse_status_success);
                DevBuf<char>          d_buffer(std::vector<char>{}, size);
                DevBuf<rocsparse_int> d_cptr(std::vector<rocsparse_int>(n + 1, 77));
                DevBuf<rocsparse_int> d_rind(std::vector<rocsparse_int>{});
                DevBuf<float>         d_cval(std::vector<float>{});
                ASSERT_EQ(rocsparse_scsr2csc(handle,
                                             m,
                                             n,
                                             0,
                                             d_val,
                                             d_ptr,
                                             d_ind,
                                             d_cval,
                                             d_rind,
                                             d_cptr,
                                             action,
                                             static_cast<rocsparse_index_base>(base),
                                             d_buffer),
                          rocsparse_status_success);
                for(rocsparse_int v : d_cptr.read(n + 1))
                {
                    EXPECT_EQ(v, base);
                }

                // No rows or no columns.
                size = 123;
                ASSERT_EQ(
                    rocsparse_csr2csc_buffer_size(handle, 0, n, 0, d_ptr, d_ind, action, &size),
                    rocsparse_status_success);
                ASSERT_EQ(rocsparse_scsr2csc(handle,
                                             0,
                                             n,
                                             0,
                                             d_val,
                                             d_ptr,
                                             d_ind,
                                             d_cval,
                                             d_rind,
                                             d_cptr,
                                             action,
                                             static_cast<rocsparse_index_base>(base),
                                             d_buffer),
                          rocsparse_status_success);
            }
        }
    }

    // ------------------------------------------------------------------------------
    // rocsparse_csr2bsr_nnz / rocsparse_scsr2bsr edge cases
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvCsr2BsrEdges)
    {
        const rocsparse_int m = 6, n = 6;
        for(int base : {0, 1})
        {
            Descr csr_descr(static_cast<rocsparse_index_base>(base));
            Descr bsr_descr(static_cast<rocsparse_index_base>(base));

            // An empty matrix: no block, and all the block row pointers are the base.
            for(rocsparse_int block_dim : {1, 2, 4})
            {
                for(rocsparse_pointer_mode mode : s_modes)
                {
                    SCOPED_TRACE("base " + std::to_string(base) + " block_dim "
                                 + std::to_string(block_dim) + " device mode "
                                 + std::to_string(mode == rocsparse_pointer_mode_device));
                    PointerModeScope      scope(handle, mode);
                    const rocsparse_int   mb = (m + block_dim - 1) / block_dim;
                    DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(m + 1, base));
                    DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{});
                    DevBuf<rocsparse_int> d_bptr(std::vector<rocsparse_int>(mb + 1, 77));
                    DevBuf<rocsparse_int> d_bnnz(std::vector<rocsparse_int>(1, 77));
                    rocsparse_int         h_bnnz = 77;
                    rocsparse_int*        t
                        = (mode == rocsparse_pointer_mode_device) ? d_bnnz.get() : &h_bnnz;
                    ASSERT_EQ(rocsparse_csr2bsr_nnz(handle,
                                                    rocsparse_direction_row,
                                                    m,
                                                    n,
                                                    csr_descr,
                                                    d_ptr,
                                                    d_ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    d_bptr,
                                                    t),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                    if(mode == rocsparse_pointer_mode_device)
                    {
                        h_bnnz = d_bnnz.read(1)[0];
                    }
                    EXPECT_EQ(h_bnnz, 0);
                    for(rocsparse_int v : d_bptr.read(mb + 1))
                    {
                        EXPECT_EQ(v, base);
                    }

                    DevBuf<float>         d_val(std::vector<float>{});
                    DevBuf<float>         d_bval(std::vector<float>{});
                    DevBuf<rocsparse_int> d_bind(std::vector<rocsparse_int>{});
                    EXPECT_EQ(rocsparse_scsr2bsr(handle,
                                                 rocsparse_direction_row,
                                                 m,
                                                 n,
                                                 csr_descr,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 block_dim,
                                                 bsr_descr,
                                                 d_bval,
                                                 d_bptr,
                                                 d_bind),
                              rocsparse_status_success);
                }
            }

            // No rows or no columns.
            for(const auto& d : {std::pair<rocsparse_int, rocsparse_int>(0, 4),
                                 std::pair<rocsparse_int, rocsparse_int>(4, 0)})
            {
                for(rocsparse_pointer_mode mode : s_modes)
                {
                    SCOPED_TRACE("base " + std::to_string(base) + " m " + std::to_string(d.first)
                                 + " n " + std::to_string(d.second) + " device mode "
                                 + std::to_string(mode == rocsparse_pointer_mode_device));
                    PointerModeScope      scope(handle, mode);
                    const rocsparse_int   block_dim = 2;
                    const rocsparse_int   mb        = (d.first + block_dim - 1) / block_dim;
                    DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(8, base));
                    DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{});
                    DevBuf<rocsparse_int> d_bptr(std::vector<rocsparse_int>(8, 77));
                    DevBuf<rocsparse_int> d_bnnz(std::vector<rocsparse_int>(1, 77));
                    rocsparse_int         h_bnnz = 77;
                    rocsparse_int*        t
                        = (mode == rocsparse_pointer_mode_device) ? d_bnnz.get() : &h_bnnz;
                    ASSERT_EQ(rocsparse_csr2bsr_nnz(handle,
                                                    rocsparse_direction_row,
                                                    d.first,
                                                    d.second,
                                                    csr_descr,
                                                    d_ptr,
                                                    d_ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    d_bptr,
                                                    t),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                    if(mode == rocsparse_pointer_mode_device)
                    {
                        h_bnnz = d_bnnz.read(1)[0];
                    }
                    EXPECT_EQ(h_bnnz, 0);
                    for(rocsparse_int v : d_bptr.read(mb + 1))
                    {
                        EXPECT_EQ(v, base);
                    }

                    DevBuf<float>         d_val(std::vector<float>{}), d_bval(std::vector<float>{});
                    DevBuf<rocsparse_int> d_bind(std::vector<rocsparse_int>{});
                    EXPECT_EQ(rocsparse_scsr2bsr(handle,
                                                 rocsparse_direction_row,
                                                 d.first,
                                                 d.second,
                                                 csr_descr,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 block_dim,
                                                 bsr_descr,
                                                 d_bval,
                                                 d_bptr,
                                                 d_bind),
                              rocsparse_status_success);
                }
            }

            // With entries, the column indices and values of the CSR matrix are required.
            {
                const HostCsr         a         = make_csr(m, n, base);
                const rocsparse_int   block_dim = 2;
                const rocsparse_int   mb        = (m + block_dim - 1) / block_dim;
                DevBuf<rocsparse_int> d_ptr(a.ptr), d_ind(a.ind);
                DevBuf<float>         d_val(a.val);
                DevBuf<rocsparse_int> d_bptr(std::vector<rocsparse_int>{}, mb + 1);
                rocsparse_int         bnnz = 0;
                EXPECT_EQ(rocsparse_csr2bsr_nnz(handle,
                                                rocsparse_direction_row,
                                                m,
                                                n,
                                                csr_descr,
                                                d_ptr,
                                                static_cast<const rocsparse_int*>(nullptr),
                                                block_dim,
                                                bsr_descr,
                                                d_bptr,
                                                &bnnz),
                          rocsparse_status_invalid_pointer);
                ASSERT_EQ(rocsparse_csr2bsr_nnz(handle,
                                                rocsparse_direction_row,
                                                m,
                                                n,
                                                csr_descr,
                                                d_ptr,
                                                d_ind,
                                                block_dim,
                                                bsr_descr,
                                                d_bptr,
                                                &bnnz),
                          rocsparse_status_success);
                ASSERT_GT(bnnz, 0);
                DevBuf<rocsparse_int> d_bind(std::vector<rocsparse_int>{}, bnnz);
                DevBuf<float>         d_bval(std::vector<float>{}, bnnz * block_dim * block_dim);
                EXPECT_EQ(rocsparse_scsr2bsr(handle,
                                             rocsparse_direction_row,
                                             m,
                                             n,
                                             csr_descr,
                                             static_cast<const float*>(nullptr),
                                             d_ptr,
                                             d_ind,
                                             block_dim,
                                             bsr_descr,
                                             d_bval,
                                             d_bptr,
                                             d_bind),
                          rocsparse_status_invalid_pointer);
                EXPECT_EQ(rocsparse_scsr2bsr(handle,
                                             rocsparse_direction_row,
                                             m,
                                             n,
                                             csr_descr,
                                             d_val,
                                             d_ptr,
                                             static_cast<const rocsparse_int*>(nullptr),
                                             block_dim,
                                             bsr_descr,
                                             d_bval,
                                             d_bptr,
                                             d_bind),
                          rocsparse_status_invalid_pointer);
            }
        }
    }
}
