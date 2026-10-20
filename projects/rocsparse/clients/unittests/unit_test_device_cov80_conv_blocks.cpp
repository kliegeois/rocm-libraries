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

// Block and ELL conversions: bsrpad_value (sorted and unsorted storage), gebsr2gebsc
// (symbolic and numeric, empty matrices), bsr2csr, ell2csr (host and device pointer
// mode, both index bases) and hyb2csr of an empty HYB matrix.

#include "unit_test_cov80_conv_common.hpp"

#include <algorithm>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    using cov80conv::DevBuf;

    rocsparse_index_base to_base(int base)
    {
        return static_cast<rocsparse_index_base>(base);
    }

    class Descr
    {
        rocsparse_mat_descr m_d = nullptr;

    public:
        explicit Descr(int base = 0, rocsparse_storage_mode storage = rocsparse_storage_mode_sorted)
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&m_d), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_index_base(m_d, to_base(base)), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_storage_mode(m_d, storage), rocsparse_status_success);
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

    // ------------------------------------------------------------------------------
    // rocsparse_sbsrpad_value
    // ------------------------------------------------------------------------------

    // m = 5 rows in 3 block rows of 2: the last block row has one row outside the
    // matrix. The block rows hold the block columns {0}, {1} and {0, 2} (or {2, 0}).
    TEST_F(HandleTest, Cov80ConvBsrPadValue)
    {
        const rocsparse_int m = 5, mb = 3, block_dim = 2;
        const float         value = 7.0f;
        for(int base : {0, 1})
        {
            for(rocsparse_storage_mode storage :
                {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted})
            {
                const bool unsorted = (storage == rocsparse_storage_mode_unsorted);
                SCOPED_TRACE("base " + std::to_string(base) + " unsorted "
                             + std::to_string(unsorted));

                const std::vector<rocsparse_int> ptr  = {base, base + 1, base + 2, base + 4};
                std::vector<rocsparse_int>       ind  = {base + 0,
                                                  base + 1,
                                                  unsorted ? base + 2 : base + 0,
                                                  unsorted ? base + 0 : base + 2};
                const rocsparse_int              nnzb = 4;

                Descr                    descr(base, storage);
                DevBuf<rocsparse_int>    d_ptr(ptr), d_ind(ind);
                const std::vector<float> val(nnzb * block_dim * block_dim, 1.0f);
                DevBuf<float>            d_val(val);

                ASSERT_EQ(rocsparse_sbsrpad_value(
                              handle, m, mb, nnzb, block_dim, value, descr, d_val, d_ptr, d_ind),
                          rocsparse_status_success);

                // Only the entry (1, 1) of the diagonal block of the last block row is set.
                std::vector<float> expected   = val;
                const int          diag_block = (unsorted ? 2 : 3);
                expected[diag_block * block_dim * block_dim + 1 * block_dim + 1] = value;
                EXPECT_EQ(d_val.read(val.size()), expected);

                // The values are required.
                EXPECT_EQ(rocsparse_sbsrpad_value(handle,
                                                  m,
                                                  mb,
                                                  nnzb,
                                                  block_dim,
                                                  value,
                                                  descr,
                                                  static_cast<float*>(nullptr),
                                                  d_ptr,
                                                  d_ind),
                          rocsparse_status_invalid_pointer);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvBsrPadValueNothingToPad)
    {
        Descr                 descr;
        const rocsparse_int   block_dim = 2;
        DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>{0, 1, 2, 3}),
            d_ind(std::vector<rocsparse_int>{0, 1, 2});
        const std::vector<float> val(3 * block_dim * block_dim, 1.0f);
        DevBuf<float>            d_val(val);

        // m = mb * block_dim: no row outside the matrix.
        EXPECT_EQ(
            rocsparse_sbsrpad_value(handle, 6, 3, 3, block_dim, 7.0f, descr, d_val, d_ptr, d_ind),
            rocsparse_status_success);
        EXPECT_EQ(d_val.read(val.size()), val);

        // No block rows, no blocks.
        EXPECT_EQ(
            rocsparse_sbsrpad_value(handle, 0, 0, 0, block_dim, 7.0f, descr, d_val, d_ptr, d_ind),
            rocsparse_status_success);
        EXPECT_EQ(
            rocsparse_sbsrpad_value(handle, 5, 3, 0, block_dim, 7.0f, descr, d_val, d_ptr, d_ind),
            rocsparse_status_success);
        EXPECT_EQ(d_val.read(val.size()), val);
    }

    // ------------------------------------------------------------------------------
    // rocsparse_sgebsr2gebsc
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvGebsr2Gebsc)
    {
        const rocsparse_int mb = 3, nb = 4, rdim = 2, cdim = 3;
        for(int base : {0, 1})
        {
            // Block row r holds the block columns 0 .. r + 1.
            std::vector<rocsparse_int> ptr = {base}, ind;
            for(rocsparse_int r = 0; r < mb; ++r)
            {
                for(rocsparse_int c = 0; c <= r + 1; ++c)
                {
                    ind.push_back(base + c);
                }
                ptr.push_back(base + static_cast<rocsparse_int>(ind.size()));
            }
            const rocsparse_int nnzb = static_cast<rocsparse_int>(ind.size());

            // Host transpose of the block pattern.
            std::vector<rocsparse_int> cptr(nb + 1, base), rind(nnzb);
            for(rocsparse_int e = 0; e < nnzb; ++e)
            {
                ++cptr[ind[e] - base + 1];
            }
            for(rocsparse_int c = 0; c < nb; ++c)
            {
                cptr[c + 1] += cptr[c] - base;
            }
            std::vector<rocsparse_int> next(cptr.begin(), cptr.end() - 1);
            for(rocsparse_int r = 0; r < mb; ++r)
            {
                for(rocsparse_int e = ptr[r] - base; e < ptr[r + 1] - base; ++e)
                {
                    rind[next[ind[e] - base]++ - base] = r + base;
                }
            }

            DevBuf<rocsparse_int>    d_ptr(ptr), d_ind(ind);
            const std::vector<float> val(nnzb * rdim * cdim, 1.0f);
            DevBuf<float>            d_val(val);
            for(rocsparse_action action : {rocsparse_action_symbolic, rocsparse_action_numeric})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " numeric "
                             + std::to_string(action == rocsparse_action_numeric));
                size_t size = 0;
                ASSERT_EQ(rocsparse_sgebsr2gebsc_buffer_size(
                              handle, mb, nb, nnzb, d_val, d_ptr, d_ind, rdim, cdim, &size),
                          rocsparse_status_success);
                DevBuf<char>          d_buffer(std::vector<char>{}, size);
                DevBuf<rocsparse_int> d_cptr(std::vector<rocsparse_int>{}, nb + 1);
                DevBuf<rocsparse_int> d_rind(std::vector<rocsparse_int>{}, nnzb);
                DevBuf<float>         d_cval(std::vector<float>{}, val.size());
                ASSERT_EQ(rocsparse_sgebsr2gebsc(handle,
                                                 mb,
                                                 nb,
                                                 nnzb,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 rdim,
                                                 cdim,
                                                 d_cval,
                                                 d_rind,
                                                 d_cptr,
                                                 action,
                                                 to_base(base),
                                                 d_buffer),
                          rocsparse_status_success);
                EXPECT_EQ(d_cptr.read(nb + 1), cptr);
                EXPECT_EQ(d_rind.read(nnzb), rind);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvGebsr2GebscEmpty)
    {
        const rocsparse_int rdim = 2, cdim = 3;
        for(int base : {0, 1})
        {
            for(const auto& d : {std::pair<rocsparse_int, rocsparse_int>(3, 4),
                                 std::pair<rocsparse_int, rocsparse_int>(0, 4),
                                 std::pair<rocsparse_int, rocsparse_int>(3, 0)})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " mb " + std::to_string(d.first)
                             + " nb " + std::to_string(d.second));
                const rocsparse_int mb = d.first, nb = d.second;
                // No blocks at all.
                DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(mb + 1, base));
                DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{});
                DevBuf<float>         d_val(std::vector<float>{});
                size_t                size = 0;
                ASSERT_EQ(rocsparse_sgebsr2gebsc_buffer_size(
                              handle, mb, nb, 0, d_val, d_ptr, d_ind, rdim, cdim, &size),
                          rocsparse_status_success);
                DevBuf<char>          d_buffer(std::vector<char>{}, size);
                DevBuf<rocsparse_int> d_cptr(std::vector<rocsparse_int>(nb + 1, 77));
                DevBuf<rocsparse_int> d_rind(std::vector<rocsparse_int>{});
                DevBuf<float>         d_cval(std::vector<float>{});
                ASSERT_EQ(rocsparse_sgebsr2gebsc(handle,
                                                 mb,
                                                 nb,
                                                 0,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 rdim,
                                                 cdim,
                                                 d_cval,
                                                 d_rind,
                                                 d_cptr,
                                                 rocsparse_action_numeric,
                                                 to_base(base),
                                                 d_buffer),
                          rocsparse_status_success);
                for(rocsparse_int v : d_cptr.read(nb + 1))
                {
                    EXPECT_EQ(v, base);
                }
            }
        }
    }

    // ------------------------------------------------------------------------------
    // rocsparse_sbsr2csr
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvBsr2CsrEdges)
    {
        const rocsparse_int block_dim = 2;
        for(int base : {0, 1})
        {
            Descr bsr_descr(base), csr_descr(base);

            // No block row or no block column: the row pointers are all the index base.
            for(const auto& d : {std::pair<rocsparse_int, rocsparse_int>(0, 3),
                                 std::pair<rocsparse_int, rocsparse_int>(3, 0)})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " mb " + std::to_string(d.first)
                             + " nb " + std::to_string(d.second));
                const rocsparse_int   mb = d.first, nb = d.second;
                DevBuf<rocsparse_int> d_bptr(std::vector<rocsparse_int>(mb + 1, base));
                DevBuf<rocsparse_int> d_bind(std::vector<rocsparse_int>{});
                DevBuf<float>         d_bval(std::vector<float>{});
                DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(mb * block_dim + 1, 77));
                DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{});
                DevBuf<float>         d_val(std::vector<float>{});
                ASSERT_EQ(rocsparse_sbsr2csr(handle,
                                             rocsparse_direction_row,
                                             mb,
                                             nb,
                                             bsr_descr,
                                             d_bval,
                                             d_bptr,
                                             d_bind,
                                             block_dim,
                                             csr_descr,
                                             d_val,
                                             d_ptr,
                                             d_ind),
                          rocsparse_status_success);
                for(rocsparse_int v : d_ptr.read(mb * block_dim + 1))
                {
                    EXPECT_EQ(v, base);
                }
            }

            // With blocks, the output arrays are required.
            const rocsparse_int   mb = 2, nb = 2, nnzb = 2;
            DevBuf<rocsparse_int> d_bptr(std::vector<rocsparse_int>{base, base + 1, base + 2});
            DevBuf<rocsparse_int> d_bind(std::vector<rocsparse_int>{base, base + 1});
            DevBuf<float>         d_bval(std::vector<float>(nnzb * block_dim * block_dim, 1.0f));
            DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>{}, mb * block_dim + 1);
            DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{}, nnzb * block_dim * block_dim);
            DevBuf<float>         d_val(std::vector<float>{}, nnzb * block_dim * block_dim);
            EXPECT_EQ(rocsparse_sbsr2csr(handle,
                                         rocsparse_direction_row,
                                         mb,
                                         nb,
                                         bsr_descr,
                                         d_bval,
                                         d_bptr,
                                         d_bind,
                                         block_dim,
                                         csr_descr,
                                         static_cast<float*>(nullptr),
                                         d_ptr,
                                         d_ind),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(rocsparse_sbsr2csr(handle,
                                         rocsparse_direction_row,
                                         mb,
                                         nb,
                                         bsr_descr,
                                         d_bval,
                                         d_bptr,
                                         d_bind,
                                         block_dim,
                                         csr_descr,
                                         d_val,
                                         d_ptr,
                                         static_cast<rocsparse_int*>(nullptr)),
                      rocsparse_status_invalid_pointer);
        }
    }

    // ------------------------------------------------------------------------------
    // rocsparse_ell2csr_nnz / rocsparse_sell2csr
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvEll2Csr)
    {
        const rocsparse_int m = 5, n = 6, width = 3;
        const rocsparse_int counts[m] = {2, 0, 3, 1, 3};
        for(int base : {0, 1})
        {
            // ELL arrays are column major, padded with -1.
            std::vector<rocsparse_int> e_ind(m * width, -1);
            std::vector<float>         e_val(m * width, 0.0f);
            std::vector<rocsparse_int> ptr = {base}, ind;
            std::vector<float>         val;
            for(rocsparse_int i = 0; i < m; ++i)
            {
                for(rocsparse_int k = 0; k < counts[i]; ++k)
                {
                    const rocsparse_int col = (i + 2 * k) % n;
                    e_ind[i + k * m]        = col + base;
                    e_val[i + k * m]        = 1.0f + i * 10 + k;
                    ind.push_back(col + base);
                    val.push_back(1.0f + i * 10 + k);
                }
                ptr.push_back(base + static_cast<rocsparse_int>(ind.size()));
            }
            const rocsparse_int nnz = static_cast<rocsparse_int>(ind.size());

            Descr                 ell_descr(base), csr_descr(base);
            DevBuf<rocsparse_int> d_eind(e_ind);
            DevBuf<float>         d_eval(e_val);
            for(rocsparse_pointer_mode mode : s_modes)
            {
                SCOPED_TRACE("base " + std::to_string(base) + " device mode "
                             + std::to_string(mode == rocsparse_pointer_mode_device));
                PointerModeScope      scope(handle, mode);
                DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>{}, m + 1);
                DevBuf<rocsparse_int> d_nnz(std::vector<rocsparse_int>(1, -1));
                rocsparse_int         h_nnz = -1;
                rocsparse_int* t = (mode == rocsparse_pointer_mode_device) ? d_nnz.get() : &h_nnz;
                ASSERT_EQ(rocsparse_ell2csr_nnz(
                              handle, m, n, ell_descr, width, d_eind, csr_descr, d_ptr, t),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                if(mode == rocsparse_pointer_mode_device)
                {
                    h_nnz = d_nnz.read(1)[0];
                }
                ASSERT_EQ(h_nnz, nnz);
                EXPECT_EQ(d_ptr.read(m + 1), ptr);

                DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{}, nnz);
                DevBuf<float>         d_val(std::vector<float>{}, nnz);
                ASSERT_EQ(rocsparse_sell2csr(handle,
                                             m,
                                             n,
                                             ell_descr,
                                             width,
                                             d_eval,
                                             d_eind,
                                             csr_descr,
                                             d_val,
                                             d_ptr,
                                             d_ind),
                          rocsparse_status_success);
                EXPECT_EQ(d_ind.read(nnz), ind);
                EXPECT_EQ(d_val.read(nnz), val);
            }
        }
    }

    TEST_F(HandleTest, Cov80ConvEll2CsrEmpty)
    {
        Descr                 ell_descr, csr_descr;
        DevBuf<rocsparse_int> d_eind(std::vector<rocsparse_int>{}, 16);
        DevBuf<float>         d_eval(std::vector<float>{}, 16);
        const rocsparse_int   cases[][3] = {{0, 4, 2}, {4, 0, 2}, {4, 4, 0}};
        for(const auto& c : cases)
        {
            SCOPED_TRACE("m " + std::to_string(c[0]) + " n " + std::to_string(c[1]) + " width "
                         + std::to_string(c[2]));
            DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(8, 0));
            DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{}, 8);
            DevBuf<float>         d_val(std::vector<float>{}, 8);
            rocsparse_int         nnz = 0;
            EXPECT_EQ(rocsparse_ell2csr_nnz(
                          handle, c[0], c[1], ell_descr, c[2], d_eind, csr_descr, d_ptr, &nnz),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_sell2csr(handle,
                                         c[0],
                                         c[1],
                                         ell_descr,
                                         c[2],
                                         d_eval,
                                         d_eind,
                                         csr_descr,
                                         d_val,
                                         d_ptr,
                                         d_ind),
                      rocsparse_status_success);
        }
    }

    // ------------------------------------------------------------------------------
    // rocsparse_hyb2csr of an empty HYB matrix
    // ------------------------------------------------------------------------------
    TEST_F(HandleTest, Cov80ConvHyb2CsrEmpty)
    {
        for(int base : {0, 1})
        {
            Descr descr(base);

            rocsparse_hyb_mat hyb = nullptr;
            ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);
            cov80conv::HybGuard hyb_guard(hyb);

            // No rows: csr2hyb clears the HYB matrix.
            DevBuf<rocsparse_int> d_ptr(std::vector<rocsparse_int>(1, base));
            DevBuf<rocsparse_int> d_ind(std::vector<rocsparse_int>{});
            DevBuf<float>         d_val(std::vector<float>{});
            ASSERT_EQ(
                rocsparse_scsr2hyb(
                    handle, 0, 4, descr, d_val, d_ptr, d_ind, hyb, 0, rocsparse_hyb_partition_auto),
                rocsparse_status_success);

            size_t size = 123;
            ASSERT_EQ(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, d_ptr, &size),
                      rocsparse_status_success);
            DevBuf<char>          d_buffer(std::vector<char>{}, size);
            DevBuf<rocsparse_int> o_ptr(std::vector<rocsparse_int>(2, 77));
            DevBuf<rocsparse_int> o_ind(std::vector<rocsparse_int>{});
            DevBuf<float>         o_val(std::vector<float>{});
            ASSERT_EQ(rocsparse_shyb2csr(handle, descr, hyb, o_val, o_ptr, o_ind, d_buffer),
                      rocsparse_status_success);
            EXPECT_EQ(o_ptr.read(1)[0], base);
        }
    }
}
