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

// sparse_to_sparse between BSR matrices with different block dimensions or directions.

#include "unit_test_device_conv_common.hpp"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

using namespace convut;

namespace
{
    using ConvSpmat = ConvTest;

    // ------------------------------------------------------------------
    // sparse_to_sparse
    // ------------------------------------------------------------------

    constexpr auto s2s_ana = rocsparse_sparse_to_sparse_stage_analysis;
    constexpr auto s2s_cmp = rocsparse_sparse_to_sparse_stage_compute;
    constexpr auto s2s_def = rocsparse_sparse_to_sparse_alg_default;

    template <typename I>
    struct HostBsr
    {
        std::vector<I>      ptr, ind;
        std::vector<double> val;
    };

    // BSR arrays with index base `base` of the row-major m x n matrix `dense` with bd x bd
    // blocks: the blocks holding an entry for which `stored` is set, in row order, zero
    // padded past the matrix edge.
    template <typename I>
    HostBsr<I> to_bsr(const std::vector<double>& dense,
                      const std::vector<bool>&   stored,
                      int64_t                    m,
                      int64_t                    n,
                      int64_t                    bd,
                      rocsparse_direction        dir,
                      rocsparse_index_base       base)
    {
        const auto in = [&](int64_t i, int64_t j) { return i < m && j < n; };
        HostBsr<I> h;
        h.ptr.push_back(static_cast<I>(base));
        for(int64_t ib = 0; ib < (m + bd - 1) / bd; ++ib)
        {
            for(int64_t jb = 0; jb < (n + bd - 1) / bd; ++jb)
            {
                bool any = false;
                for(int64_t k = 0; k < bd * bd; ++k)
                {
                    const int64_t i = ib * bd + k / bd, j = jb * bd + k % bd;
                    any = any || (in(i, j) && stored[i * n + j]);
                }
                if(!any)
                    continue;
                h.ind.push_back(static_cast<I>(jb + static_cast<I>(base)));
                for(int64_t k = 0; k < bd * bd; ++k)
                {
                    const int64_t i = ib * bd + (dir == dir_r ? k / bd : k % bd);
                    const int64_t j = jb * bd + (dir == dir_r ? k % bd : k / bd);
                    h.val.push_back(in(i, j) ? dense[i * n + j] : 0.0);
                }
            }
            h.ptr.push_back(static_cast<I>(h.ind.size()) + static_cast<I>(base));
        }
        return h;
    }

    // An m x n source matrix in bd_s x bd_s blocks, converted to BSR targets with every
    // other block dimension from 1 to 4 and either direction. The source blocks hold
    // zeros, and blocks past the matrix edge pad the source; both are stored entries, so
    // the target holds every block that overlaps a source block. A target block grid
    // covers the padded source. The target arrays must be the host BSR arrays of the
    // padded source. A change of layout needs permissive mode; 1 x 1 blocks with another
    // direction are a plain copy.
    template <typename I>
    void sparse_to_sparse_bsr_reblock(rocsparse_handle     handle,
                                      int64_t              m,
                                      int64_t              n,
                                      int64_t              bd_s,
                                      rocsparse_direction  dir_s,
                                      rocsparse_index_base base)
    {
        constexpr auto      f64   = rocsparse_datatype_f64_r;
        const auto          itype = sizeof(I) == 4 ? i32 : rocsparse_indextype_i64;
        std::vector<double> dense(m * n, 0.0);
        std::vector<bool>   nonzero(m * n, false);
        for(int64_t i = 0; i < m; ++i)
            for(int64_t j = 0; j < n; ++j)
                if(j + 1 >= i && (i + 2 * j) % 5 != 0)
                {
                    dense[i * n + j]   = static_cast<double>(i * n + j + 1);
                    nonzero[i * n + j] = true;
                }
        const HostBsr<I> src    = to_bsr<I>(dense, nonzero, m, n, bd_s, dir_s, base);
        const int64_t    mb_s   = (m + bd_s - 1) / bd_s;
        const int64_t    nb_s   = (n + bd_s - 1) / bd_s;
        const int64_t    nnzb_s = static_cast<int64_t>(src.ind.size());

        // The padded source and its stored entries, as the intermediate CSR matrix holds them.
        const int64_t       pm = mb_s * bd_s, pn = nb_s * bd_s;
        std::vector<double> pdense(pm * pn, 0.0);
        std::vector<bool>   pstored(pm * pn, false);
        for(int64_t ib = 0; ib < mb_s; ++ib)
            for(I p = src.ptr[ib] - base; p < src.ptr[ib + 1] - base; ++p)
                for(int64_t k = 0; k < bd_s * bd_s; ++k)
                {
                    const int64_t i = ib * bd_s + (dir_s == dir_r ? k / bd_s : k % bd_s);
                    const int64_t j
                        = (src.ind[p] - base) * bd_s + (dir_s == dir_r ? k % bd_s : k / bd_s);
                    pdense[i * pn + j]  = src.val[p * bd_s * bd_s + k];
                    pstored[i * pn + j] = true;
                }

        DevBuf<I>      s_ptr(src.ptr), s_ind(src.ind);
        DevBuf<double> s_val(src.val);
        for(int64_t bd = 1; bd <= 4; ++bd)
        {
            for(auto dir : {dir_r, dir_c})
            {
                if(bd == bd_s && dir == dir_s)
                    continue;
                SCOPED_TRACE("bd " + std::to_string(bd) + " dir " + std::to_string(dir));
                const bool     reblock = bd != bd_s || (bd_s > 1 && dir != dir_s);
                const auto     want    = to_bsr<I>(pdense, pstored, pm, pn, bd, dir, base);
                const int64_t  mb      = (pm + bd - 1) / bd;
                const int64_t  nb      = (pn + bd - 1) / bd;
                const int64_t  cap     = static_cast<int64_t>(want.ind.size()) + 4;
                DevBuf<I>      t_ptr(std::vector<I>(mb + 1, -7)), t_ind(std::vector<I>(cap, -7));
                DevBuf<double> t_val(std::vector<double>(cap * bd * bd, -7.0));
                Spmat          source, target;
                ASSERT_EQ(rocsparse_create_bsr_descr(&source,
                                                     mb_s,
                                                     nb_s,
                                                     nnzb_s,
                                                     dir_s,
                                                     bd_s,
                                                     s_ptr,
                                                     s_ind,
                                                     s_val,
                                                     itype,
                                                     itype,
                                                     base,
                                                     f64),
                          st_ok);
                ASSERT_EQ(
                    rocsparse_create_bsr_descr(
                        &target, mb, nb, 1, dir, bd, t_ptr, t_ind, t_val, itype, itype, base, f64),
                    st_ok);
                if(reblock)
                {
                    S2sDescr strict;
                    ASSERT_EQ(
                        rocsparse_create_sparse_to_sparse_descr(&strict, source, target, s2s_def),
                        st_ok);
                    size_t size = 0;
                    EXPECT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                                  handle, strict, source, target, s2s_ana, &size),
                              rocsparse_status_not_implemented);
                }
                S2sDescr s2s;
                ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(&s2s, source, target, s2s_def),
                          st_ok);
                if(reblock)
                    ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(s2s), st_ok);
                for(auto stage : {s2s_ana, s2s_cmp})
                {
                    size_t size = 0;
                    ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                                  handle, s2s, source, target, stage, &size),
                              st_ok);
                    DevBuf<char> buffer(size);
                    ASSERT_EQ(rocsparse_sparse_to_sparse(
                                  handle, s2s, source, target, stage, size, buffer),
                              st_ok);
                    if(stage == s2s_ana)
                        ASSERT_EQ(rocsparse_bsr_set_pointers(target, t_ptr, t_ind, t_val), st_ok);
                }
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                int64_t t_mb = -1, t_nb = -1, t_nnzb = -1;
                EXPECT_EQ(rocsparse_spmat_get_size(target, &t_mb, &t_nb, &t_nnzb), st_ok);
                EXPECT_EQ(t_nnzb, static_cast<int64_t>(want.ind.size()));
                // Nothing is written past the target entries.
                std::vector<I>      want_ind = want.ind;
                std::vector<double> want_val = want.val;
                want_ind.resize(cap, -7);
                want_val.resize(cap * bd * bd, -7.0);
                EXPECT_EQ(t_ptr.read(), want.ptr);
                EXPECT_EQ(t_ind.read(), want_ind);
                EXPECT_EQ(t_val.read(), want_val);
            }
        }
    }

    TEST_F(ConvSpmat, SparseToSparseBsrReblock)
    {
        {
            SCOPED_TRACE("4 x 4, i32, base 0, 2 x 2 row-major source");
            sparse_to_sparse_bsr_reblock<int32_t>(handle, 4, 4, 2, dir_r, base0);
        }
        {
            SCOPED_TRACE("4 x 4, i64, base 1, 2 x 2 column-major source");
            sparse_to_sparse_bsr_reblock<int64_t>(handle, 4, 4, 2, dir_c, base1);
        }
        {
            SCOPED_TRACE("4 x 6, i32, base 1, 2 x 2 column-major source");
            sparse_to_sparse_bsr_reblock<int32_t>(handle, 4, 6, 2, dir_c, base1);
        }
        {
            SCOPED_TRACE("4 x 4, i64, base 0, 3 x 3 row-major source padded to 6 x 6");
            sparse_to_sparse_bsr_reblock<int64_t>(handle, 4, 4, 3, dir_r, base0);
        }
        {
            SCOPED_TRACE("5 x 7, i32, base 0, 1 x 1 source");
            sparse_to_sparse_bsr_reblock<int32_t>(handle, 5, 7, 1, dir_r, base0);
        }
    }

    // A BSR -> BSR conversion with another block dimension keeps the index and value types:
    // a 32-bit to 64-bit index change, or a double to float value change with both value
    // arrays set, returns rocsparse_status_not_implemented before the analysis, also in
    // permissive mode.
    TEST_F(ConvSpmat, SparseToSparseBsrReblockTypes)
    {
        constexpr auto            f32 = rocsparse_datatype_f32_r, f64 = rocsparse_datatype_f64_r;
        constexpr auto            i64 = rocsparse_indextype_i64;
        const std::vector<double> val = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        DevBuf<int32_t>           s_ptr(std::vector<int32_t>{0, 2, 3});
        DevBuf<int32_t>           s_ind(std::vector<int32_t>{0, 1, 1});
        DevBuf<double>            s_val(val);
        DevBuf<int64_t>           t_ptr64(5), t_ind64(16);
        DevBuf<int32_t>           t_ptr32(5), t_ind32(16);
        DevBuf<double>            t_val64(16);
        DevBuf<float>             t_val32(16);
        for(bool index_change : {true, false})
        {
            SCOPED_TRACE(index_change ? "i32 -> i64 indices" : "f64 -> f32 values");
            Spmat source, target;
            ASSERT_EQ(rocsparse_create_bsr_descr(
                          &source, 2, 2, 3, dir_r, 2, s_ptr, s_ind, s_val, i32, i32, base0, f64),
                      st_ok);
            if(index_change)
                ASSERT_EQ(rocsparse_create_bsr_descr(&target,
                                                     4,
                                                     4,
                                                     0,
                                                     dir_r,
                                                     1,
                                                     t_ptr64,
                                                     t_ind64,
                                                     t_val64,
                                                     i64,
                                                     i64,
                                                     base0,
                                                     f64),
                          st_ok);
            else
                ASSERT_EQ(rocsparse_create_bsr_descr(&target,
                                                     4,
                                                     4,
                                                     0,
                                                     dir_r,
                                                     1,
                                                     t_ptr32,
                                                     t_ind32,
                                                     t_val32,
                                                     i32,
                                                     i32,
                                                     base0,
                                                     f32),
                          st_ok);
            S2sDescr s2s;
            ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(&s2s, source, target, s2s_def),
                      st_ok);
            ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(s2s), st_ok);
            size_t size = 0;
            EXPECT_EQ(
                rocsparse_sparse_to_sparse_buffer_size(handle, s2s, source, target, s2s_ana, &size),
                rocsparse_status_not_implemented);
        }
    }

    // A 4 x 4 matrix in 2 x 2 blocks (0,0), (0,1) and (1,1), converted to a BSR target that
    // declares the same 2 x 2 block grid with 1 x 1 blocks: the target cannot hold the
    // matrix, so the conversion returns rocsparse_status_invalid_size, also without
    // permissive mode, and nothing is written past the target arrays.
    TEST_F(ConvSpmat, SparseToSparseBsrTargetTooSmall)
    {
        constexpr auto             f64 = rocsparse_datatype_f64_r;
        constexpr int64_t          mb = 2, nnzb = 3, pad = 64;
        const std::vector<double>  val = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        DevBuf<int32_t>            s_ptr(std::vector<int32_t>{0, 2, 3});
        DevBuf<int32_t>            s_ind(std::vector<int32_t>{0, 1, 1});
        DevBuf<double>             s_val(val);
        const std::vector<int32_t> guard_i(mb + 1 + pad, -7);
        const std::vector<double>  guard_v(nnzb + pad, -7.0);
        for(bool permissive : {false, true})
        {
            SCOPED_TRACE("permissive " + std::to_string(permissive));
            DevBuf<int32_t> t_ptr(guard_i), t_ind(guard_i);
            DevBuf<double>  t_val(guard_v);
            Spmat           source, target;
            ASSERT_EQ(
                rocsparse_create_bsr_descr(
                    &source, mb, mb, nnzb, dir_r, 2, s_ptr, s_ind, s_val, i32, i32, base0, f64),
                st_ok);
            ASSERT_EQ(
                rocsparse_create_bsr_descr(
                    &target, mb, mb, nnzb, dir_r, 1, t_ptr, t_ind, t_val, i32, i32, base0, f64),
                st_ok);
            S2sDescr s2s;
            ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(&s2s, source, target, s2s_def),
                      st_ok);
            if(permissive)
                ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(s2s), st_ok);
            rocsparse_status st = st_ok;
            for(auto stage : {s2s_ana, s2s_cmp})
            {
                size_t size = 0;
                if(st == st_ok)
                    st = rocsparse_sparse_to_sparse_buffer_size(
                        handle, s2s, source, target, stage, &size);
                if(st == st_ok)
                {
                    DevBuf<char> buffer(size);
                    st = rocsparse_sparse_to_sparse(
                        handle, s2s, source, target, stage, size, buffer);
                }
            }
            EXPECT_EQ(st, rocsparse_status_invalid_size);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            const auto tail = [](const auto& v, int64_t from) {
                return std::vector<typename std::decay_t<decltype(v)>::value_type>(v.begin() + from,
                                                                                   v.end());
            };
            EXPECT_EQ(tail(t_ptr.read(), mb + 1), tail(guard_i, mb + 1));
            EXPECT_EQ(tail(t_ind.read(), nnzb), tail(guard_i, nnzb));
            EXPECT_EQ(tail(t_val.read(), nnzb), tail(guard_v, nnzb));
        }
    }
} // namespace
