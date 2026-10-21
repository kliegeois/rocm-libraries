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

// General block CSR (gebsr) and block / ELL conversions.

#include "unit_test_device_conv_common.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

using namespace convut;

namespace
{
    using ConvGebsr = ConvTest;

    // ------------------------------------------------------------------------------
    // ell2csr_nnz / sell2csr
    // ------------------------------------------------------------------------------

    TEST_F(ConvGebsr, Ell2CsrEmpty)
    {
        MatDescr      ell_descr;
        DevBuf<I>     d_eind(16), d_ind(8);
        DevBuf<float> d_eval(16), d_val(8);
        for(const auto& c : {std::array<I, 4>{0, 4, 2, 0}, {4, 0, 2, 1}, {4, 4, 0, 0}})
        {
            for(bool device_mode : {false, true})
            {
                SCOPED_TRACE("m " + std::to_string(c[0]) + " n " + std::to_string(c[1]) + " width "
                             + std::to_string(c[2]) + " base " + std::to_string(c[3])
                             + " device mode " + std::to_string(device_mode));
                MatDescr  csr_descr(base_of(c[3]));
                DevBuf<I> d_ptr(std::vector<I>(8, 77));
                // An empty CSR matrix: no entries, every row pointer is the index base, and
                // nothing past the m + 1 pointers is written.
                EXPECT_EQ(
                    nnz_via(
                        handle,
                        device_mode,
                        [&](I* nnz) {
                            return rocsparse_ell2csr_nnz(
                                handle, c[0], c[1], ell_descr, c[2], d_eind, csr_descr, d_ptr, nnz);
                        }),
                    0);
                std::vector<I> want(8, 77);
                std::fill(want.begin(), want.begin() + c[0] + 1, c[3]);
                EXPECT_EQ(d_ptr.read(8), want);
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
                          st_ok);
            }
        }
    }

    // rocsparse_sparse_to_sparse reads the nnz (or the ELL width) of the target into a host
    // variable. ELL -> CSR, CSR -> BSR and CSR -> ELL must convert empty matrices and the
    // 4 x 4 identity in both pointer modes, with 32-bit and 64-bit row pointers, and leave
    // the pointer mode of the handle unchanged.
    TEST_F(ConvGebsr, SparseToSparseNnz)
    {
        // Source and target formats, m, n, ELL width of the source (the 4 x 4 identity has
        // width 1) and index base.
        struct Case
        {
            rocsparse_format sf, tf;
            int64_t          m, n, width;
            int              base;
        };
        constexpr auto ell = rocsparse_format_ell, csr = rocsparse_format_csr,
                       bsr = rocsparse_format_bsr;
        const Case cases[] = {{ell, csr, 0, 4, 2, 0},
                              {ell, csr, 4, 0, 0, 1},
                              {ell, csr, 4, 4, 0, 0},
                              {ell, csr, 4, 4, 1, 0},
                              {ell, csr, 4, 4, 1, 1},
                              {csr, bsr, 0, 4, 0, 0},
                              {csr, bsr, 4, 0, 0, 1},
                              {csr, bsr, 4, 4, 1, 0},
                              {csr, bsr, 4, 4, 1, 1},
                              {csr, ell, 0, 4, 0, 0},
                              {csr, ell, 4, 0, 0, 1},
                              {csr, ell, 4, 4, 1, 0},
                              {csr, ell, 4, 4, 1, 1}};
        const auto run     = [&](auto ptr_tag, const Case& c, rocsparse_pointer_mode mode) {
            using P             = decltype(ptr_tag);
            const auto ptr_type = sizeof(P) == 4 ? i32 : rocsparse_indextype_i64;
            const bool identity = c.m == 4 && c.n == 4 && c.width == 1;
            const auto base     = base_of(c.base);
            const int  b        = c.base;
            SCOPED_TRACE(std::string(c.sf == ell ? "ell" : "csr") + " -> "
                         + (c.tf == csr   ? "csr"
                                : c.tf == bsr ? "bsr"
                                              : "ell")
                         + " row pointer bytes " + std::to_string(sizeof(P)) + " m "
                         + std::to_string(c.m) + " n " + std::to_string(c.n) + " width "
                         + std::to_string(c.width) + " base " + std::to_string(b) + " pointer mode "
                         + std::to_string(mode));
            PointerModeScope pm(handle, mode);

            std::vector<P>       h_ptr(c.m + 1, P(b));
            std::vector<int32_t> h_ind(16, 0);
            std::vector<float>   h_val(16, 0.0f);
            if(identity)
            {
                for(int i = 0; i < 4; ++i)
                {
                    h_ptr[i + 1] = P(b + i + 1);
                    h_ind[i]     = b + i;
                    h_val[i]     = float(i + 1);
                }
            }
            DevBuf<P>       s_ptr(h_ptr), t_ptr(std::vector<P>(8, 77));
            DevBuf<int32_t> s_ind(h_ind), t_ind(std::vector<int32_t>(16, 77));
            DevBuf<float>   s_val(h_val), t_val(std::vector<float>(16, -1.0f));
            const int64_t   mb = (c.m + 1) / 2, nb = (c.n + 1) / 2;
            Spmat           source, target;
            if(c.sf == ell)
                ASSERT_EQ(rocsparse_create_ell_descr(
                              &source, c.m, c.n, s_ind, s_val, c.width, i32, base, f32),
                          st_ok);
            else
                ASSERT_EQ(rocsparse_create_csr_descr(&source,
                                                     c.m,
                                                     c.n,
                                                     identity ? 4 : 0,
                                                     s_ptr,
                                                     s_ind,
                                                     s_val,
                                                     ptr_type,
                                                     i32,
                                                     base,
                                                     f32),
                          st_ok);
            if(c.tf == csr)
                ASSERT_EQ(rocsparse_create_csr_descr(
                              &target, c.m, c.n, 0, t_ptr, t_ind, t_val, ptr_type, i32, base, f32),
                          st_ok);
            else if(c.tf == bsr)
                ASSERT_EQ(rocsparse_create_bsr_descr(&target,
                                                     mb,
                                                     nb,
                                                     0,
                                                     dir_r,
                                                     2,
                                                     t_ptr,
                                                     t_ind,
                                                     t_val,
                                                     ptr_type,
                                                     i32,
                                                     base,
                                                     f32),
                          st_ok);
            else
                ASSERT_EQ(
                    rocsparse_create_ell_descr(&target, c.m, c.n, t_ind, t_val, 0, i32, base, f32),
                    st_ok);
            S2sDescr descr;
            ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                          &descr, source, target, rocsparse_sparse_to_sparse_alg_default),
                      st_ok);
            for(auto stage : {rocsparse_sparse_to_sparse_stage_analysis,
                              rocsparse_sparse_to_sparse_stage_compute})
            {
                size_t size = 0;
                ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                              handle, descr, source, target, stage, &size),
                          st_ok);
                DevBuf<char> buffer(size);
                ASSERT_EQ(
                    rocsparse_sparse_to_sparse(handle, descr, source, target, stage, size, buffer),
                    st_ok);
            }
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            rocsparse_pointer_mode after = rocsparse_pointer_mode(-1);
            EXPECT_EQ(rocsparse_get_pointer_mode(handle, &after), st_ok);
            EXPECT_EQ(after, mode);

            // The identity has 4 entries in 4 rows, or 2 diagonal 2 x 2 blocks in 2 block
            // rows, or width 1. Nothing is written past the expected entries.
            std::vector<P>       want_ptr(8, 77);
            std::vector<int32_t> want_ind(16, 77);
            const int64_t        rows = c.tf == bsr ? mb : c.m;
            const int64_t        nnz  = identity ? (c.tf == bsr ? 2 : 4) : 0;
            if(c.tf != ell)
            {
                for(int64_t i = 0; i <= rows; ++i)
                    want_ptr[i] = P(b + (identity ? i : 0));
                EXPECT_EQ(t_ptr.read(8), want_ptr);
                int64_t t_rows = -1, t_cols = -1, t_nnz = -1;
                EXPECT_EQ(rocsparse_spmat_get_size(target, &t_rows, &t_cols, &t_nnz), st_ok);
                EXPECT_EQ(t_nnz, nnz);
            }
            else
            {
                int64_t              e_rows = -1, e_cols = -1, e_width = -1;
                void *               e_ind = nullptr, *e_val = nullptr;
                rocsparse_indextype  e_itype;
                rocsparse_index_base e_base;
                rocsparse_datatype   e_dtype;
                EXPECT_EQ(rocsparse_ell_get(target,
                                            &e_rows,
                                            &e_cols,
                                            &e_ind,
                                            &e_val,
                                            &e_width,
                                            &e_itype,
                                            &e_base,
                                            &e_dtype),
                          st_ok);
                EXPECT_EQ(e_width, identity ? 1 : 0);
            }
            for(int64_t k = 0; k < nnz; ++k)
                want_ind[k] = b + int32_t(k);
            EXPECT_EQ(t_ind.read(16), want_ind);

            // CSR and ELL (column-major, width 1) hold the diagonal; BSR holds the two
            // diagonal 2 x 2 blocks, row-major.
            std::vector<float> want_val(16, -1.0f);
            if(identity)
            {
                const std::vector<float> v = c.tf == bsr
                                                     ? std::vector<float>{1, 0, 0, 2, 3, 0, 0, 4}
                                                     : std::vector<float>{1, 2, 3, 4};
                for(size_t k = 0; k < v.size(); ++k)
                    want_val[k] = v[k];
            }
            EXPECT_EQ(t_val.read(16), want_val);
        };
        for(const auto& c : cases)
            for(auto mode : s_pointer_modes)
            {
                run(int32_t(), c, mode);
                run(int64_t(), c, mode);
            }
    }
} // namespace
