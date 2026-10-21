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

// Host paths of the library internals that the public clients never execute:
//   - the CSR triangular split.
// Coverage counts a line only when every branch of that line has run, so each
// check is taken on both sides. The CSR triangular split launches a (tiny)
// kernel.

#include "unit_test_utils.hpp"

#include "rocsparse_csxsldu.hpp"
#include "rocsparse_handle.hpp"

#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    rocsparse_status sldu_buffer_size(rocsparse_handle    handle,
                                      rocsparse_direction dir,
                                      rocsparse_direction ldir,
                                      rocsparse_direction udir,
                                      size_t*             bytes)
    {
        return rocsparse::csxsldu_buffer_size_template<float, rocsparse_int, rocsparse_int>(
            handle,
            dir,
            4,
            4,
            4,
            nullptr,
            nullptr,
            nullptr,
            rocsparse_index_base_zero,
            rocsparse_diag_type_unit,
            ldir,
            rocsparse_diag_type_unit,
            udir,
            bytes);
    }

    rocsparse_status sldu_preprocess(rocsparse_handle            handle,
                                     rocsparse_direction         dir,
                                     rocsparse_diag_type         ldiag,
                                     rocsparse_direction         ldir,
                                     rocsparse_diag_type         udiag,
                                     rocsparse_direction         udir,
                                     rocsparse_int*              scratch,
                                     rocsparse_int*              lnnz_out,
                                     rocsparse_int*              unnz_out,
                                     std::vector<rocsparse_int>* lptr_out = nullptr,
                                     std::vector<rocsparse_int>* uptr_out = nullptr)
    {
        const std::vector<rocsparse_int> sentinel{-7, -7, -7};
        device_vector<rocsparse_int>     ptr{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<rocsparse_int>     ind{std::vector<rocsparse_int>{0, 1}};
        device_vector<float>             val{std::vector<float>{1.0f, 1.0f}};
        device_vector<rocsparse_int>     lptr{sentinel};
        device_vector<rocsparse_int>     uptr{sentinel};
        if(!ptr.ptr || !ind.ptr || !val.ptr || !lptr.ptr || !uptr.ptr)
        {
            return rocsparse_status_memory_error;
        }
        rocsparse_int          lnnz = -1;
        rocsparse_int          unnz = -1;
        const rocsparse_status status
            = rocsparse::csxsldu_preprocess_template<float, rocsparse_int, rocsparse_int>(
                handle,
                dir,
                2,
                2,
                2,
                ptr.ptr,
                ind.ptr,
                val.ptr,
                rocsparse_index_base_zero,
                ldiag,
                ldir,
                &lnnz,
                lptr.ptr,
                rocsparse_index_base_zero,
                udiag,
                udir,
                &unnz,
                uptr.ptr,
                rocsparse_index_base_zero,
                scratch);
        if(lnnz_out)
            *lnnz_out = lnnz;
        if(unnz_out)
            *unnz_out = unnz;
        if(lptr_out)
            *lptr_out = to_host(lptr);
        if(uptr_out)
            *uptr_out = to_host(uptr);
        return status;
    }
    class HostInternalBranches : public HandleTest
    {
    };
} // namespace

// Buffer-size has three direction combinations. Preprocess's column walk and
// the two mixed diag-type launches are not on the itilu0 path.
TEST_F(HostInternalBranches, csxsldu_directions)
{
    constexpr auto row    = rocsparse_direction_row;
    constexpr auto column = rocsparse_direction_column;
    constexpr auto entry  = (4 + 1) * sizeof(rocsparse_int);

    const struct
    {
        rocsparse_direction dir, ldir, udir;
        size_t              bytes;
    } sizes[] = {{row, column, column, 2 * entry},
                 {row, row, row, 0},
                 {row, row, column, entry},
                 {column, row, row, 2 * entry},
                 {column, column, row, entry}};
    for(const auto& s : sizes)
    {
        size_t bytes = 99;
        EXPECT_EQ(sldu_buffer_size(handle, s.dir, s.ldir, s.udir, &bytes),
                  rocsparse_status_success);
        EXPECT_EQ(bytes, s.bytes);
    }

    device_vector<rocsparse_int> scratch{(size_t)8};
    ASSERT_TRUE(scratch.ptr);

    // 2x2 diagonal. Unit triangle stores nothing; non-unit stores the diagonal.
    // Row direction with a column factor writes the scanned pointers into scratch.
    auto expect_split = [&](rocsparse_diag_type        ldiag,
                            rocsparse_diag_type        udiag,
                            rocsparse_int              want_lnnz,
                            rocsparse_int              want_unnz,
                            std::vector<rocsparse_int> want) {
        ASSERT_EQ(hipMemset(scratch.ptr, 0x7f, 8 * sizeof(rocsparse_int)), hipSuccess);
        rocsparse_int lnnz = -1;
        rocsparse_int unnz = -1;
        EXPECT_EQ(
            sldu_preprocess(handle, row, ldiag, column, udiag, column, scratch.ptr, &lnnz, &unnz),
            rocsparse_status_success);
        EXPECT_EQ(lnnz, want_lnnz);
        EXPECT_EQ(unnz, want_unnz);
        ASSERT_EQ(to_host(scratch.ptr, want.size()), want);
    };
    expect_split(rocsparse_diag_type_unit, rocsparse_diag_type_non_unit, 0, 2, {0, 0, 0, 0, 1, 2});
    expect_split(rocsparse_diag_type_non_unit, rocsparse_diag_type_unit, 2, 0, {0, 1, 2, 0, 0, 0});

    // Factors in the input direction are written to lptr / uptr directly.
    std::vector<rocsparse_int> lptr, uptr;
    rocsparse_int              lnnz = -1, unnz = -1;
    EXPECT_EQ(sldu_preprocess(handle,
                              row,
                              rocsparse_diag_type_unit,
                              row,
                              rocsparse_diag_type_non_unit,
                              row,
                              scratch.ptr,
                              &lnnz,
                              &unnz,
                              &lptr,
                              &uptr),
              rocsparse_status_success);
    EXPECT_EQ(lnnz, 0);
    EXPECT_EQ(unnz, 2);
    EXPECT_EQ(lptr, (std::vector<rocsparse_int>{0, 0, 0}));
    EXPECT_EQ(uptr, (std::vector<rocsparse_int>{0, 1, 2}));

    // Column direction with column factors: both factors in the input direction, so the
    // pointers go to lptr / uptr directly. Every element of lptr / uptr starts at the
    // sentinel -7, including element 0. A zero handle buffer makes preprocess allocate
    // its scratch.
    auto*  raw          = reinterpret_cast<_rocsparse_handle*>(handle);
    size_t saved_buffer = raw->buffer_size;
    raw->buffer_size    = 0;
    lnnz = unnz = -1;
    EXPECT_EQ(sldu_preprocess(handle,
                              column,
                              rocsparse_diag_type_unit,
                              column,
                              rocsparse_diag_type_unit,
                              column,
                              scratch.ptr,
                              &lnnz,
                              &unnz,
                              &lptr,
                              &uptr),
              rocsparse_status_success);
    raw->buffer_size = saved_buffer;
    EXPECT_EQ(lnnz, 0);
    EXPECT_EQ(unnz, 0);
    EXPECT_EQ(lptr, (std::vector<rocsparse_int>{0, 0, 0}));
    EXPECT_EQ(uptr, (std::vector<rocsparse_int>{0, 0, 0}));

    // Same walk with a non-unit lower factor: the diagonal goes to L.
    lnnz = unnz = -1;
    EXPECT_EQ(sldu_preprocess(handle,
                              column,
                              rocsparse_diag_type_non_unit,
                              column,
                              rocsparse_diag_type_unit,
                              column,
                              scratch.ptr,
                              &lnnz,
                              &unnz,
                              &lptr,
                              &uptr),
              rocsparse_status_success);
    EXPECT_EQ(lnnz, 2);
    EXPECT_EQ(unnz, 0);
    EXPECT_EQ(lptr, (std::vector<rocsparse_int>{0, 1, 2}));
    EXPECT_EQ(uptr, (std::vector<rocsparse_int>{0, 0, 0}));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

namespace
{
    struct sldu_case
    {
        rocsparse_direction        dir;
        rocsparse_int              m, n;
        std::vector<rocsparse_int> ptr, ind;
        rocsparse_index_base       base;
        rocsparse_diag_type        ldiag;
        rocsparse_direction        ldir;
        rocsparse_index_base       lbase;
        rocsparse_diag_type        udiag;
        rocsparse_direction        udir;
        rocsparse_index_base       ubase;
    };

    // Host reference of the L / U pointers that preprocess computes in the input
    // direction: one entry per row (row direction) or column (column direction) plus
    // one, starting at the factor's index base. At most one diagonal type is non-unit,
    // and the diagonal goes to that factor.
    void sldu_reference(const sldu_case&            c,
                        std::vector<rocsparse_int>& lptr,
                        std::vector<rocsparse_int>& uptr)
    {
        const rocsparse_int nseq = (c.dir == rocsparse_direction_row) ? c.m : c.n;
        lptr.assign(1, c.lbase);
        uptr.assign(1, c.ubase);
        for(rocsparse_int s = 0; s < nseq; ++s)
        {
            rocsparse_int l = 0, u = 0;
            for(rocsparse_int k = c.ptr[s] - c.base; k < c.ptr[s + 1] - c.base; ++k)
            {
                const rocsparse_int other = c.ind[k] - c.base;
                const rocsparse_int row   = (c.dir == rocsparse_direction_row) ? s : other;
                const rocsparse_int col   = (c.dir == rocsparse_direction_row) ? other : s;
                if(row > col)
                    ++l;
                else if(row < col)
                    ++u;
                else if(c.ldiag == rocsparse_diag_type_non_unit)
                    ++l;
                else if(c.udiag == rocsparse_diag_type_non_unit)
                    ++u;
            }
            lptr.push_back(lptr.back() + l);
            uptr.push_back(uptr.back() + u);
        }
    }
} // namespace

// Non-square inputs, one-based indices, distinct L / U bases, strictly lower and upper
// entries, and factors in the other direction (their pointers go to the scratch buffer,
// L first, U after it). Every pointer and both counts are compared with the host
// reference; pointer arrays preprocess must not write keep their -7 sentinel.
TEST_F(HostInternalBranches, csxsldu_preprocess_reference)
{
    constexpr auto row    = rocsparse_direction_row;
    constexpr auto column = rocsparse_direction_column;
    constexpr auto unit   = rocsparse_diag_type_unit;
    constexpr auto nonu   = rocsparse_diag_type_non_unit;
    constexpr auto zero   = rocsparse_index_base_zero;
    constexpr auto one    = rocsparse_index_base_one;

    // 2x3: (0,0) (0,1) (0,2) / (1,0) (1,1) (1,2), stored by column.
    const std::vector<rocsparse_int> p23{0, 2, 4, 6}, i23{0, 1, 0, 1, 0, 1};
    // The same matrix one-based.
    const std::vector<rocsparse_int> p23b{1, 3, 5, 7}, i23b{1, 2, 1, 2, 1, 2};
    // 3x2: (0,0) / (1,0) (1,1) / (2,0), stored by column.
    const std::vector<rocsparse_int> p32{0, 3, 4}, i32{0, 1, 2, 1};
    // 3x2: (0,0) / (1,0) (1,1) / (2,0) (2,1), stored by row, one-based.
    const std::vector<rocsparse_int> r32b{1, 2, 4, 6}, j32b{1, 1, 2, 1, 2};

    const sldu_case cases[] = {
        {column, 2, 3, p23, i23, zero, unit, column, zero, nonu, column, zero},
        {column, 3, 2, p32, i32, zero, nonu, column, zero, unit, column, zero},
        {column, 2, 3, p23b, i23b, one, unit, column, one, nonu, column, zero},
        {column, 2, 3, p23, i23, zero, unit, row, one, nonu, row, one},
        {column, 2, 3, p23, i23, zero, nonu, column, zero, unit, row, one},
        {column, 2, 3, p23b, i23b, one, nonu, row, one, unit, column, zero},
        {row, 3, 2, r32b, j32b, one, unit, row, one, nonu, column, one},
        // 3x0 by column: one pointer and no entries; ind holds a dummy so that the
        // device allocation is not empty.
        {column, 3, 0, {1}, {1}, one, unit, column, one, nonu, column, zero},
    };

    int index = 0;
    for(const sldu_case& c : cases)
    {
        SCOPED_TRACE("case " + std::to_string(index++));
        const rocsparse_int              nseq = (c.dir == row) ? c.m : c.n;
        const rocsparse_int              nnz  = static_cast<rocsparse_int>(c.ind.size());
        const std::vector<rocsparse_int> sentinel(nseq + 1, -7);
        device_vector<rocsparse_int>     ptr{c.ptr};
        device_vector<rocsparse_int>     ind{c.ind};
        device_vector<float>             val{std::vector<float>(nnz, 1.0f)};
        device_vector<rocsparse_int>     lptr{sentinel};
        device_vector<rocsparse_int>     uptr{sentinel};
        device_vector<rocsparse_int>     scratch{std::vector<rocsparse_int>(2 * (nseq + 1), -7)};
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && lptr.ptr && uptr.ptr && scratch.ptr);

        rocsparse_int          lnnz = -1, unnz = -1;
        const rocsparse_status status
            = rocsparse::csxsldu_preprocess_template<float, rocsparse_int, rocsparse_int>(
                handle,
                c.dir,
                c.m,
                c.n,
                nnz,
                ptr.ptr,
                ind.ptr,
                val.ptr,
                c.base,
                c.ldiag,
                c.ldir,
                &lnnz,
                lptr.ptr,
                c.lbase,
                c.udiag,
                c.udir,
                &unnz,
                uptr.ptr,
                c.ubase,
                scratch.ptr);
        EXPECT_EQ(status, rocsparse_status_success);
        if(status != rocsparse_status_success)
            continue;

        std::vector<rocsparse_int> want_l, want_u;
        sldu_reference(c, want_l, want_u);
        const bool   l_scratch = (c.ldir != c.dir);
        const bool   u_scratch = (c.udir != c.dir);
        const size_t u_offset  = l_scratch ? nseq + 1 : 0;
        const auto   got_l     = l_scratch ? to_host(scratch.ptr, nseq + 1) : to_host(lptr);
        const auto   got_u = u_scratch ? to_host(scratch.ptr + u_offset, nseq + 1) : to_host(uptr);
        EXPECT_EQ(got_l, want_l);
        EXPECT_EQ(got_u, want_u);
        EXPECT_EQ(lnnz, want_l.back() - c.lbase);
        EXPECT_EQ(unnz, want_u.back() - c.ubase);
        if(l_scratch)
            EXPECT_EQ(to_host(lptr), sentinel);
        if(u_scratch)
            EXPECT_EQ(to_host(uptr), sentinel);
    }

    // Both diagonals non-unit is rejected by the count kernel dispatch, in both directions.
    for(const rocsparse_direction d : {row, column})
    {
        rocsparse_int lnnz = -1, unnz = -1;
        EXPECT_EQ(sldu_preprocess(handle, d, nonu, d, nonu, d, nullptr, &lnnz, &unnz),
                  rocsparse_status_invalid_value);
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}
