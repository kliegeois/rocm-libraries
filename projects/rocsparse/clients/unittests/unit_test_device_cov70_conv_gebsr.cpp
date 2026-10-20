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

// General block CSR (gebsr) conversions on matrices whose row and column block
// dimensions differ: gebsr2csr, gebsr2gebsr and csr2gebsr. Every conversion is
// checked on the full output: the result is expanded to a dense matrix on the host
// and compared against the dense matrix the input describes.

#include "unit_test_device_cov70_conv_common.hpp"

#include "rocsparse_handle.hpp"

#include <algorithm>
#include <memory>
#include <string>

using namespace rocsparse_ut;
using namespace cov70conv;

namespace
{
    struct ShrinkHandleBuffer
    {
        rocsparse_handle handle;
        size_t           saved;
        explicit ShrinkHandleBuffer(rocsparse_handle h)
            : handle(h)
            , saved(h->buffer_size)
        {
            h->buffer_size = 0;
        }
        ~ShrinkHandleBuffer()
        {
            handle->buffer_size = saved;
        }
    };

    struct PointerModeScope
    {
        rocsparse_handle       handle;
        rocsparse_pointer_mode saved;
        PointerModeScope(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
            , saved(h->pointer_mode)
        {
            EXPECT_EQ(rocsparse_set_pointer_mode(h, mode), rocsparse_status_success);
        }
        ~PointerModeScope()
        {
            (void)rocsparse_set_pointer_mode(handle, saved);
        }
    };

    rocsparse_index_base to_base(int base)
    {
        return base == 0 ? rocsparse_index_base_zero : rocsparse_index_base_one;
    }

    const char* dir_name(rocsparse_direction dir)
    {
        return dir == rocsparse_direction_row ? "row" : "column";
    }

    // The input gebsr matrix: 3 x 3 blocks, block row 1 is empty,
    //   row 0: block columns {0, 2}, row 2: block columns {1, 2}.
    HostGebsr<float> make_gebsr(rocsparse_direction dir, int rbd, int cbd, int base)
    {
        HostGebsr<float> g;
        g.dir     = dir;
        g.mb      = 3;
        g.nb      = 3;
        g.rbd     = rbd;
        g.cbd     = cbd;
        g.base    = base;
        g.row_ptr = {base + 0, base + 2, base + 2, base + 4};
        g.col_ind = {base + 0, base + 2, base + 1, base + 2};
        g.val.assign(static_cast<size_t>(4) * rbd * cbd, 0.0f);
        for(int k = 0; k < 4; ++k)
        {
            for(int ii = 0; ii < rbd; ++ii)
            {
                for(int jj = 0; jj < cbd; ++jj)
                {
                    const size_t off = (dir == rocsparse_direction_row)
                                           ? static_cast<size_t>(k) * rbd * cbd + ii * cbd + jj
                                           : static_cast<size_t>(k) * rbd * cbd + jj * rbd + ii;
                    g.val[off] = pattern_value(static_cast<size_t>(k) * rbd * cbd + ii * cbd + jj);
                }
            }
        }
        return g;
    }

    // Row-major dense expansion of a gebsr matrix, zero padded to rows x cols.
    std::vector<float> gebsr_to_dense(const HostGebsr<float>& g, int rows, int cols)
    {
        std::vector<float> dense(static_cast<size_t>(rows) * cols, 0.0f);
        for(int ib = 0; ib < g.mb; ++ib)
        {
            for(int k = g.row_ptr[ib] - g.base; k < g.row_ptr[ib + 1] - g.base; ++k)
            {
                const int jb = g.col_ind[k] - g.base;
                for(int ii = 0; ii < g.rbd; ++ii)
                {
                    for(int jj = 0; jj < g.cbd; ++jj)
                    {
                        const size_t off
                            = (g.dir == rocsparse_direction_row)
                                  ? static_cast<size_t>(k) * g.rbd * g.cbd + ii * g.cbd + jj
                                  : static_cast<size_t>(k) * g.rbd * g.cbd + jj * g.rbd + ii;
                        const int row = ib * g.rbd + ii;
                        const int col = jb * g.cbd + jj;
                        EXPECT_LT(row, rows);
                        EXPECT_LT(col, cols);
                        if(row < rows && col < cols)
                        {
                            dense[static_cast<size_t>(row) * cols + col] = g.val[off];
                        }
                    }
                }
            }
        }
        return dense;
    }

    // Row-major dense expansion of a csr matrix.
    std::vector<float> csr_to_dense(const std::vector<int>&   row_ptr,
                                    const std::vector<int>&   col_ind,
                                    const std::vector<float>& val,
                                    int                       base,
                                    int                       rows,
                                    int                       cols)
    {
        std::vector<float> dense(static_cast<size_t>(rows) * cols, 0.0f);
        for(int i = 0; i < rows; ++i)
        {
            for(int k = row_ptr[i] - base; k < row_ptr[i + 1] - base; ++k)
            {
                dense[static_cast<size_t>(i) * cols + (col_ind[k] - base)] = val[k];
            }
        }
        return dense;
    }

    // Read a device gebsr matrix back to the host.
    HostGebsr<float> read_gebsr(rocsparse_direction         dir,
                                int                         mb,
                                int                         nb,
                                int                         rbd,
                                int                         cbd,
                                int                         base,
                                const device_vector<int>&   row_ptr,
                                const device_vector<int>&   col_ind,
                                const device_vector<float>& val)
    {
        HostGebsr<float> g;
        g.dir          = dir;
        g.mb           = mb;
        g.nb           = nb;
        g.rbd          = rbd;
        g.cbd          = cbd;
        g.base         = base;
        g.row_ptr      = to_host(row_ptr.ptr, static_cast<size_t>(mb) + 1);
        const int nnzb = g.row_ptr.back() - base;
        g.col_ind      = to_host(col_ind.ptr, static_cast<size_t>(nnzb));
        g.val          = to_host(val.ptr, static_cast<size_t>(nnzb) * rbd * cbd);
        return g;
    }

    // Dense matrix with an irregular pattern (an empty row, rows of different
    // lengths), values distinct and nonzero where stored.
    std::vector<float> make_csr_dense(int m, int n)
    {
        std::vector<float> dense(static_cast<size_t>(m) * n, 0.0f);
        for(int i = 0; i < m; ++i)
        {
            if(i % 11 == 5)
            {
                continue;
            }
            for(int j = 0; j < n; ++j)
            {
                if((i * 3 + j * 5) % 7 == 0 || (i + j) % 13 == 0)
                {
                    dense[static_cast<size_t>(i) * n + j]
                        = pattern_value(static_cast<size_t>(i) * n + j);
                }
            }
        }
        return dense;
    }

    void dense_to_csr(const std::vector<float>& dense,
                      int                       m,
                      int                       n,
                      int                       base,
                      std::vector<int>&         row_ptr,
                      std::vector<int>&         col_ind,
                      std::vector<float>&       val)
    {
        row_ptr.assign(1, base);
        col_ind.clear();
        val.clear();
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(dense[static_cast<size_t>(i) * n + j] != 0.0f)
                {
                    col_ind.push_back(j + base);
                    val.push_back(dense[static_cast<size_t>(i) * n + j]);
                }
            }
            row_ptr.push_back(static_cast<int>(val.size()) + base);
        }
    }

    // Number of nonzero (rbd x cbd) blocks of every block row of a row-major dense
    // matrix whose dimensions are multiples of the block dimensions.
    std::vector<int> nonzero_blocks_per_row(
        const std::vector<float>& dense, int rows, int cols, int rbd, int cbd)
    {
        std::vector<int> counts(static_cast<size_t>(rows / rbd), 0);
        for(int ib = 0; ib < rows / rbd; ++ib)
        {
            for(int jb = 0; jb < cols / cbd; ++jb)
            {
                bool any = false;
                for(int ii = 0; ii < rbd && !any; ++ii)
                    for(int jj = 0; jj < cbd && !any; ++jj)
                        any = dense[static_cast<size_t>(ib * rbd + ii) * cols + jb * cbd + jj]
                              != 0.0f;
                counts[static_cast<size_t>(ib)] += any ? 1 : 0;
            }
        }
        return counts;
    }

    // Block row pointer differences must equal the host block counts.
    void expect_block_rows(const std::vector<int>& row_ptr,
                           int                     base,
                           const std::vector<int>& expected_counts)
    {
        ASSERT_EQ(row_ptr.size(), expected_counts.size() + 1);
        for(size_t ib = 0; ib < expected_counts.size(); ++ib)
        {
            EXPECT_EQ(row_ptr[ib + 1] - row_ptr[ib], expected_counts[ib]) << "block row " << ib;
        }
        EXPECT_EQ(row_ptr[0], base);
    }

    // Block dimensions of the gebsr2csr dispatch buckets (<= 2, <= 4, <= 8, <= 16,
    // <= 32, <= 64, <= 128). The row dimension i and the column dimension i of the
    // two lists fall into the same bucket, so a grid over both lists reaches every
    // pair of buckets including the seven diagonal ones.
    const int s_row_dims[] = {1, 3, 6, 12, 24, 48, 96};
    const int s_col_dims[] = {2, 4, 8, 16, 32, 64, 128};

    // gebsr2csr of the 3 x 3 block test matrix, compared with the host reference.
    void run_gebsr2csr(rocsparse_handle    handle,
                       rocsparse_direction dir,
                       int                 rbd,
                       int                 cbd,
                       int                 bsr_base,
                       int                 csr_base)
    {
        SCOPED_TRACE(std::string("dir=") + dir_name(dir) + " rbd=" + std::to_string(rbd)
                     + " cbd=" + std::to_string(cbd) + " bsr_base=" + std::to_string(bsr_base)
                     + " csr_base=" + std::to_string(csr_base));

        const HostGebsr<float> g = make_gebsr(dir, rbd, cbd, bsr_base);
        const int              m = g.mb * rbd;
        const int              n = g.nb * cbd;

        std::vector<int>   ref_ptr, ref_ind;
        std::vector<float> ref_val;
        host_gebsr2csr(g, csr_base, ref_ptr, ref_ind, ref_val);
        ASSERT_EQ(ref_val.size(), static_cast<size_t>(4) * rbd * cbd);

        MatDescr bsr_descr(to_base(bsr_base)), csr_descr(to_base(csr_base));
        ASSERT_TRUE(bsr_descr.d && csr_descr.d);

        device_vector<int>   d_bsr_ptr(g.row_ptr);
        device_vector<int>   d_bsr_ind(g.col_ind);
        device_vector<float> d_bsr_val(g.val);
        device_vector<int>   d_csr_ptr(static_cast<size_t>(m) + 1);
        device_vector<int>   d_csr_ind(ref_val.size());
        device_vector<float> d_csr_val(ref_val.size());
        ASSERT_TRUE(d_bsr_ptr.ptr && d_bsr_ind.ptr && d_bsr_val.ptr && d_csr_ptr.ptr
                    && d_csr_ind.ptr && d_csr_val.ptr);

        ASSERT_EQ(rocsparse_sgebsr2csr(handle,
                                       dir,
                                       g.mb,
                                       g.nb,
                                       bsr_descr.d,
                                       d_bsr_val,
                                       d_bsr_ptr,
                                       d_bsr_ind,
                                       rbd,
                                       cbd,
                                       csr_descr.d,
                                       d_csr_val,
                                       d_csr_ptr,
                                       d_csr_ind),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        EXPECT_EQ(to_host(d_csr_ptr.ptr, static_cast<size_t>(m) + 1), ref_ptr);
        EXPECT_EQ(to_host(d_csr_ind.ptr, ref_ind.size()), ref_ind);
        EXPECT_EQ(to_host(d_csr_val.ptr, ref_val.size()), ref_val);

        // The CSR matrix describes the same dense matrix as the gebsr one.
        EXPECT_EQ(csr_to_dense(ref_ptr, ref_ind, ref_val, csr_base, m, n), gebsr_to_dense(g, m, n));
    }

    class Cov70ConvGebsr : public HandleTest
    {
    };
}

// gebsr2csr for every pair of dispatch buckets of the row and the column block
// dimension (49 pairs, the seven diagonal ones included), both block directions;
// the index bases of the two matrices are varied over the grid.
TEST_F(Cov70ConvGebsr, Gebsr2CsrBucketGrid)
{
    int combo = 0;
    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(int rbd : s_row_dims)
        {
            for(int cbd : s_col_dims)
            {
                const int bsr_base = combo % 2;
                const int csr_base = (combo / 2) % 2;
                ++combo;
                run_gebsr2csr(handle, dir, rbd, cbd, bsr_base, csr_base);
                if(::testing::Test::HasFatalFailure())
                {
                    return;
                }
            }
        }
    }
}

// Equal row and column block dimensions are converted through the BSR code path.
TEST_F(Cov70ConvGebsr, Gebsr2CsrEqualBlockDims)
{
    int combo = 0;
    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(int bd : {1, 3, 24, 96})
        {
            for(int bases = 0; bases < 4; ++bases)
            {
                ++combo;
                run_gebsr2csr(handle, dir, bd, bd, bases % 2, bases / 2);
                if(::testing::Test::HasFatalFailure())
                {
                    return;
                }
            }
        }
    }
    EXPECT_GT(combo, 0);
}

// gebsr2csr (3 x 5 blocks) in the other precisions, empty matrices and invalid
// arguments.
TEST_F(Cov70ConvGebsr, Gebsr2CsrPrecisionsAndArguments)
{
    const int rbd = 3, cbd = 5;

    // double and complex precisions on one combination.
    {
        const HostGebsr<float> g = make_gebsr(rocsparse_direction_row, rbd, cbd, 0);
        const int              m = g.mb * rbd;
        std::vector<int>       ref_ptr, ref_ind;
        std::vector<float>     ref_val;
        host_gebsr2csr(g, 0, ref_ptr, ref_ind, ref_val);

        MatDescr bsr_descr, csr_descr;
        ASSERT_TRUE(bsr_descr.d && csr_descr.d);
        device_vector<int> d_bsr_ptr(g.row_ptr);
        device_vector<int> d_bsr_ind(g.col_ind);
        device_vector<int> d_csr_ptr(static_cast<size_t>(m) + 1);
        device_vector<int> d_csr_ind(ref_val.size());

        {
            std::vector<double>   hv(g.val.begin(), g.val.end());
            device_vector<double> d_bsr_val(hv);
            device_vector<double> d_csr_val(ref_val.size());
            ASSERT_EQ(rocsparse_dgebsr2csr(handle,
                                           rocsparse_direction_row,
                                           g.mb,
                                           g.nb,
                                           bsr_descr.d,
                                           d_bsr_val,
                                           d_bsr_ptr,
                                           d_bsr_ind,
                                           rbd,
                                           cbd,
                                           csr_descr.d,
                                           d_csr_val,
                                           d_csr_ptr,
                                           d_csr_ind),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            std::vector<double> expected(ref_val.begin(), ref_val.end());
            EXPECT_EQ(to_host(d_csr_ptr.ptr, static_cast<size_t>(m) + 1), ref_ptr);
            EXPECT_EQ(to_host(d_csr_ind.ptr, ref_ind.size()), ref_ind);
            EXPECT_EQ(to_host(d_csr_val.ptr, expected.size()), expected);
        }
        {
            std::vector<rocsparse_float_complex> hv;
            for(size_t i = 0; i < g.val.size(); ++i)
            {
                hv.push_back(rocsparse_float_complex(g.val[i], -g.val[i]));
            }
            device_vector<rocsparse_float_complex> d_bsr_val(hv);
            device_vector<rocsparse_float_complex> d_csr_val(ref_val.size());
            ASSERT_EQ(rocsparse_cgebsr2csr(handle,
                                           rocsparse_direction_row,
                                           g.mb,
                                           g.nb,
                                           bsr_descr.d,
                                           d_bsr_val,
                                           d_bsr_ptr,
                                           d_bsr_ind,
                                           rbd,
                                           cbd,
                                           csr_descr.d,
                                           d_csr_val,
                                           d_csr_ptr,
                                           d_csr_ind),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            std::vector<rocsparse_float_complex> expected;
            for(float v : ref_val)
            {
                expected.push_back(rocsparse_float_complex(v, -v));
            }
            EXPECT_EQ(to_host(d_csr_val.ptr, expected.size()), expected);
        }
        {
            std::vector<rocsparse_double_complex> hv;
            for(size_t i = 0; i < g.val.size(); ++i)
            {
                hv.push_back(rocsparse_double_complex(g.val[i], 2.0 * g.val[i]));
            }
            device_vector<rocsparse_double_complex> d_bsr_val(hv);
            device_vector<rocsparse_double_complex> d_csr_val(ref_val.size());
            ASSERT_EQ(rocsparse_zgebsr2csr(handle,
                                           rocsparse_direction_row,
                                           g.mb,
                                           g.nb,
                                           bsr_descr.d,
                                           d_bsr_val,
                                           d_bsr_ptr,
                                           d_bsr_ind,
                                           rbd,
                                           cbd,
                                           csr_descr.d,
                                           d_csr_val,
                                           d_csr_ptr,
                                           d_csr_ind),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            std::vector<rocsparse_double_complex> expected;
            for(float v : ref_val)
            {
                expected.push_back(rocsparse_double_complex(v, 2.0 * v));
            }
            EXPECT_EQ(to_host(d_csr_val.ptr, expected.size()), expected);
        }
    }

    // Empty matrices: the row pointer is filled with the index base.
    for(int base : {0, 1})
    {
        for(int mb_nb : {0, 1})
        {
            SCOPED_TRACE("base=" + std::to_string(base) + " mb_nb=" + std::to_string(mb_nb));
            const int mb = mb_nb == 0 ? 0 : 2;
            const int nb = mb_nb == 0 ? 3 : 0;
            const int m  = mb * rbd;

            MatDescr bsr_descr, csr_descr(to_base(base));
            ASSERT_TRUE(bsr_descr.d && csr_descr.d);
            device_vector<int> d_bsr_ptr(std::vector<int>(static_cast<size_t>(mb) + 1, 0));
            device_vector<int> d_csr_ptr(std::vector<int>(static_cast<size_t>(m) + 1, -3));
            ASSERT_TRUE(d_bsr_ptr.ptr && d_csr_ptr.ptr);

            EXPECT_EQ(rocsparse_sgebsr2csr(handle,
                                           rocsparse_direction_row,
                                           mb,
                                           nb,
                                           bsr_descr.d,
                                           nullptr,
                                           d_bsr_ptr,
                                           nullptr,
                                           rbd,
                                           cbd,
                                           csr_descr.d,
                                           nullptr,
                                           d_csr_ptr,
                                           nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(to_host(d_csr_ptr.ptr, static_cast<size_t>(m) + 1),
                      std::vector<int>(static_cast<size_t>(m) + 1, base));
        }
    }

    // Invalid arguments.
    {
        const HostGebsr<float> g = make_gebsr(rocsparse_direction_row, rbd, cbd, 0);
        const int              m = g.mb * rbd;
        MatDescr               bsr_descr, csr_descr;
        device_vector<int>     d_bsr_ptr(g.row_ptr);
        device_vector<int>     d_bsr_ind(g.col_ind);
        device_vector<float>   d_bsr_val(g.val);
        device_vector<int>     d_csr_ptr(static_cast<size_t>(m) + 1);
        device_vector<int>     d_csr_ind(static_cast<size_t>(4) * rbd * cbd);
        device_vector<float>   d_csr_val(static_cast<size_t>(4) * rbd * cbd);
        ASSERT_TRUE(d_bsr_ptr.ptr && d_bsr_ind.ptr && d_bsr_val.ptr && d_csr_ptr.ptr
                    && d_csr_ind.ptr && d_csr_val.ptr);

        auto call = [&](rocsparse_handle    h,
                        rocsparse_direction dir,
                        int                 rb,
                        int                 cb,
                        const float*        bval,
                        const int*          bind,
                        float*              cval,
                        int*                cind) {
            return rocsparse_sgebsr2csr(h,
                                        dir,
                                        g.mb,
                                        g.nb,
                                        bsr_descr.d,
                                        bval,
                                        d_bsr_ptr,
                                        bind,
                                        rb,
                                        cb,
                                        csr_descr.d,
                                        cval,
                                        d_csr_ptr,
                                        cind);
        };

        EXPECT_EQ(call(handle,
                       rocsparse_direction_row,
                       rbd,
                       cbd,
                       d_bsr_val,
                       d_bsr_ind,
                       d_csr_val,
                       d_csr_ind),
                  rocsparse_status_success);
        EXPECT_EQ(call(nullptr,
                       rocsparse_direction_row,
                       rbd,
                       cbd,
                       d_bsr_val,
                       d_bsr_ind,
                       d_csr_val,
                       d_csr_ind),
                  rocsparse_status_invalid_handle);
        // An invalid direction is not documented: the library convention for an
        // invalid enumeration is invalid_value.
        COV70_EXPECT_STATUS_IN(call(handle,
                                    static_cast<rocsparse_direction>(77),
                                    rbd,
                                    cbd,
                                    d_bsr_val,
                                    d_bsr_ind,
                                    d_csr_val,
                                    d_csr_ind),
                               rocsparse_status_invalid_value,
                               rocsparse_status_invalid_size);
        EXPECT_EQ(call(handle,
                       rocsparse_direction_row,
                       0,
                       cbd,
                       d_bsr_val,
                       d_bsr_ind,
                       d_csr_val,
                       d_csr_ind),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(call(handle,
                       rocsparse_direction_row,
                       rbd,
                       0,
                       d_bsr_val,
                       d_bsr_ind,
                       d_csr_val,
                       d_csr_ind),
                  rocsparse_status_invalid_size);
        // Arrays that cannot hold the nnzb blocks found in the row pointer.
        EXPECT_EQ(call(handle,
                       rocsparse_direction_row,
                       rbd,
                       cbd,
                       nullptr,
                       d_bsr_ind,
                       d_csr_val,
                       d_csr_ind),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       rocsparse_direction_row,
                       rbd,
                       cbd,
                       d_bsr_val,
                       nullptr,
                       d_csr_val,
                       d_csr_ind),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       rocsparse_direction_row,
                       rbd,
                       cbd,
                       d_bsr_val,
                       d_bsr_ind,
                       nullptr,
                       d_csr_ind),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       rocsparse_direction_row,
                       rbd,
                       cbd,
                       d_bsr_val,
                       d_bsr_ind,
                       d_csr_val,
                       nullptr),
                  rocsparse_status_invalid_pointer);

        // Only the general matrix type is supported.
        ASSERT_EQ(rocsparse_set_mat_type(bsr_descr.d, rocsparse_matrix_type_symmetric),
                  rocsparse_status_success);
        // Only general matrices are documented as supported; the status of the
        // refusal is not documented.
        COV70_EXPECT_REJECTED(call(
            handle, rocsparse_direction_row, rbd, cbd, d_bsr_val, d_bsr_ind, d_csr_val, d_csr_ind));
        ASSERT_EQ(rocsparse_set_mat_type(bsr_descr.d, rocsparse_matrix_type_general),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_type(csr_descr.d, rocsparse_matrix_type_symmetric),
                  rocsparse_status_success);
        // Only general matrices are documented as supported; the status of the
        // refusal is not documented.
        COV70_EXPECT_REJECTED(call(
            handle, rocsparse_direction_row, rbd, cbd, d_bsr_val, d_bsr_ind, d_csr_val, d_csr_ind));
    }
}

namespace
{
    // gebsr -> gebsr with the output block dimensions (rbd_c, cbd_c).
    void run_gebsr2gebsr(rocsparse_handle    handle,
                         rocsparse_direction dir,
                         int                 rbd_a,
                         int                 cbd_a,
                         int                 rbd_c,
                         int                 cbd_c,
                         int                 base_a,
                         int                 base_c,
                         bool                device_mode,
                         bool                shrink)
    {
        SCOPED_TRACE(std::string("dir=") + dir_name(dir) + " A=" + std::to_string(rbd_a) + "x"
                     + std::to_string(cbd_a) + " C=" + std::to_string(rbd_c) + "x"
                     + std::to_string(cbd_c) + " base_a=" + std::to_string(base_a) + " base_c="
                     + std::to_string(base_c) + " device_mode=" + std::to_string(device_mode)
                     + " shrink=" + std::to_string(shrink));

        const HostGebsr<float> a    = make_gebsr(dir, rbd_a, cbd_a, base_a);
        const int              m    = a.mb * rbd_a;
        const int              n    = a.nb * cbd_a;
        const int              mb_c = (m + rbd_c - 1) / rbd_c;
        const int              nb_c = (n + cbd_c - 1) / cbd_c;
        const int              nnzb = 4;

        MatDescr descr_a(to_base(base_a)), descr_c(to_base(base_c));
        ASSERT_TRUE(descr_a.d && descr_c.d);

        device_vector<int>   d_ptr_a(a.row_ptr);
        device_vector<int>   d_ind_a(a.col_ind);
        device_vector<float> d_val_a(a.val);
        ASSERT_TRUE(d_ptr_a.ptr && d_ind_a.ptr && d_val_a.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_sgebsr2gebsr_buffer_size(handle,
                                                     dir,
                                                     a.mb,
                                                     a.nb,
                                                     nnzb,
                                                     descr_a.d,
                                                     d_val_a,
                                                     d_ptr_a,
                                                     d_ind_a,
                                                     rbd_a,
                                                     cbd_a,
                                                     rbd_c,
                                                     cbd_c,
                                                     &buffer_size),
                  rocsparse_status_success);
        // The size is the library's choice (the paths with an output row block
        // dimension above 32 need scratch space, the others none); the routines
        // below run with exactly the reported size.

        device_vector<char> temp(std::max(buffer_size, size_t(1)));
        ASSERT_TRUE(temp.ptr);
        device_vector<int> d_ptr_c(static_cast<size_t>(mb_c) + 1);
        ASSERT_TRUE(d_ptr_c.ptr);

        PointerModeScope pm(
            handle, device_mode ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
        std::unique_ptr<ShrinkHandleBuffer> shrinker;
        if(shrink)
        {
            shrinker.reset(new ShrinkHandleBuffer(handle));
        }

        int                nnz_host = -1;
        device_vector<int> nnz_dev(static_cast<size_t>(1));
        ASSERT_TRUE(nnz_dev.ptr);
        int* nnz_ptr = device_mode ? nnz_dev.ptr : &nnz_host;

        ASSERT_EQ(rocsparse_gebsr2gebsr_nnz(handle,
                                            dir,
                                            a.mb,
                                            a.nb,
                                            nnzb,
                                            descr_a.d,
                                            d_ptr_a,
                                            d_ind_a,
                                            rbd_a,
                                            cbd_a,
                                            descr_c.d,
                                            d_ptr_c,
                                            rbd_c,
                                            cbd_c,
                                            nnz_ptr,
                                            temp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const int nnzb_c = device_mode ? to_host(nnz_dev.ptr, 1)[0] : nnz_host;

        // Same dense matrix, padded with zeros to the output block grid. All stored
        // values of the input are nonzero, so the output blocks that hold a nonzero
        // are exactly the blocks the conversion has to keep.
        const int                rows = mb_c * rbd_c;
        const int                cols = nb_c * cbd_c;
        std::vector<float>       expected(static_cast<size_t>(rows) * cols, 0.0f);
        const std::vector<float> dense_a = gebsr_to_dense(a, m, n);
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                expected[static_cast<size_t>(i) * cols + j]
                    = dense_a[static_cast<size_t>(i) * n + j];
            }
        }
        const std::vector<int> expected_counts
            = nonzero_blocks_per_row(expected, rows, cols, rbd_c, cbd_c);
        int expected_nnzb = 0;
        for(int cnt : expected_counts)
            expected_nnzb += cnt;
        ASSERT_GT(expected_nnzb, 0);
        EXPECT_EQ(nnzb_c, expected_nnzb);
        ASSERT_EQ(nnzb_c, expected_nnzb);
        const std::vector<int> nnz_row_ptr = to_host(d_ptr_c.ptr, static_cast<size_t>(mb_c) + 1);
        EXPECT_EQ(nnz_row_ptr.back() - base_c, nnzb_c);
        expect_block_rows(nnz_row_ptr, base_c, expected_counts);

        device_vector<int>   d_ind_c(static_cast<size_t>(nnzb_c));
        device_vector<float> d_val_c(static_cast<size_t>(nnzb_c) * rbd_c * cbd_c);
        ASSERT_TRUE(d_ind_c.ptr && d_val_c.ptr);

        ASSERT_EQ(rocsparse_sgebsr2gebsr(handle,
                                         dir,
                                         a.mb,
                                         a.nb,
                                         nnzb,
                                         descr_a.d,
                                         d_val_a,
                                         d_ptr_a,
                                         d_ind_a,
                                         rbd_a,
                                         cbd_a,
                                         descr_c.d,
                                         d_val_c,
                                         d_ptr_c,
                                         d_ind_c,
                                         rbd_c,
                                         cbd_c,
                                         temp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const HostGebsr<float> c
            = read_gebsr(dir, mb_c, nb_c, rbd_c, cbd_c, base_c, d_ptr_c, d_ind_c, d_val_c);

        // The block column indices of every block row are strictly increasing and in range.
        for(int ib = 0; ib < mb_c; ++ib)
        {
            for(int k = c.row_ptr[ib] - base_c; k < c.row_ptr[ib + 1] - base_c; ++k)
            {
                EXPECT_GE(c.col_ind[k] - base_c, 0);
                EXPECT_LT(c.col_ind[k] - base_c, nb_c);
                if(k > c.row_ptr[ib] - base_c)
                {
                    EXPECT_GT(c.col_ind[k], c.col_ind[k - 1]);
                }
            }
        }

        // The converted matrix holds the same dense matrix.
        expect_block_rows(c.row_ptr, base_c, expected_counts);
        EXPECT_EQ(gebsr_to_dense(c, rows, cols), expected);
    }
}

// gebsr2gebsr over a grid of output block sizes: the fast kernels (row block
// dimension <= 32, per power of two bucket) and the CSR based path (> 32), both
// directions. For every output row block dimension the 16 runs (direction, input
// block size, output column block dimension) use all 16 combinations of
// (base of A, base of C, host/device pointer mode, large/small handle buffer), so
// each combination occurs on both code paths and for every bucket.
TEST_F(Cov70ConvGebsr, Gebsr2GebsrGrid)
{
    int dir_index = 0;
    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        int dims_index = 0;
        for(auto dims_a : {std::pair<int, int>{3, 2}, std::pair<int, int>{4, 5}})
        {
            for(int rbd_c : {1, 2, 3, 4, 7, 8, 13, 16, 25, 32, 40, 64})
            {
                int cbd_index = 0;
                for(int cbd_c : {2, 5, 16, 33})
                {
                    const int  combo       = cbd_index + 4 * (dims_index + 2 * dir_index);
                    const int  base_a      = combo % 2;
                    const int  base_c      = (combo / 2) % 2;
                    const bool device_mode = (combo / 4) % 2 == 1;
                    const bool shrink      = (combo / 8) % 2 == 1;
                    ++cbd_index;
                    run_gebsr2gebsr(handle,
                                    dir,
                                    dims_a.first,
                                    dims_a.second,
                                    rbd_c,
                                    cbd_c,
                                    base_a,
                                    base_c,
                                    device_mode,
                                    shrink);
                    if(::testing::Test::HasFatalFailure())
                    {
                        return;
                    }
                }
            }
            ++dims_index;
        }
        ++dir_index;
    }
}

// gebsr2gebsr with empty matrices: buffer size of the two paths, nnz of a matrix
// without blocks (host and device pointer modes) and the compute quick return.
TEST_F(Cov70ConvGebsr, Gebsr2GebsrEmpty)
{
    const int rbd_a = 3, cbd_a = 2;
    const int mb = 2, nb = 3;

    for(int rbd_c : {4, 40})
    {
        const int cbd_c = 3;

        // Without block rows / columns.
        for(int case_id = 0; case_id < 2; ++case_id)
        {
            SCOPED_TRACE("rbd_c=" + std::to_string(rbd_c) + " case=" + std::to_string(case_id));
            const int mb_e = case_id == 0 ? 0 : mb;
            const int nb_e = case_id == 0 ? nb : 0;

            MatDescr descr_a, descr_c;
            ASSERT_TRUE(descr_a.d && descr_c.d);
            device_vector<int> d_ptr_a(std::vector<int>(static_cast<size_t>(mb_e) + 1, 0));
            device_vector<int> d_ptr_c(std::vector<int>(static_cast<size_t>(8), -4));
            ASSERT_TRUE(d_ptr_a.ptr && d_ptr_c.ptr);

            size_t buffer_size = 1;
            ASSERT_EQ(rocsparse_sgebsr2gebsr_buffer_size(handle,
                                                         rocsparse_direction_row,
                                                         mb_e,
                                                         nb_e,
                                                         0,
                                                         descr_a.d,
                                                         nullptr,
                                                         d_ptr_a,
                                                         nullptr,
                                                         rbd_a,
                                                         cbd_a,
                                                         rbd_c,
                                                         cbd_c,
                                                         &buffer_size),
                      rocsparse_status_success);
            // The size is the library's choice; the nnz routine below gets a buffer of
            // exactly the reported size (at least one byte, a null buffer is rejected
            // on the CSR based path even for empty matrices).

            device_vector<char> temp(std::max(buffer_size, size_t(1)));
            ASSERT_TRUE(temp.ptr);

            int nnz_total = -1;
            EXPECT_EQ(rocsparse_gebsr2gebsr_nnz(handle,
                                                rocsparse_direction_row,
                                                mb_e,
                                                nb_e,
                                                0,
                                                descr_a.d,
                                                d_ptr_a,
                                                nullptr,
                                                rbd_a,
                                                cbd_a,
                                                descr_c.d,
                                                d_ptr_c,
                                                rbd_c,
                                                cbd_c,
                                                &nnz_total,
                                                temp.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(nnz_total, 0);
            {
                // The output row pointer is the index base (zero) for every output
                // block row; entries beyond it keep their initial value.
                const int              mb_c = (mb_e * rbd_a + rbd_c - 1) / rbd_c;
                const std::vector<int> got  = to_host(d_ptr_c.ptr, static_cast<size_t>(8));
                for(int ib = 0; ib <= mb_c; ++ib)
                {
                    EXPECT_EQ(got[static_cast<size_t>(ib)], 0) << "entry " << ib;
                }
                for(int ib = mb_c + 1; ib < 8; ++ib)
                {
                    EXPECT_EQ(got[static_cast<size_t>(ib)], -4) << "entry " << ib;
                }
            }

            EXPECT_EQ(rocsparse_sgebsr2gebsr(handle,
                                             rocsparse_direction_row,
                                             mb_e,
                                             nb_e,
                                             0,
                                             descr_a.d,
                                             nullptr,
                                             d_ptr_a,
                                             nullptr,
                                             rbd_a,
                                             cbd_a,
                                             descr_c.d,
                                             nullptr,
                                             d_ptr_c,
                                             nullptr,
                                             rbd_c,
                                             cbd_c,
                                             temp.ptr),
                      rocsparse_status_success);
        }

        // Block rows and columns present but no blocks: the output row pointer is
        // filled with the base and no entries are reported.
        for(int base_c : {0, 1})
        {
            for(rocsparse_pointer_mode mode :
                {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device})
            {
                SCOPED_TRACE("rbd_c=" + std::to_string(rbd_c) + " base_c=" + std::to_string(base_c)
                             + " device_mode="
                             + std::to_string(mode == rocsparse_pointer_mode_device));

                const int m    = mb * rbd_a;
                const int mb_c = (m + rbd_c - 1) / rbd_c;

                MatDescr descr_a, descr_c(to_base(base_c));
                ASSERT_TRUE(descr_a.d && descr_c.d);
                PointerModeScope pm(handle, mode);

                device_vector<int> d_ptr_a(std::vector<int>(static_cast<size_t>(mb) + 1, 0));
                device_vector<int> d_ptr_c(std::vector<int>(static_cast<size_t>(mb_c) + 1, -4));
                device_vector<int> d_nnz(std::vector<int>{-6});
                ASSERT_TRUE(d_ptr_a.ptr && d_ptr_c.ptr && d_nnz.ptr);
                int  nnz_host = -8;
                int* nnz_ptr  = (mode == rocsparse_pointer_mode_device) ? d_nnz.ptr : &nnz_host;
                device_vector<char> temp(static_cast<size_t>(1) << 12);
                ASSERT_TRUE(temp.ptr);

                ASSERT_EQ(rocsparse_gebsr2gebsr_nnz(handle,
                                                    rocsparse_direction_row,
                                                    mb,
                                                    nb,
                                                    0,
                                                    descr_a.d,
                                                    d_ptr_a,
                                                    nullptr,
                                                    rbd_a,
                                                    cbd_a,
                                                    descr_c.d,
                                                    d_ptr_c,
                                                    rbd_c,
                                                    cbd_c,
                                                    nnz_ptr,
                                                    temp.ptr),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                const int nnz_c
                    = (mode == rocsparse_pointer_mode_device) ? to_host(d_nnz.ptr, 1)[0] : nnz_host;
                EXPECT_EQ(nnz_c, 0);
                EXPECT_EQ(to_host(d_ptr_c.ptr, static_cast<size_t>(mb_c) + 1),
                          std::vector<int>(static_cast<size_t>(mb_c) + 1, base_c));
            }
        }
    }
}

namespace
{
    void run_csr2gebsr(rocsparse_handle    handle,
                       rocsparse_direction dir,
                       int                 rbd,
                       int                 cbd,
                       int                 csr_base,
                       int                 bsr_base,
                       bool                device_mode,
                       bool                shrink)
    {
        SCOPED_TRACE(std::string("dir=") + dir_name(dir) + " block=" + std::to_string(rbd) + "x"
                     + std::to_string(cbd) + " csr_base=" + std::to_string(csr_base) + " bsr_base="
                     + std::to_string(bsr_base) + " device_mode=" + std::to_string(device_mode)
                     + " shrink=" + std::to_string(shrink));

        const int                m = 37, n = 29;
        const std::vector<float> dense = make_csr_dense(m, n);
        std::vector<int>         h_ptr, h_ind;
        std::vector<float>       h_val;
        dense_to_csr(dense, m, n, csr_base, h_ptr, h_ind, h_val);
        const int mb = (m + rbd - 1) / rbd;
        const int nb = (n + cbd - 1) / cbd;

        MatDescr csr_descr(to_base(csr_base)), bsr_descr(to_base(bsr_base));
        ASSERT_TRUE(csr_descr.d && bsr_descr.d);

        device_vector<int>   d_ptr(h_ptr);
        device_vector<int>   d_ind(h_ind);
        device_vector<float> d_val(h_val);
        ASSERT_TRUE(d_ptr.ptr && d_ind.ptr && d_val.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_scsr2gebsr_buffer_size(
                      handle, dir, m, n, csr_descr.d, d_val, d_ptr, d_ind, rbd, cbd, &buffer_size),
                  rocsparse_status_success);
        device_vector<char> temp(std::max(buffer_size, size_t(1)));
        ASSERT_TRUE(temp.ptr);

        device_vector<int> d_bsr_ptr(static_cast<size_t>(mb) + 1);
        ASSERT_TRUE(d_bsr_ptr.ptr);

        PointerModeScope pm(
            handle, device_mode ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
        std::unique_ptr<ShrinkHandleBuffer> shrinker;
        if(shrink)
        {
            shrinker.reset(new ShrinkHandleBuffer(handle));
        }

        int                nnz_host = -1;
        device_vector<int> nnz_dev(static_cast<size_t>(1));
        ASSERT_TRUE(nnz_dev.ptr);
        int* nnz_ptr = device_mode ? nnz_dev.ptr : &nnz_host;

        ASSERT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          dir,
                                          m,
                                          n,
                                          csr_descr.d,
                                          d_ptr,
                                          d_ind,
                                          bsr_descr.d,
                                          d_bsr_ptr,
                                          rbd,
                                          cbd,
                                          nnz_ptr,
                                          temp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const int nnzb = device_mode ? to_host(nnz_dev.ptr, 1)[0] : nnz_host;

        // The blocks to keep are those that hold at least one stored (nonzero) entry.
        const int          rows = mb * rbd;
        const int          cols = nb * cbd;
        std::vector<float> expected(static_cast<size_t>(rows) * cols, 0.0f);
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                expected[static_cast<size_t>(i) * cols + j] = dense[static_cast<size_t>(i) * n + j];
            }
        }
        const std::vector<int> expected_counts
            = nonzero_blocks_per_row(expected, rows, cols, rbd, cbd);
        int expected_nnzb = 0;
        for(int cnt : expected_counts)
            expected_nnzb += cnt;
        ASSERT_GT(expected_nnzb, 0);
        EXPECT_EQ(nnzb, expected_nnzb);
        ASSERT_EQ(nnzb, expected_nnzb);
        const std::vector<int> nnz_row_ptr = to_host(d_bsr_ptr.ptr, static_cast<size_t>(mb) + 1);
        EXPECT_EQ(nnz_row_ptr.back() - bsr_base, nnzb);
        expect_block_rows(nnz_row_ptr, bsr_base, expected_counts);

        device_vector<int>   d_bsr_ind(static_cast<size_t>(nnzb));
        device_vector<float> d_bsr_val(static_cast<size_t>(nnzb) * rbd * cbd);
        ASSERT_TRUE(d_bsr_ind.ptr && d_bsr_val.ptr);

        ASSERT_EQ(rocsparse_scsr2gebsr(handle,
                                       dir,
                                       m,
                                       n,
                                       csr_descr.d,
                                       d_val,
                                       d_ptr,
                                       d_ind,
                                       bsr_descr.d,
                                       d_bsr_val,
                                       d_bsr_ptr,
                                       d_bsr_ind,
                                       rbd,
                                       cbd,
                                       temp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const HostGebsr<float> b
            = read_gebsr(dir, mb, nb, rbd, cbd, bsr_base, d_bsr_ptr, d_bsr_ind, d_bsr_val);

        // Block columns of every block row are strictly increasing and in range.
        for(int ib = 0; ib < mb; ++ib)
        {
            for(int k = b.row_ptr[ib] - bsr_base; k < b.row_ptr[ib + 1] - bsr_base; ++k)
            {
                EXPECT_GE(b.col_ind[k] - bsr_base, 0);
                EXPECT_LT(b.col_ind[k] - bsr_base, nb);
                if(k > b.row_ptr[ib] - bsr_base)
                {
                    EXPECT_GT(b.col_ind[k], b.col_ind[k - 1]);
                }
            }
        }

        expect_block_rows(b.row_ptr, bsr_base, expected_counts);
        EXPECT_EQ(gebsr_to_dense(b, rows, cols), expected);
    }
}

// csr2gebsr over a grid of block sizes covering the buckets of the kernel dispatch
// for both directions. The index bases, pointer modes and scratch buffer situations
// (handle buffer large enough or too small) are varied across the grid; not every
// combination occurs for every block size.
TEST_F(Cov70ConvGebsr, Csr2GebsrGrid)
{
    int combo = 0;
    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(int rbd : {1, 2, 3, 4, 7, 8, 13, 16, 25, 32, 33, 64})
        {
            for(int cbd : {1, 3, 8, 17, 33})
            {
                const int  csr_base    = combo % 2;
                const int  bsr_base    = (combo / 2) % 2;
                const bool device_mode = (combo / 4) % 2 == 1;
                const bool shrink      = (combo / 8) % 2 == 1;
                ++combo;
                run_csr2gebsr(handle, dir, rbd, cbd, csr_base, bsr_base, device_mode, shrink);
                if(::testing::Test::HasFatalFailure())
                {
                    return;
                }
            }
        }
    }
}

// csr2gebsr quick returns and pointer checks.
TEST_F(Cov70ConvGebsr, Csr2GebsrEmptyAndArguments)
{
    const int rbd = 3, cbd = 4;

    // m == 0 or n == 0.
    for(int base : {0, 1})
    {
        for(rocsparse_pointer_mode mode :
            {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device})
        {
            for(int shape = 0; shape < 2; ++shape)
            {
                const int m = shape == 0 ? 0 : 7;
                const int n = shape == 0 ? 5 : 0;
                SCOPED_TRACE("base=" + std::to_string(base) + " shape=" + std::to_string(shape)
                             + " device_mode="
                             + std::to_string(mode == rocsparse_pointer_mode_device));

                const int mb = (m + rbd - 1) / rbd;
                MatDescr  csr_descr, bsr_descr(to_base(base));
                ASSERT_TRUE(csr_descr.d && bsr_descr.d);
                PointerModeScope pm(handle, mode);

                device_vector<int> d_csr_ptr(std::vector<int>(static_cast<size_t>(m) + 1, 0));
                device_vector<int> d_bsr_ptr(std::vector<int>(static_cast<size_t>(mb) + 1, -2));
                device_vector<int> d_nnz(std::vector<int>{-5});
                ASSERT_TRUE(d_csr_ptr.ptr && d_bsr_ptr.ptr && d_nnz.ptr);
                int  nnz_host = -9;
                int* nnz_ptr  = (mode == rocsparse_pointer_mode_device) ? d_nnz.ptr : &nnz_host;

                ASSERT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  m,
                                                  n,
                                                  csr_descr.d,
                                                  d_csr_ptr,
                                                  nullptr,
                                                  bsr_descr.d,
                                                  d_bsr_ptr,
                                                  rbd,
                                                  cbd,
                                                  nnz_ptr,
                                                  nullptr),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                const int nnz
                    = (mode == rocsparse_pointer_mode_device) ? to_host(d_nnz.ptr, 1)[0] : nnz_host;
                EXPECT_EQ(nnz, 0);
                EXPECT_EQ(to_host(d_bsr_ptr.ptr, static_cast<size_t>(mb) + 1),
                          std::vector<int>(static_cast<size_t>(mb) + 1, base));

                // The compute routine returns immediately.
                EXPECT_EQ(rocsparse_scsr2gebsr(handle,
                                               rocsparse_direction_row,
                                               m,
                                               n,
                                               csr_descr.d,
                                               nullptr,
                                               d_csr_ptr,
                                               nullptr,
                                               bsr_descr.d,
                                               nullptr,
                                               d_bsr_ptr,
                                               nullptr,
                                               rbd,
                                               cbd,
                                               nullptr),
                          rocsparse_status_success);
            }
        }
    }

    // Missing arrays while the row pointer announces entries.
    {
        const int                m = 37, n = 29;
        const std::vector<float> dense = make_csr_dense(m, n);
        std::vector<int>         h_ptr, h_ind;
        std::vector<float>       h_val;
        dense_to_csr(dense, m, n, 0, h_ptr, h_ind, h_val);
        const int mb = (m + rbd - 1) / rbd;

        MatDescr             csr_descr, bsr_descr;
        device_vector<int>   d_ptr(h_ptr);
        device_vector<int>   d_ind(h_ind);
        device_vector<float> d_val(h_val);
        device_vector<int>   d_bsr_ptr(static_cast<size_t>(mb) + 1);
        device_vector<char>  temp(static_cast<size_t>(1) << 16);
        ASSERT_TRUE(d_ptr.ptr && d_ind.ptr && d_val.ptr && d_bsr_ptr.ptr && temp.ptr);

        int nnz = 0;
        EXPECT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          rocsparse_direction_row,
                                          m,
                                          n,
                                          csr_descr.d,
                                          d_ptr,
                                          nullptr,
                                          bsr_descr.d,
                                          d_bsr_ptr,
                                          rbd,
                                          cbd,
                                          &nnz,
                                          temp.ptr),
                  rocsparse_status_invalid_pointer);

        ASSERT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          rocsparse_direction_row,
                                          m,
                                          n,
                                          csr_descr.d,
                                          d_ptr,
                                          d_ind,
                                          bsr_descr.d,
                                          d_bsr_ptr,
                                          rbd,
                                          cbd,
                                          &nnz,
                                          temp.ptr),
                  rocsparse_status_success);
        ASSERT_GT(nnz, 0);
        device_vector<int>   d_bsr_ind(static_cast<size_t>(nnz));
        device_vector<float> d_bsr_val(static_cast<size_t>(nnz) * rbd * cbd);

        auto compute = [&](const float* cval, const int* cind, float* bval, int* bind) {
            return rocsparse_scsr2gebsr(handle,
                                        rocsparse_direction_row,
                                        m,
                                        n,
                                        csr_descr.d,
                                        cval,
                                        d_ptr,
                                        cind,
                                        bsr_descr.d,
                                        bval,
                                        d_bsr_ptr,
                                        bind,
                                        rbd,
                                        cbd,
                                        temp.ptr);
        };
        EXPECT_EQ(compute(nullptr, d_ind, d_bsr_val, d_bsr_ind), rocsparse_status_invalid_pointer);
        EXPECT_EQ(compute(d_val, nullptr, d_bsr_val, d_bsr_ind), rocsparse_status_invalid_pointer);

        // Invalid block dimensions and directions.
        EXPECT_EQ(rocsparse_scsr2gebsr(handle,
                                       rocsparse_direction_row,
                                       m,
                                       n,
                                       csr_descr.d,
                                       d_val,
                                       d_ptr,
                                       d_ind,
                                       bsr_descr.d,
                                       d_bsr_val,
                                       d_bsr_ptr,
                                       d_bsr_ind,
                                       0,
                                       cbd,
                                       temp.ptr),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_scsr2gebsr(handle,
                                       rocsparse_direction_row,
                                       m,
                                       n,
                                       csr_descr.d,
                                       d_val,
                                       d_ptr,
                                       d_ind,
                                       bsr_descr.d,
                                       d_bsr_val,
                                       d_bsr_ptr,
                                       d_bsr_ind,
                                       rbd,
                                       0,
                                       temp.ptr),
                  rocsparse_status_invalid_size);
        // The library convention for an invalid enumeration is invalid_value; the
        // header does not document it.
        COV70_EXPECT_STATUS_IN(rocsparse_scsr2gebsr(handle,
                                                    static_cast<rocsparse_direction>(55),
                                                    m,
                                                    n,
                                                    csr_descr.d,
                                                    d_val,
                                                    d_ptr,
                                                    d_ind,
                                                    bsr_descr.d,
                                                    d_bsr_val,
                                                    d_bsr_ptr,
                                                    d_bsr_ind,
                                                    rbd,
                                                    cbd,
                                                    temp.ptr),
                               rocsparse_status_invalid_value,
                               rocsparse_status_invalid_size);
    }
}
