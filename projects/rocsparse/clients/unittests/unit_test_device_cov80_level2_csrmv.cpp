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

//
// Device unit tests of the CSR matrix vector product on matrices of every shape
// class: many short rows, many medium rows, long rows, a single very long row
// among short ones, and runs of consecutive long rows. They exercise the
// granularity choices of the row split, adaptive, LRB and nnz split algorithms
// through the v2 interface and compare the products with a host product.
//
#include "unit_test_cov80_level2_common.hpp"

#include <algorithm>

using namespace cov80l2;

namespace
{
    // Row i holds `len(i)` distinct sorted columns; the columns of a row are
    // (start + k * step) mod n for k < len with n prime and larger than every
    // row length, so they never repeat.
    template <typename T, typename LenFn>
    host_csr<T> make_shaped(int32_t m, int32_t n, int base, LenFn len_of)
    {
        host_csr<T> a;
        a.m    = m;
        a.n    = n;
        a.base = 0;
        a.ptr.push_back(0);
        int64_t counter = 0;
        for(int32_t i = 0; i < m; ++i)
        {
            const int32_t        len   = std::min<int32_t>(len_of(i), n - 1);
            const int32_t        start = (i * 17) % n;
            const int32_t        step  = 1 + (i % 7) * 2;
            std::vector<int32_t> cols;
            for(int32_t k = 0; k < len; ++k)
                cols.push_back((start + static_cast<int64_t>(k) * step) % n);
            std::sort(cols.begin(), cols.end());
            for(int32_t j : cols)
            {
                a.ind.push_back(j);
                a.val.push_back(value_of<T>(counter++, true));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        rebase(a, base);
        return a;
    }

    struct csr_case
    {
        rocsparse_spmv_alg  alg    = rocsparse_spmv_alg_default;
        rocsparse_operation trans  = rocsparse_operation_none;
        bool                device = false;
    };

    template <typename T>
    void run_csr(rocsparse_handle handle, const host_csr<T>& A, const csr_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "csrmv alg=" << c.alg << " trans=" << c.trans << " device=" << c.device
                     << " base=" << A.base << " m=" << A.m << " n=" << A.n << " nnz=" << A.nnz());

        const size_t xs = trans_size(c.trans, A.m, A.n, true);
        const size_t ys = trans_size(c.trans, A.m, A.n, false);

        const T alpha = rocsparse_ut::scalar<T>(1.5f);
        const T beta  = rocsparse_ut::scalar<T>(-0.5f);

        const std::vector<T> x    = make_vector<T>(xs, 2);
        const std::vector<T> y0   = make_vector<T>(ys, 9);
        const std::vector<T> want = csr_gemv(A, c.trans, alpha, x, beta, y0);

        device_csr<T>    dA(A);
        device_vector<T> dx(x), dy(y0), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             A.m,
                                             A.n,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             to_base(A.base),
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, xs, dx, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, ys, dy, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);

        scoped_pointer_mode mode(
            handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_alg, c.alg),
                  rocsparse_status_success);
        ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_operation, c.trans),
                  rocsparse_status_success);
        ASSERT_EQ(
            set_input(
                handle, descr.d, rocsparse_spmv_input_scalar_datatype, rocsparse_ut::dt_of<T>()),
            rocsparse_status_success);
        ASSERT_EQ(
            set_input(
                handle, descr.d, rocsparse_spmv_input_compute_datatype, rocsparse_ut::dt_of<T>()),
            rocsparse_status_success);

        const void* palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha);
        const void* pbeta
            = c.device ? static_cast<const void*>(dbeta.ptr) : static_cast<const void*>(&beta);

        for(const auto stage : {rocsparse_v2_spmv_stage_analysis, rocsparse_v2_spmv_stage_compute})
        {
            size_t bytes = 0;
            ASSERT_EQ(rocsparse_v2_spmv_buffer_size(
                          handle, descr.d, mat.d, vx.d, vy.d, stage, &bytes, nullptr),
                      rocsparse_status_success);
            device_vector<char> dbuf(std::max<size_t>(bytes, 1));
            ASSERT_TRUE(dbuf.ptr);
            ASSERT_EQ(rocsparse_v2_spmv(handle,
                                        descr.d,
                                        palpha,
                                        mat.d,
                                        vx.d,
                                        pbeta,
                                        vy.d,
                                        stage,
                                        bytes,
                                        bytes ? static_cast<void*>(dbuf.ptr) : nullptr,
                                        nullptr),
                      rocsparse_status_success);
        }
        EXPECT_TRUE(expect_near(want, to_host(dy), "csrmv y"));
    }

    class Cov80L2Csrmv : public rocsparse_ut::HandleTest
    {
    protected:
        // Every algorithm on one matrix; the pointer mode alternates with the
        // algorithm and the operation, the transposed product is only requested
        // from the algorithms that implement it (documented as not deterministic).
        void sweep(const host_csr<float>& A, bool with_transpose)
        {
            int n = 0;
            for(const auto alg : {rocsparse_spmv_alg_csr_rowsplit,
                                  rocsparse_spmv_alg_csr_adaptive,
                                  rocsparse_spmv_alg_csr_lrb,
                                  rocsparse_spmv_alg_csr_nnzsplit})
                for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    const bool trans_ok = (alg == rocsparse_spmv_alg_csr_rowsplit
                                           || alg == rocsparse_spmv_alg_csr_nnzsplit);
                    if(trans != rocsparse_operation_none && !(trans_ok && with_transpose))
                        continue;
                    csr_case c;
                    c.alg    = alg;
                    c.trans  = trans;
                    c.device = ((n++) % 3) == 1;
                    run_csr<float>(handle, A, c);
                    if(HasFailure())
                        return;
                }
        }
    };

    TEST_F(Cov80L2Csrmv, many_short_rows)
    {
        // 12000 rows of 30 entries.
        sweep(make_shaped<float>(12000, 5003, 0, [](int32_t) { return 30; }), true);
    }

    TEST_F(Cov80L2Csrmv, many_medium_rows)
    {
        // 12000 rows of 40 entries, one based.
        sweep(make_shaped<float>(12000, 5003, 1, [](int32_t) { return 40; }), true);
    }

    TEST_F(Cov80L2Csrmv, dense_rows)
    {
        // 3000 rows of 130 entries.
        sweep(make_shaped<float>(3000, 2011, 0, [](int32_t) { return 130; }), true);
    }

    TEST_F(Cov80L2Csrmv, one_very_long_row_among_short_ones)
    {
        sweep(make_shaped<float>(
                  6000, 4001, 0, [](int32_t i) { return i == 2500 ? 3500 : 2 + (i % 5); }),
              true);
    }

    TEST_F(Cov80L2Csrmv, consecutive_long_rows)
    {
        // Runs of consecutive rows that are longer than a workgroup can reduce at once.
        sweep(make_shaped<float>(
                  3000, 6007, 1, [](int32_t i) { return (i % 100) < 12 ? 4500 : 1 + (i % 4); }),
              false);
    }

    TEST_F(Cov80L2Csrmv, mixed_lengths_with_empty_rows)
    {
        sweep(make_shaped<float>(9000,
                                 3001,
                                 0,
                                 [](int32_t i) {
                                     if(i % 11 == 0)
                                         return 0;
                                     if(i % 97 == 5)
                                         return 700;
                                     return 1 + (i % 61);
                                 }),
              true);
    }
}
