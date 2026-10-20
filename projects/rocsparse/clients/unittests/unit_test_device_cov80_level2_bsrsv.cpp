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
// Device unit tests of the block triangular solve rocsparse_Xbsrsv_*: every block
// dimension class, both fill modes, unit and non-unit diagonals, no transpose and
// transpose, both block storage directions, index bases and pointer modes. The
// blocks of the matrices are strictly diagonally dominant (non-unit diagonal) or
// small enough for the triangular recurrence to stay bounded (unit diagonal), and
// the diagonal blocks are triangular, so the host reference is well posed.
//
#include "unit_test_cov80_level2_common.hpp"

using namespace cov80l2;

namespace
{
    // The analysis reads the column index that precedes or follows a block row: keep
    // one guard element (never a valid index) on each side.
    inline std::vector<int32_t> guarded(const std::vector<int32_t>& v)
    {
        std::vector<int32_t> g(v.size() + 2, -7);
        std::copy(v.begin(), v.end(), g.begin() + 1);
        return g;
    }

    template <typename T>
    T mk_alpha(float a)
    {
        return rocsparse_ut::scalar<T>(a);
    }
    template <>
    rocsparse_float_complex mk_alpha<rocsparse_float_complex>(float a)
    {
        return rocsparse_float_complex(a, 0.25f);
    }
    template <>
    rocsparse_double_complex mk_alpha<rocsparse_double_complex>(float a)
    {
        return rocsparse_double_complex(a, 0.25f);
    }

    template <typename T>
    struct bsrsv_matrix
    {
        int32_t             mb   = 0;
        int32_t             bd   = 0;
        int                 base = 0;
        rocsparse_direction dir  = rocsparse_direction_row;
        // block structure (with the index base) and values in the block storage order
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;
        // the same matrix as a scalar CSR matrix (index base 0)
        host_csr<T> scalar;

        int32_t nnzb() const
        {
            return static_cast<int32_t>(ind.size());
        }
    };

    // Block rows with the block columns {i - 3, i - 1, i} (lower) or {i, i + 1, i + 3}
    // (upper). With `diag_blocks` false the diagonal block is left out.
    template <typename T>
    bsrsv_matrix<T> make_bsrsv(int32_t             mb,
                               int32_t             bd,
                               rocsparse_fill_mode fill,
                               rocsparse_direction dir,
                               int                 base,
                               bool                diag_blocks = true)
    {
        bsrsv_matrix<T> a;
        a.mb   = mb;
        a.bd   = bd;
        a.base = base;
        a.dir  = dir;

        const bool lower = (fill == rocsparse_fill_mode_lower);

        // Off-diagonal entries are scaled so that a scalar row holds at most 3 * bd of them
        // with a combined magnitude below 1.
        const T scale = rocsparse_ut::scalar<T>(1.0f / (12.0f * static_cast<float>(bd)));

        std::vector<std::vector<int32_t>> cols(mb);
        for(int32_t i = 0; i < mb; ++i)
        {
            std::vector<int32_t> c;
            if(lower)
            {
                if(i >= 3)
                    c.push_back(i - 3);
                if(i >= 1)
                    c.push_back(i - 1);
                if(diag_blocks)
                    c.push_back(i);
            }
            else
            {
                if(diag_blocks)
                    c.push_back(i);
                if(i + 1 < mb)
                    c.push_back(i + 1);
                if(i + 3 < mb)
                    c.push_back(i + 3);
            }
            cols[i] = c;
        }

        const size_t blk = static_cast<size_t>(bd) * bd;
        a.ptr.push_back(base);
        a.scalar.m    = mb * bd;
        a.scalar.n    = mb * bd;
        a.scalar.base = 0;
        a.scalar.ptr.assign(1, 0);
        int64_t counter = 0;

        // value of every scalar entry, by (scalar row, scalar column)
        std::vector<std::vector<std::pair<int32_t, T>>> rows(static_cast<size_t>(mb) * bd);

        for(int32_t i = 0; i < mb; ++i)
        {
            for(int32_t j : cols[i])
            {
                const size_t first = a.val.size();
                a.val.resize(first + blk, T{});
                for(int32_t r = 0; r < bd; ++r)
                    for(int32_t c = 0; c < bd; ++c)
                    {
                        T v{};
                        if(i == j)
                        {
                            // triangular diagonal block
                            const bool keep = lower ? (c <= r) : (c >= r);
                            if(keep)
                            {
                                v = value_of<T>(counter++, true);
                                v = (r == c) ? v + rocsparse_ut::scalar<T>(5.0f) : v * scale;
                            }
                        }
                        else
                        {
                            v = value_of<T>(counter++, true) * scale;
                        }
                        const size_t pos
                            = (dir == rocsparse_direction_row) ? r * bd + c : c * bd + r;
                        a.val[first + pos] = v;
                        if(magnitude(v) != 0.0)
                            rows[static_cast<size_t>(i) * bd + r].push_back({j * bd + c, v});
                    }
                a.ind.push_back(j + base);
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()) + base);
        }
        for(auto& r : rows)
        {
            std::sort(
                r.begin(), r.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            for(const auto& e : r)
            {
                a.scalar.ind.push_back(e.first);
                a.scalar.val.push_back(e.second);
            }
            a.scalar.ptr.push_back(static_cast<int32_t>(a.scalar.ind.size()));
        }
        return a;
    }

    template <typename T>
    struct bsrsv_api;

#define COV80_BSRSV(P, TYPE)                                  \
    template <>                                               \
    struct bsrsv_api<TYPE>                                    \
    {                                                         \
        template <typename... Args>                           \
        static rocsparse_status buffer_size(Args... args)     \
        {                                                     \
            return rocsparse_##P##bsrsv_buffer_size(args...); \
        }                                                     \
        template <typename... Args>                           \
        static rocsparse_status analysis(Args... args)        \
        {                                                     \
            return rocsparse_##P##bsrsv_analysis(args...);    \
        }                                                     \
        template <typename... Args>                           \
        static rocsparse_status solve(Args... args)           \
        {                                                     \
            return rocsparse_##P##bsrsv_solve(args...);       \
        }                                                     \
    };
    COV80_BSRSV(s, float)
    COV80_BSRSV(d, double)
    COV80_BSRSV(c, rocsparse_float_complex)
    COV80_BSRSV(z, rocsparse_double_complex)
#undef COV80_BSRSV

    struct solve_case
    {
        int32_t             bd          = 4;
        int32_t             mb          = 12;
        rocsparse_fill_mode fill        = rocsparse_fill_mode_lower;
        rocsparse_diag_type diag        = rocsparse_diag_type_non_unit;
        rocsparse_operation trans       = rocsparse_operation_none;
        rocsparse_direction dir         = rocsparse_direction_row;
        int                 base        = 0;
        bool                device      = false;
        float               alpha       = 1.0f;
        bool                diag_blocks = true;
    };

    template <typename T>
    void run_bsrsv(rocsparse_handle handle, const solve_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "bsrsv bd=" << c.bd << " mb=" << c.mb << " fill=" << c.fill
                     << " diag=" << c.diag << " trans=" << c.trans << " dir=" << c.dir
                     << " base=" << c.base << " device=" << c.device << " alpha=" << c.alpha);

        const bsrsv_matrix<T> A = make_bsrsv<T>(c.mb, c.bd, c.fill, c.dir, c.base, c.diag_blocks);
        const int32_t         m = c.mb * c.bd;

        const T              alpha = mk_alpha<T>(c.alpha);
        const std::vector<T> x     = make_vector<T>(m, 3);

        device_vector<int32_t> dptr(A.ptr), dind_buf(guarded(A.ind));
        const int32_t*         dind_ptr = dind_buf.ptr ? dind_buf.ptr + 1 : nullptr;
        device_vector<T>       dval(A.val), dx(x), dy(std::vector<T>(m, T{})),
            dalpha(std::vector<T>(1, alpha));
        ASSERT_TRUE(dptr.ptr && dind_ptr && dval.ptr && dx.ptr && dy.ptr && dalpha.ptr);

        mat_descr descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr.d, to_base(c.base)), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr.d, c.fill), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr.d, c.diag), rocsparse_status_success);
        mat_info info;
        ASSERT_TRUE(info.d);

        scoped_pointer_mode mode(
            handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);

        size_t bytes = 0;
        ASSERT_EQ(bsrsv_api<T>::buffer_size(handle,
                                            c.dir,
                                            c.trans,
                                            c.mb,
                                            A.nnzb(),
                                            descr.d,
                                            dval.ptr,
                                            dptr.ptr,
                                            dind_ptr,
                                            c.bd,
                                            info.d,
                                            &bytes),
                  rocsparse_status_success);
        device_vector<char> dbuf(std::max<size_t>(bytes, 1));
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(bsrsv_api<T>::analysis(handle,
                                         c.dir,
                                         c.trans,
                                         c.mb,
                                         A.nnzb(),
                                         descr.d,
                                         dval.ptr,
                                         dptr.ptr,
                                         dind_ptr,
                                         c.bd,
                                         info.d,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         dbuf.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(bsrsv_api<T>::solve(handle,
                                      c.dir,
                                      c.trans,
                                      c.mb,
                                      A.nnzb(),
                                      c.device ? dalpha.ptr : &alpha,
                                      descr.d,
                                      dval.ptr,
                                      dptr.ptr,
                                      dind_ptr,
                                      c.bd,
                                      info.d,
                                      dx.ptr,
                                      dy.ptr,
                                      rocsparse_solve_policy_auto,
                                      dbuf.ptr),
                  rocsparse_status_success);

        // The position lives in host or device memory according to the pointer mode.
        rocsparse_int                position = 0;
        device_vector<rocsparse_int> dposition(std::vector<rocsparse_int>(1, 0));
        ASSERT_TRUE(dposition.ptr);
        ASSERT_EQ(rocsparse_bsrsv_zero_pivot(handle, info.d, c.device ? dposition.ptr : &position),
                  rocsparse_status_success);
        if(c.device)
            position = to_host(dposition)[0];
        EXPECT_EQ(position, -1);

        const std::vector<T> want
            = host_triangular_solve<T>(A.scalar, c.fill, c.diag, c.trans, alpha, x);
        for(const auto& v : want)
            ASSERT_TRUE(std::isfinite(magnitude(v)));
        EXPECT_TRUE(expect_near(want, to_host(dy), "bsrsv y"));

        ASSERT_EQ(rocsparse_bsrsv_clear(handle, info.d), rocsparse_status_success);
    }

    class Cov80L2Bsrsv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep_dims(const std::vector<int32_t>& dims, int32_t mb)
        {
            int n = 0;
            for(const int32_t bd : dims)
                for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                    for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                        for(const auto trans :
                            {rocsparse_operation_none, rocsparse_operation_transpose})
                        {
                            solve_case c;
                            c.bd     = bd;
                            c.mb     = mb;
                            c.fill   = fill;
                            c.diag   = diag;
                            c.trans  = trans;
                            c.dir    = ((n / 2) % 2) ? rocsparse_direction_column
                                                     : rocsparse_direction_row;
                            c.base   = (n / 3) % 2;
                            c.device = (n % 3) == 1;
                            c.alpha  = (n % 4 == 3) ? -2.0f : 1.0f;
                            ++n;
                            run_bsrsv<T>(handle, c);
                            if(HasFailure())
                                return;
                        }
        }
    };

    TEST_F(Cov80L2Bsrsv, small_blocks_float)
    {
        sweep_dims<float>({1, 2, 3, 4, 5, 6, 8}, 14);
    }

    TEST_F(Cov80L2Bsrsv, medium_blocks_float)
    {
        sweep_dims<float>({9, 12, 16, 17, 24, 32}, 8);
    }

    TEST_F(Cov80L2Bsrsv, large_blocks_float)
    {
        sweep_dims<float>({33, 40}, 6);
    }

    TEST_F(Cov80L2Bsrsv, double_and_complex)
    {
        sweep_dims<double>({3, 9, 20}, 8);
        sweep_dims<rocsparse_float_complex>({4, 17}, 8);
        sweep_dims<rocsparse_double_complex>({2, 33}, 6);
    }

    TEST_F(Cov80L2Bsrsv, missing_diagonal_block_is_a_zero_pivot)
    {
        // A non-unit diagonal needs a diagonal block: the solve reports a zero pivot.
        for(const int32_t bd : {2, 9})
        {
            const bsrsv_matrix<float> A = make_bsrsv<float>(
                8, bd, rocsparse_fill_mode_lower, rocsparse_direction_row, 0, false);
            const int32_t          m = 8 * bd;
            device_vector<int32_t> dptr(A.ptr), dind_buf(guarded(A.ind));
            const int32_t*         dind_ptr = dind_buf.ptr ? dind_buf.ptr + 1 : nullptr;
            device_vector<float>   dval(A.val), dx(make_vector<float>(m, 3)),
                dy(std::vector<float>(m, 0.0f));
            ASSERT_TRUE(dptr.ptr && dind_ptr && dval.ptr && dx.ptr && dy.ptr);

            mat_descr descr;
            ASSERT_TRUE(descr.d);
            ASSERT_EQ(rocsparse_set_mat_fill_mode(descr.d, rocsparse_fill_mode_lower),
                      rocsparse_status_success);
            mat_info info;
            ASSERT_TRUE(info.d);

            size_t bytes = 0;
            ASSERT_EQ(rocsparse_sbsrsv_buffer_size(handle,
                                                   rocsparse_direction_row,
                                                   rocsparse_operation_none,
                                                   8,
                                                   A.nnzb(),
                                                   descr.d,
                                                   dval.ptr,
                                                   dptr.ptr,
                                                   dind_ptr,
                                                   bd,
                                                   info.d,
                                                   &bytes),
                      rocsparse_status_success);
            device_vector<char> dbuf(std::max<size_t>(bytes, 1));
            ASSERT_TRUE(dbuf.ptr);
            ASSERT_EQ(rocsparse_sbsrsv_analysis(handle,
                                                rocsparse_direction_row,
                                                rocsparse_operation_none,
                                                8,
                                                A.nnzb(),
                                                descr.d,
                                                dval.ptr,
                                                dptr.ptr,
                                                dind_ptr,
                                                bd,
                                                info.d,
                                                rocsparse_analysis_policy_reuse,
                                                rocsparse_solve_policy_auto,
                                                dbuf.ptr),
                      rocsparse_status_success);
            const float alpha = 1.0f;
            ASSERT_EQ(rocsparse_sbsrsv_solve(handle,
                                             rocsparse_direction_row,
                                             rocsparse_operation_none,
                                             8,
                                             A.nnzb(),
                                             &alpha,
                                             descr.d,
                                             dval.ptr,
                                             dptr.ptr,
                                             dind_ptr,
                                             bd,
                                             info.d,
                                             dx.ptr,
                                             dy.ptr,
                                             rocsparse_solve_policy_auto,
                                             dbuf.ptr),
                      rocsparse_status_success);
            rocsparse_int position = -1;
            EXPECT_EQ(rocsparse_bsrsv_zero_pivot(handle, info.d, &position),
                      rocsparse_status_zero_pivot);
            EXPECT_EQ(rocsparse_bsrsv_clear(handle, info.d), rocsparse_status_success);
        }
    }
}
