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

//
// Device unit tests of the sparse matrix-vector products with a CSR matrix:
//   * the legacy rocsparse_Xcsrmv / rocsparse_Xcsrmv_analysis routines,
//   * the v2 spmv descriptor API (rocsparse_v2_spmv) with its algorithms,
//     mixed scalar / compute precisions and extra terms (gamma * z),
//   * the algorithm / format validation of rocsparse_spmv and rocsparse_v2_spmv.
// Results are compared with a host product on the same data.
//
#include "unit_test_device_cov70_level2_common.hpp"

#include <deque>
#include <memory>
#include <string>

using namespace cov70l2;

namespace
{
    // ----- helpers -----------------------------------------------------------------
    template <typename T>
    T make_scalar(float re, float)
    {
        return rocsparse_ut::scalar<T>(re);
    }
    template <>
    inline rocsparse_float_complex make_scalar<rocsparse_float_complex>(float re, float im)
    {
        return rocsparse_float_complex(re, im);
    }
    template <>
    inline rocsparse_double_complex make_scalar<rocsparse_double_complex>(float re, float im)
    {
        return rocsparse_double_complex(re, im);
    }

    template <typename To, typename From>
    inline To convert_scalar(const From& v)
    {
        return static_cast<To>(v);
    }

    template <typename CT, typename AT>
    host_csr<CT> cast_csr(const host_csr<AT>& a)
    {
        host_csr<CT> r;
        r.m    = a.m;
        r.n    = a.n;
        r.base = a.base;
        r.ptr  = a.ptr;
        r.ind  = a.ind;
        r.val.assign(a.val.begin(), a.val.end());
        return r;
    }

    // Rectangular CSR with rows of very different length: every seventh row is
    // empty, a few rows are (almost) dense when `big` is set, the others have one
    // to nine entries. With `empty` the matrix has no entry at all.
    template <typename T>
    host_csr<T> make_irregular_csr(int32_t m, int32_t n, int base, bool big, bool empty)
    {
        host_csr<T> a;
        a.m    = m;
        a.n    = n;
        a.base = 0;
        a.ptr.push_back(0);
        int64_t seed = 0;
        for(int32_t i = 0; i < m; ++i)
        {
            int32_t len;
            if(empty || i % 7 == 3)
                len = 0;
            else if(big && i % 13 == 5)
                len = std::min<int32_t>(n, 200);
            else
                len = std::min<int32_t>(n, 1 + (i * 5) % 9);
            std::vector<int32_t> cols;
            if(len > 0)
            {
                const int32_t stride = std::max<int32_t>(1, n / len);
                const int32_t start  = (i * 7) % n;
                for(int32_t k = 0; k < len; ++k)
                    cols.push_back((start + k * stride) % n);
                std::sort(cols.begin(), cols.end());
                cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            }
            for(int32_t j : cols)
            {
                a.ind.push_back(j);
                a.val.push_back(value_of<T>(seed++, true));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        rebase(a, base);
        return a;
    }

    // Full symmetric matrix from the stored lower triangle.
    template <typename T>
    host_csr<T> symmetrize_lower(const host_csr<T>& low)
    {
        const int32_t                                   n = low.m;
        std::vector<std::vector<std::pair<int32_t, T>>> rows(n);
        for(int32_t i = 0; i < n; ++i)
            for(int32_t k = low.ptr[i] - low.base; k < low.ptr[i + 1] - low.base; ++k)
            {
                const int32_t j = low.ind[k] - low.base;
                rows[i].push_back({j, low.val[k]});
                if(i != j)
                    rows[j].push_back({i, low.val[k]});
            }
        host_csr<T> s;
        s.m    = n;
        s.n    = n;
        s.base = low.base;
        s.ptr.push_back(low.base);
        for(int32_t i = 0; i < n; ++i)
        {
            std::sort(rows[i].begin(), rows[i].end(), [](const auto& l, const auto& r) {
                return l.first < r.first;
            });
            for(const auto& e : rows[i])
            {
                s.ind.push_back(e.first + low.base);
                s.val.push_back(e.second);
            }
            s.ptr.push_back(static_cast<int32_t>(s.ind.size()) + low.base);
        }
        return s;
    }

    // ----- legacy csrmv ----------------------------------------------------------------
#define L2_CSRMV_WRAPPERS(P, TYPE)                                                    \
    inline rocsparse_status csrmv_analysis(rocsparse_handle          handle,          \
                                           rocsparse_operation       trans,           \
                                           int32_t                   m,               \
                                           int32_t                   n,               \
                                           int32_t                   nnz,             \
                                           const rocsparse_mat_descr descr,           \
                                           const TYPE*               val,             \
                                           const int32_t*            ptr,             \
                                           const int32_t*            ind,             \
                                           rocsparse_mat_info        info)            \
    {                                                                                 \
        return rocsparse_##P##csrmv_analysis(                                         \
            handle, trans, m, n, nnz, descr, val, ptr, ind, info);                    \
    }                                                                                 \
    inline rocsparse_status csrmv(rocsparse_handle          handle,                   \
                                  rocsparse_operation       trans,                    \
                                  int32_t                   m,                        \
                                  int32_t                   n,                        \
                                  int32_t                   nnz,                      \
                                  const TYPE*               alpha,                    \
                                  const rocsparse_mat_descr descr,                    \
                                  const TYPE*               val,                      \
                                  const int32_t*            ptr,                      \
                                  const int32_t*            ind,                      \
                                  rocsparse_mat_info        info,                     \
                                  const TYPE*               x,                        \
                                  const TYPE*               beta,                     \
                                  TYPE*                     y)                        \
    {                                                                                 \
        return rocsparse_##P##csrmv(                                                  \
            handle, trans, m, n, nnz, alpha, descr, val, ptr, ind, info, x, beta, y); \
    }

    L2_CSRMV_WRAPPERS(s, float)
    L2_CSRMV_WRAPPERS(d, double)
    L2_CSRMV_WRAPPERS(c, rocsparse_float_complex)
    L2_CSRMV_WRAPPERS(z, rocsparse_double_complex)
#undef L2_CSRMV_WRAPPERS

    struct csrmv_case
    {
        rocsparse_operation   trans     = rocsparse_operation_none;
        rocsparse_matrix_type mtype     = rocsparse_matrix_type_general;
        int                   info_mode = 0; // 0: no info, 1: fresh info, 2: analysed info
        bool                  device    = false;
        int                   base      = 0;
        int32_t               m         = 37;
        int32_t               n         = 29;
        float                 alpha     = 1.5f;
        float                 beta      = -0.5f;
        bool                  empty     = false;
        bool                  big       = false;
    };

    template <typename T>
    void run_csrmv(rocsparse_handle handle, const csrmv_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "csrmv trans=" << c.trans << " type=" << c.mtype << " info=" << c.info_mode
                     << " device=" << c.device << " base=" << c.base << " m=" << c.m << " n=" << c.n
                     << " alpha=" << c.alpha << " beta=" << c.beta << " empty=" << c.empty);

        const bool  sym = (c.mtype == rocsparse_matrix_type_symmetric);
        host_csr<T> store;
        host_csr<T> ref;
        if(sym)
        {
            store = make_triangular_pattern<T>(c.n, 3, true, false, true, c.base);
            ref   = symmetrize_lower(store);
        }
        else
        {
            store = make_irregular_csr<T>(c.m, c.n, c.base, c.big, c.empty);
            ref   = store;
        }
        const int32_t m  = store.m;
        const int32_t n  = store.n;
        const int32_t xs = (c.trans == rocsparse_operation_none) ? n : m;
        const int32_t ys = (c.trans == rocsparse_operation_none) ? m : n;

        const std::vector<T> x     = make_vector<T>(xs, 2);
        const std::vector<T> y0    = make_vector<T>(ys, 9);
        const T              alpha = make_scalar<T>(c.alpha, -0.25f * (c.alpha != 0.0f));
        const T              beta  = make_scalar<T>(c.beta, 0.25f * (c.beta != 1.0f));
        const std::vector<T> want  = csr_gemv(ref, c.trans, alpha, x, beta, y0);

        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        ASSERT_EQ(rocsparse_set_mat_type(descr, c.mtype), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);

        device_csr<T>    dA(store);
        device_vector<T> dx(x), dy(y0), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr);

        if(c.info_mode == 2)
        {
            ASSERT_EQ(csrmv_analysis(
                          handle, c.trans, m, n, store.nnz(), descr, dA.val, dA.ptr, dA.ind, info),
                      rocsparse_status_success);
        }

        {
            scoped_pointer_mode mode(
                handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
            ASSERT_EQ(csrmv(handle,
                            c.trans,
                            m,
                            n,
                            store.nnz(),
                            c.device ? dalpha.ptr : &alpha,
                            descr,
                            dA.val,
                            dA.ptr,
                            dA.ind,
                            c.info_mode == 0 ? nullptr : info.d,
                            dx,
                            c.device ? dbeta.ptr : &beta,
                            dy),
                      rocsparse_status_success);
        }
        EXPECT_TRUE(expect_near(want, to_host(dy), "csrmv y"));
        EXPECT_EQ(rocsparse_csrmv_clear(handle, info), rocsparse_status_success);
    }

    class Cov70L2Csrmv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep(bool all)
        {
            int k = 0;
            for(const auto trans : {rocsparse_operation_none,
                                    rocsparse_operation_transpose,
                                    rocsparse_operation_conjugate_transpose})
                for(const int info_mode : {0, 1, 2})
                    for(const auto mtype :
                        {rocsparse_matrix_type_general, rocsparse_matrix_type_symmetric})
                    {
                        if(mtype == rocsparse_matrix_type_symmetric
                           && std::is_same<typename real_of<T>::type, T>::value == false)
                            continue; // symmetric products are only checked for real data
                        if(!all && (k++ % 2) == 1)
                            continue;
                        csrmv_case c;
                        c.trans     = trans;
                        c.info_mode = info_mode;
                        c.mtype     = mtype;
                        c.device    = (info_mode == 1);
                        c.base      = (trans == rocsparse_operation_transpose) ? 1 : 0;
                        run_csrmv<T>(handle, c);
                        if(::testing::Test::HasFailure())
                            return;
                    }
        }
    };

    TEST_F(Cov70L2Csrmv, float_all_variants)
    {
        sweep<float>(true);
    }

    TEST_F(Cov70L2Csrmv, double_and_complex)
    {
        sweep<double>(false);
        sweep<rocsparse_float_complex>(false);
        sweep<rocsparse_double_complex>(false);
    }

    TEST_F(Cov70L2Csrmv, scalars_that_skip_or_scale)
    {
        // alpha == 0 leaves y = beta * y (and y unchanged for beta == 1).
        for(const float alpha : {0.0f, 1.0f})
            for(const float beta : {0.0f, 1.0f, 0.5f})
                for(const bool device : {false, true})
                    for(const int info_mode : {0, 2})
                    {
                        csrmv_case c;
                        c.alpha     = alpha;
                        c.beta      = beta;
                        c.device    = device;
                        c.info_mode = info_mode;
                        run_csrmv<double>(handle, c);
                        if(::testing::Test::HasFailure())
                            return;
                    }
    }

    TEST_F(Cov70L2Csrmv, long_rows)
    {
        for(const int info_mode : {0, 2})
            for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                csrmv_case c;
                c.m         = 70;
                c.n         = 400;
                c.big       = true;
                c.info_mode = info_mode;
                c.trans     = trans;
                run_csrmv<float>(handle, c);
                if(::testing::Test::HasFailure())
                    return;
            }
    }

    TEST_F(Cov70L2Csrmv, empty_matrix_scales_y)
    {
        for(const bool device : {false, true})
            for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                for(const int info_mode : {0, 1, 2})
                {
                    csrmv_case c;
                    c.empty     = true;
                    c.device    = device;
                    c.trans     = trans;
                    c.info_mode = info_mode;
                    run_csrmv<float>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(Cov70L2Csrmv, zero_sized_dimensions)
    {
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        device_vector<float>   one(std::vector<float>(4, 1.0f)), y(std::vector<float>(4, 3.0f));
        device_vector<int32_t> ptr(std::vector<int32_t>(8, 0)), ind(std::vector<int32_t>(4, 0));
        ASSERT_TRUE(one.ptr && y.ptr && ptr.ptr && ind.ptr);
        const float alpha = 1.0f;
        const float beta  = 2.0f;

        // m == 0: nothing to update.
        EXPECT_EQ(
            csrmv_analysis(
                handle, rocsparse_operation_none, 0, 4, 0, descr, one.ptr, ptr.ptr, ind.ptr, info),
            rocsparse_status_success);
        EXPECT_EQ(csrmv(handle,
                        rocsparse_operation_none,
                        0,
                        4,
                        0,
                        &alpha,
                        descr,
                        one.ptr,
                        ptr.ptr,
                        ind.ptr,
                        info,
                        one,
                        &beta,
                        y),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(std::vector<float>(4, 3.0f), to_host(y), "csrmv y (m=0)"));

        // n == 0 with a non transposed product: y = beta * y.
        EXPECT_EQ(
            csrmv_analysis(
                handle, rocsparse_operation_none, 4, 0, 0, descr, one.ptr, ptr.ptr, ind.ptr, info),
            rocsparse_status_success);
        EXPECT_EQ(csrmv(handle,
                        rocsparse_operation_none,
                        4,
                        0,
                        0,
                        &alpha,
                        descr,
                        one.ptr,
                        ptr.ptr,
                        ind.ptr,
                        info,
                        one,
                        &beta,
                        y),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(std::vector<float>(4, 6.0f), to_host(y), "csrmv y (n=0)"));
    }

    TEST_F(Cov70L2Csrmv, argument_checks)
    {
        const host_csr<float> A = make_irregular_csr<float>(12, 9, 0, false, false);
        device_csr<float>     dA(A);
        device_vector<float>  dx(make_vector<float>(9, 1)), dy(make_vector<float>(12, 2));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        const float   alpha = 1.0f;
        const float   beta  = 0.0f;
        const auto    op    = rocsparse_operation_none;
        const int32_t nnz   = A.nnz();

        auto call
            = [&](rocsparse_handle    h,
                  rocsparse_operation t,
                  int32_t             m,
                  int32_t             n,
                  int32_t             z,
                  const float*        pa,
                  rocsparse_mat_descr d,
                  const float*        v,
                  const int32_t*      p,
                  const int32_t*      i,
                  const float*        px,
                  const float*        pb,
                  float* py) { return csrmv(h, t, m, n, z, pa, d, v, p, i, info, px, pb, py); };
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, descr, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_success);
        EXPECT_EQ(
            call(nullptr, op, 12, 9, nnz, &alpha, descr, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_invalid_handle);
        EXPECT_EQ(call(handle,
                       static_cast<rocsparse_operation>(77),
                       12,
                       9,
                       nnz,
                       &alpha,
                       descr,
                       dA.val,
                       dA.ptr,
                       dA.ind,
                       dx,
                       &beta,
                       dy),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(
            call(handle, op, -1, 9, nnz, &alpha, descr, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_invalid_size);
        EXPECT_EQ(
            call(handle, op, 12, -1, nnz, &alpha, descr, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_invalid_size);
        EXPECT_EQ(call(handle, op, 12, 9, -1, &alpha, descr, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, nullptr, descr, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, nullptr, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, descr, nullptr, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, descr, dA.val, nullptr, dA.ind, dx, &beta, dy),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, descr, dA.val, dA.ptr, nullptr, dx, &beta, dy),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, descr, dA.val, dA.ptr, dA.ind, nullptr, &beta, dy),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, descr, dA.val, dA.ptr, dA.ind, dx, nullptr, dy),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, descr, dA.val, dA.ptr, dA.ind, dx, &beta, nullptr),
            rocsparse_status_invalid_pointer);

        mat_descr unsorted;
        ASSERT_EQ(rocsparse_set_mat_storage_mode(unsorted, rocsparse_storage_mode_unsorted),
                  rocsparse_status_success);
        EXPECT_EQ(
            call(handle, op, 12, 9, nnz, &alpha, unsorted, dA.val, dA.ptr, dA.ind, dx, &beta, dy),
            rocsparse_status_requires_sorted_storage);

        // analysis
        EXPECT_EQ(csrmv_analysis(nullptr, op, 12, 9, nnz, descr, dA.val, dA.ptr, dA.ind, info),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(csrmv_analysis(handle, op, 12, 9, nnz, nullptr, dA.val, dA.ptr, dA.ind, info),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(csrmv_analysis(handle,
                                 op,
                                 12,
                                 9,
                                 nnz,
                                 descr,
                                 static_cast<const float*>(nullptr),
                                 dA.ptr,
                                 dA.ind,
                                 info),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(csrmv_analysis(handle, op, 12, 9, nnz, descr, dA.val, nullptr, dA.ind, info),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(csrmv_analysis(handle, op, 12, 9, nnz, descr, dA.val, dA.ptr, nullptr, info),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(csrmv_analysis(handle, op, -1, 9, nnz, descr, dA.val, dA.ptr, dA.ind, info),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(csrmv_analysis(handle, op, 12, 9, nnz, unsorted, dA.val, dA.ptr, dA.ind, info),
                  rocsparse_status_requires_sorted_storage);
        EXPECT_EQ(rocsparse_csrmv_clear(nullptr, info), rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_csrmv_clear(handle, nullptr), rocsparse_status_invalid_pointer);
    }

    // ----- v2 spmv descriptor API ----------------------------------------------------------
    struct spmv_descr_guard
    {
        rocsparse_spmv_descr d = nullptr;
        spmv_descr_guard()
        {
            if(rocsparse_create_spmv_descr(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~spmv_descr_guard()
        {
            if(d)
                (void)rocsparse_destroy_spmv_descr(d);
        }
        spmv_descr_guard(const spmv_descr_guard&) = delete;
        spmv_descr_guard& operator=(const spmv_descr_guard&) = delete;
    };

    template <typename V>
    rocsparse_status set_input(rocsparse_handle     handle,
                               rocsparse_spmv_descr d,
                               rocsparse_spmv_input input,
                               const V&             value)
    {
        return rocsparse_spmv_set_input(handle, d, input, &value, sizeof(V), nullptr);
    }

    inline rocsparse_status set_basic_inputs(rocsparse_handle     handle,
                                             rocsparse_spmv_descr d,
                                             rocsparse_spmv_alg   alg,
                                             rocsparse_operation  trans,
                                             rocsparse_datatype   scalar_type,
                                             rocsparse_datatype   compute_type)
    {
        rocsparse_status st = set_input(handle, d, rocsparse_spmv_input_alg, alg);
        if(st != rocsparse_status_success)
            return st;
        st = set_input(handle, d, rocsparse_spmv_input_operation, trans);
        if(st != rocsparse_status_success)
            return st;
        st = set_input(handle, d, rocsparse_spmv_input_scalar_datatype, scalar_type);
        if(st != rocsparse_status_success)
            return st;
        return set_input(handle, d, rocsparse_spmv_input_compute_datatype, compute_type);
    }

    struct v2_case
    {
        rocsparse_spmv_alg  alg          = rocsparse_spmv_alg_default;
        rocsparse_operation trans        = rocsparse_operation_none;
        int                 extras       = 0;
        bool                device       = false;
        int                 base         = 0;
        bool                starting_ids = false;
        // 0: general data, 1: matrix without entries, 2: alpha = 0 and beta = 1
        int     special    = 0;
        bool    zero_gamma = false;
        bool    big        = false;
        int32_t m          = 41;
        int32_t n          = 33;
    };

    // A: matrix precision, CT: precision of x, y and of the computation, ST: precision of alpha,
    // beta and gamma.
    template <typename AT, typename CT, typename ST>
    void run_v2(rocsparse_handle handle, const v2_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "v2_spmv alg=" << c.alg << " trans=" << c.trans << " extras=" << c.extras
                     << " device=" << c.device << " base=" << c.base << " special=" << c.special
                     << " zero_gamma=" << c.zero_gamma << " ids=" << c.starting_ids);

        const host_csr<AT> A  = make_irregular_csr<AT>(c.m, c.n, c.base, c.big, c.special == 1);
        const host_csr<CT> Ac = cast_csr<CT>(A);
        const int32_t      xs = (c.trans == rocsparse_operation_none) ? c.n : c.m;
        const int32_t      ys = (c.trans == rocsparse_operation_none) ? c.m : c.n;

        const std::vector<CT> x  = make_vector<CT>(xs, 2);
        const std::vector<CT> y0 = make_vector<CT>(ys, 9);
        const ST              alpha
            = (c.special == 2) ? make_scalar<ST>(0.0f, 0.0f) : make_scalar<ST>(1.5f, -0.5f);
        const ST beta
            = (c.special == 2) ? make_scalar<ST>(1.0f, 0.0f) : make_scalar<ST>(-0.5f, 0.25f);

        std::vector<ST>              gamma(c.extras > 0 ? c.extras : 1);
        std::vector<std::vector<CT>> z(c.extras);
        for(int e = 0; e < c.extras; ++e)
        {
            gamma[e] = c.zero_gamma ? make_scalar<ST>(0.0f, 0.0f)
                                    : make_scalar<ST>(0.5f + 0.25f * e, 0.25f * (e + 1));
            z[e]     = make_vector<CT>(ys, 20 + e);
        }

        std::vector<CT> want
            = csr_gemv<CT>(Ac, c.trans, convert_scalar<CT>(alpha), x, convert_scalar<CT>(beta), y0);
        for(int e = 0; e < c.extras; ++e)
            for(int32_t j = 0; j < ys; ++j)
                want[j] = want[j] + convert_scalar<CT>(gamma[e]) * z[e][j];

        device_csr<AT>    dA(A);
        device_vector<CT> dx(x), dy(y0);
        device_vector<ST> dalpha(std::vector<ST>(1, alpha)), dbeta(std::vector<ST>(1, beta));
        device_vector<ST> dgamma(gamma);
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr && dgamma.ptr);
        std::deque<device_vector<CT>> dz;
        for(int e = 0; e < c.extras; ++e)
        {
            dz.emplace_back(z[e]);
            ASSERT_TRUE(dz.back().ptr);
        }

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             c.m,
                                             c.n,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             to_base(c.base),
                                             rocsparse_ut::dt_of<AT>()),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, xs, dx, rocsparse_ut::dt_of<CT>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, ys, dy, rocsparse_ut::dt_of<CT>()),
                  rocsparse_status_success);

        scoped_pointer_mode mode(
            handle, c.device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_basic_inputs(handle,
                                   descr.d,
                                   c.alg,
                                   c.trans,
                                   rocsparse_ut::dt_of<ST>(),
                                   rocsparse_ut::dt_of<CT>()),
                  rocsparse_status_success);
        if(c.starting_ids)
            ASSERT_EQ(
                set_input(handle, descr.d, rocsparse_spmv_input_nnz_use_starting_block_ids, true),
                rocsparse_status_success);

        dnvec_guard                              vgamma;
        std::deque<dnvec_guard>                  vz;
        std::vector<rocsparse_const_dnvec_descr> zs;
        if(c.extras > 0)
        {
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vgamma.d,
                                                   c.extras,
                                                   c.device ? static_cast<void*>(dgamma.ptr)
                                                            : static_cast<void*>(gamma.data()),
                                                   rocsparse_ut::dt_of<ST>()),
                      rocsparse_status_success);
            for(int e = 0; e < c.extras; ++e)
            {
                vz.emplace_back();
                ASSERT_EQ(rocsparse_create_dnvec_descr(
                              &vz.back().d, ys, dz[e].ptr, rocsparse_ut::dt_of<CT>()),
                          rocsparse_status_success);
                zs.push_back(vz.back().d);
            }
            ASSERT_EQ(
                rocsparse_spmv_set_extra(handle, descr.d, c.extras, vgamma.d, zs.data(), nullptr),
                rocsparse_status_success);
        }

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
                                        bytes ? dbuf.ptr : nullptr,
                                        nullptr),
                      rocsparse_status_success);
        }
        EXPECT_TRUE(expect_near(want, to_host(dy), "v2_spmv y"));
    }

    class Cov70L2V2Spmv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename AT, typename CT, typename ST>
        void sweep(const std::vector<rocsparse_spmv_alg>& algs, int max_extras, bool both_trans)
        {
            int k = 0;
            for(const auto alg : algs)
                for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    const bool trans_ok = (alg == rocsparse_spmv_alg_default
                                           || alg == rocsparse_spmv_alg_csr_rowsplit
                                           || alg == rocsparse_spmv_alg_csr_nnzsplit);
                    if(trans != rocsparse_operation_none && (!trans_ok || !both_trans))
                        continue;
                    for(int extras = 0; extras <= max_extras;
                        extras += (max_extras > 0 ? max_extras : 1))
                    {
                        v2_case c;
                        c.alg          = alg;
                        c.trans        = trans;
                        c.extras       = extras;
                        c.device       = ((k >> 1) & 1) != 0;
                        c.base         = k & 1;
                        c.starting_ids = (alg == rocsparse_spmv_alg_csr_nnzsplit) && (k % 3 == 0);
                        ++k;
                        run_v2<AT, CT, ST>(handle, c);
                        if(::testing::Test::HasFailure())
                            return;
                    }
                }
        }
    };

    const std::vector<rocsparse_spmv_alg> all_csr_algs = {rocsparse_spmv_alg_default,
                                                          rocsparse_spmv_alg_csr_rowsplit,
                                                          rocsparse_spmv_alg_csr_adaptive,
                                                          rocsparse_spmv_alg_csr_lrb,
                                                          rocsparse_spmv_alg_csr_nnzsplit};

    TEST_F(Cov70L2V2Spmv, float_algorithms_with_extras)
    {
        sweep<float, float, float>(all_csr_algs, 2, true);
    }

    TEST_F(Cov70L2V2Spmv, double_algorithms_with_extras)
    {
        sweep<double, double, double>(all_csr_algs, 1, false);
    }

    TEST_F(Cov70L2V2Spmv, complex_algorithms_with_extras)
    {
        sweep<rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex>(
            {rocsparse_spmv_alg_default,
             rocsparse_spmv_alg_csr_adaptive,
             rocsparse_spmv_alg_csr_lrb},
            2,
            false);
        sweep<rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex>(
            {rocsparse_spmv_alg_csr_rowsplit, rocsparse_spmv_alg_csr_nnzsplit}, 1, true);
    }

    TEST_F(Cov70L2V2Spmv, scalar_type_differs_from_compute_type)
    {
        // Single precision scalars and matrix, double precision vectors.
        sweep<float, double, float>(
            {rocsparse_spmv_alg_default, rocsparse_spmv_alg_csr_adaptive}, 0, true);
        sweep<float, double, double>(
            {rocsparse_spmv_alg_csr_rowsplit, rocsparse_spmv_alg_csr_nnzsplit}, 0, true);
        sweep<rocsparse_float_complex, rocsparse_double_complex, rocsparse_double_complex>(
            {rocsparse_spmv_alg_csr_rowsplit, rocsparse_spmv_alg_csr_adaptive}, 0, false);
    }

    TEST_F(Cov70L2V2Spmv, long_rows)
    {
        for(const auto alg : all_csr_algs)
        {
            v2_case c;
            c.alg = alg;
            c.big = true;
            c.m   = 90;
            c.n   = 500;
            run_v2<float, float, float>(handle, c);
            if(::testing::Test::HasFailure())
                return;
        }
    }

    TEST_F(Cov70L2V2Spmv, matrix_without_entries_and_extras)
    {
        // The matrix is never read: y = beta * y + sum_i gamma_i z_i.
        for(const bool device : {false, true})
            for(const int extras : {0, 2})
                for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    v2_case c;
                    c.special = 1;
                    c.extras  = extras;
                    c.device  = device;
                    c.trans   = trans;
                    c.alg     = rocsparse_spmv_alg_csr_rowsplit;
                    run_v2<double, double, double>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(Cov70L2V2Spmv, alpha_zero_beta_one)
    {
        // With alpha = 0 and beta = 1 (host pointers) only the extra terms act. When all
        // gamma are zero too, y must not change at all.
        for(const bool zero_gamma : {false, true})
            for(const int extras : {0, 2})
                for(const bool device : {false, true})
                {
                    v2_case c;
                    c.special    = 2;
                    c.extras     = extras;
                    c.zero_gamma = zero_gamma;
                    c.device     = device;
                    run_v2<double, double, double>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(Cov70L2V2Spmv, stage_order_and_input_errors)
    {
        const host_csr<float> A = make_irregular_csr<float>(12, 9, 0, false, false);
        device_csr<float>     dA(A);
        device_vector<float>  dx(make_vector<float>(9, 1)), dy(make_vector<float>(12, 2));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);
        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             12,
                                             9,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, 9, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, 12, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        size_t      bytes = 0;

        auto spmv = [&](rocsparse_v2_spmv_stage stage) {
            return rocsparse_v2_spmv(
                handle, descr.d, &alpha, mat.d, vx.d, &beta, vy.d, stage, 0, nullptr, nullptr);
        };

        // Inputs that were never set.
        EXPECT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                descr.d,
                                                mat.d,
                                                vx.d,
                                                vy.d,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &bytes,
                                                nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_invalid_value);

        // set_input argument validation.
        const auto alg = rocsparse_spmv_alg_default;
        EXPECT_EQ(rocsparse_spmv_set_input(
                      nullptr, descr.d, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spmv_set_input(
                      handle, nullptr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spmv_set_input(
                handle, descr.d, static_cast<rocsparse_spmv_input>(77), &alg, sizeof(alg), nullptr),
            rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spmv_set_input(
                      handle, descr.d, rocsparse_spmv_input_alg, nullptr, sizeof(alg), nullptr),
                  rocsparse_status_invalid_pointer);
        for(const auto input : {rocsparse_spmv_input_alg,
                                rocsparse_spmv_input_operation,
                                rocsparse_spmv_input_scalar_datatype,
                                rocsparse_spmv_input_compute_datatype,
                                rocsparse_spmv_input_nnz_use_starting_block_ids,
                                rocsparse_spmv_input_enable_extra})
        {
            // a bool is one byte, so use another wrong size for that input
            const size_t wrong_size
                = (input == rocsparse_spmv_input_nnz_use_starting_block_ids) ? 4 : 1;
            EXPECT_EQ(rocsparse_spmv_set_input(handle, descr.d, input, &alg, wrong_size, nullptr),
                      rocsparse_status_invalid_size);
        }
        // enable_extra needs extras to be set first.
        const int32_t enable = 1;
        EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, enable),
                  rocsparse_status_invalid_value);

        ASSERT_EQ(set_basic_inputs(handle,
                                   descr.d,
                                   rocsparse_spmv_alg_csr_rowsplit,
                                   rocsparse_operation_none,
                                   rocsparse_datatype_f32_r,
                                   rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        // Compute before analysis.
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_compute), rocsparse_status_invalid_value);
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_success);
        // Analysis twice.
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_invalid_value);
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_compute), rocsparse_status_success);
        // Analysis after compute.
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_invalid_value);
        // The algorithm and the operation cannot change once the analysis ran.
        EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_alg, alg),
                  rocsparse_status_internal_error);
        const auto trans = rocsparse_operation_transpose;
        EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_operation, trans),
                  rocsparse_status_internal_error);

        // Invalid stage / pointers of the compute entry points.
        EXPECT_EQ(rocsparse_v2_spmv(handle,
                                    descr.d,
                                    &alpha,
                                    mat.d,
                                    vx.d,
                                    &beta,
                                    vy.d,
                                    static_cast<rocsparse_v2_spmv_stage>(77),
                                    0,
                                    nullptr,
                                    nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_v2_spmv(handle,
                                    descr.d,
                                    nullptr,
                                    mat.d,
                                    vx.d,
                                    &beta,
                                    vy.d,
                                    rocsparse_v2_spmv_stage_compute,
                                    0,
                                    nullptr,
                                    nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_v2_spmv(handle,
                                    descr.d,
                                    &alpha,
                                    mat.d,
                                    vx.d,
                                    nullptr,
                                    vy.d,
                                    rocsparse_v2_spmv_stage_compute,
                                    0,
                                    nullptr,
                                    nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_v2_spmv(handle,
                                    descr.d,
                                    &alpha,
                                    mat.d,
                                    vx.d,
                                    &beta,
                                    vy.d,
                                    rocsparse_v2_spmv_stage_compute,
                                    0,
                                    dy.ptr,
                                    nullptr),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_v2_spmv(handle,
                                    descr.d,
                                    &alpha,
                                    mat.d,
                                    vx.d,
                                    &beta,
                                    vy.d,
                                    rocsparse_v2_spmv_stage_compute,
                                    8,
                                    nullptr,
                                    nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                descr.d,
                                                mat.d,
                                                vx.d,
                                                vy.d,
                                                static_cast<rocsparse_v2_spmv_stage>(77),
                                                &bytes,
                                                nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                descr.d,
                                                mat.d,
                                                vx.d,
                                                vy.d,
                                                rocsparse_v2_spmv_stage_compute,
                                                nullptr,
                                                nullptr),
                  rocsparse_status_invalid_pointer);

        // The result of the successful compute is the product A x.
        const std::vector<float> want = csr_gemv<float>(A,
                                                        rocsparse_operation_none,
                                                        1.0f,
                                                        make_vector<float>(9, 1),
                                                        0.0f,
                                                        make_vector<float>(12, 2));
        EXPECT_TRUE(expect_near(want, to_host(dy), "v2_spmv y"));
    }

    TEST_F(Cov70L2V2Spmv, set_extra_errors)
    {
        const int32_t         ys = 12;
        device_vector<double> gamma_dev(std::vector<double>(2, 1.0)),
            z0(make_vector<double>(ys, 1)), z1(make_vector<double>(ys, 2)),
            z_short(make_vector<double>(ys - 1, 3));
        device_vector<float> z_f32(make_vector<float>(ys, 4)),
            gamma_f32(std::vector<float>(2, 1.0f));
        ASSERT_TRUE(gamma_dev.ptr && z0.ptr && z1.ptr && z_short.ptr && z_f32.ptr && gamma_f32.ptr);

        const auto  f64 = rocsparse_datatype_f64_r;
        dnvec_guard g1, g2, a, b, shorter, single, gf32;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&g1.d, 1, gamma_dev, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&g2.d, 2, gamma_dev, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&a.d, ys, z0, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&b.d, ys, z1, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&shorter.d, ys - 1, z_short, f64),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&single.d, ys, z_f32, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&gf32.d, 1, gamma_f32, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        rocsparse_const_dnvec_descr zs[2] = {a.d, b.d};

        // Datatypes not set yet.
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, zs, nullptr),
                  rocsparse_status_invalid_value);
        ASSERT_EQ(
            set_basic_inputs(
                handle, descr.d, rocsparse_spmv_alg_default, rocsparse_operation_none, f64, f64),
            rocsparse_status_success);

        EXPECT_EQ(rocsparse_spmv_set_extra(nullptr, descr.d, 2, g2.d, zs, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, nullptr, 2, g2.d, zs, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 0, g2.d, zs, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, nullptr, zs, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        // gamma with the wrong datatype or size.
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 1, gf32.d, zs, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g1.d, zs, nullptr),
                  rocsparse_status_invalid_size);
        // z vectors with a null entry, the wrong datatype or different sizes.
        rocsparse_const_dnvec_descr z_null[2] = {a.d, nullptr};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, z_null, nullptr),
                  rocsparse_status_invalid_pointer);
        rocsparse_const_dnvec_descr z_type[1] = {single.d};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 1, g1.d, z_type, nullptr),
                  rocsparse_status_invalid_value);
        rocsparse_const_dnvec_descr z_size[2] = {a.d, shorter.d};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, z_size, nullptr),
                  rocsparse_status_invalid_size);

        // Valid extras, then toggled with enable_extra and removed again.
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, zs, nullptr),
                  rocsparse_status_success);
        for(const int32_t enable : {0, 1})
            EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, enable),
                      rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmv_clear_extra(handle, descr.d, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmv_clear_extra(nullptr, descr.d, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spmv_clear_extra(handle, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);

        // Different scalar and compute datatypes do not support extras.
        spmv_descr_guard mixed;
        ASSERT_TRUE(mixed.d);
        ASSERT_EQ(set_basic_inputs(handle,
                                   mixed.d,
                                   rocsparse_spmv_alg_default,
                                   rocsparse_operation_none,
                                   rocsparse_datatype_f32_r,
                                   f64),
                  rocsparse_status_success);
        rocsparse_const_dnvec_descr z_one[1] = {a.d};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, mixed.d, 1, gf32.d, z_one, nullptr),
                  rocsparse_status_not_implemented);
    }

    // ----- algorithm / format validation -----------------------------------------------------
    enum class mv_format
    {
        csr,
        csc,
        coo,
        coo_aos,
        ell,
        bsr,
        bell,
        sell
    };

    bool alg_valid_for(mv_format f, rocsparse_spmv_alg alg)
    {
        if(alg == rocsparse_spmv_alg_default)
            return f != mv_format::bell;
        switch(f)
        {
        case mv_format::csr:
        case mv_format::csc:
            return alg == rocsparse_spmv_alg_csr_rowsplit || alg == rocsparse_spmv_alg_csr_adaptive
                   || alg == rocsparse_spmv_alg_csr_lrb || alg == rocsparse_spmv_alg_csr_nnzsplit;
        case mv_format::coo:
        case mv_format::coo_aos:
            return alg == rocsparse_spmv_alg_coo || alg == rocsparse_spmv_alg_coo_atomic;
        case mv_format::ell:
            return alg == rocsparse_spmv_alg_ell;
        case mv_format::bsr:
            return alg == rocsparse_spmv_alg_bsr;
        case mv_format::sell:
            return alg == rocsparse_spmv_alg_sell;
        case mv_format::bell:
            return false;
        }
        return false;
    }

    class Cov70L2SpmvAlgs : public rocsparse_ut::HandleTest
    {
    protected:
        static constexpr int32_t dim = 4;

        // Builds a small descriptor of each format. The arrays are valid device
        // memory but only descriptors are inspected by the stages used below.
        bool make_matrix(mv_format f, spmat_guard& mat)
        {
            idx = std::make_unique<device_vector<int32_t>>(std::vector<int32_t>(64, 0));
            val = std::make_unique<device_vector<float>>(std::vector<float>(64, 1.0f));
            if(!idx->ptr || !val->ptr)
                return false;
            const auto it = rocsparse_indextype_i32;
            const auto b  = rocsparse_index_base_zero;
            const auto dt = rocsparse_datatype_f32_r;
            switch(f)
            {
            case mv_format::csr:
                return rocsparse_create_csr_descr(
                           &mat.d, dim, dim, dim, idx->ptr, idx->ptr, val->ptr, it, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::csc:
                return rocsparse_create_csc_descr(
                           &mat.d, dim, dim, dim, idx->ptr, idx->ptr, val->ptr, it, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::coo:
                return rocsparse_create_coo_descr(
                           &mat.d, dim, dim, dim, idx->ptr, idx->ptr, val->ptr, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::coo_aos:
                return rocsparse_create_coo_aos_descr(
                           &mat.d, dim, dim, dim, idx->ptr, val->ptr, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::ell:
                return rocsparse_create_ell_descr(
                           &mat.d, dim, dim, idx->ptr, val->ptr, 1, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::bsr:
                return rocsparse_create_bsr_descr(&mat.d,
                                                  2,
                                                  2,
                                                  2,
                                                  rocsparse_direction_row,
                                                  2,
                                                  idx->ptr,
                                                  idx->ptr,
                                                  val->ptr,
                                                  it,
                                                  it,
                                                  b,
                                                  dt)
                       == rocsparse_status_success;
            case mv_format::bell:
                return rocsparse_create_bell_descr(&mat.d,
                                                   dim,
                                                   dim,
                                                   rocsparse_direction_row,
                                                   2,
                                                   1,
                                                   idx->ptr,
                                                   val->ptr,
                                                   it,
                                                   b,
                                                   dt)
                       == rocsparse_status_success;
            case mv_format::sell:
                return rocsparse_create_sell_descr(
                           &mat.d, dim, dim, dim, 2, 4, idx->ptr, idx->ptr, val->ptr, it, it, b, dt)
                       == rocsparse_status_success;
            }
            return false;
        }

        std::unique_ptr<device_vector<int32_t>> idx;
        std::unique_ptr<device_vector<float>>   val;
    };

    const std::vector<rocsparse_spmv_alg> every_alg = {rocsparse_spmv_alg_default,
                                                       rocsparse_spmv_alg_csr_rowsplit,
                                                       rocsparse_spmv_alg_csr_adaptive,
                                                       rocsparse_spmv_alg_csr_lrb,
                                                       rocsparse_spmv_alg_csr_nnzsplit,
                                                       rocsparse_spmv_alg_coo,
                                                       rocsparse_spmv_alg_coo_atomic,
                                                       rocsparse_spmv_alg_ell,
                                                       rocsparse_spmv_alg_sell,
                                                       rocsparse_spmv_alg_bsr};

    TEST_F(Cov70L2SpmvAlgs, spmv_buffer_size_checks_algorithm_per_format)
    {
        device_vector<float> dx(std::vector<float>(dim, 1.0f)), dy(std::vector<float>(dim, 1.0f));
        ASSERT_TRUE(dx.ptr && dy.ptr);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, dim, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, dim, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        for(const auto f : {mv_format::csr,
                            mv_format::csc,
                            mv_format::coo,
                            mv_format::coo_aos,
                            mv_format::ell,
                            mv_format::bsr,
                            mv_format::bell,
                            mv_format::sell})
        {
            spmat_guard mat;
            ASSERT_TRUE(make_matrix(f, mat)) << "format " << static_cast<int>(f);
            for(const auto alg : every_alg)
            {
                SCOPED_TRACE(::testing::Message()
                             << "format " << static_cast<int>(f) << " alg " << alg);
                size_t                 bytes = 99;
                const rocsparse_status st    = rocsparse_spmv(handle,
                                                           rocsparse_operation_none,
                                                           &alpha,
                                                           mat.d,
                                                           vx.d,
                                                           &beta,
                                                           vy.d,
                                                           rocsparse_datatype_f32_r,
                                                           alg,
                                                           rocsparse_spmv_stage_buffer_size,
                                                           &bytes,
                                                           nullptr);
                if(!alg_valid_for(f, alg))
                {
                    EXPECT_EQ(st, rocsparse_status_invalid_value);
                }
                else if(f == mv_format::sell)
                {
                    // Sliced ELL is only available through rocsparse_v2_spmv.
                    EXPECT_EQ(st, rocsparse_status_not_implemented);
                }
                else
                {
                    EXPECT_EQ(st, rocsparse_status_success);
                    EXPECT_EQ(bytes, size_t{0});
                }
            }
        }
    }

    TEST_F(Cov70L2SpmvAlgs, v2_spmv_analysis_rejects_algorithm_of_other_format)
    {
        device_vector<float> dx(std::vector<float>(dim, 1.0f)), dy(std::vector<float>(dim, 1.0f));
        ASSERT_TRUE(dx.ptr && dy.ptr);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, dim, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, dim, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        for(const auto f : {mv_format::csr,
                            mv_format::csc,
                            mv_format::coo,
                            mv_format::coo_aos,
                            mv_format::ell,
                            mv_format::bsr,
                            mv_format::bell,
                            mv_format::sell})
        {
            spmat_guard mat;
            ASSERT_TRUE(make_matrix(f, mat)) << "format " << static_cast<int>(f);
            for(const auto alg : every_alg)
            {
                if(alg_valid_for(f, alg))
                    continue; // the valid combinations are exercised with real data elsewhere
                SCOPED_TRACE(::testing::Message()
                             << "format " << static_cast<int>(f) << " alg " << alg);
                spmv_descr_guard descr;
                ASSERT_TRUE(descr.d);
                ASSERT_EQ(set_basic_inputs(handle,
                                           descr.d,
                                           alg,
                                           rocsparse_operation_none,
                                           rocsparse_datatype_f32_r,
                                           rocsparse_datatype_f32_r),
                          rocsparse_status_success);
                EXPECT_EQ(rocsparse_v2_spmv(handle,
                                            descr.d,
                                            &alpha,
                                            mat.d,
                                            vx.d,
                                            &beta,
                                            vy.d,
                                            rocsparse_v2_spmv_stage_analysis,
                                            0,
                                            nullptr,
                                            nullptr),
                          rocsparse_status_invalid_value);
            }
        }
    }
} // namespace
