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
// Device unit tests that drive the public C API dispatch of the generic sparse
// products and triangular solves: the matrix format and algorithm switches of SpMV
// and v2 SpMV, the segmented-atomic SpMM, the analysis policies and scalar
// conversions of SpTRSV and SpTRSM, and the csrsm block size ladder.
//
// The matrices are identities, so the dispatch (not the arithmetic) is under test:
// every solve or product of ones must return ones. An algorithm that the library does
// not implement for a configuration ends a call quietly, any other failure is
// reported.
//
#include "unit_test_device_level2_common.hpp"

#include <numeric>

using namespace rocsparse_ut;
using namespace level2;

namespace
{
    constexpr auto f32   = rocsparse_datatype_f32_r;
    constexpr auto f64   = rocsparse_datatype_f64_r;
    constexpr auto op_n  = rocsparse_operation_none;
    constexpr auto op_t  = rocsparse_operation_transpose;
    constexpr auto op_c  = rocsparse_operation_conjugate_transpose;
    constexpr auto col   = rocsparse_order_column;
    constexpr auto row   = rocsparse_order_row;
    constexpr auto lo    = rocsparse_fill_mode_lower;
    constexpr auto up    = rocsparse_fill_mode_upper;
    constexpr auto nu    = rocsparse_diag_type_non_unit;
    constexpr auto un    = rocsparse_diag_type_unit;
    constexpr auto reuse = rocsparse_analysis_policy_reuse;
    constexpr auto force = rocsparse_analysis_policy_force;

    // ----- staged calls ------------------------------------------------------------

    // True when the next stage can run. A stage that is not implemented ends the call
    // without a failure.
    bool proceed(rocsparse_status status, const char* what)
    {
        if(status == rocsparse_status_success)
            return true;
        EXPECT_EQ(status, rocsparse_status_not_implemented) << what;
        return false;
    }

    // SpMV and SpMM: one buffer, sized by the first stage, shared by the following ones.
    // Returns true when the compute stage ran.
    template <typename Stage, typename Call>
    bool run_buffered(
        Call call, Stage size, Stage preprocess, Stage compute, int preprocess_passes = 1)
    {
        size_t bytes = 0;
        if(!proceed(call(size, &bytes, nullptr), "buffer_size"))
            return false;
        device_vector<char> buffer(std::max<size_t>(bytes, 1));
        EXPECT_NE(buffer.ptr, nullptr);
        void* p = (bytes && buffer.ptr) ? buffer.ptr : nullptr;
        for(int pass = 0; pass < preprocess_passes; ++pass)
            if(!proceed(call(preprocess, &bytes, p), "preprocess"))
                return false;
        const bool ok = proceed(call(compute, &bytes, p), "compute");
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return ok;
    }

    // v2 SpMV, SpTRSV and SpTRSM: every stage queries its own buffer size.
    // Returns true when the last stage ran.
    template <typename Stage, typename Size, typename Run>
    bool run_stages(std::initializer_list<Stage> stages, Size size, Run run)
    {
        for(const Stage stage : stages)
        {
            size_t bytes = 0;
            if(!proceed(size(stage, &bytes), "buffer_size"))
                return false;
            device_vector<char> buffer(std::max<size_t>(bytes, 1));
            EXPECT_NE(buffer.ptr, nullptr);
            if(!proceed(run(stage, bytes, (bytes && buffer.ptr) ? buffer.ptr : nullptr), "stage"))
                return false;
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        }
        return true;
    }

    template <typename T>
    void expect_ones(const device_vector<T>& v)
    {
        for(const T& e : to_host(v))
            EXPECT_EQ(e, T(1));
    }

    template <typename I = int32_t>
    std::vector<I> counting(int64_t count)
    {
        std::vector<I> v(count);
        std::iota(v.begin(), v.end(), I(0));
        return v;
    }

    // ----- data ----------------------------------------------------------------

    enum class mat_format
    {
        csr,
        csc,
        coo,
        coo_aos,
        ell,
        bsr_row,
        bsr_col,
        sell
    };

    // The n x n identity in the formats of the tests, with x = 1 and y = 0.
    template <typename T>
    struct identity
    {
        const int32_t          n;
        device_vector<int32_t> ptr, idx, aos, slices;
        device_vector<T>       val, x, y;

        explicit identity(int32_t size = 3)
            : n(size)
            , ptr(counting(size + 1))
            , idx(counting(size))
            , aos(interleaved(size))
            , slices(std::vector<int32_t>{0, size})
            , val(std::vector<T>(size, T(1)))
            , x(std::vector<T>(size, T(1)))
            , y(std::vector<T>(size, T(0)))
        {
        }

        bool ok() const
        {
            return ptr.ptr && idx.ptr && aos.ptr && slices.ptr && val.ptr && x.ptr && y.ptr;
        }

        rocsparse_status create(mat_format format, spmat_guard& A) const
        {
            const auto it = rocsparse_indextype_i32;
            const auto b  = rocsparse_index_base_zero;
            const auto dt = rocsparse_ut::dt_of<T>();
            switch(format)
            {
            case mat_format::csr:
                return rocsparse_create_csr_descr(&A.d, n, n, n, ptr, idx, val, it, it, b, dt);
            case mat_format::csc:
                return rocsparse_create_csc_descr(&A.d, n, n, n, ptr, idx, val, it, it, b, dt);
            case mat_format::coo:
                return rocsparse_create_coo_descr(&A.d, n, n, n, idx, idx, val, it, b, dt);
            case mat_format::coo_aos:
                return rocsparse_create_coo_aos_descr(&A.d, n, n, n, aos, val, it, b, dt);
            case mat_format::ell:
                return rocsparse_create_ell_descr(&A.d, n, n, idx, val, 1, it, b, dt);
            case mat_format::bsr_row:
            case mat_format::bsr_col:
                return rocsparse_create_bsr_descr(&A.d,
                                                  n,
                                                  n,
                                                  n,
                                                  format == mat_format::bsr_row
                                                      ? rocsparse_direction_row
                                                      : rocsparse_direction_column,
                                                  1,
                                                  ptr,
                                                  idx,
                                                  val,
                                                  it,
                                                  it,
                                                  b,
                                                  dt);
            case mat_format::sell:
                // one slice of n rows
                return rocsparse_create_sell_descr(
                    &A.d, n, n, n, n, n, slices, idx, val, it, it, b, dt);
            }
            return rocsparse_status_invalid_value;
        }

        rocsparse_status vectors(dnvec_guard& vx, dnvec_guard& vy) const
        {
            const rocsparse_status st
                = rocsparse_create_dnvec_descr(&vx.d, n, x, rocsparse_ut::dt_of<T>());
            return st != rocsparse_status_success
                       ? st
                       : rocsparse_create_dnvec_descr(&vy.d, n, y, rocsparse_ut::dt_of<T>());
        }

    private:
        static std::vector<int32_t> interleaved(int32_t count)
        {
            std::vector<int32_t> v(2 * count);
            for(int32_t i = 0; i < 2 * count; ++i)
                v[i] = i / 2;
            return v;
        }
    };

    rocsparse_status
        set_triangular(rocsparse_spmat_descr A, rocsparse_fill_mode fill, rocsparse_diag_type diag)
    {
        const rocsparse_status st
            = rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill, sizeof(fill));
        return st != rocsparse_status_success
                   ? st
                   : rocsparse_spmat_set_attribute(
                         A, rocsparse_spmat_diag_type, &diag, sizeof(diag));
    }

    // A dense matrix in either order, filled with one value; the storage has no padding.
    template <typename T>
    struct dense_matrix
    {
        int64_t          rows, cols, ld;
        device_vector<T> data;
        dnmat_guard      desc;

        dense_matrix(int64_t r, int64_t c, rocsparse_order order, T fill)
            : rows(r)
            , cols(c)
            , ld(std::max<int64_t>(1, order == rocsparse_order_column ? r : c))
            , data(std::vector<T>(
                  std::max<int64_t>(1, ld * (order == rocsparse_order_column ? c : r)), fill))
        {
            if(data.ptr)
                (void)rocsparse_create_dnmat_descr(
                    &desc.d, rows, cols, ld, data.ptr, rocsparse_ut::dt_of<T>(), order);
        }

        bool ok() const
        {
            return data.ptr && desc.d;
        }
    };

    template <typename V>
    rocsparse_status trs_input(rocsparse_handle       h,
                               rocsparse_sptrsv_descr d,
                               rocsparse_sptrsv_input input,
                               const V&               value)
    {
        return rocsparse_sptrsv_set_input(h, d, input, &value, sizeof(V), nullptr);
    }

    template <typename V>
    rocsparse_status trs_input(rocsparse_handle       h,
                               rocsparse_sptrsm_descr d,
                               rocsparse_sptrsm_input input,
                               const V&               value)
    {
        return rocsparse_sptrsm_set_input(h, d, input, &value, sizeof(V), nullptr);
    }

    // ----- SpMV --------------------------------------------------------------------

    struct spmv_call
    {
        rocsparse_spmv_alg  alg;
        rocsparse_operation op;
        bool                again = false; // preprocess a second time
    };

    // The calls share one matrix: a later algorithm must not reuse analysis data of an
    // earlier one (nnz-split reads arrays the other analyses never fill).
    struct spmv_case
    {
        mat_format             format;
        std::vector<spmv_call> calls;
    };

    std::vector<spmv_case> spmv_cases()
    {
        const auto               def = rocsparse_spmv_alg_default;
        std::vector<spmv_case>   v;
        const rocsparse_spmv_alg csr_algs[] = {def,
                                               rocsparse_spmv_alg_csr_rowsplit,
                                               rocsparse_spmv_alg_csr_adaptive,
                                               rocsparse_spmv_alg_csr_lrb,
                                               rocsparse_spmv_alg_csr_nnzsplit};
        for(const auto alg : csr_algs)
            for(const auto op : all_ops)
                v.push_back({mat_format::csr, {{alg, op, alg == def && op == op_n}}});
        v.push_back({mat_format::csc, {{def, op_n, true}, {def, op_c}}});
        for(const auto alg : csr_algs)
            if(alg != def)
                v.push_back({mat_format::csc, {{alg, op_n}}});
        const auto coo    = rocsparse_spmv_alg_coo;
        const auto atomic = rocsparse_spmv_alg_coo_atomic;
        v.push_back(
            {mat_format::coo, {{def, op_n, true}, {coo, op_n}, {atomic, op_n}, {coo, op_t}}});
        v.push_back({mat_format::coo_aos, {{def, op_n, true}, {coo, op_n}, {atomic, op_n}}});
        v.push_back(
            {mat_format::ell,
             {{def, op_n, true}, {rocsparse_spmv_alg_ell, op_n}, {rocsparse_spmv_alg_ell, op_c}}});
        for(const auto dir : {mat_format::bsr_row, mat_format::bsr_col})
            v.push_back({dir, {{def, op_n, true}, {rocsparse_spmv_alg_bsr, op_t}}});
        return v;
    }

    // The v2 calls: the same table is used, `again` meaning a second analysis.
    std::vector<spmv_case> v2_spmv_cases()
    {
        const auto             def = rocsparse_spmv_alg_default;
        std::vector<spmv_case> v;
        v.push_back({mat_format::csr,
                     {{rocsparse_spmv_alg_csr_nnzsplit, op_n},
                      {rocsparse_spmv_alg_csr_lrb, op_c},
                      {rocsparse_spmv_alg_csr_adaptive, op_n},
                      {rocsparse_spmv_alg_csr_adaptive, op_n}}});
        v.push_back({mat_format::csc, {{rocsparse_spmv_alg_csr_nnzsplit, op_n}, {def, op_t}}});
        v.push_back({mat_format::coo,
                     {{rocsparse_spmv_alg_coo_atomic, op_n}, {rocsparse_spmv_alg_coo, op_c}}});
        v.push_back({mat_format::coo_aos, {{rocsparse_spmv_alg_coo, op_t}}});
        v.push_back({mat_format::ell, {{rocsparse_spmv_alg_ell, op_c}}});
        v.push_back({mat_format::bsr_col, {{rocsparse_spmv_alg_bsr, op_n}}});
        v.push_back({mat_format::sell, {{rocsparse_spmv_alg_sell, op_n}, {def, op_t}}});
        return v;
    }

    class SpmvDispatch : public HandleTest, public ::testing::WithParamInterface<spmv_case>
    {
    };

    TEST_P(SpmvDispatch, identity)
    {
        const spmv_case& c = GetParam();
        identity<float>  id;
        spmat_guard      A;
        dnvec_guard      x, y;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(c.format, A), rocsparse_status_success);
        ASSERT_EQ(id.vectors(x, y), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const float         alpha = 1.0f;
        const float         beta  = 0.0f;
        for(const spmv_call& k : c.calls)
        {
            const bool ran = run_buffered(
                [&](rocsparse_spmv_stage stage, size_t* bytes, void* buffer) {
                    return rocsparse_spmv(handle,
                                          k.op,
                                          &alpha,
                                          A.d,
                                          x.d,
                                          &beta,
                                          y.d,
                                          f32,
                                          k.alg,
                                          stage,
                                          bytes,
                                          buffer);
                },
                rocsparse_spmv_stage_buffer_size,
                rocsparse_spmv_stage_preprocess,
                rocsparse_spmv_stage_compute,
                k.again ? 2 : 1);
            if(ran)
                expect_ones(id.y);
        }
    }

    INSTANTIATE_TEST_SUITE_P(L2Device, SpmvDispatch, ::testing::ValuesIn(spmv_cases()));

    class V2SpmvDispatch : public HandleTest, public ::testing::WithParamInterface<spmv_case>
    {
    };

    TEST_P(V2SpmvDispatch, identity)
    {
        const spmv_case& c = GetParam();
        identity<float>  id(c.format == mat_format::sell ? 2 : 3);
        spmat_guard      A;
        dnvec_guard      x, y;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(c.format, A), rocsparse_status_success);
        ASSERT_EQ(id.vectors(x, y), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const float         alpha = 1.0f;
        const float         beta  = 0.0f;
        for(const spmv_call& k : c.calls)
        {
            spmv_descr_guard descr;
            ASSERT_TRUE(descr.d);
            ASSERT_EQ(set_basic_inputs(handle, descr.d, k.alg, k.op, f32, f32),
                      rocsparse_status_success);
            ASSERT_EQ(
                set_input(handle, descr.d, rocsparse_spmv_input_nnz_use_starting_block_ids, true),
                rocsparse_status_success);
            const bool ran = run_stages(
                {rocsparse_v2_spmv_stage_analysis, rocsparse_v2_spmv_stage_compute},
                [&](rocsparse_v2_spmv_stage stage, size_t* bytes) {
                    return rocsparse_v2_spmv_buffer_size(
                        handle, descr.d, A.d, x.d, y.d, stage, bytes, nullptr);
                },
                [&](rocsparse_v2_spmv_stage stage, size_t bytes, void* buffer) {
                    return rocsparse_v2_spmv(handle,
                                             descr.d,
                                             &alpha,
                                             A.d,
                                             x.d,
                                             &beta,
                                             y.d,
                                             stage,
                                             bytes,
                                             buffer,
                                             nullptr);
                });
            if(ran)
                expect_ones(id.y);
        }
    }

    INSTANTIATE_TEST_SUITE_P(L2Device, V2SpmvDispatch, ::testing::ValuesIn(v2_spmv_cases()));

    class V2SpmvExtras : public HandleTest
    {
    };

    TEST_F(V2SpmvExtras, datatypes_and_batches)
    {
        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        auto set_types = [&](rocsparse_datatype scalar, rocsparse_datatype compute) {
            ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_scalar_datatype, scalar),
                      rocsparse_status_success);
            ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_compute_datatype, compute),
                      rocsparse_status_success);
        };
        auto vector = [](dnvec_guard& g, const auto& data, rocsparse_datatype type) {
            ASSERT_EQ(rocsparse_create_dnvec_descr(&g.d, data.n, data.ptr, type),
                      rocsparse_status_success);
        };

        // Enabling the extras requires extras to be set.
        EXPECT_NE(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, int32_t(1)),
                  rocsparse_status_success);

        // Integer extras.
        {
            set_types(rocsparse_datatype_i32_r, rocsparse_datatype_i32_r);
            device_vector<int32_t> gamma(std::vector<int32_t>{1, 1}),
                z0(std::vector<int32_t>(3, 1)), z1(std::vector<int32_t>(3, 2));
            dnvec_guard g, a, b;
            vector(g, gamma, rocsparse_datatype_i32_r);
            vector(a, z0, rocsparse_datatype_i32_r);
            vector(b, z1, rocsparse_datatype_i32_r);
            rocsparse_const_dnvec_descr zs[2] = {a.d, b.d};
            EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g.d, zs, nullptr),
                      rocsparse_status_success);
            for(const int32_t enable : {1, 0})
                EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, enable),
                          rocsparse_status_success);
            EXPECT_EQ(rocsparse_spmv_clear_extra(handle, descr.d, nullptr),
                      rocsparse_status_success);
        }

        // A datatype without extras.
        {
            set_types(rocsparse_datatype_f16_r, rocsparse_datatype_f16_r);
            device_vector<float> gamma(std::vector<float>(1, 1.0f)), z(std::vector<float>(3, 1.0f));
            dnvec_guard          g, a;
            vector(g, gamma, rocsparse_datatype_f16_r);
            vector(a, z, rocsparse_datatype_f16_r);
            rocsparse_const_dnvec_descr zs[1] = {a.d};
            EXPECT_NE(rocsparse_spmv_set_extra(handle, descr.d, 1, g.d, zs, nullptr),
                      rocsparse_status_success);
        }

        // A batch_count other than 1 on the matrix or a vector.
        identity<float> id;
        spmat_guard     A;
        dnvec_guard     x, y;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(mat_format::csr, A), rocsparse_status_success);
        ASSERT_EQ(id.vectors(x, y), rocsparse_status_success);
        spmv_descr_guard plain;
        ASSERT_TRUE(plain.d);
        ASSERT_EQ(set_basic_inputs(handle, plain.d, rocsparse_spmv_alg_default, op_n, f32, f32),
                  rocsparse_status_success);
        const float alpha              = 1.0f;
        const float beta               = 0.0f;
        auto        expect_unsupported = [&](rocsparse_v2_spmv_stage stage) {
            size_t bytes = 0;
            EXPECT_NE(rocsparse_v2_spmv_buffer_size(
                          handle, plain.d, A.d, x.d, y.d, stage, &bytes, nullptr),
                      rocsparse_status_success);
            EXPECT_NE(
                rocsparse_v2_spmv(
                    handle, plain.d, &alpha, A.d, x.d, &beta, y.d, stage, 0, nullptr, nullptr),
                rocsparse_status_success);
        };
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 2, id.n + 1, id.n),
                  rocsparse_status_success);
        expect_unsupported(rocsparse_v2_spmv_stage_analysis);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 1, 0, 0), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(x.d, 2, id.n), rocsparse_status_success);
        expect_unsupported(rocsparse_v2_spmv_stage_analysis);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(x.d, 1, id.n), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(y.d, 2, id.n), rocsparse_status_success);
        expect_unsupported(rocsparse_v2_spmv_stage_compute);
    }

    // ----- SpMM ----------------------------------------------------------------------

    // A 2 x 2 identity COO matrix times ones. n selects the main or remainder part of
    // the segmented-atomic kernel, and the orders and the operation of B select the
    // clauses of its layout condition.
    struct coomm_case
    {
        int                 n;
        rocsparse_order     order_b;
        rocsparse_operation trans_b;
        rocsparse_order     order_c;
    };

    // n 1..15 with every layout was swept: these are the cases that reach distinct code.
    std::vector<coomm_case> coomm_cases()
    {
        return {{2, row, op_t, col},
                {2, row, op_n, row},
                {5, col, op_n, col},
                {5, row, op_c, col},
                {6, col, op_c, col},
                {9, row, op_n, row},
                {12, row, op_n, row},
                {12, row, op_c, col},
                {13, col, op_c, col},
                {13, row, op_c, row},
                {14, col, op_n, col},
                {14, col, op_t, row},
                {15, col, op_n, col},
                {15, col, op_t, col}};
    }

    class CoommSegmentedAtomic : public HandleTest, public ::testing::WithParamInterface<coomm_case>
    {
    };

    TEST_P(CoommSegmentedAtomic, identity)
    {
        const coomm_case& c = GetParam();
        const bool        t = (c.trans_b != op_n);
        identity<float>   id(2);
        spmat_guard       A;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(mat_format::coo, A), rocsparse_status_success);
        dense_matrix<float> B(t ? c.n : 2, t ? 2 : c.n, c.order_b, 1.0f);
        dense_matrix<float> C(2, c.n, c.order_c, 0.0f);
        ASSERT_TRUE(B.ok() && C.ok());
        scoped_pointer_mode host(handle, false);
        const float         alpha = 1.0f;
        const float         beta  = 0.0f;
        const bool          ran   = run_buffered(
            [&](rocsparse_spmm_stage stage, size_t* bytes, void* buffer) {
                return rocsparse_spmm(handle,
                                      op_n,
                                      c.trans_b,
                                      &alpha,
                                      A.d,
                                      B.desc.d,
                                      &beta,
                                      C.desc.d,
                                      f32,
                                      rocsparse_spmm_alg_coo_segmented_atomic,
                                      stage,
                                      bytes,
                                      buffer);
            },
            rocsparse_spmm_stage_buffer_size,
            rocsparse_spmm_stage_preprocess,
            rocsparse_spmm_stage_compute);
        if(ran)
            expect_ones(C.data);
    }

    INSTANTIATE_TEST_SUITE_P(L2Device, CoommSegmentedAtomic, ::testing::ValuesIn(coomm_cases()));

    // ----- SpTRSV ----------------------------------------------------------------------

    struct sptrsv_options
    {
        rocsparse_operation       op          = op_n;
        rocsparse_datatype        scalar      = f32;
        rocsparse_datatype        compute     = f32;
        rocsparse_analysis_policy policy      = rocsparse_analysis_policy_reuse;
        const void*               alpha       = nullptr;
        bool                      diagonal    = false; // diagonal solve instead of triangular
        rocsparse_singularity     singularity = rocsparse_singularity_none;
        int64_t                   singularity_position = -1;
    };

    bool
        configure_sptrsv(rocsparse_handle handle, rocsparse_sptrsv_descr d, const sptrsv_options& o)
    {
        bool ok  = true;
        auto put = [&](rocsparse_sptrsv_input input, const auto& value) {
            ok = ok && trs_input(handle, d, input, value) == rocsparse_status_success;
        };
        put(rocsparse_sptrsv_input_alg, rocsparse_sptrsv_alg_default);
        put(rocsparse_sptrsv_input_operation, o.op);
        put(rocsparse_sptrsv_input_scalar_datatype, o.scalar);
        put(rocsparse_sptrsv_input_compute_datatype, o.compute);
        put(rocsparse_sptrsv_input_analysis_policy, o.policy);
        // the pointer itself is the input
        ok = ok
             && rocsparse_sptrsv_set_input(handle,
                                           d,
                                           rocsparse_sptrsv_input_scalar_alpha,
                                           o.alpha,
                                           sizeof(const void*),
                                           nullptr)
                    == rocsparse_status_success;
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        put(rocsparse_sptrsv_input_solve_mode,
            o.diagonal ? rocsparse_solve_mode_diagonal : rocsparse_solve_mode_triangular);
        if(o.diagonal)
            put(rocsparse_sptrsv_input_diagonal_modifier, rocsparse_diagonal_modifier_absolute);
#endif
        return ok;
    }

    // Returns true when the solve ran.
    bool run_sptrsv(rocsparse_handle      handle,
                    rocsparse_spmat_descr A,
                    rocsparse_dnvec_descr x,
                    rocsparse_dnvec_descr y,
                    const sptrsv_options& o)
    {
        sptrsv_descr_guard descr;
        EXPECT_NE(descr.d, nullptr);
        if(!descr.d
           || !(configure_sptrsv(handle, descr.d, o) || (ADD_FAILURE() << "configure", false)))
            return false;

        // The outputs are written through host pointers.
        int64_t               position    = -1;
        rocsparse_singularity singularity = rocsparse_singularity_none;
        auto                  query       = [&](rocsparse_sptrsv_output output, auto* value) {
            scoped_pointer_mode host(handle, false);
            return rocsparse_sptrsv_get_output(
                handle, descr.d, output, value, sizeof(*value), nullptr);
        };
        // nothing has been analysed yet
        (void)query(rocsparse_sptrsv_output_singularity_position, &position);
        (void)query(rocsparse_sptrsv_output_singularity, &singularity);

        const bool ran = run_stages(
            {rocsparse_sptrsv_stage_analysis, rocsparse_sptrsv_stage_compute},
            [&](rocsparse_sptrsv_stage stage, size_t* bytes) {
                return rocsparse_sptrsv_buffer_size(
                    handle, descr.d, A, x, y, stage, bytes, nullptr);
            },
            [&](rocsparse_sptrsv_stage stage, size_t bytes, void* buffer) {
                return rocsparse_sptrsv(handle, descr.d, A, x, y, stage, bytes, buffer, nullptr);
            });
        if(!ran)
            return false;

        EXPECT_EQ(query(rocsparse_sptrsv_output_singularity_position, &position),
                  rocsparse_status_success);
        EXPECT_EQ(query(rocsparse_sptrsv_output_singularity, &singularity),
                  rocsparse_status_success);
        EXPECT_EQ(singularity, o.singularity);
        EXPECT_EQ(position, o.singularity_position);
        const rocsparse_status st = query(rocsparse_sptrsv_output_zero_pivot_position, &position);
        EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_zero_pivot);
        return true;
    }

    struct sptrsv_case
    {
        mat_format                format;
        rocsparse_fill_mode       fill;
        rocsparse_diag_type       diag;
        rocsparse_analysis_policy policy;
        rocsparse_operation       op;
        bool                      diagonal_solve = false;
    };

    std::vector<sptrsv_case> sptrsv_cases()
    {
        using F = mat_format;
        std::vector<sptrsv_case> v{{F::coo, lo, nu, reuse, op_c},
                                   {F::coo, up, un, force, op_c},
                                   {F::csc, up, un, reuse, op_n},
                                   {F::ell, lo, nu, force, op_n},
                                   {F::ell, lo, un, reuse, op_n},
                                   {F::ell, up, nu, force, op_t}};
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        for(const auto format : {F::csr, F::csc})
            v.push_back({format, lo, nu, force, op_n, true});
#endif
        return v;
    }

    class SptrsvDispatch : public HandleTest, public ::testing::WithParamInterface<sptrsv_case>
    {
    };

    TEST_P(SptrsvDispatch, identity)
    {
        const sptrsv_case& c = GetParam();
        identity<float>    id;
        spmat_guard        A;
        dnvec_guard        x, y;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(c.format, A), rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, c.fill, c.diag), rocsparse_status_success);
        ASSERT_EQ(id.vectors(x, y), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const float         alpha = 1.0f;
        sptrsv_options      o;
        o.op       = c.op;
        o.policy   = c.policy;
        o.alpha    = &alpha;
        o.diagonal = c.diagonal_solve;
        if(run_sptrsv(handle, A.d, x.d, y.d, o))
            expect_ones(id.y);
    }

    INSTANTIATE_TEST_SUITE_P(L2Device, SptrsvDispatch, ::testing::ValuesIn(sptrsv_cases()));

    class SptrsvMisc : public HandleTest
    {
    };

    TEST_F(SptrsvMisc, missing_diagonal_is_a_symbolic_singularity)
    {
        // Row 1 has no diagonal entry.
        device_vector<int32_t> ptr(std::vector<int32_t>{0, 1, 2}), ind(std::vector<int32_t>{0, 0});
        device_vector<float>   val(std::vector<float>(2, 1.0f)), xv(std::vector<float>(2, 1.0f)),
            yv(std::vector<float>(2, 0.0f));
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && xv.ptr && yv.ptr);
        spmat_guard A;
        dnvec_guard x, y;
        ASSERT_EQ(rocsparse_create_csr_descr(&A.d,
                                             2,
                                             2,
                                             2,
                                             ptr,
                                             ind,
                                             val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             f32),
                  rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&x.d, 2, xv, f32), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&y.d, 2, yv, f32), rocsparse_status_success);
        const float    alpha = 1.0f;
        sptrsv_options o;
        o.alpha                = &alpha;
        o.singularity          = rocsparse_singularity_symbolic;
        o.singularity_position = 1;
        run_sptrsv(handle, A.d, x.d, y.d, o);
    }

    TEST_F(SptrsvMisc, scalar_type_differs_from_compute_type)
    {
        // The matrix and the vectors are f64, the scalar is f32.
        identity<double> id;
        spmat_guard      A;
        dnvec_guard      x, y;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(mat_format::csr, A), rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(id.vectors(x, y), rocsparse_status_success);
        const float          alpha_host = 1.0f;
        device_vector<float> alpha_device(std::vector<float>(1, 1.0f));
        ASSERT_TRUE(alpha_device.ptr);

        sptrsv_options o;
        o.scalar  = f32;
        o.compute = f64;
        o.alpha   = &alpha_host;
        {
            scoped_pointer_mode host(handle, false);
            if(run_sptrsv(handle, A.d, x.d, y.d, o))
                expect_ones(id.y);
        }
        // The library takes the scalar of a device pointer mode from the handle here, so
        // the product is not checked: this call only reaches the conversion.
        scoped_pointer_mode device(handle, true);
        o.op     = op_t;
        o.policy = rocsparse_analysis_policy_force;
        o.alpha  = alpha_device.ptr;
        run_sptrsv(handle, A.d, x.d, y.d, o);
    }

    // ----- SpTRSM ----------------------------------------------------------------------

    struct sptrsm_options
    {
        rocsparse_operation       op_a     = op_n;
        rocsparse_operation       op_x     = op_n;
        rocsparse_datatype        scalar   = f32;
        rocsparse_datatype        compute  = f32;
        rocsparse_analysis_policy policy   = rocsparse_analysis_policy_reuse;
        const void*               alpha    = nullptr;
        bool                      diagonal = false; // diagonal solve instead of triangular
    };

    bool
        configure_sptrsm(rocsparse_handle handle, rocsparse_sptrsm_descr d, const sptrsm_options& o)
    {
        bool ok  = true;
        auto put = [&](rocsparse_sptrsm_input input, const auto& value) {
            ok = ok && trs_input(handle, d, input, value) == rocsparse_status_success;
        };
        put(rocsparse_sptrsm_input_alg, rocsparse_sptrsm_alg_default);
        put(rocsparse_sptrsm_input_operation_A, o.op_a);
        put(rocsparse_sptrsm_input_operation_X, o.op_x);
        put(rocsparse_sptrsm_input_scalar_datatype, o.scalar);
        put(rocsparse_sptrsm_input_compute_datatype, o.compute);
        put(rocsparse_sptrsm_input_analysis_policy, o.policy);
        ok = ok
             && rocsparse_sptrsm_set_input(handle,
                                           d,
                                           rocsparse_sptrsm_input_scalar_alpha,
                                           o.alpha,
                                           sizeof(const void*),
                                           nullptr)
                    == rocsparse_status_success;
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        put(rocsparse_sptrsm_input_solve_mode,
            o.diagonal ? rocsparse_solve_mode_diagonal : rocsparse_solve_mode_triangular);
        if(o.diagonal)
            put(rocsparse_sptrsm_input_diagonal_modifier, rocsparse_diagonal_modifier_none);
#endif
        return ok;
    }

    // Returns true when the solve ran.
    bool run_sptrsm(rocsparse_handle      handle,
                    rocsparse_spmat_descr A,
                    rocsparse_dnmat_descr X,
                    rocsparse_dnmat_descr Y,
                    const sptrsm_options& o)
    {
        sptrsm_descr_guard descr;
        EXPECT_NE(descr.d, nullptr);
        if(!descr.d
           || !(configure_sptrsm(handle, descr.d, o) || (ADD_FAILURE() << "configure", false)))
            return false;
        const bool ran = run_stages(
            {rocsparse_sptrsm_stage_analysis, rocsparse_sptrsm_stage_compute},
            [&](rocsparse_sptrsm_stage stage, size_t* bytes) {
                return rocsparse_sptrsm_buffer_size(
                    handle, descr.d, A, X, Y, stage, bytes, nullptr);
            },
            [&](rocsparse_sptrsm_stage stage, size_t bytes, void* buffer) {
                return rocsparse_sptrsm(handle, descr.d, A, X, Y, stage, bytes, buffer, nullptr);
            });
        if(!ran)
            return false;
        // The output is written through a host pointer.
        scoped_pointer_mode host(handle, false);
        int64_t             position = 0;
        EXPECT_EQ(rocsparse_sptrsm_get_output(handle,
                                              descr.d,
                                              rocsparse_sptrsm_output_zero_pivot_position,
                                              &position,
                                              sizeof(position),
                                              nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(position, -1);
        return true;
    }

    // X is op(X) = m x nrhs stored as given, Y is m x nrhs.
    struct sptrsm_case
    {
        mat_format                format;
        rocsparse_operation       op_a;
        rocsparse_operation       op_x;
        rocsparse_order           order_x;
        rocsparse_order           order_y;
        int                       nrhs;
        rocsparse_analysis_policy policy;
        bool                      diagonal_solve = false;
    };

    std::vector<sptrsm_case> sptrsm_cases()
    {
        using F = mat_format;
        std::vector<sptrsm_case> v{{F::csr, op_c, op_n, col, col, 0, reuse},
                                   {F::coo, op_c, op_t, row, row, 2, reuse},
                                   {F::coo, op_t, op_n, row, col, 0, reuse},
                                   {F::coo, op_t, op_t, row, row, 0, force},
                                   {F::csc, op_n, op_t, col, col, 2, reuse},
                                   {F::csc, op_c, op_c, row, col, 2, force},
                                   {F::csc, op_t, op_n, row, col, 0, reuse}};
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        v.push_back({F::csr, op_n, op_n, col, col, 2, force, true});
#endif
        return v;
    }

    class SptrsmDispatch : public HandleTest, public ::testing::WithParamInterface<sptrsm_case>
    {
    };

    TEST_P(SptrsmDispatch, identity)
    {
        const sptrsm_case& c = GetParam();
        identity<float>    id;
        spmat_guard        A;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(c.format, A), rocsparse_status_success);
        // The diagonal solve ignores the stored diagonal.
        ASSERT_EQ(set_triangular(
                      A.d,
                      c.diagonal_solve ? rocsparse_fill_mode_upper : rocsparse_fill_mode_lower,
                      c.diagonal_solve ? rocsparse_diag_type_unit : rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        const bool          t = (c.op_x != op_n);
        dense_matrix<float> X(t ? c.nrhs : id.n, t ? id.n : c.nrhs, c.order_x, 1.0f);
        dense_matrix<float> Y(id.n, c.nrhs, c.order_y, 0.0f);
        ASSERT_TRUE(X.ok() && Y.ok());
        scoped_pointer_mode host(handle, false);
        const float         alpha = 1.0f;
        sptrsm_options      o;
        o.op_a     = c.op_a;
        o.op_x     = c.op_x;
        o.policy   = c.policy;
        o.alpha    = &alpha;
        o.diagonal = c.diagonal_solve;
        if(run_sptrsm(handle, A.d, X.desc.d, Y.desc.d, o) && c.nrhs > 0)
            expect_ones(Y.data);
    }

    INSTANTIATE_TEST_SUITE_P(L2Device, SptrsmDispatch, ::testing::ValuesIn(sptrsm_cases()));

    class SptrsmMisc : public HandleTest
    {
    };

    TEST_F(SptrsmMisc, scalar_type_differs_from_compute_type)
    {
        // The matrix and the matrices are f64, the scalar is f32.
        identity<double> id;
        spmat_guard      A;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(mat_format::csr, A), rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_unit),
                  rocsparse_status_success);
        dense_matrix<double> X(id.n, 2, rocsparse_order_column, 1.0);
        dense_matrix<double> Y(id.n, 2, rocsparse_order_row, 0.0);
        ASSERT_TRUE(X.ok() && Y.ok());
        const float          alpha_host = 1.0f;
        device_vector<float> alpha_device(std::vector<float>(1, 1.0f));
        ASSERT_TRUE(alpha_device.ptr);

        sptrsm_options o;
        o.scalar  = f32;
        o.compute = f64;
        o.op_a    = op_c;
        o.alpha   = &alpha_host;
        {
            scoped_pointer_mode host(handle, false);
            run_sptrsm(handle, A.d, X.desc.d, Y.desc.d, o);
        }
        scoped_pointer_mode device(handle, true);
        o.op_a   = op_n;
        o.policy = rocsparse_analysis_policy_force;
        o.alpha  = alpha_device.ptr;
        if(run_sptrsm(handle, A.d, X.desc.d, Y.desc.d, o))
            expect_ones(Y.data);
    }

    TEST_F(SptrsmMisc, batches_are_not_supported)
    {
        identity<float> id;
        spmat_guard     A;
        ASSERT_TRUE(id.ok());
        ASSERT_EQ(id.create(mat_format::csr, A), rocsparse_status_success);
        dense_matrix<float> X(id.n, 1, rocsparse_order_column, 1.0f);
        dense_matrix<float> Y(id.n, 1, rocsparse_order_column, 0.0f);
        ASSERT_TRUE(X.ok() && Y.ok());
        sptrsm_descr_guard descr;
        const float        alpha = 1.0f;
        sptrsm_options     o;
        o.alpha = &alpha;
        ASSERT_TRUE(descr.d);
        ASSERT_TRUE(configure_sptrsm(handle, descr.d, o));
        // The batch checks are in rocsparse_sptrsm, not in the buffer size query.
        auto expect_unsupported = [&](rocsparse_sptrsm_stage stage) {
            EXPECT_NE(rocsparse_sptrsm(
                          handle, descr.d, A.d, X.desc.d, Y.desc.d, stage, 0, nullptr, nullptr),
                      rocsparse_status_success);
        };
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 2, id.n + 1, id.n),
                  rocsparse_status_success);
        expect_unsupported(rocsparse_sptrsm_stage_analysis);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 1, 0, 0), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(X.desc.d, 2, id.n), rocsparse_status_success);
        expect_unsupported(rocsparse_sptrsm_stage_analysis);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(X.desc.d, 1, id.n), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(Y.desc.d, 2, 0), rocsparse_status_success);
        expect_unsupported(rocsparse_sptrsm_stage_compute);
    }

    // ----- csrsm --------------------------------------------------------------------------

    // blockdim starts at 512 and shifts until it is the smallest power of two that is
    // >= nrhs and >= 64, then the matching kernel is launched. nrhs = 1 stays on the
    // csrsv path, 2, 65, 129, 257 and 513 select the block sizes 64 to 1024.
    struct csrsm_case
    {
        rocsparse_int             nrhs;
        rocsparse_operation       op_a, op_b;
        rocsparse_fill_mode       fill;
        rocsparse_diag_type       diag;
        rocsparse_analysis_policy policy;
    };

    // The full cross product of operations, fill modes, diagonals and policies was
    // swept; these cases reach distinct code.
    std::vector<csrsm_case> csrsm_cases()
    {
        return {{0, op_t, op_t, up, nu, force},
                {1, op_n, op_n, up, un, reuse},
                {1, op_n, op_c, up, nu, force},
                {65, op_c, op_t, lo, un, reuse},
                {129, op_c, op_n, up, un, force},
                {257, op_n, op_t, lo, nu, force},
                {513, op_c, op_c, up, un, force}};
    }

    class CsrsmDispatch : public HandleTest, public ::testing::WithParamInterface<csrsm_case>
    {
    };

    TEST_P(CsrsmDispatch, identity)
    {
        const csrsm_case&   c   = GetParam();
        const rocsparse_int m   = 4;
        const rocsparse_int ldb = (c.op_b == op_n) ? m : std::max<rocsparse_int>(c.nrhs, 1);
        const rocsparse_int nb  = ldb * ((c.op_b == op_n) ? std::max<rocsparse_int>(c.nrhs, 1) : m);
        device_vector<rocsparse_int> ptr(counting<rocsparse_int>(m + 1)),
            ind(counting<rocsparse_int>(m));
        device_vector<float> val(std::vector<float>(m, 1.0f)), B(std::vector<float>(nb, 1.0f));
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && B.ptr);
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, c.fill), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, c.diag), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const float         alpha  = 1.0f;
        const auto          policy = rocsparse_solve_policy_auto;

        size_t bytes = 0;
        ASSERT_EQ(rocsparse_scsrsm_buffer_size(handle,
                                               c.op_a,
                                               c.op_b,
                                               m,
                                               c.nrhs,
                                               m,
                                               &alpha,
                                               descr,
                                               val,
                                               ptr,
                                               ind,
                                               B,
                                               ldb,
                                               info,
                                               policy,
                                               &bytes),
                  rocsparse_status_success);
        device_vector<char> buffer(std::max<size_t>(bytes, 1));
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_scsrsm_analysis(handle,
                                            c.op_a,
                                            c.op_b,
                                            m,
                                            c.nrhs,
                                            m,
                                            &alpha,
                                            descr,
                                            val,
                                            ptr,
                                            ind,
                                            B,
                                            ldb,
                                            info,
                                            c.policy,
                                            policy,
                                            buffer),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_scsrsm_solve(handle,
                                         c.op_a,
                                         c.op_b,
                                         m,
                                         c.nrhs,
                                         m,
                                         &alpha,
                                         descr,
                                         val,
                                         ptr,
                                         ind,
                                         B,
                                         ldb,
                                         info,
                                         policy,
                                         buffer),
                  rocsparse_status_success);
        rocsparse_int pivot = 0;
        ASSERT_EQ(rocsparse_csrsm_zero_pivot(handle, info, &pivot), rocsparse_status_success);
        EXPECT_EQ(pivot, -1);
        if(c.nrhs > 0)
            expect_ones(B);
        EXPECT_EQ(rocsparse_csrsm_clear(handle, info), rocsparse_status_success);
    }

    INSTANTIATE_TEST_SUITE_P(L2Device, CsrsmDispatch, ::testing::ValuesIn(csrsm_cases()));

    class CsrsmQuickReturn : public HandleTest
    {
    };

    TEST_F(CsrsmQuickReturn, empty_system)
    {
        // m == 0 returns before the pointers are checked.
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        device_vector<float>         dummy(std::vector<float>(1, 0.0f));
        device_vector<rocsparse_int> index(std::vector<rocsparse_int>(1, 0));
        ASSERT_TRUE(dummy.ptr && index.ptr);
        const float alpha = 1.0f;
        size_t      bytes = 0;
        ASSERT_EQ(rocsparse_scsrsm_buffer_size(handle,
                                               op_n,
                                               op_n,
                                               0,
                                               0,
                                               0,
                                               &alpha,
                                               descr,
                                               dummy,
                                               index,
                                               index,
                                               dummy,
                                               1,
                                               info,
                                               rocsparse_solve_policy_auto,
                                               &bytes),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_scsrsm_solve(handle,
                                         op_n,
                                         op_t,
                                         0,
                                         1,
                                         0,
                                         &alpha,
                                         descr,
                                         dummy,
                                         index,
                                         index,
                                         dummy,
                                         1,
                                         info,
                                         rocsparse_solve_policy_auto,
                                         dummy.ptr),
                  rocsparse_status_success);
    }
} // namespace
