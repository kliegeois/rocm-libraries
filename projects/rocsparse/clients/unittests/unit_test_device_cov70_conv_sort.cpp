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

// Device unit tests for the sparse sort routines: the generic rocsparse_spsort
// API (COO / CSR / CSC, 32 and 64 bit indices, strided batches, in place and out
// of place) and the legacy coosort / csrsort / cscsort entry points with and
// without a permutation. Every output (offsets, indices, values and the
// permutation) is compared with a host reference sort.

#include "unit_test_device_cov70_conv_common.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

using namespace rocsparse_ut;
using namespace cov70conv;

namespace
{
    // ------------------------------------------------------------------
    // Host matrices
    // ------------------------------------------------------------------

    // Unique (row, col) positions of an M x N matrix in a scrambled order.
    std::vector<std::pair<int, int>> scrambled_positions(int M, int N)
    {
        std::vector<std::pair<int, int>> pos;
        for(int i = 0; i < M; ++i)
            for(int j = 0; j < N; ++j)
                if((i * 7 + j * 3) % 4 == 0)
                    pos.emplace_back(i, j);
        const size_t                     n = pos.size();
        std::vector<std::pair<int, int>> out(n);
        for(size_t k = 0; k < n; ++k)
            out[(k * 10007) % n] = pos[k];
        return out;
    }

    // Compressed major index structure with unsorted minor indices in each major slice.
    struct Compressed
    {
        int              M = 0, N = 0;
        std::vector<int> offsets; // size M + 1, zero based
        std::vector<int> minor; // zero based, unsorted inside every slice
    };

    Compressed make_compressed(int M, int N)
    {
        Compressed c;
        c.M = M;
        c.N = N;
        c.offsets.assign(static_cast<size_t>(M) + 1, 0);
        for(int r = 0; r < M; ++r)
        {
            const int cnt = (r * 3 + 1) % (N / 2 + 2); // includes empty slices
            const int s   = (r * 5) % N;
            for(int k = 0; k < cnt; ++k)
                c.minor.push_back((s + k * 5) % N); // N = 13 and 5 are coprime: unique
            c.offsets[static_cast<size_t>(r) + 1] = static_cast<int>(c.minor.size());
        }
        return c;
    }

    // ------------------------------------------------------------------
    // spsort driver
    // ------------------------------------------------------------------

    rocsparse_status run_spsort(rocsparse_handle            handle,
                                rocsparse_const_spmat_descr source,
                                rocsparse_spmat_descr       target,
                                rocsparse_spsort_alg        alg,
                                bool                        set_direction,
                                rocsparse_direction         dir)
    {
        rocsparse_spsort_descr descr = nullptr;
        EXPECT_EQ(rocsparse_spsort_descr_create(handle, &descr, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        if(set_direction)
        {
            EXPECT_EQ(
                rocsparse_spsort_set_input(
                    handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr),
                rocsparse_status_success);
        }

        size_t analysis_size = 77;
        EXPECT_EQ(rocsparse_spsort_buffer_size(handle,
                                               descr,
                                               source,
                                               target,
                                               rocsparse_spsort_stage_analysis,
                                               &analysis_size,
                                               nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(analysis_size, size_t(0));
        rocsparse_status status = rocsparse_spsort(
            handle, descr, source, target, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr);
        EXPECT_EQ(status, rocsparse_status_success);

        size_t size = 0;
        EXPECT_EQ(
            rocsparse_spsort_buffer_size(
                handle, descr, source, target, rocsparse_spsort_stage_compute, &size, nullptr),
            rocsparse_status_success);
        EXPECT_GT(size, size_t(0));
        void* buffer = nullptr;
        EXPECT_EQ(hipMalloc(&buffer, size), hipSuccess);

        status = rocsparse_spsort(
            handle, descr, source, target, rocsparse_spsort_stage_compute, size, buffer, nullptr);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(hipFree(buffer), hipSuccess);
        EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
        return status;
    }

    // ------------------------------------------------------------------
    // COO
    // ------------------------------------------------------------------

    struct CooCfg
    {
        rocsparse_direction  dir;
        int                  base;
        int                  batch;
        int64_t              stride; // > nnz leaves a padding between the batches
        bool                 in_place;
        bool                 broadcast; // a single source matrix sorted into every batch
        rocsparse_spsort_alg alg;
    };

    template <typename I, typename T>
    void run_coo(rocsparse_handle handle, const CooCfg& cfg)
    {
        const int  M = 23, N = 17;
        const auto pos = scrambled_positions(M, N);
        const int  nnz = static_cast<int>(pos.size());
        ASSERT_GT(nnz, 20);

        const int     B_dst = cfg.batch;
        const int     B_src = cfg.broadcast ? 1 : cfg.batch;
        const int64_t s_src = cfg.broadcast ? 0 : cfg.stride;
        const int64_t s_dst = cfg.stride;
        const size_t  n_src = static_cast<size_t>((B_src - 1) * s_src + nnz);
        const size_t  n_dst = static_cast<size_t>((B_dst - 1) * s_dst + nnz);

        const I        sent_i = static_cast<I>(-5);
        const T        sent_v = make_val<T>(-5, -5);
        std::vector<I> h_row(n_src, sent_i), h_col(n_src, sent_i);
        std::vector<T> h_val(n_src, sent_v);
        for(int b = 0; b < B_src; ++b)
            for(int k = 0; k < nnz; ++k)
            {
                h_row[static_cast<size_t>(b * s_src + k)] = static_cast<I>(pos[k].first + cfg.base);
                h_col[static_cast<size_t>(b * s_src + k)]
                    = static_cast<I>(pos[k].second + cfg.base);
                h_val[static_cast<size_t>(b * s_src + k)] = make_val<T>(1 + k + 1000 * b, k % 5);
            }

        // Reference permutation.
        std::vector<int> perm(static_cast<size_t>(nnz));
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
            const auto ka = (cfg.dir == rocsparse_direction_row)
                                ? std::make_pair(pos[a].first, pos[a].second)
                                : std::make_pair(pos[a].second, pos[a].first);
            const auto kb = (cfg.dir == rocsparse_direction_row)
                                ? std::make_pair(pos[b].first, pos[b].second)
                                : std::make_pair(pos[b].second, pos[b].first);
            return ka < kb;
        });

        device_vector<I> d_row(h_row), d_col(h_col);
        device_vector<T> d_val(h_val);
        std::vector<I>   init_i(n_dst, sent_i);
        std::vector<T>   init_v(n_dst, sent_v);
        device_vector<I> t_row(cfg.in_place ? std::vector<I>() : init_i);
        device_vector<I> t_col(cfg.in_place ? std::vector<I>() : init_i);
        device_vector<T> t_val(cfg.in_place ? std::vector<T>() : init_v);
        I*               pr = cfg.in_place ? d_row.ptr : t_row.ptr;
        I*               pc = cfg.in_place ? d_col.ptr : t_col.ptr;
        T*               pv = cfg.in_place ? d_val.ptr : t_val.ptr;
        ASSERT_NE(d_row.ptr, nullptr);
        ASSERT_NE(pr, nullptr);

        rocsparse_spmat_descr src = nullptr, dst = nullptr;
        const auto idx_base = cfg.base ? rocsparse_index_base_one : rocsparse_index_base_zero;
        ASSERT_EQ(
            rocsparse_create_coo_descr(
                &src, M, N, nnz, d_row.ptr, d_col.ptr, d_val.ptr, it_of<I>(), idx_base, dt_of<T>()),
            rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_descr(
                      &dst, M, N, nnz, pr, pc, pv, it_of<I>(), idx_base, dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(src, B_src, s_src), rocsparse_status_success);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(dst, B_dst, s_dst), rocsparse_status_success);

        ASSERT_EQ(run_spsort(handle, src, dst, cfg.alg, true, cfg.dir), rocsparse_status_success);

        // Expected target.
        std::vector<I> e_row = cfg.in_place ? h_row : init_i;
        std::vector<I> e_col = cfg.in_place ? h_col : init_i;
        std::vector<T> e_val = cfg.in_place ? h_val : init_v;
        for(int b = 0; b < B_dst; ++b)
            for(int k = 0; k < nnz; ++k)
            {
                const int    p  = perm[static_cast<size_t>(k)];
                const size_t at = static_cast<size_t>(b * s_dst + k);
                e_row[at]       = static_cast<I>(pos[p].first + cfg.base);
                e_col[at]       = static_cast<I>(pos[p].second + cfg.base);
                const int sb    = cfg.broadcast ? 0 : b;
                e_val[at]       = make_val<T>(1 + p + 1000 * sb, p % 5);
            }
        EXPECT_EQ(to_host(pr, n_dst), e_row);
        EXPECT_EQ(to_host(pc, n_dst), e_col);
        const std::vector<T> g_val = to_host(pv, n_dst);
        ASSERT_EQ(g_val.size(), e_val.size());
        for(size_t k = 0; k < g_val.size(); ++k)
        {
            EXPECT_EQ(g_val[k], e_val[k]) << "value " << k;
        }

        EXPECT_EQ(rocsparse_destroy_spmat_descr(src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(dst), rocsparse_status_success);
    }

    template <typename I, typename T>
    void run_coo_grid(rocsparse_handle handle)
    {
        const auto row = rocsparse_direction_row;
        const auto col = rocsparse_direction_column;
        const auto def = rocsparse_spsort_alg_default;
        const auto rdx = rocsparse_spsort_alg_radix_sort;
        // dir, base, batch, stride, in_place, broadcast, alg
        run_coo<I, T>(handle, {row, 0, 1, 0, false, false, def});
        run_coo<I, T>(handle, {col, 1, 1, 0, false, false, rdx});
        run_coo<I, T>(handle, {row, 1, 1, 0, true, false, rdx});
        run_coo<I, T>(handle, {col, 0, 1, 0, true, false, def});
        run_coo<I, T>(handle, {row, 0, 3, 130, false, false, def});
        run_coo<I, T>(handle, {col, 1, 2, 300, true, false, rdx});
        run_coo<I, T>(handle, {row, 1, 3, 125, false, true, def});
    }

    TEST_F(HandleTest, Cov70ConvSpsortCooInt32Float)
    {
        run_coo_grid<int32_t, float>(handle);
    }
    TEST_F(HandleTest, Cov70ConvSpsortCooInt64Double)
    {
        run_coo_grid<int64_t, double>(handle);
    }
    TEST_F(HandleTest, Cov70ConvSpsortCooInt32FloatComplex)
    {
        run_coo_grid<int32_t, rocsparse_float_complex>(handle);
    }
    TEST_F(HandleTest, Cov70ConvSpsortCooInt64DoubleComplex)
    {
        run_coo_grid<int64_t, rocsparse_double_complex>(handle);
    }

    // ------------------------------------------------------------------
    // CSR / CSC
    // ------------------------------------------------------------------

    struct CsxCfg
    {
        bool                 csc;
        int                  base;
        int                  batch;
        int64_t              off_stride; // 0: the offsets are shared by the batches
        int64_t              cv_stride; // stride of the indices and the values
        bool                 in_place;
        bool                 broadcast;
        rocsparse_spsort_alg alg;
    };

    template <typename P, typename J, typename T>
    void run_csx(rocsparse_handle handle, const CsxCfg& cfg)
    {
        // The major dimension is the number of rows for CSR and of columns for CSC.
        const int        M = 37, N = 13;
        const Compressed c   = make_compressed(M, N);
        const int        nnz = static_cast<int>(c.minor.size());
        ASSERT_GT(nnz, 20);

        const int     B_dst  = cfg.batch;
        const int     B_src  = cfg.broadcast ? 1 : cfg.batch;
        const int64_t os_src = cfg.broadcast ? 0 : cfg.off_stride;
        const int64_t cs_src = cfg.broadcast ? 0 : cfg.cv_stride;
        const size_t  no_src = static_cast<size_t>((B_src - 1) * os_src + M + 1);
        const size_t  nc_src = static_cast<size_t>((B_src - 1) * cs_src + nnz);
        const size_t  no_dst = static_cast<size_t>((B_dst - 1) * cfg.off_stride + M + 1);
        const size_t  nc_dst = static_cast<size_t>((B_dst - 1) * cfg.cv_stride + nnz);

        const P sent_p = static_cast<P>(-5);
        const J sent_j = static_cast<J>(-5);
        const T sent_v = make_val<T>(-5, -5);

        std::vector<P> h_off(no_src, sent_p);
        std::vector<J> h_ind(nc_src, sent_j);
        std::vector<T> h_val(nc_src, sent_v);
        for(int b = 0; b < B_src; ++b)
        {
            if(b == 0 || os_src != 0)
                for(int r = 0; r <= M; ++r)
                    h_off[static_cast<size_t>(b * os_src + r)]
                        = static_cast<P>(c.offsets[static_cast<size_t>(r)] + cfg.base);
            for(int k = 0; k < nnz; ++k)
            {
                h_ind[static_cast<size_t>(b * cs_src + k)]
                    = static_cast<J>(c.minor[static_cast<size_t>(k)] + cfg.base);
                h_val[static_cast<size_t>(b * cs_src + k)] = make_val<T>(1 + k + 1000 * b, k % 5);
            }
        }

        // Reference: sort the minor indices of every slice.
        std::vector<int> perm(static_cast<size_t>(nnz));
        std::iota(perm.begin(), perm.end(), 0);
        for(int r = 0; r < M; ++r)
        {
            std::stable_sort(perm.begin() + c.offsets[static_cast<size_t>(r)],
                             perm.begin() + c.offsets[static_cast<size_t>(r) + 1],
                             [&](int a, int b) {
                                 return c.minor[static_cast<size_t>(a)]
                                        < c.minor[static_cast<size_t>(b)];
                             });
        }

        device_vector<P> d_off(h_off);
        device_vector<J> d_ind(h_ind);
        device_vector<T> d_val(h_val);
        std::vector<P>   init_p(no_dst, sent_p);
        std::vector<J>   init_j(nc_dst, sent_j);
        std::vector<T>   init_v(nc_dst, sent_v);
        device_vector<P> t_off(cfg.in_place ? std::vector<P>() : init_p);
        device_vector<J> t_ind(cfg.in_place ? std::vector<J>() : init_j);
        device_vector<T> t_val(cfg.in_place ? std::vector<T>() : init_v);
        P*               po = cfg.in_place ? d_off.ptr : t_off.ptr;
        J*               pj = cfg.in_place ? d_ind.ptr : t_ind.ptr;
        T*               pv = cfg.in_place ? d_val.ptr : t_val.ptr;
        ASSERT_NE(d_off.ptr, nullptr);
        ASSERT_NE(po, nullptr);
        ASSERT_NE(pj, nullptr);

        const auto idx_base       = cfg.base ? rocsparse_index_base_one : rocsparse_index_base_zero;
        rocsparse_spmat_descr src = nullptr, dst = nullptr;
        if(cfg.csc)
        {
            ASSERT_EQ(rocsparse_create_csc_descr(&src,
                                                 N,
                                                 M,
                                                 nnz,
                                                 d_off.ptr,
                                                 d_ind.ptr,
                                                 d_val.ptr,
                                                 it_of<P>(),
                                                 it_of<J>(),
                                                 idx_base,
                                                 dt_of<T>()),
                      rocsparse_status_success);
            ASSERT_EQ(
                rocsparse_create_csc_descr(
                    &dst, N, M, nnz, po, pj, pv, it_of<P>(), it_of<J>(), idx_base, dt_of<T>()),
                rocsparse_status_success);
            ASSERT_EQ(rocsparse_csc_set_strided_batch(src, B_src, os_src, cs_src),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_csc_set_strided_batch(dst, B_dst, cfg.off_stride, cfg.cv_stride),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&src,
                                                 M,
                                                 N,
                                                 nnz,
                                                 d_off.ptr,
                                                 d_ind.ptr,
                                                 d_val.ptr,
                                                 it_of<P>(),
                                                 it_of<J>(),
                                                 idx_base,
                                                 dt_of<T>()),
                      rocsparse_status_success);
            ASSERT_EQ(
                rocsparse_create_csr_descr(
                    &dst, M, N, nnz, po, pj, pv, it_of<P>(), it_of<J>(), idx_base, dt_of<T>()),
                rocsparse_status_success);
            ASSERT_EQ(rocsparse_csr_set_strided_batch(src, B_src, os_src, cs_src),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, B_dst, cfg.off_stride, cfg.cv_stride),
                      rocsparse_status_success);
        }

        ASSERT_EQ(run_spsort(handle, src, dst, cfg.alg, false, rocsparse_direction_row),
                  rocsparse_status_success);

        std::vector<P> e_off = cfg.in_place ? h_off : init_p;
        std::vector<J> e_ind = cfg.in_place ? h_ind : init_j;
        std::vector<T> e_val = cfg.in_place ? h_val : init_v;
        for(int b = 0; b < B_dst; ++b)
        {
            if(b == 0 || cfg.off_stride != 0)
                for(int r = 0; r <= M; ++r)
                    e_off[static_cast<size_t>(b * cfg.off_stride + r)]
                        = static_cast<P>(c.offsets[static_cast<size_t>(r)] + cfg.base);
            for(int k = 0; k < nnz; ++k)
            {
                const int    p  = perm[static_cast<size_t>(k)];
                const size_t at = static_cast<size_t>(b * cfg.cv_stride + k);
                e_ind[at]       = static_cast<J>(c.minor[static_cast<size_t>(p)] + cfg.base);
                const int sb    = cfg.broadcast ? 0 : b;
                e_val[at]       = make_val<T>(1 + p + 1000 * sb, p % 5);
            }
        }
        EXPECT_EQ(to_host(po, no_dst), e_off);
        EXPECT_EQ(to_host(pj, nc_dst), e_ind);
        const std::vector<T> g_val = to_host(pv, nc_dst);
        ASSERT_EQ(g_val.size(), e_val.size());
        for(size_t k = 0; k < g_val.size(); ++k)
        {
            EXPECT_EQ(g_val[k], e_val[k]) << "value " << k;
        }

        EXPECT_EQ(rocsparse_destroy_spmat_descr(src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(dst), rocsparse_status_success);
    }

    template <typename P, typename J, typename T>
    void run_csx_grid(rocsparse_handle handle)
    {
        const auto def = rocsparse_spsort_alg_default;
        const auto rdx = rocsparse_spsort_alg_radix_sort;
        for(const bool csc : {false, true})
        {
            // csc, base, batch, off_stride, cv_stride, in_place, broadcast, alg
            run_csx<P, J, T>(handle, {csc, 0, 1, 0, 0, false, false, def});
            run_csx<P, J, T>(handle, {csc, 1, 1, 0, 0, false, false, rdx});
            run_csx<P, J, T>(handle, {csc, 1, 1, 0, 0, true, false, def});
            run_csx<P, J, T>(handle, {csc, 0, 1, 0, 0, true, false, rdx});
            // shared offsets, strided indices and values
            run_csx<P, J, T>(handle, {csc, 0, 3, 0, 140, false, false, def});
            // one offsets array per batch, padded
            run_csx<P, J, T>(handle, {csc, 1, 2, 50, 150, true, false, rdx});
            run_csx<P, J, T>(handle, {csc, 0, 3, 40, 135, false, false, def});
            // a single source sorted into every batch
            run_csx<P, J, T>(handle, {csc, 1, 3, 0, 140, false, true, def});
            run_csx<P, J, T>(handle, {csc, 0, 2, 41, 133, false, true, rdx});
        }
    }

    TEST_F(HandleTest, Cov70ConvSpsortCsxInt32Int32Float)
    {
        run_csx_grid<int32_t, int32_t, float>(handle);
    }
    TEST_F(HandleTest, Cov70ConvSpsortCsxInt32Int64Double)
    {
        run_csx_grid<int32_t, int64_t, double>(handle);
    }
    TEST_F(HandleTest, Cov70ConvSpsortCsxInt64Int32FloatComplex)
    {
        run_csx_grid<int64_t, int32_t, rocsparse_float_complex>(handle);
    }
    TEST_F(HandleTest, Cov70ConvSpsortCsxInt64Int64DoubleComplex)
    {
        run_csx_grid<int64_t, int64_t, rocsparse_double_complex>(handle);
    }

    // ------------------------------------------------------------------
    // spsort descriptor and argument validation
    // ------------------------------------------------------------------

    TEST_F(HandleTest, Cov70ConvSpsortDescrInputs)
    {
        rocsparse_spsort_descr descr = nullptr;
        EXPECT_EQ(rocsparse_spsort_descr_create(nullptr, &descr, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spsort_descr_create(handle, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        ASSERT_EQ(rocsparse_spsort_descr_create(handle, &descr, nullptr), rocsparse_status_success);

        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_radix_sort;
        const rocsparse_direction  dir = rocsparse_direction_column;
        EXPECT_EQ(rocsparse_spsort_set_input(
                      nullptr, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, nullptr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spsort_set_input(handle,
                                             descr,
                                             static_cast<rocsparse_spsort_input>(100),
                                             &alg,
                                             sizeof(alg),
                                             nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, nullptr, sizeof(alg), nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg) - 1, nullptr),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir) + 1, nullptr),
            rocsparse_status_invalid_size);
        const rocsparse_direction bad_dir = static_cast<rocsparse_direction>(100);
        EXPECT_EQ(rocsparse_spsort_set_input(handle,
                                             descr,
                                             rocsparse_spsort_input_direction,
                                             &bad_dir,
                                             sizeof(bad_dir),
                                             nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr),
                  rocsparse_status_success);

        // Once the analysis has run, the inputs can no longer be changed.
        const int              nnz = 4;
        std::vector<int32_t>   ptr = {0, 2, 4}, ind = {1, 0, 1, 0};
        std::vector<float>     val = {1, 2, 3, 4};
        device_vector<int32_t> d_ptr(ptr), d_ind(ind);
        device_vector<float>   d_val(val);
        rocsparse_spmat_descr  mat = nullptr;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                             2,
                                             2,
                                             nnz,
                                             d_ptr.ptr,
                                             d_ind.ptr,
                                             d_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_spsort(
                handle, descr, mat, mat, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr),
            rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr),
                  rocsparse_status_invalid_value);
        // The analysis cannot be executed twice.
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, mat, mat, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr),
            rocsparse_status_invalid_value);

        EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_descr_destroy(nullptr, descr, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
        // A null descriptor is accepted.
        EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, nullptr, nullptr),
                  rocsparse_status_success);
    }

    TEST_F(HandleTest, Cov70ConvSpsortInvalidArguments)
    {
        const int              M = 3, N = 4, nnz = 5;
        std::vector<int32_t>   ptr = {0, 2, 3, 5}, ind = {3, 1, 2, 2, 0};
        std::vector<float>     val = {1, 2, 3, 4, 5};
        device_vector<int32_t> d_ptr(ptr), d_ind(ind), d_ptr2(ptr), d_ind2(ind);
        device_vector<float>   d_val(val), d_val2(val);
        device_vector<int64_t> d_ind64(std::vector<int64_t>(ind.begin(), ind.end()));
        device_vector<int32_t> d_ival(ind);
        ASSERT_NE(d_ptr.ptr, nullptr);

        const auto make_csr = [&](rocsparse_spmat_descr* d,
                                  int64_t                rows,
                                  int64_t                cols,
                                  int64_t                nz,
                                  void*                  p,
                                  void*                  i,
                                  void*                  v,
                                  rocsparse_indextype    jt,
                                  rocsparse_index_base   base,
                                  rocsparse_datatype     dt) {
            return rocsparse_create_csr_descr(
                d, rows, cols, nz, p, i, v, rocsparse_indextype_i32, jt, base, dt);
        };
        rocsparse_spmat_descr src = nullptr, dst = nullptr;
        ASSERT_EQ(make_csr(&src,
                           M,
                           N,
                           nnz,
                           d_ptr.ptr,
                           d_ind.ptr,
                           d_val.ptr,
                           rocsparse_indextype_i32,
                           rocsparse_index_base_zero,
                           rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(make_csr(&dst,
                           M,
                           N,
                           nnz,
                           d_ptr2.ptr,
                           d_ind2.ptr,
                           d_val2.ptr,
                           rocsparse_indextype_i32,
                           rocsparse_index_base_zero,
                           rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        rocsparse_spsort_descr descr = nullptr;
        ASSERT_EQ(rocsparse_spsort_descr_create(handle, &descr, nullptr), rocsparse_status_success);
        size_t     size        = 0;
        const auto buffer_size = [&](rocsparse_spsort_descr      dd,
                                     rocsparse_const_spmat_descr s,
                                     rocsparse_spmat_descr       t,
                                     rocsparse_spsort_stage      stage) {
            return rocsparse_spsort_buffer_size(handle, dd, s, t, stage, &size, nullptr);
        };

        // Null arguments.
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      nullptr, descr, src, dst, rocsparse_spsort_stage_compute, &size, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(buffer_size(nullptr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(buffer_size(descr, nullptr, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(buffer_size(descr, src, nullptr, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(buffer_size(descr, src, dst, static_cast<rocsparse_spsort_stage>(100)),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);

        // The algorithm has not been set.
        EXPECT_EQ(buffer_size(descr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_value);
        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;
        ASSERT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_success);

        // Mismatching target.
        rocsparse_spmat_descr other = nullptr;
        ASSERT_EQ(make_csr(&other,
                           M,
                           N + 1,
                           nnz,
                           d_ptr2.ptr,
                           d_ind2.ptr,
                           d_val2.ptr,
                           rocsparse_indextype_i32,
                           rocsparse_index_base_zero,
                           rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, other, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(other), rocsparse_status_success);

        ASSERT_EQ(make_csr(&other,
                           M,
                           N,
                           nnz,
                           d_ptr2.ptr,
                           d_ind64.ptr,
                           d_val2.ptr,
                           rocsparse_indextype_i64,
                           rocsparse_index_base_zero,
                           rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, other, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(other), rocsparse_status_success);

        ASSERT_EQ(make_csr(&other,
                           M,
                           N,
                           nnz,
                           d_ptr2.ptr,
                           d_ind2.ptr,
                           d_val2.ptr,
                           rocsparse_indextype_i32,
                           rocsparse_index_base_one,
                           rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, other, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(other), rocsparse_status_success);

        ASSERT_EQ(rocsparse_create_coo_descr(&other,
                                             M,
                                             N,
                                             nnz,
                                             d_ind2.ptr,
                                             d_ind2.ptr,
                                             d_val2.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, other, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(other), rocsparse_status_success);

        // Data types that the value gather does not support.
        rocsparse_spmat_descr isrc = nullptr, idst = nullptr;
        ASSERT_EQ(make_csr(&isrc,
                           M,
                           N,
                           nnz,
                           d_ptr.ptr,
                           d_ind.ptr,
                           d_ival.ptr,
                           rocsparse_indextype_i32,
                           rocsparse_index_base_zero,
                           rocsparse_datatype_i32_r),
                  rocsparse_status_success);
        ASSERT_EQ(make_csr(&idst,
                           M,
                           N,
                           nnz,
                           d_ptr2.ptr,
                           d_ind2.ptr,
                           d_ival.ptr,
                           rocsparse_indextype_i32,
                           rocsparse_index_base_zero,
                           rocsparse_datatype_i32_r),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, isrc, idst, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(isrc), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(idst), rocsparse_status_success);

        // Batch counts must agree, and the batches of an overlapping layout are rejected.
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 2, 0, nnz), rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_success); // a single source is sorted into every batch
        ASSERT_EQ(rocsparse_csr_set_strided_batch(src, 2, M + 1, nnz), rocsparse_status_success);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 3, M + 1, nnz), rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_value);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 2, 1, nnz), rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_size);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 2, M + 1, nnz - 1),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size(descr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_invalid_size);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(src, 1, 0, 0), rocsparse_status_success);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 1, 0, 0), rocsparse_status_success);

        // The stages must run in order and the buffer must be large enough.
        EXPECT_EQ(rocsparse_spsort(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, src, dst, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr),
            rocsparse_status_success);
        ASSERT_EQ(buffer_size(descr, src, dst, rocsparse_spsort_stage_compute),
                  rocsparse_status_success);
        ASSERT_GT(size, size_t(1));
        void* buffer = nullptr;
        ASSERT_EQ(hipMalloc(&buffer, size), hipSuccess);
        EXPECT_EQ(rocsparse_spsort(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, 0, buffer, nullptr),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, src, dst, rocsparse_spsort_stage_compute, size, nullptr, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, src, dst, rocsparse_spsort_stage_compute, size - 1, buffer, nullptr),
            rocsparse_status_invalid_size);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, src, dst, rocsparse_spsort_stage_compute, size, buffer, nullptr),
            rocsparse_status_success);
        // The analysis cannot follow the compute stage.
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, src, dst, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr),
            rocsparse_status_invalid_value);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(hipFree(buffer), hipSuccess);

        // The sorted output of the successful run.
        EXPECT_EQ(to_host(d_ptr2), ptr);
        EXPECT_EQ(to_host(d_ind2), (std::vector<int32_t>{1, 3, 2, 0, 2}));
        EXPECT_EQ(to_host(d_val2), (std::vector<float>{2, 1, 3, 5, 4}));

        // Argument checks of rocsparse_spsort itself.
        EXPECT_EQ(
            rocsparse_spsort(
                nullptr, descr, src, dst, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
            rocsparse_status_invalid_handle);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, nullptr, src, dst, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, nullptr, dst, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, src, nullptr, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spsort(handle,
                                   descr,
                                   src,
                                   dst,
                                   static_cast<rocsparse_spsort_stage>(100),
                                   0,
                                   nullptr,
                                   nullptr),
                  rocsparse_status_invalid_value);

        EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(dst), rocsparse_status_success);
    }

    TEST_F(HandleTest, Cov70ConvSpsortCooDirectionRequired)
    {
        const int              nnz = 3;
        std::vector<int32_t>   row = {1, 0, 1}, col = {0, 1, 1};
        std::vector<float>     val = {1, 2, 3};
        device_vector<int32_t> d_row(row), d_col(col), d_row2(row), d_col2(col);
        device_vector<float>   d_val(val), d_val2(val);
        rocsparse_spmat_descr  src = nullptr, dst = nullptr;
        ASSERT_EQ(rocsparse_create_coo_descr(&src,
                                             2,
                                             2,
                                             nnz,
                                             d_row.ptr,
                                             d_col.ptr,
                                             d_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_descr(&dst,
                                             2,
                                             2,
                                             nnz,
                                             d_row2.ptr,
                                             d_col2.ptr,
                                             d_val2.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_spsort_descr descr = nullptr;
        ASSERT_EQ(rocsparse_spsort_descr_create(handle, &descr, nullptr), rocsparse_status_success);
        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;
        ASSERT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        size_t size = 0;
        // Only the COO format uses the direction, and it is mandatory there.
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, &size, nullptr),
                  rocsparse_status_invalid_value);

        // COO overlapping batches.
        const rocsparse_direction dir = rocsparse_direction_row;
        ASSERT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, &size, nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(src, 2, nnz - 1), rocsparse_status_success);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(dst, 2, nnz - 1), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, &size, nullptr),
                  rocsparse_status_invalid_size);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(dst, 2, nnz), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, &size, nullptr),
                  rocsparse_status_invalid_size);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(src, 2, nnz), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      handle, descr, src, dst, rocsparse_spsort_stage_compute, &size, nullptr),
                  rocsparse_status_success);

        EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(dst), rocsparse_status_success);
    }

    TEST_F(HandleTest, Cov70ConvSpsortUnsupportedFormats)
    {
        const int              nnz = 3;
        std::vector<int32_t>   ind = {1, 0, 1};
        std::vector<float>     val = {1, 2, 3};
        device_vector<int32_t> d_ind(ind), d_ind2(ind);
        device_vector<float>   d_val(val), d_val2(val);
        rocsparse_spmat_descr  src = nullptr, dst = nullptr;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&src,
                                                 2,
                                                 2,
                                                 nnz,
                                                 d_ind.ptr,
                                                 d_val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&dst,
                                                 2,
                                                 2,
                                                 nnz,
                                                 d_ind2.ptr,
                                                 d_val2.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_spsort_descr descr = nullptr;
        ASSERT_EQ(rocsparse_spsort_descr_create(handle, &descr, nullptr), rocsparse_status_success);
        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;
        ASSERT_EQ(rocsparse_spsort_set_input(
                      handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        size_t size = 0;
        EXPECT_EQ(rocsparse_spsort_buffer_size(
                      handle, descr, src, dst, rocsparse_spsort_stage_analysis, &size, nullptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(
            rocsparse_spsort(
                handle, descr, src, dst, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr),
            rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(dst), rocsparse_status_success);
    }

    // ------------------------------------------------------------------
    // Legacy COO sort
    // ------------------------------------------------------------------

    TEST_F(HandleTest, Cov70ConvCoosortLegacy)
    {
        const int  M = 23, N = 17;
        const auto pos = scrambled_positions(M, N);
        const int  nnz = static_cast<int>(pos.size());

        for(const bool by_row : {true, false})
        {
            for(const bool with_perm : {true, false})
            {
                std::vector<int> h_row(static_cast<size_t>(nnz)), h_col(static_cast<size_t>(nnz));
                for(int k = 0; k < nnz; ++k)
                {
                    h_row[static_cast<size_t>(k)] = pos[static_cast<size_t>(k)].first;
                    h_col[static_cast<size_t>(k)] = pos[static_cast<size_t>(k)].second;
                }
                std::vector<int> ref(static_cast<size_t>(nnz));
                std::iota(ref.begin(), ref.end(), 0);
                std::stable_sort(ref.begin(), ref.end(), [&](int a, int b) {
                    const auto ka = by_row ? std::make_pair(h_row[a], h_col[a])
                                           : std::make_pair(h_col[a], h_row[a]);
                    const auto kb = by_row ? std::make_pair(h_row[b], h_col[b])
                                           : std::make_pair(h_col[b], h_row[b]);
                    return ka < kb;
                });

                device_vector<int> d_row(h_row), d_col(h_col), d_perm(static_cast<size_t>(nnz));
                ASSERT_NE(d_row.ptr, nullptr);
                ASSERT_NE(d_perm.ptr, nullptr);
                size_t size = 0;
                ASSERT_EQ(rocsparse_coosort_buffer_size(handle, M, N, nnz, d_row, d_col, &size),
                          rocsparse_status_success);
                device_vector<char> d_buf(std::max(size, size_t(1)));
                ASSERT_NE(d_buf.ptr, nullptr);
                if(with_perm)
                {
                    ASSERT_EQ(rocsparse_create_identity_permutation(handle, nnz, d_perm),
                              rocsparse_status_success);
                }
                int* perm = with_perm ? d_perm.ptr : nullptr;
                if(by_row)
                {
                    ASSERT_EQ(
                        rocsparse_coosort_by_row(handle, M, N, nnz, d_row, d_col, perm, d_buf),
                        rocsparse_status_success);
                }
                else
                {
                    ASSERT_EQ(
                        rocsparse_coosort_by_column(handle, M, N, nnz, d_row, d_col, perm, d_buf),
                        rocsparse_status_success);
                }
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                std::vector<int> e_row(static_cast<size_t>(nnz)), e_col(static_cast<size_t>(nnz));
                for(int k = 0; k < nnz; ++k)
                {
                    e_row[static_cast<size_t>(k)]
                        = h_row[static_cast<size_t>(ref[static_cast<size_t>(k)])];
                    e_col[static_cast<size_t>(k)]
                        = h_col[static_cast<size_t>(ref[static_cast<size_t>(k)])];
                }
                EXPECT_EQ(to_host(d_row), e_row) << "by_row " << by_row << " perm " << with_perm;
                EXPECT_EQ(to_host(d_col), e_col) << "by_row " << by_row << " perm " << with_perm;
                if(with_perm)
                {
                    EXPECT_EQ(to_host(d_perm), ref) << "by_row " << by_row;
                }
            }
        }

        // Empty matrices return immediately.
        device_vector<int>  d_row(std::vector<int>{0}), d_col(std::vector<int>{0});
        device_vector<char> d_buf(size_t(64));
        EXPECT_EQ(rocsparse_coosort_by_row(handle, 0, N, 0, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_coosort_by_row(handle, M, 0, 0, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_coosort_by_column(handle, 0, N, 0, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_coosort_by_column(handle, M, N, 0, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_success);

        // Argument validation.
        EXPECT_EQ(rocsparse_coosort_by_row(nullptr, M, N, 1, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_coosort_by_row(handle, -1, N, 1, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_coosort_by_row(handle, M, N, 1, nullptr, d_col, nullptr, d_buf),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_coosort_by_row(handle, M, N, 1, d_row, nullptr, nullptr, d_buf),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_coosort_by_row(handle, M, N, 1, d_row, d_col, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_coosort_by_column(nullptr, M, N, 1, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_coosort_by_column(handle, M, -1, 1, d_row, d_col, nullptr, d_buf),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_coosort_by_column(handle, M, N, 1, nullptr, d_col, nullptr, d_buf),
                  rocsparse_status_invalid_pointer);
    }

    // ------------------------------------------------------------------
    // Legacy CSR / CSC sort
    // ------------------------------------------------------------------

    TEST_F(HandleTest, Cov70ConvCsrCscsortLegacy)
    {
        const int        M = 37, N = 13;
        const Compressed c   = make_compressed(M, N);
        const int        nnz = static_cast<int>(c.minor.size());

        for(const bool csc : {false, true})
        {
            for(const int base : {0, 1})
            {
                for(const bool with_perm : {true, false})
                {
                    MatDescr descr(base ? rocsparse_index_base_one : rocsparse_index_base_zero);
                    std::vector<int> h_off(c.offsets), h_ind(c.minor);
                    for(int& v : h_off)
                        v += base;
                    for(int& v : h_ind)
                        v += base;

                    std::vector<int> ref(static_cast<size_t>(nnz));
                    std::iota(ref.begin(), ref.end(), 0);
                    for(int r = 0; r < M; ++r)
                    {
                        std::stable_sort(ref.begin() + c.offsets[static_cast<size_t>(r)],
                                         ref.begin() + c.offsets[static_cast<size_t>(r) + 1],
                                         [&](int a, int b) {
                                             return c.minor[static_cast<size_t>(a)]
                                                    < c.minor[static_cast<size_t>(b)];
                                         });
                    }

                    device_vector<int> d_off(h_off), d_ind(h_ind), d_perm(static_cast<size_t>(nnz));
                    ASSERT_NE(d_off.ptr, nullptr);
                    ASSERT_NE(d_perm.ptr, nullptr);
                    // CSR has M rows and N columns, CSC has N rows and M columns.
                    const int rows = csc ? N : M;
                    const int cols = csc ? M : N;
                    size_t    size = 0;
                    if(csc)
                    {
                        ASSERT_EQ(rocsparse_cscsort_buffer_size(
                                      handle, rows, cols, nnz, d_off, d_ind, &size),
                                  rocsparse_status_success);
                    }
                    else
                    {
                        ASSERT_EQ(rocsparse_csrsort_buffer_size(
                                      handle, rows, cols, nnz, d_off, d_ind, &size),
                                  rocsparse_status_success);
                    }
                    device_vector<char> d_buf(std::max(size, size_t(1)));
                    ASSERT_NE(d_buf.ptr, nullptr);
                    if(with_perm)
                    {
                        ASSERT_EQ(rocsparse_create_identity_permutation(handle, nnz, d_perm),
                                  rocsparse_status_success);
                    }
                    int* perm = with_perm ? d_perm.ptr : nullptr;
                    if(csc)
                    {
                        ASSERT_EQ(rocsparse_cscsort(
                                      handle, rows, cols, nnz, descr, d_off, d_ind, perm, d_buf),
                                  rocsparse_status_success);
                    }
                    else
                    {
                        ASSERT_EQ(rocsparse_csrsort(
                                      handle, rows, cols, nnz, descr, d_off, d_ind, perm, d_buf),
                                  rocsparse_status_success);
                    }
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                    std::vector<int> e_ind(static_cast<size_t>(nnz));
                    for(int k = 0; k < nnz; ++k)
                        e_ind[static_cast<size_t>(k)]
                            = c.minor[static_cast<size_t>(ref[static_cast<size_t>(k)])] + base;
                    EXPECT_EQ(to_host(d_off), h_off);
                    EXPECT_EQ(to_host(d_ind), e_ind)
                        << "csc " << csc << " base " << base << " perm " << with_perm;
                    if(with_perm)
                    {
                        EXPECT_EQ(to_host(d_perm), ref);
                    }
                }
            }
        }
    }
} // namespace
