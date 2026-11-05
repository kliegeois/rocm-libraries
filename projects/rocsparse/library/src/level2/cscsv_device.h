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

#pragma once

#include "rocsparse_common.hpp"

// CSC solve of A * y = alpha * x with A triangular in CSC: once y_c is known, column c
// pushes a_rc * y_c into sum[r] and decrements rem[r]; unknown r is ready when rem[r] is zero.
namespace rocsparse
{
    template <uint32_t NUM_LANES, typename T>
    __device__ __forceinline__ T cscsv_group_broadcast(T v)
    {
        return __shfl(v, 0, NUM_LANES);
    }

    template <uint32_t NUM_LANES>
    __device__ __forceinline__ int cscsv_group_any(int v)
    {
        for(uint32_t o = NUM_LANES / 2; o > 0; o >>= 1)
        {
            v |= __shfl_xor(v, o, NUM_LANES);
        }
        return v;
    }

    // One group of NUM_LANES lanes per column: diagonal entry and dependency counts.
    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_analysis_kernel(J m,
                               const I* __restrict__ csc_col_ptr,
                               const J* __restrict__ csc_row_ind,
                               I* __restrict__ csc_diag_ind,
                               J* __restrict__ dependency_count,
                               J* __restrict__ zero_pivot,
                               rocsparse_index_base idx_base,
                               rocsparse_diag_type  diag_type,
                               rocsparse_fill_mode  fill_mode)
    {
        const uint32_t lid = hipThreadIdx_x & (NUM_LANES - 1);
        const uint32_t wid = hipThreadIdx_x / NUM_LANES;
        const J        col = hipBlockIdx_x * (BLOCKSIZE / NUM_LANES) + wid;

        if(col < m)
        {
            const I col_begin = csc_col_ptr[col] - idx_base;
            const I col_end   = csc_col_ptr[col + 1] - idx_base;

            int has_diag = 0;
            for(I k = col_begin + lid; k < col_end; k += NUM_LANES)
            {
                const J row = csc_row_ind[k] - idx_base;
                if(row == col)
                {
                    csc_diag_ind[col] = k;
                    has_diag          = 1;
                }
                else if((fill_mode == rocsparse_fill_mode_lower) ? (row > col) : (row < col))
                {
                    rocsparse::atomic_add(&dependency_count[row], static_cast<J>(1));
                }
            }

            has_diag = rocsparse::cscsv_group_any<NUM_LANES>(has_diag);
            if(lid == 0 && !has_diag && diag_type == rocsparse_diag_type_non_unit)
            {
                rocsparse::atomic_min(zero_pivot, col + idx_base);
            }
        }
    }

    // Longest column without atomics, as in doti: each block reduces a grid-stride range of
    // columns into workspace[hipBlockIdx_x], then a single block reduces the partial results.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_max_nnz_part1(int64_t m,
                             const I* __restrict__ csc_col_ptr,
                             I* __restrict__ workspace)
    {
        __shared__ I shared[BLOCKSIZE];

        I nnz = 0;
        for(int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; col < m;
            col += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            nnz = rocsparse::max(nnz, static_cast<I>(csc_col_ptr[col + 1] - csc_col_ptr[col]));
        }
        shared[hipThreadIdx_x] = nnz;
        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(hipThreadIdx_x, shared);

        if(hipThreadIdx_x == 0)
        {
            workspace[hipBlockIdx_x] = shared[0];
        }
    }

    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_max_nnz_part2(I* __restrict__ workspace)
    {
        __shared__ I shared[BLOCKSIZE];

        shared[hipThreadIdx_x] = workspace[hipThreadIdx_x];
        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(hipThreadIdx_x, shared);

        if(hipThreadIdx_x == 0)
        {
            workspace[0] = shared[0];
        }
    }

    // Resets the number of unresolved dependencies of every unknown, for each batch instance,
    // from the counts of the analysis. The solve decrements these counters.
    template <uint32_t BLOCKSIZE, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_init_kernel(J       m,
                           int64_t batch_count,
                           const J* __restrict__ dependency_count,
                           J* __restrict__ rem)
    {
        for(int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            i < m * batch_count;
            i += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            rem[i] = dependency_count[i % m];
        }
    }

    // The entries of each column are split into tasks of at most NUM_LANES * ITEMS entries, so
    // that a long column is pushed by several groups at once instead of one.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_count_tasks_kernel(int64_t m,
                                  const I* __restrict__ csc_col_ptr,
                                  int64_t chunk,
                                  I* __restrict__ task_count)
    {
        const int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(col < m)
        {
            const int64_t len = csc_col_ptr[col + 1] - csc_col_ptr[col];
            task_count[col]   = static_cast<I>(rocsparse::max(static_cast<int64_t>(1),
                                                            (len + chunk - 1) / chunk));
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_expand_tasks_kernel(int64_t m,
                                   const I* __restrict__ csc_col_ptr,
                                   const I* __restrict__ task_ptr,
                                   int64_t chunk,
                                   J* __restrict__ task_col,
                                   I* __restrict__ task_begin,
                                   rocsparse_index_base idx_base)
    {
        const int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(col < m)
        {
            const I begin = csc_col_ptr[col] - idx_base;
            for(I t = task_ptr[col]; t < task_ptr[col + 1]; ++t)
            {
                task_col[t]   = static_cast<J>(col);
                task_begin[t] = begin + static_cast<I>((t - task_ptr[col]) * chunk);
            }
        }
    }

    // Level of each unknown: one plus the highest level of the unknowns it depends on. Same
    // traversal as the solve, in elimination order, with integer atomics only.
    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, uint32_t ITEMS, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_level_kernel(int64_t num_tasks,
                            const I* __restrict__ csc_col_ptr,
                            const J* __restrict__ csc_row_ind,
                            const J* __restrict__ task_col,
                            const I* __restrict__ task_begin,
                            J* __restrict__ rem,
                            int32_t* __restrict__ level,
                            int32_t* __restrict__ ticket,
                            rocsparse_index_base idx_base,
                            rocsparse_fill_mode  fill_mode)
    {
        __shared__ int32_t block_id;
        if(hipThreadIdx_x == 0)
        {
            block_id = rocsparse::atomic_add(ticket, 1);
        }
        __syncthreads();

        const uint32_t lid = hipThreadIdx_x & (NUM_LANES - 1);
        const uint32_t wid = hipThreadIdx_x / NUM_LANES;
        const int64_t  w   = static_cast<int64_t>(block_id) * (BLOCKSIZE / NUM_LANES) + wid;
        if(w >= num_tasks)
        {
            return;
        }

        const bool    lower   = (fill_mode == rocsparse_fill_mode_lower);
        const int64_t t       = lower ? w : num_tasks - 1 - w;
        const J       col     = task_col[t];
        const I       kbeg    = task_begin[t];
        const I       col_end = csc_col_ptr[col + 1] - idx_base;

        J    rows[ITEMS];
        bool deps[ITEMS];
        for(uint32_t i = 0; i < ITEMS; ++i)
        {
            const I k = kbeg + lid + i * NUM_LANES;
            rows[i]   = (k < col_end) ? csc_row_ind[k] - idx_base : col;
            deps[i]   = lower ? (rows[i] > col) : (rows[i] < col);
        }

        bool done = false;
        while(!done)
        {
            J r = (lid == 0)
                      ? rocsparse::atomic_load(&rem[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                      : static_cast<J>(0);
            r   = rocsparse::cscsv_group_broadcast<NUM_LANES>(r);
            if(r == 0)
            {
                rocsparse::acquire_done_rows();
                const int32_t lvl
                    = rocsparse::atomic_load(&level[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                      + 1;
                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        rocsparse::atomic_max(&level[rows[i]], lvl);
                    }
                }
                rocsparse::release_done_row_data();
                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        __scoped_atomic_fetch_sub(&rem[rows[i]],
                                                  static_cast<J>(1),
                                                  __ATOMIC_RELAXED,
                                                  __MEMORY_SCOPE_DEVICE);
                    }
                }
                done = true;
            }
            else
            {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_task_keys_kernel(int64_t num_tasks,
                                const J* __restrict__ task_col,
                                const int32_t* __restrict__ level,
                                int32_t* __restrict__ keys,
                                I* __restrict__ perm,
                                int32_t natural_order,
                                rocsparse_fill_mode fill_mode)
    {
        const int64_t t = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(t < num_tasks)
        {
            keys[t] = natural_order ? static_cast<int32_t>((fill_mode == rocsparse_fill_mode_lower)
                                                               ? t
                                                               : num_tasks - 1 - t)
                                    : level[task_col[t]];
            perm[t] = static_cast<I>(t);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_task_gather_kernel(int64_t num_tasks,
                                  const I* __restrict__ perm,
                                  const J* __restrict__ task_col,
                                  const I* __restrict__ task_begin,
                                  J* __restrict__ sorted_task_col,
                                  I* __restrict__ sorted_task_begin)
    {
        const int64_t t = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(t < num_tasks)
        {
            const I p            = perm[t];
            sorted_task_col[t]   = task_col[p];
            sorted_task_begin[t] = task_begin[p];
        }
    }

    // One group of NUM_LANES lanes per task, tasks sorted by level. Every task of a column waits
    // for the column to be ready and computes its unknown from the same final partial sum; the
    // first task of the column stores it. Blocks take an atomic ticket instead of hipBlockIdx_x,
    // so a group only waits on groups that are already running.
    template <uint32_t BLOCKSIZE,
              uint32_t NUM_LANES,
              uint32_t ITEMS,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_DEVICE_ILF void cscsv_solve_device(int64_t num_tasks,
                                                 T       alpha,
                                                 const I* __restrict__ csc_col_ptr,
                                                 const J* __restrict__ csc_row_ind,
                                                 const T* __restrict__ csc_val,
                                                 const I* __restrict__ csc_diag_ind,
                                                 const J* __restrict__ task_col,
                                                 const I* __restrict__ task_begin,
                                                 const T* __restrict__ x,
                                                 int64_t x_inc,
                                                 T*      y,
                                                 int64_t y_inc,
                                                 J* __restrict__ rem,
                                                 T* __restrict__ sum,
                                                 int32_t* __restrict__ ticket,
                                                 J*                   zero_pivot,
                                                 rocsparse_index_base idx_base,
                                                 rocsparse_fill_mode  fill_mode,
                                                 rocsparse_diag_type  diag_type)
    {
        __shared__ int32_t block_id;
        if(hipThreadIdx_x == 0)
        {
            block_id = rocsparse::atomic_add(ticket, 1);
        }
        __syncthreads();

        const uint32_t lid = hipThreadIdx_x & (NUM_LANES - 1);
        const uint32_t wid = hipThreadIdx_x / NUM_LANES;
        const int64_t  t   = static_cast<int64_t>(block_id) * (BLOCKSIZE / NUM_LANES) + wid;

        if(t >= num_tasks)
        {
            return;
        }

        const bool lower     = (fill_mode == rocsparse_fill_mode_lower);
        const J    col       = task_col[t];
        const I    kbeg      = task_begin[t];
        const I    col_begin = csc_col_ptr[col] - idx_base;
        const I    col_end   = csc_col_ptr[col + 1] - idx_base;
        const bool owner     = (kbeg == col_begin);

        // Off the critical path: everything that does not depend on other unknowns.
        T inv_diag = static_cast<T>(1);
        if(diag_type == rocsparse_diag_type_non_unit)
        {
            const I d = csc_diag_ind[col];
            if(d >= 0)
            {
                T diag = csc_val[d];
                if(diag == static_cast<T>(0))
                {
                    if(owner && lid == 0)
                    {
                        rocsparse::atomic_min(zero_pivot, col + idx_base);
                    }
                    diag = static_cast<T>(1);
                }
                inv_diag = static_cast<T>(1) / diag;
            }
        }
        const T ax = alpha * x[col * x_inc];

        J    rows[ITEMS];
        T    vals[ITEMS];
        bool deps[ITEMS];
        for(uint32_t i = 0; i < ITEMS; ++i)
        {
            const I k = kbeg + lid + i * NUM_LANES;
            rows[i]   = (k < col_end) ? csc_row_ind[k] - idx_base : col;
            vals[i]   = (k < col_end) ? csc_val[k] : static_cast<T>(0);
            deps[i]   = lower ? (rows[i] > col) : (rows[i] < col);
        }

        bool done = false;
        while(!done)
        {
            J r = (lid == 0)
                      ? rocsparse::atomic_load(&rem[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                      : static_cast<J>(0);
            r   = rocsparse::cscsv_group_broadcast<NUM_LANES>(r);

            if(r == 0)
            {
                rocsparse::acquire_done_rows();

                const T s
                    = rocsparse::atomic_load(&sum[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                T yc = ax - s;
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    yc = yc * inv_diag;
                }

                if(owner && lid == 0)
                {
                    y[col * y_inc] = yc;
                }

                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        rocsparse::atomic_add(&sum[rows[i]], vals[i] * yc);
                    }
                }

                // One release for all the additions of this lane.
                rocsparse::release_done_row_data();

                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        __scoped_atomic_fetch_sub(&rem[rows[i]],
                                                  static_cast<J>(1),
                                                  __ATOMIC_RELAXED,
                                                  __MEMORY_SCOPE_DEVICE);
                    }
                }
                done = true;
            }
            else
            {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
}
