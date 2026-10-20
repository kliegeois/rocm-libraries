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

// Helpers shared by the cov80 precond tests: the generic descriptor API
// (spilu0 / spic0 / spildlt0) driven on CSR / BSR matrices of every value and
// index type, checked against the host references of the cov70 common header
// (diagonally dominant matrices, finite references, tolerances that scale with
// the row length). The wavefront size of the handle is never overridden.

#pragma once

#include "unit_test_device_cov70_precond_common.hpp"

#include <limits>
#include <memory>
#include <string>

namespace cov80
{
    using namespace cov70;

    // Restores the size of the internal buffer of the handle on scope exit.
    struct ScopedBufferSize
    {
        rocsparse_handle handle;
        size_t           saved;

        ScopedBufferSize(rocsparse_handle h, size_t size)
            : handle(h)
            , saved(h->buffer_size)
        {
            handle->buffer_size = size;
        }

        ~ScopedBufferSize()
        {
            handle->buffer_size = saved;
        }

        ScopedBufferSize(const ScopedBufferSize&) = delete;
        ScopedBufferSize& operator=(const ScopedBufferSize&) = delete;
    };

    enum class Kind
    {
        ilu,
        ic,
        ildlt
    };
    enum class In
    {
        alg,
        policy,
        dtype,
        boost_enable,
        boost_tol,
        boost_val,
        sing_tol
    };
    enum class Stage
    {
        analysis,
        compute
    };

    template <Kind K>
    struct api;

#define COV80_API(KIND, NAME)                                                                   \
    template <>                                                                                 \
    struct api<KIND>                                                                            \
    {                                                                                           \
        using descr_t = rocsparse_##NAME##_descr;                                               \
        static rocsparse_status create(rocsparse_handle handle, descr_t* descr)                 \
        {                                                                                       \
            return rocsparse_##NAME##_descr_create(handle, descr, nullptr);                     \
        }                                                                                       \
        static rocsparse_status destroy(rocsparse_handle handle, descr_t descr)                 \
        {                                                                                       \
            return rocsparse_##NAME##_descr_destroy(handle, descr, nullptr);                    \
        }                                                                                       \
        static rocsparse_##NAME##_stage stage(Stage s)                                          \
        {                                                                                       \
            return (s == Stage::analysis) ? rocsparse_##NAME##_stage_analysis                   \
                                          : rocsparse_##NAME##_stage_compute;                   \
        }                                                                                       \
        static rocsparse_##NAME##_input input(In in)                                            \
        {                                                                                       \
            switch(in)                                                                          \
            {                                                                                   \
            case In::alg:                                                                       \
                return rocsparse_##NAME##_input_alg;                                            \
            case In::policy:                                                                    \
                return rocsparse_##NAME##_input_analysis_policy;                                \
            case In::dtype:                                                                     \
                return rocsparse_##NAME##_input_compute_datatype;                               \
            case In::boost_enable:                                                              \
                return rocsparse_##NAME##_input_boost_enable;                                   \
            case In::boost_tol:                                                                 \
                return rocsparse_##NAME##_input_boost_tolerance;                                \
            case In::boost_val:                                                                 \
                return rocsparse_##NAME##_input_boost_value;                                    \
            case In::sing_tol:                                                                  \
                return rocsparse_##NAME##_input_singularity_tolerance;                          \
            }                                                                                   \
            return rocsparse_##NAME##_input_alg;                                                \
        }                                                                                       \
        static rocsparse_status set(rocsparse_handle         handle,                            \
                                    descr_t                  descr,                             \
                                    rocsparse_##NAME##_input in,                                \
                                    const void*              data,                              \
                                    size_t                   size)                              \
        {                                                                                       \
            return rocsparse_##NAME##_set_input(handle, descr, in, data, size, nullptr);        \
        }                                                                                       \
        static rocsparse_status buffer_size(rocsparse_handle            handle,                 \
                                            descr_t                     descr,                  \
                                            rocsparse_const_spmat_descr A,                      \
                                            rocsparse_const_spmat_descr P,                      \
                                            rocsparse_##NAME##_stage    stage,                  \
                                            size_t*                     size)                   \
        {                                                                                       \
            return rocsparse_##NAME##_buffer_size(handle, descr, A, P, stage, size, nullptr);   \
        }                                                                                       \
        static rocsparse_status run(rocsparse_handle            handle,                         \
                                    descr_t                     descr,                          \
                                    rocsparse_const_spmat_descr A,                              \
                                    rocsparse_spmat_descr       P,                              \
                                    rocsparse_##NAME##_stage    stage,                          \
                                    size_t                      size,                           \
                                    void*                       buffer)                         \
        {                                                                                       \
            return rocsparse_##NAME(handle, descr, A, P, stage, size, buffer, nullptr);         \
        }                                                                                       \
        static rocsparse_status                                                                 \
            get(rocsparse_handle handle, descr_t descr, bool position, void* data, size_t size) \
        {                                                                                       \
            return rocsparse_##NAME##_get_output(                                               \
                handle,                                                                         \
                descr,                                                                          \
                position ? rocsparse_##NAME##_output_singularity_position                       \
                         : rocsparse_##NAME##_output_singularity,                               \
                data,                                                                           \
                size,                                                                           \
                nullptr);                                                                       \
        }                                                                                       \
    }

    COV80_API(Kind::ilu, spilu0);
    COV80_API(Kind::ic, spic0);
    COV80_API(Kind::ildlt, spildlt0);
#undef COV80_API

    template <Kind K>
    rocsparse_status set_value(rocsparse_handle         handle,
                               typename api<K>::descr_t descr,
                               In                       in,
                               const void*              data,
                               size_t                   size)
    {
        return api<K>::set(handle, descr, api<K>::input(in), data, size);
    }

    struct run_options
    {
        rocsparse_analysis_policy policy       = rocsparse_analysis_policy_reuse;
        bool                      boost        = false;
        double                    boost_tol    = 0.0;
        double                    boost_val    = 1.0;
        bool                      set_sing_tol = false;
        double                    sing_tol     = 0.0;
        bool                      tol_after    = false; // set the tolerance after the analysis
        size_t device_memory = 0; // > 0: size of the device memory of the handle (bytes)
    };

    template <typename T>
    struct run_result
    {
        rocsparse_status                   analysis_status = rocsparse_status_internal_error;
        rocsparse_status                   compute_status  = rocsparse_status_internal_error;
        std::vector<T>                     values;
        std::vector<rocsparse_singularity> singularity;
        std::vector<int64_t>               position;
    };

    // Destroys the descriptor when a failed ASSERT leaves the function early.
    template <Kind K>
    struct descr_guard
    {
        rocsparse_handle         handle;
        typename api<K>::descr_t descr = nullptr;

        explicit descr_guard(rocsparse_handle h)
            : handle(h)
        {
        }

        ~descr_guard()
        {
            if(descr)
                (void)api<K>::destroy(handle, descr);
        }

        descr_guard(const descr_guard&) = delete;
        descr_guard& operator=(const descr_guard&) = delete;
    };

    // Analysis then compute through the generic API on matrix `A`. `val` is the
    // device value array (nvals entries, all batches) that holds the result.
    template <Kind K, typename T>
    void run_factorization(rocsparse_handle        handle,
                           rocsparse_spmat_descr   A,
                           const device_vector<T>& val,
                           int                     batch,
                           const run_options&      opt,
                           run_result<T>&          out)
    {
        descr_guard<K> guard(handle);
        auto&          descr = guard.descr;
        // Shrinks the size of the internal buffer of the handle (the buffer itself is not
        // touched, so the library only ever sees a smaller buffer than the real one).
        std::unique_ptr<ScopedBufferSize> shrink;
        if(opt.device_memory > 0)
            shrink = std::make_unique<ScopedBufferSize>(handle, opt.device_memory);
        ASSERT_EQ(api<K>::create(handle, &descr), rocsparse_status_success);

        const rocsparse_datatype dtype = dt_of<T>();
        ASSERT_EQ((set_value<K>(handle, descr, In::dtype, &dtype, sizeof(dtype))),
                  rocsparse_status_success);
        const int32_t alg = 0;
        ASSERT_EQ((set_value<K>(handle, descr, In::alg, &alg, sizeof(int32_t))),
                  rocsparse_status_success);
        ASSERT_EQ((set_value<K>(handle, descr, In::policy, &opt.policy, sizeof(opt.policy))),
                  rocsparse_status_success);

        const int32_t boost_enable  = opt.boost ? 1 : 0;
        const double  boost_tol     = opt.boost_tol;
        const T       boost_val     = make_value<T>(opt.boost_val, 0.0);
        const void*   boost_val_ptr = &boost_val;
        const double  sing_tol      = opt.sing_tol;
        const void*   sing_tol_ptr  = &sing_tol;
        if(opt.boost)
        {
            ASSERT_EQ(
                (set_value<K>(handle, descr, In::boost_enable, &boost_enable, sizeof(int32_t))),
                rocsparse_status_success);
            ASSERT_EQ((set_value<K>(handle, descr, In::boost_tol, &boost_tol, sizeof(double))),
                      rocsparse_status_success);
            ASSERT_EQ((set_value<K>(handle, descr, In::boost_val, boost_val_ptr, sizeof(void*))),
                      rocsparse_status_success);
        }
        if(opt.set_sing_tol && !opt.tol_after)
        {
            ASSERT_EQ((set_value<K>(handle, descr, In::sing_tol, sing_tol_ptr, sizeof(void*))),
                      rocsparse_status_success);
        }

        size_t analysis_size = 0;
        ASSERT_EQ(api<K>::buffer_size(
                      handle, descr, A, A, api<K>::stage(Stage::analysis), &analysis_size),
                  rocsparse_status_success);
        device_vector<char> analysis_buffer{analysis_size ? analysis_size : size_t{1}};
        ASSERT_TRUE(analysis_buffer.ptr);
        out.analysis_status = api<K>::run(handle,
                                          descr,
                                          A,
                                          A,
                                          api<K>::stage(Stage::analysis),
                                          analysis_size,
                                          analysis_size ? analysis_buffer.ptr : nullptr);
        ASSERT_EQ(out.analysis_status, rocsparse_status_success);

        if(opt.set_sing_tol && opt.tol_after)
        {
            ASSERT_EQ((set_value<K>(handle, descr, In::sing_tol, sing_tol_ptr, sizeof(void*))),
                      rocsparse_status_success);
        }

        size_t compute_size = 0;
        ASSERT_EQ(
            api<K>::buffer_size(handle, descr, A, A, api<K>::stage(Stage::compute), &compute_size),
            rocsparse_status_success);
        device_vector<char> compute_buffer{compute_size ? compute_size : size_t{1}};
        ASSERT_TRUE(compute_buffer.ptr);

        out.compute_status = api<K>::run(handle,
                                         descr,
                                         A,
                                         A,
                                         api<K>::stage(Stage::compute),
                                         compute_size,
                                         compute_size ? compute_buffer.ptr : nullptr);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        out.values = to_host(val);

        // API quirk: data_size_in_bytes must be the size of ONE element (any other
        // value is rejected), but the library writes one element per batch entry. A
        // caller that follows the documented size for a batched matrix therefore
        // needs a buffer of `batch` elements. The buffers below hold one extra
        // sentinel entry to prove that exactly `batch` entries are written.
        const auto    sing_sentinel = static_cast<rocsparse_singularity>(-123);
        const int64_t pos_sentinel  = -987654321;
        out.singularity.assign(static_cast<size_t>(batch) + 1, rocsparse_singularity_numeric_near);
        out.singularity.back() = sing_sentinel;
        out.position.assign(static_cast<size_t>(batch) + 1, -2);
        out.position.back() = pos_sentinel;
        ASSERT_EQ(api<K>::get(
                      handle, descr, false, out.singularity.data(), sizeof(rocsparse_singularity)),
                  rocsparse_status_success);
        ASSERT_EQ(api<K>::get(handle, descr, true, out.position.data(), sizeof(int64_t)),
                  rocsparse_status_success);
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        EXPECT_EQ(out.singularity.back(), sing_sentinel);
        EXPECT_EQ(out.position.back(), pos_sentinel);
        out.singularity.pop_back();
        out.position.pop_back();

        EXPECT_EQ(api<K>::destroy(handle, descr), rocsparse_status_success);
        descr = nullptr;
    }

    // CSR matrix on the device, optionally batched (each batch entry scales the
    // base values by 1 + 0.25 * b).
    template <typename T, typename I, typename J>
    struct dev_csr
    {
        csr_matrix<T>              base;
        std::vector<csr_matrix<T>> batches;
        device_vector<I>           row;
        device_vector<J>           col;
        device_vector<T>           val;
        rocsparse_spmat_descr      A = nullptr;

        dev_csr()               = default;
        dev_csr(const dev_csr&) = delete;
        dev_csr& operator=(const dev_csr&) = delete;

        ~dev_csr()
        {
            if(A)
                (void)rocsparse_destroy_spmat_descr(A);
        }

        // `m` is zero based; the arrays handed to the library are shifted by one
        // when `index_base` is one.
        void init(const csr_matrix<T>& m,
                  int                  batch,
                  bool                 shared_row_ptr,
                  rocsparse_index_base index_base = rocsparse_index_base_zero)
        {
            const I shift = (index_base == rocsparse_index_base_one) ? 1 : 0;
            base          = m;
            batches.clear();
            std::vector<I> hrow;
            std::vector<J> hcol;
            std::vector<T> hval;
            for(int b = 0; b < batch; ++b)
            {
                csr_matrix<T> mb = m;
                const double  s  = 1.0 + 0.25 * b;
                for(auto& v : mb.val)
                    v = make_value<T>(re(v) * s, im(v) * s);
                batches.push_back(mb);
                if(b == 0 || !shared_row_ptr)
                {
                    for(const auto p : m.ptr)
                        hrow.push_back(static_cast<I>(p) + shift);
                }
                for(const auto c : m.ind)
                    hcol.push_back(static_cast<J>(c) + static_cast<J>(shift));
                hval.insert(hval.end(), mb.val.begin(), mb.val.end());
            }
            upload(row, hrow);
            upload(col, hcol);
            upload(val, hval);
            ASSERT_TRUE(row.ptr && col.ptr && val.ptr);
            ASSERT_EQ(rocsparse_create_csr_descr(&A,
                                                 m.n,
                                                 m.n,
                                                 m.nnz(),
                                                 row.ptr,
                                                 col.ptr,
                                                 val.ptr,
                                                 it_of<I>(),
                                                 it_of<J>(),
                                                 index_base,
                                                 dt_of<T>()),
                      rocsparse_status_success);
            if(batch > 1)
            {
                ASSERT_EQ(rocsparse_csr_set_strided_batch(
                              A, batch, shared_row_ptr ? 0 : m.n + 1, m.nnz()),
                          rocsparse_status_success);
            }
        }
    };

    template <typename T>
    struct dev_bsr
    {
        bsr_matrix<T>          base;
        device_vector<int32_t> row;
        device_vector<int32_t> col;
        device_vector<T>       val;
        rocsparse_spmat_descr  A = nullptr;

        dev_bsr()               = default;
        dev_bsr(const dev_bsr&) = delete;
        dev_bsr& operator=(const dev_bsr&) = delete;

        ~dev_bsr()
        {
            if(A)
                (void)rocsparse_destroy_spmat_descr(A);
        }

        void init(const bsr_matrix<T>& m,
                  rocsparse_index_base index_base = rocsparse_index_base_zero)
        {
            const int32_t shift       = (index_base == rocsparse_index_base_one) ? 1 : 0;
            base                      = m;
            std::vector<int32_t> hrow = m.ptr;
            std::vector<int32_t> hcol = m.ind;
            for(auto& p : hrow)
                p += shift;
            for(auto& c : hcol)
                c += shift;
            upload(row, hrow);
            upload(col, hcol);
            upload(val, m.val);
            ASSERT_TRUE(row.ptr && col.ptr && val.ptr);
            ASSERT_EQ(rocsparse_create_bsr_descr(&A,
                                                 m.mb,
                                                 m.mb,
                                                 m.nnzb(),
                                                 m.dir,
                                                 m.block_dim,
                                                 row.ptr,
                                                 col.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 index_base,
                                                 dt_of<T>()),
                      rocsparse_status_success);
        }
    };

    template <typename F>
    void for_all_types(F&& f)
    {
        f(float{});
        f(double{});
        f(rocsparse_float_complex{});
        f(rocsparse_double_complex{});
    }

    // Expected factor of a scalar matrix for each kind. For IC(0) only the
    // lower triangle is a factor; the entries above it are not compared.
    template <Kind K, typename T>
    std::vector<T> reference(const csr_matrix<T>& a, const run_options& opt)
    {
        if constexpr(K == Kind::ilu)
            return host_ilu0(a, opt.boost ? opt.boost_tol : -1.0, opt.boost_val);
        else if constexpr(K == Kind::ic)
            return keep_lower_only(a, host_ic0(a));
        else
            return host_ildlt0(a);
    }

    // Compares one factorized scalar matrix with the host reference and checks the residual.
    template <Kind K, typename T>
    void check_factor(const csr_matrix<T>&  a,
                      const std::vector<T>& got_in,
                      const run_options&    opt,
                      const std::string&    what)
    {
        SCOPED_TRACE(what);
        std::vector<T> got = got_in;
        if constexpr(K == Kind::ic)
            got = keep_lower_only(a, got_in);
        expect_close(got, reference<K>(a, opt), tolerance<T>(max_row_length(a)));
        if(!opt.boost)
        {
            double res = 0.0;
            if constexpr(K == Kind::ilu)
                res = ilu_pattern_residual(a, got);
            else if constexpr(K == Kind::ic)
                res = ic_pattern_residual(a, got);
            else
                res = ildlt_pattern_residual(a, got);
            expect_small_residual(res, a);
        }
    }

    template <Kind K, typename T, typename I = int32_t, typename J = int32_t>
    void check_csr_matrix(rocsparse_handle     handle,
                          const csr_matrix<T>& m,
                          const run_options&   opt,
                          const std::string&   what,
                          rocsparse_index_base base   = rocsparse_index_base_zero,
                          run_result<T>*       result = nullptr)
    {
        SCOPED_TRACE(what);
        dev_csr<T, I, J> d;
        d.init(m, 1, true, base);
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<K, T>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        check_factor<K>(d.base, out.values, opt, what);
        EXPECT_EQ(out.singularity[0], rocsparse_singularity_none);
        EXPECT_EQ(out.position[0], -1);
        if(result != nullptr)
            *result = out;
    }
} // namespace cov80
