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

// Generic incomplete-factorization API (rocsparse_spilu0 / spic0 / spildlt0):
// descriptor inputs and outputs, argument validation, stage ordering, CSR and
// BSR matrices, batched CSR, numeric boost, singularity reporting, and the
// kernel regimes selected by the row length (hash vs binary search).
//
// Every factorization is compared with a host reference and with the
// residual of the factorization on the sparsity pattern.

#include "unit_test_device_cov70_precond_common.hpp"

#include <limits>
#include <string>

using namespace cov70;

namespace
{
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

#define COV70_API(KIND, NAME)                                                                   \
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

    COV70_API(Kind::ilu, spilu0);
    COV70_API(Kind::ic, spic0);
    COV70_API(Kind::ildlt, spildlt0);
#undef COV70_API

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
        int                       wavefront    = 0;
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
        typename api<K>::descr_t descr = nullptr;
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

        if(opt.wavefront > 0)
        {
            ScopedWavefront guard(handle, opt.wavefront);
            out.compute_status = api<K>::run(handle,
                                             descr,
                                             A,
                                             A,
                                             api<K>::stage(Stage::compute),
                                             compute_size,
                                             compute_size ? compute_buffer.ptr : nullptr);
        }
        else
        {
            out.compute_status = api<K>::run(handle,
                                             descr,
                                             A,
                                             A,
                                             api<K>::stage(Stage::compute),
                                             compute_size,
                                             compute_size ? compute_buffer.ptr : nullptr);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        out.values = to_host(val);

        // The singularity output is written once per batch entry even though
        // the size argument describes a single element.
        out.singularity.assign(static_cast<size_t>(batch), rocsparse_singularity_numeric_near);
        out.position.assign(static_cast<size_t>(batch), -2);
        ASSERT_EQ(api<K>::get(
                      handle, descr, false, out.singularity.data(), sizeof(rocsparse_singularity)),
                  rocsparse_status_success);
        ASSERT_EQ(api<K>::get(handle, descr, true, out.position.data(), sizeof(int64_t)),
                  rocsparse_status_success);
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

        EXPECT_EQ(api<K>::destroy(handle, descr), rocsparse_status_success);
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

        void init(const csr_matrix<T>& m, int batch, bool shared_row_ptr)
        {
            base = m;
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
                    hrow.insert(hrow.end(), m.ptr.begin(), m.ptr.end());
                hcol.insert(hcol.end(), m.ind.begin(), m.ind.end());
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
                                                 rocsparse_index_base_zero,
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

        void init(const bsr_matrix<T>& m)
        {
            base = m;
            upload(row, m.ptr);
            upload(col, m.ind);
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
                                                 rocsparse_index_base_zero,
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
        expect_close(got, reference<K>(a, opt), 4.0);
        if(!opt.boost)
        {
            double res = 0.0;
            if constexpr(K == Kind::ilu)
                res = ilu_pattern_residual(a, got);
            else if constexpr(K == Kind::ic)
                res = ic_pattern_residual(a, got);
            else
                res = ildlt_pattern_residual(a, got);
            EXPECT_LT(res, 50.0 * tolerance<T>());
        }
    }

    template <Kind K, typename T, typename I = int32_t, typename J = int32_t>
    void check_csr_case(rocsparse_handle   handle,
                        int32_t            grid,
                        int                batch,
                        bool               shared_row_ptr,
                        const run_options& opt)
    {
        const bool       skew = (K == Kind::ilu);
        dev_csr<T, I, J> d;
        d.init(make_grid_matrix<T>(grid, skew), batch, shared_row_ptr);
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<K, T>(handle, d.A, d.val, batch, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        const size_t nnz = static_cast<size_t>(d.base.nnz());
        ASSERT_EQ(out.values.size(), nnz * static_cast<size_t>(batch));
        for(int b = 0; b < batch; ++b)
        {
            const std::vector<T> got(out.values.begin() + b * nnz,
                                     out.values.begin() + (b + 1) * nnz);
            check_factor<K>(d.batches[b], got, opt, "batch " + std::to_string(b));
            EXPECT_EQ(out.singularity[b], rocsparse_singularity_none) << "batch " << b;
            EXPECT_EQ(out.position[b], -1) << "batch " << b;
        }
    }

    template <Kind K, typename T>
    void check_bsr_case(rocsparse_handle    handle,
                        int32_t             mb,
                        int32_t             block_dim,
                        rocsparse_direction dir,
                        const run_options&  opt,
                        bool                verify = true)
    {
        const bool skew = (K == Kind::ilu);
        dev_bsr<T> d;
        d.init(make_block_matrix<T>(mb, block_dim, dir, skew));
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<K, T>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        ASSERT_EQ(out.values.size(), d.base.val.size());
        if(!verify)
        {
            // Launch-only check, see the wave64 tests: only the status is asserted,
            // the output of a 64-lane kernel on 32-lane hardware is not meaningful.
            return;
        }
        const csr_matrix<T> scalar = expand_bsr(d.base);
        const csr_matrix<T> got    = expand_bsr([&] {
            bsr_matrix<T> m = d.base;
            m.val           = out.values;
            return m;
        }());
        check_factor<K>(scalar,
                        got.val,
                        opt,
                        "mb=" + std::to_string(mb) + " block_dim=" + std::to_string(block_dim));
        EXPECT_EQ(out.singularity[0], rocsparse_singularity_none);
        EXPECT_EQ(out.position[0], -1);
    }
}

class Cov70PrecondGeneric : public HandleTest
{
};

// ---------------------------------------------------------------------------
// CSR
// ---------------------------------------------------------------------------
TEST_F(Cov70PrecondGeneric, spilu0_csr_types_and_policies)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        run_options opt;
        check_csr_case<Kind::ilu, T>(handle, 6, 1, true, opt);
        opt.policy = rocsparse_analysis_policy_force;
        check_csr_case<Kind::ilu, T>(handle, 5, 1, true, opt);
    });
}

TEST_F(Cov70PrecondGeneric, spic0_csr_types_and_policies)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        run_options opt;
        check_csr_case<Kind::ic, T>(handle, 6, 1, true, opt);
        opt.policy = rocsparse_analysis_policy_force;
        check_csr_case<Kind::ic, T>(handle, 5, 1, true, opt);
    });
}

TEST_F(Cov70PrecondGeneric, spilu0_spic0_csr_int64_indices)
{
    run_options opt;
    check_csr_case<Kind::ilu, double, int64_t, int32_t>(handle, 5, 1, true, opt);
    check_csr_case<Kind::ilu, float, int64_t, int64_t>(handle, 5, 1, true, opt);
    check_csr_case<Kind::ic, double, int64_t, int32_t>(handle, 5, 1, true, opt);
    check_csr_case<Kind::ic, float, int64_t, int64_t>(handle, 5, 1, true, opt);
}

TEST_F(Cov70PrecondGeneric, spilu0_spic0_csr_batched)
{
    run_options opt;
    for(const bool shared : {true, false})
    {
        check_csr_case<Kind::ilu, double>(handle, 4, 3, shared, opt);
        check_csr_case<Kind::ic, double>(handle, 4, 3, shared, opt);
    }
    check_csr_case<Kind::ilu, rocsparse_float_complex>(handle, 4, 2, false, opt);
    check_csr_case<Kind::ic, rocsparse_double_complex>(handle, 4, 2, false, opt);
}

TEST_F(Cov70PrecondGeneric, spilu0_csr_numeric_boost)
{
    run_options opt;
    opt.boost     = true;
    opt.boost_tol = 1.0e6; // every pivot is boosted
    opt.boost_val = 2.0;
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        check_csr_case<Kind::ilu, T>(handle, 5, 1, true, opt);
    });
    // Boost enabled but no pivot is small enough: same factor as without boost.
    opt.boost_tol = 1.0e-12;
    check_csr_case<Kind::ilu, double>(handle, 5, 1, true, opt);
    // The boosted run on two batch entries.
    opt.boost_tol = 1.0e6;
    check_csr_case<Kind::ilu, float>(handle, 4, 2, false, opt);
}

TEST_F(Cov70PrecondGeneric, spic0_csr_boost_inputs_are_accepted)
{
    // spic0 stores the boost parameters but its kernels do not apply them.
    run_options opt;
    opt.boost     = true;
    opt.boost_tol = 1.0e6;
    opt.boost_val = 2.0;
    opt.boost     = true;
    dev_csr<double, int32_t, int32_t> d;
    d.init(make_grid_matrix<double>(4, false), 1, true);
    ASSERT_NE(d.A, nullptr);
    run_result<double> out;
    run_factorization<Kind::ic, double>(handle, d.A, d.val, 1, opt, out);
    ASSERT_EQ(out.compute_status, rocsparse_status_success);
    const auto want = keep_lower_only(d.base, host_ic0(d.base));
    expect_close(keep_lower_only(d.base, out.values), want, 4.0);
}

// ---------------------------------------------------------------------------
// Singularity reporting
// ---------------------------------------------------------------------------
namespace
{
    // A grid matrix whose diagonal entry of `row` is removed (symbolic singularity).
    template <typename T>
    csr_matrix<T> without_diagonal(csr_matrix<T> a, int32_t row)
    {
        for(int32_t p = a.ptr[row]; p < a.ptr[row + 1]; ++p)
        {
            if(a.ind[p] == row)
            {
                a.ind.erase(a.ind.begin() + p);
                a.val.erase(a.val.begin() + p);
                for(int32_t r = row + 1; r <= a.n; ++r)
                    --a.ptr[r];
                break;
            }
        }
        return a;
    }

    template <Kind K, typename T>
    void singular_case(rocsparse_handle      handle,
                       const csr_matrix<T>&  a,
                       const run_options&    opt,
                       rocsparse_singularity want_singularity,
                       int64_t               want_position)
    {
        dev_csr<T, int32_t, int32_t> d;
        d.init(a, 1, true);
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<K, T>(handle, d.A, d.val, 1, opt, out);
        EXPECT_EQ(out.compute_status, rocsparse_status_success);
        EXPECT_EQ(out.singularity[0], want_singularity);
        EXPECT_EQ(out.position[0], want_position);
    }
}

TEST_F(Cov70PrecondGeneric, spilu0_spic0_symbolic_singularity)
{
    const csr_matrix<double> a = without_diagonal(make_grid_matrix<double>(4, false), 5);
    run_options              opt;
    singular_case<Kind::ilu, double>(handle, a, opt, rocsparse_singularity_symbolic, 5);
    singular_case<Kind::ic, double>(handle, a, opt, rocsparse_singularity_symbolic, 5);
    opt.policy = rocsparse_analysis_policy_force;
    singular_case<Kind::ilu, double>(handle, a, opt, rocsparse_singularity_symbolic, 5);
}

TEST_F(Cov70PrecondGeneric, spilu0_exact_zero_pivot)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        auto a  = make_grid_matrix<T>(4, false);
        // Row 3 has pivot exactly 0 once its diagonal is zeroed and nothing
        // above it couples to row 3 through the lower triangle's updates.
        a.val[a.ptr[0]] = make_value<T>(0.0);
        run_options opt;
        singular_case<Kind::ilu, T>(handle, a, opt, rocsparse_singularity_numeric_exact, 0);
    });
}

TEST_F(Cov70PrecondGeneric, spic0_exact_zero_pivot)
{
    auto a          = make_grid_matrix<double>(4, false);
    a.val[a.ptr[0]] = 0.0;
    run_options opt;
    singular_case<Kind::ic, double>(handle, a, opt, rocsparse_singularity_numeric_exact, 0);
}

TEST_F(Cov70PrecondGeneric, spilu0_spic0_near_singularity_tolerance)
{
    for(const bool after : {false, true})
    {
        run_options opt;
        opt.set_sing_tol = true;
        opt.sing_tol     = 1.0e3; // every pivot is within the tolerance
        opt.tol_after    = after;
        const auto a     = make_grid_matrix<double>(4, false);
        singular_case<Kind::ilu, double>(handle, a, opt, rocsparse_singularity_numeric_near, 0);
        singular_case<Kind::ic, double>(handle, a, opt, rocsparse_singularity_numeric_near, 0);
        opt.sing_tol = 1.0e-12; // no pivot is
        singular_case<Kind::ilu, double>(handle, a, opt, rocsparse_singularity_none, -1);
        singular_case<Kind::ic, double>(handle, a, opt, rocsparse_singularity_none, -1);
    }
}

// ---------------------------------------------------------------------------
// BSR
// ---------------------------------------------------------------------------
TEST_F(Cov70PrecondGeneric, spilu0_bsr_block_dims)
{
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(const int32_t block_dim : {1, 2, 3, 4, 8, 9, 16, 17, 32})
        {
            run_options opt;
            check_bsr_case<Kind::ilu, double>(handle, 4, block_dim, dir, opt);
        }
        run_options opt;
        check_bsr_case<Kind::ilu, float>(handle, 4, 5, dir, opt);
        check_bsr_case<Kind::ilu, rocsparse_float_complex>(handle, 3, 6, dir, opt);
        check_bsr_case<Kind::ilu, rocsparse_double_complex>(handle, 3, 7, dir, opt);
    }
}

TEST_F(Cov70PrecondGeneric, spic0_bsr_block_dims)
{
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(const int32_t block_dim : {1, 2, 3, 4, 8, 9, 16, 17, 32})
        {
            run_options opt;
            check_bsr_case<Kind::ic, double>(handle, 4, block_dim, dir, opt);
        }
        run_options opt;
        check_bsr_case<Kind::ic, float>(handle, 4, 5, dir, opt);
        check_bsr_case<Kind::ic, rocsparse_float_complex>(handle, 3, 6, dir, opt);
        check_bsr_case<Kind::ic, rocsparse_double_complex>(handle, 3, 7, dir, opt);
    }
}

// A handle that reports a 64 wide wavefront selects the specialised block
// kernels (2_8 / 9_32 / 33_64 for ILU, 2_8_unrolled / 9_16 / 17_32 for IC).
// They are written for 64 lanes: on this 32-lane hardware the kernels with
// block_dim <= 4 still produce the exact factor (checked against the host
// reference), while the wider ones produce non-finite values. For those the test only
// asserts that the launch succeeds; the numeric result of the factorization is
// covered by the wavefront-32 tests above.
TEST_F(Cov70PrecondGeneric, spilu0_bsr_wave64_kernels)
{
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        run_options opt;
        opt.wavefront = 64;
        for(const int32_t block_dim : {2, 4})
        {
            check_bsr_case<Kind::ilu, double>(handle, 4, block_dim, dir, opt);
            check_bsr_case<Kind::ilu, float>(handle, 3, block_dim, dir, opt);
        }
        check_bsr_case<Kind::ilu, rocsparse_float_complex>(handle, 3, 3, dir, opt);
        check_bsr_case<Kind::ilu, rocsparse_double_complex>(handle, 3, 2, dir, opt);
        for(const int32_t block_dim : {8, 9, 16, 32, 33, 64})
        {
            check_bsr_case<Kind::ilu, double>(handle, 3, block_dim, dir, opt, false);
        }
        check_bsr_case<Kind::ilu, float>(handle, 3, 20, dir, opt, false);
        check_bsr_case<Kind::ilu, rocsparse_float_complex>(handle, 3, 12, dir, opt, false);
        check_bsr_case<Kind::ilu, rocsparse_double_complex>(handle, 3, 20, dir, opt, false);
    }
}

TEST_F(Cov70PrecondGeneric, spic0_bsr_wave64_kernels)
{
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        run_options opt;
        opt.wavefront = 64;
        for(const int32_t block_dim : {2, 4})
        {
            check_bsr_case<Kind::ic, double>(handle, 4, block_dim, dir, opt);
            check_bsr_case<Kind::ic, float>(handle, 3, block_dim, dir, opt);
        }
        check_bsr_case<Kind::ic, rocsparse_float_complex>(handle, 3, 3, dir, opt);
        check_bsr_case<Kind::ic, rocsparse_double_complex>(handle, 3, 2, dir, opt);
        for(const int32_t block_dim : {8, 9, 16, 17, 32})
        {
            check_bsr_case<Kind::ic, double>(handle, 3, block_dim, dir, opt, false);
        }
        check_bsr_case<Kind::ic, float>(handle, 3, 24, dir, opt, false);
        check_bsr_case<Kind::ic, rocsparse_float_complex>(handle, 3, 12, dir, opt, false);
        check_bsr_case<Kind::ic, rocsparse_double_complex>(handle, 3, 20, dir, opt, false);
    }
}

TEST_F(Cov70PrecondGeneric, spilu0_spic0_bsr_force_policy)
{
    run_options opt;
    opt.policy = rocsparse_analysis_policy_force;
    check_bsr_case<Kind::ilu, double>(handle, 4, 4, rocsparse_direction_row, opt);
    check_bsr_case<Kind::ic, double>(handle, 4, 4, rocsparse_direction_row, opt);
}

TEST_F(Cov70PrecondGeneric, spilu0_bsr_numeric_boost)
{
    run_options opt;
    opt.boost     = true;
    opt.boost_tol = 1.0e6;
    opt.boost_val = 2.0;
    check_bsr_case<Kind::ilu, double>(handle, 3, 4, rocsparse_direction_row, opt);
    check_bsr_case<Kind::ilu, float>(handle, 3, 3, rocsparse_direction_column, opt);
}

// ---------------------------------------------------------------------------
// Row-length regimes of IC(0) and ILDLT(0)
// ---------------------------------------------------------------------------
TEST_F(Cov70PrecondGeneric, spic0_csr_binsearch_fat_row)
{
    // Rows longer than 512 entries select the binary-search IC(0) kernel.
    run_options opt;
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        dev_csr<T, int32_t, int32_t> d;
        // Real values: the library's complex IC(0) update is checked on the grid matrices.
        d.init(make_fat_row_matrix<T>(520, false), 1, true);
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<Kind::ic, T>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        check_factor<Kind::ic>(d.base, out.values, opt, "fat row");
        EXPECT_EQ(out.singularity[0], rocsparse_singularity_none);
        EXPECT_EQ(out.position[0], -1);
    });
}

TEST_F(Cov70PrecondGeneric, spildlt0_csr_hash_row_lengths)
{
    // Row lengths up to 512 select the hash kernel; the table size grows with the length.
    // The lengths stay below the table sizes (32, 64, ..., 512): a row that fills the
    // whole shared-memory table makes the lookup of a missing key probe forever.
    run_options opt;
    for(const int32_t fat : {8, 31, 33, 63, 65, 127, 129, 255, 257, 511})
    {
        dev_csr<double, int32_t, int32_t> d;
        d.init(make_fat_row_matrix<double>(fat), 1, true);
        ASSERT_NE(d.A, nullptr);
        run_result<double> out;
        run_factorization<Kind::ildlt, double>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        check_factor<Kind::ildlt>(d.base, out.values, opt, "fat=" + std::to_string(fat));
        EXPECT_EQ(out.singularity[0], rocsparse_singularity_none);
        EXPECT_EQ(out.position[0], -1);
    }
}

TEST_F(Cov70PrecondGeneric, spildlt0_csr_hash_types_and_indices)
{
    run_options opt;
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        dev_csr<T, int32_t, int32_t> d;
        d.init(make_fat_row_matrix<T>(40), 1, true);
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<Kind::ildlt, T>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        check_factor<Kind::ildlt>(d.base, out.values, opt, "types");
    });
    {
        dev_csr<float, int64_t, int32_t> d;
        d.init(make_fat_row_matrix<float>(20), 1, true);
        ASSERT_NE(d.A, nullptr);
        run_result<float> out;
        run_factorization<Kind::ildlt, float>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        check_factor<Kind::ildlt>(d.base, out.values, opt, "i64/i32");
    }
    {
        dev_csr<double, int64_t, int64_t> d;
        d.init(make_fat_row_matrix<double>(20), 1, true);
        ASSERT_NE(d.A, nullptr);
        run_result<double> out;
        run_factorization<Kind::ildlt, double>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        check_factor<Kind::ildlt>(d.base, out.values, opt, "i64/i64");
    }
}

TEST_F(Cov70PrecondGeneric, spildlt0_csr_singularity_outputs)
{
    // Near singularity through the tolerance, exact zero pivot and a missing diagonal.
    {
        run_options opt;
        opt.set_sing_tol = true;
        opt.sing_tol     = 1.0e3;
        singular_case<Kind::ildlt, double>(
            handle, make_fat_row_matrix<double>(10), opt, rocsparse_singularity_numeric_near, 0);
    }
    {
        auto a          = make_fat_row_matrix<double>(10);
        a.val[a.ptr[0]] = 0.0;
        run_options opt;
        singular_case<Kind::ildlt, double>(handle, a, opt, rocsparse_singularity_numeric_exact, 0);
    }
    {
        run_options opt;
        singular_case<Kind::ildlt, double>(handle,
                                           without_diagonal(make_fat_row_matrix<double>(10), 4),
                                           opt,
                                           rocsparse_singularity_symbolic,
                                           4);
    }
}

// ---------------------------------------------------------------------------
// Descriptor, input, output and stage validation
// ---------------------------------------------------------------------------
namespace
{
    template <Kind K>
    void validate_descriptor_api(rocsparse_handle handle)
    {
        using A                   = api<K>;
        typename A::descr_t descr = nullptr;

        // create / destroy
        EXPECT_EQ(A::create(handle, nullptr), rocsparse_status_invalid_pointer);
        ASSERT_EQ(A::create(handle, &descr), rocsparse_status_success);
        ASSERT_NE(descr, nullptr);

        // set_input argument checks
        const rocsparse_datatype dtype = rocsparse_datatype_f64_r;
        EXPECT_EQ(A::set(nullptr, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::set(handle, nullptr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            A::set(
                handle, descr, static_cast<decltype(A::input(In::alg))>(99), &dtype, sizeof(dtype)),
            rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::dtype), nullptr, sizeof(dtype)),
                  rocsparse_status_invalid_pointer);

        // Wrong sizes.
        const int32_t alg = 0;
        EXPECT_EQ(A::set(handle, descr, A::input(In::alg), &alg, 1),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::dtype), &dtype, 1),
                  rocsparse_status_invalid_value);
        const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;
        EXPECT_EQ(A::set(handle, descr, A::input(In::policy), &policy, 1),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_enable), &alg, 1),
                  rocsparse_status_invalid_value);
        const double value = 1.0;
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_tol), &value, 3),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_val), &value, 1),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::sing_tol), &value, 1),
                  rocsparse_status_invalid_value);

        // Valid inputs: the boost tolerance may be a float or a double.
        const float ftol = 1.0f;
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_tol), &ftol, sizeof(float)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_tol), &value, sizeof(double)),
                  rocsparse_status_success);
        const int32_t enable = 1;
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_enable), &enable, sizeof(int32_t)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_val), &value, sizeof(void*)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::sing_tol), &value, sizeof(void*)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::alg), &alg, sizeof(alg)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::policy), &policy, sizeof(policy)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_success);

        // get_output argument checks
        rocsparse_singularity singularity = rocsparse_singularity_none;
        int64_t               position    = 0;
        EXPECT_EQ(A::get(nullptr, descr, false, &singularity, sizeof(singularity)),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::get(handle, nullptr, false, &singularity, sizeof(singularity)),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::get(handle, descr, false, nullptr, sizeof(singularity)),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::get(handle, descr, false, &singularity, 0), rocsparse_status_invalid_size);
        EXPECT_EQ(A::get(handle, descr, true, &position, 0), rocsparse_status_invalid_size);
        EXPECT_EQ(A::get(handle, descr, false, &singularity, 1), rocsparse_status_invalid_value);
        EXPECT_EQ(A::get(handle, descr, true, &position, 1), rocsparse_status_invalid_value);

        // Destroying a null descriptor is allowed.
        EXPECT_EQ(A::destroy(handle, nullptr), rocsparse_status_success);
        EXPECT_EQ(A::destroy(handle, descr), rocsparse_status_success);
    }

    template <Kind K>
    void validate_stage_api(rocsparse_handle handle)
    {
        using A = api<K>;
        dev_csr<double, int32_t, int32_t> d;
        d.init(make_grid_matrix<double>(3, false), 1, true);
        ASSERT_NE(d.A, nullptr);
        dev_csr<double, int32_t, int32_t> other;
        other.init(make_grid_matrix<double>(3, false), 1, true);
        ASSERT_NE(other.A, nullptr);

        typename A::descr_t descr = nullptr;
        ASSERT_EQ(A::create(handle, &descr), rocsparse_status_success);

        size_t size = 0;
        // buffer_size argument checks.
        EXPECT_EQ(A::buffer_size(nullptr, descr, d.A, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::buffer_size(handle, nullptr, d.A, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::buffer_size(handle, descr, nullptr, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::buffer_size(handle, descr, d.A, nullptr, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::buffer_size(handle, descr, d.A, other.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(A::buffer_size(handle,
                                 descr,
                                 d.A,
                                 d.A,
                                 static_cast<decltype(A::stage(Stage::analysis))>(7),
                                 &size),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::analysis), nullptr),
                  rocsparse_status_invalid_pointer);
        ASSERT_EQ(A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);

        // run argument checks.
        EXPECT_EQ(A::run(nullptr, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::run(handle, nullptr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::run(handle, descr, nullptr, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::run(handle, descr, d.A, nullptr, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::run(handle, descr, d.A, other.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(A::run(handle,
                         descr,
                         d.A,
                         d.A,
                         static_cast<decltype(A::stage(Stage::analysis))>(7),
                         size,
                         buffer.ptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), 0, buffer.ptr),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, nullptr),
                  rocsparse_status_invalid_pointer);

        // The descriptor datatype must match the matrix.
        const rocsparse_datatype wrong = rocsparse_datatype_f32_r;
        ASSERT_EQ(A::set(handle, descr, A::input(In::dtype), &wrong, sizeof(wrong)),
                  rocsparse_status_success);
        EXPECT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_not_implemented);
        const rocsparse_datatype dtype = rocsparse_datatype_f64_r;
        ASSERT_EQ(A::set(handle, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_success);

        // The compute stage needs the analysis first; the analysis policy must be valid.
        EXPECT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::compute), size, buffer.ptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_value);
        const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;
        ASSERT_EQ(A::set(handle, descr, A::input(In::policy), &policy, sizeof(policy)),
                  rocsparse_status_success);
        ASSERT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_success);

        // The analysis cannot run twice and the descriptor can no longer be reconfigured.
        EXPECT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_value);
        const int32_t alg = 0;
        EXPECT_EQ(A::set(handle, descr, A::input(In::alg), &alg, sizeof(alg)),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::policy), &policy, sizeof(policy)),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_invalid_value);

        size_t compute_size = 0;
        ASSERT_EQ(A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::compute), &compute_size),
                  rocsparse_status_success);
        device_vector<char> compute_buffer{compute_size ? compute_size : size_t{1}};
        ASSERT_TRUE(compute_buffer.ptr);
        ASSERT_EQ(A::run(handle,
                         descr,
                         d.A,
                         d.A,
                         A::stage(Stage::compute),
                         compute_size,
                         compute_size ? compute_buffer.ptr : nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // After the compute stage the analysis cannot run again.
        EXPECT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_value);

        EXPECT_EQ(A::destroy(handle, descr), rocsparse_status_success);
    }
}

TEST_F(Cov70PrecondGeneric, spilu0_descriptor_and_input_validation)
{
    validate_descriptor_api<Kind::ilu>(handle);
}

TEST_F(Cov70PrecondGeneric, spic0_descriptor_and_input_validation)
{
    validate_descriptor_api<Kind::ic>(handle);
}

TEST_F(Cov70PrecondGeneric, spildlt0_descriptor_and_input_validation)
{
    validate_descriptor_api<Kind::ildlt>(handle);
}

TEST_F(Cov70PrecondGeneric, spilu0_stage_validation)
{
    validate_stage_api<Kind::ilu>(handle);
}

TEST_F(Cov70PrecondGeneric, spic0_stage_validation)
{
    validate_stage_api<Kind::ic>(handle);
}

TEST_F(Cov70PrecondGeneric, spildlt0_stage_validation)
{
    validate_stage_api<Kind::ildlt>(handle);
}
