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

    // The singularity position is reported in the index base of the matrix; -1 means
    // "no singularity" in either base.
    inline int64_t position_in_base(int64_t zero_based, rocsparse_index_base base)
    {
        return (zero_based < 0 || base == rocsparse_index_base_zero) ? zero_based : zero_based + 1;
    }

    // Factorization of the (batched) grid matrix.
    template <Kind K, typename T, typename I = int32_t, typename J = int32_t>
    void check_csr_case(rocsparse_handle     handle,
                        int32_t              grid,
                        int                  batch,
                        bool                 shared_row_ptr,
                        const run_options&   opt,
                        rocsparse_index_base base = rocsparse_index_base_zero)
    {
        const bool       skew = (K == Kind::ilu);
        dev_csr<T, I, J> d;
        d.init(make_grid_matrix<T>(grid, skew), batch, shared_row_ptr, base);
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

    // Factorization of the BSR matrix `m` through the generic API.
    //
    // verify == true: the factor must match the host reference (the scalar
    // expansion factorized on the host) and no singularity may be reported.
    //
    // verify == false is used for the wave64-specialised kernels forced onto a 32-lane
    // GPU by the wave64 tests: the kernels are written for 64 lanes and, run with 32,
    // produce wrong (for block dimensions above 4 even non-finite) values, so the values
    // are not compared and finiteness cannot be asserted either. The call must still
    // succeed, the sparsity pattern the library was given must be unchanged, and the
    // factorization must have written to the values. On a native 64-lane GPU the tests
    // verify the values.
    template <Kind K, typename T>
    void check_bsr_matrix(rocsparse_handle     handle,
                          const bsr_matrix<T>& m,
                          const run_options&   opt,
                          bool                 verify,
                          const std::string&   what,
                          rocsparse_index_base base = rocsparse_index_base_zero)
    {
        SCOPED_TRACE(what);
        dev_bsr<T> d;
        d.init(m, base);
        ASSERT_NE(d.A, nullptr);
        // Whatever the wave size, the matrix must admit the factorization.
        expect_diagonally_dominant(expand_bsr(m));
        run_result<T> out;
        run_factorization<K, T>(handle, d.A, d.val, 1, opt, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        ASSERT_EQ(out.values.size(), d.base.val.size());
        if(!verify)
        {
            const int32_t shift = (base == rocsparse_index_base_one) ? 1 : 0;
            const auto    row   = to_host(d.row);
            const auto    col   = to_host(d.col);
            ASSERT_EQ(row.size(), d.base.ptr.size());
            ASSERT_EQ(col.size(), d.base.ind.size());
            for(size_t i = 0; i < row.size(); ++i)
                EXPECT_EQ(row[i], d.base.ptr[i] + shift) << "row pointer " << i;
            for(size_t i = 0; i < col.size(); ++i)
                EXPECT_EQ(col[i], d.base.ind[i] + shift) << "column index " << i;
            bool written = false;
            for(size_t i = 0; i < out.values.size(); ++i)
                written = written || !(to_cd(out.values[i]) == to_cd(d.base.val[i]));
            EXPECT_TRUE(written) << "the kernel left every value unchanged";
            return;
        }
        const csr_matrix<T> scalar = expand_bsr(d.base);
        const csr_matrix<T> got    = expand_bsr([&] {
            bsr_matrix<T> f = d.base;
            f.val           = out.values;
            return f;
        }());
        check_factor<K>(scalar, got.val, opt, "scalar expansion");
        EXPECT_EQ(out.singularity[0], rocsparse_singularity_none);
        EXPECT_EQ(out.position[0], -1);
    }

    template <Kind K, typename T>
    void check_bsr_case(rocsparse_handle     handle,
                        int32_t              mb,
                        int32_t              block_dim,
                        rocsparse_direction  dir,
                        const run_options&   opt,
                        bool                 verify = true,
                        rocsparse_index_base base   = rocsparse_index_base_zero)
    {
        const bool skew = (K == Kind::ilu);
        check_bsr_matrix<K, T>(handle,
                               make_block_matrix<T>(mb, block_dim, dir, skew),
                               opt,
                               verify,
                               "mb=" + std::to_string(mb)
                                   + " block_dim=" + std::to_string(block_dim),
                               base);
    }
}

class Cov70PrecondGeneric : public HandleTest
{
};

// ---------------------------------------------------------------------------
// Test matrices and helpers shared by the factorization tests below
// ---------------------------------------------------------------------------
namespace
{
    // Factorization of one scalar CSR matrix through the generic API, compared
    // with the host reference. `result` (optional) receives the raw outputs.
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

    // Position of the diagonal entry of `row` in the CSR arrays.
    template <typename T>
    size_t diagonal_position(const csr_matrix<T>& a, int32_t row)
    {
        for(int32_t p = a.ptr[row]; p < a.ptr[row + 1]; ++p)
            if(a.ind[p] == row)
                return static_cast<size_t>(p);
        return static_cast<size_t>(-1);
    }

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

    // `want_position` is zero based; the library reports it in the matrix index base.
    template <Kind K, typename T>
    void singular_case(rocsparse_handle      handle,
                       const csr_matrix<T>&  a,
                       const run_options&    opt,
                       rocsparse_singularity want_singularity,
                       int64_t               want_position,
                       rocsparse_index_base  base = rocsparse_index_base_zero)
    {
        dev_csr<T, int32_t, int32_t> d;
        d.init(a, 1, true, base);
        ASSERT_NE(d.A, nullptr);
        run_result<T> out;
        run_factorization<K, T>(handle, d.A, d.val, 1, opt, out);
        EXPECT_EQ(out.compute_status, rocsparse_status_success);
        EXPECT_EQ(out.singularity[0], want_singularity);
        EXPECT_EQ(out.position[0], position_in_base(want_position, base));
    }

    // Pivot of every row as the singularity test sees it: |u_ii| for ILU(0) (the test
    // is |u_ii| <= tol) and the pivot before the square root, l_ii^2, for IC(0) (the
    // test is l_ii^2 <= tol^2).
    template <Kind K>
    std::vector<double> pivots_of(const csr_matrix<double>& a)
    {
        const auto          f = (K == Kind::ic) ? host_ic0(a) : host_ilu0(a);
        std::vector<double> pivots;
        for(int32_t r = 0; r < a.n; ++r)
        {
            const double v = f[diagonal_position(a, r)];
            pivots.push_back(K == Kind::ic ? v * v : std::abs(v));
        }
        return pivots;
    }

    // Zero based row of the first pivot that is at most `threshold`, -1 if none.
    int64_t first_pivot_at_most(const std::vector<double>& pivots, double threshold)
    {
        for(size_t i = 0; i < pivots.size(); ++i)
            if(pivots[i] <= threshold)
                return static_cast<int64_t>(i);
        return -1;
    }

    // True when no pivot lies within a relative 1e-6 of `threshold`, so that host and
    // device rounding cannot disagree on which side of it a pivot falls.
    bool clear_of_pivots(const std::vector<double>& pivots, double threshold)
    {
        for(const double p : pivots)
            if(std::abs(p - threshold) <= 1.0e-6 * threshold)
                return false;
        return true;
    }

    template <Kind K>
    void near_singular_case(rocsparse_handle          handle,
                            const csr_matrix<double>& a,
                            double                    tol,
                            bool                      after_analysis,
                            int64_t                   want_position,
                            rocsparse_index_base      base = rocsparse_index_base_zero)
    {
        SCOPED_TRACE("tol=" + std::to_string(tol) + " after=" + std::to_string(after_analysis));
        run_options opt;
        opt.set_sing_tol = true;
        opt.sing_tol     = tol;
        opt.tol_after    = after_analysis;
        singular_case<K, double>(handle,
                                 a,
                                 opt,
                                 want_position < 0 ? rocsparse_singularity_none
                                                   : rocsparse_singularity_numeric_near,
                                 want_position,
                                 base);
    }
}

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

// On the grid matrix a row never shares a column below the diagonal with a row it
// depends on, so every IC(0) inner sum is zero (the band tests below have non-zero
// sums). The complex conjugation order of the library's complex IC(0) update
// (see spic0_csr_nonzero_inner_sums_hash) is therefore invisible here, and the
// complex types are safe to use on this matrix.
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
    // Grid matrix: the IC(0) inner sums are zero, see spic0_csr_types_and_policies.
    check_csr_case<Kind::ic, rocsparse_double_complex>(handle, 4, 2, false, opt);
}

// ILU(0) of banded matrices: every row has several entries on both sides of the
// diagonal that the elimination updates.
TEST_F(Cov70PrecondGeneric, spilu0_csr_band_matrices)
{
    run_options opt;
    // Longest rows of 3, 6, 21 and 41 entries.
    for(const int32_t w : {2, 5, 20, 40})
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            check_csr_matrix<Kind::ilu, T>(
                handle, make_lower_band_matrix<T>(60, w), opt, "w=" + std::to_string(w));
        });
    }
}

// IC(0) inner sums with several non-zero terms (rows that share columns below the
// diagonal) through the hash kernel. The real types are checked with real data; the
// complex types only with real-valued entries: the library's complex IC(0) kernels
// accumulate fma(val[k], conj(val[idx]), sum), which is the conjugate of the sum
// L * L^H requires, so a complex matrix with non-real products does not reproduce
// the host factorization (library bug, not exercised here). With real values the
// products are real and the conjugation order does not matter.
TEST_F(Cov70PrecondGeneric, spic0_csr_nonzero_inner_sums_hash)
{
    run_options opt;
    // Longest rows of 3, 6, 21 and 41 entries (never 32, 64, ..., see
    // make_fat_row_matrix).
    for(const int32_t w : {2, 5, 20, 40})
    {
        check_csr_matrix<Kind::ic, float>(
            handle, make_lower_band_matrix<float>(60, w, false), opt, "band float");
        check_csr_matrix<Kind::ic, double>(
            handle, make_lower_band_matrix<double>(60, w, false), opt, "band double");
        check_csr_matrix<Kind::ic, rocsparse_float_complex>(
            handle, make_lower_band_matrix<rocsparse_float_complex>(60, w, false), opt, "band cf");
        check_csr_matrix<Kind::ic, rocsparse_double_complex>(
            handle, make_lower_band_matrix<rocsparse_double_complex>(60, w, false), opt, "band cd");
    }
    // A dense last row over narrow band rows: table sizes 32 .. 512 (lengths below
    // the sizes) and band 3 so that the sums have three terms.
    for(const int32_t fat : {8, 40, 100, 300, 511})
    {
        check_csr_matrix<Kind::ic, double>(
            handle, make_fat_row_matrix<double>(fat, false, 3), opt, "fat " + std::to_string(fat));
        check_csr_matrix<Kind::ic, float>(
            handle, make_fat_row_matrix<float>(fat, false, 3), opt, "fat " + std::to_string(fat));
    }
    check_csr_matrix<Kind::ic, rocsparse_double_complex>(
        handle, make_fat_row_matrix<rocsparse_double_complex>(100, false, 3), opt, "fat cd");
}

TEST_F(Cov70PrecondGeneric, spilu0_spic0_csr_one_based)
{
    run_options opt;
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        check_csr_case<Kind::ilu, T>(handle, 5, 1, true, opt, rocsparse_index_base_one);
        // Grid matrix: the IC(0) inner sums are zero, see spic0_csr_types_and_policies.
        check_csr_case<Kind::ic, T>(handle, 5, 1, true, opt, rocsparse_index_base_one);
    });
    check_csr_case<Kind::ilu, double>(handle, 4, 3, false, opt, rocsparse_index_base_one);
    check_csr_case<Kind::ic, double>(handle, 4, 3, true, opt, rocsparse_index_base_one);
    check_csr_case<Kind::ilu, double, int64_t, int32_t>(
        handle, 5, 1, true, opt, rocsparse_index_base_one);
    check_csr_case<Kind::ic, float, int64_t, int64_t>(
        handle, 5, 1, true, opt, rocsparse_index_base_one);
    // Band and fat-row matrices (non-zero inner sums) in the hash and binary-search kernels.
    check_csr_matrix<Kind::ic, double>(handle,
                                       make_lower_band_matrix<double>(60, 5, false),
                                       opt,
                                       "band",
                                       rocsparse_index_base_one);
    check_csr_matrix<Kind::ic, double>(handle,
                                       make_fat_row_matrix<double>(520, false, 3),
                                       opt,
                                       "fat 520",
                                       rocsparse_index_base_one);
    check_csr_matrix<Kind::ilu, double>(
        handle, make_lower_band_matrix<double>(60, 5), opt, "band", rocsparse_index_base_one);
    check_csr_matrix<Kind::ilu, double>(handle,
                                        make_fat_row_matrix<double>(520, true, 3),
                                        opt,
                                        "fat 520",
                                        rocsparse_index_base_one);
    check_csr_matrix<Kind::ildlt, double>(
        handle, make_fat_row_matrix<double>(40, true, 3), opt, "fat 40", rocsparse_index_base_one);
}

// The ILU(0) boost replaces a pivot whose magnitude is at most the boost tolerance
// by the boost value (its magnitude) with the sign / phase of the pivot.
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

namespace
{
    // Tridiagonal matrix of order 12 with weak off-diagonal entries (the pivots stay
    // close to the diagonal) and three small diagonal entries:
    //   row 3: -0.5 (+0.2i for complex types), a negative real part
    //   row 6:  0.4 (0 when `zero_pivot_row6`; row 6 then also loses its lower
    //           entry so that its pivot is exactly 0)
    //   row 9:  2.5
    // All the other diagonal entries are 5.
    template <typename T>
    csr_matrix<T> make_boost_matrix(bool zero_pivot_row6)
    {
        csr_matrix<T> a;
        a.n = 12;
        a.ptr.push_back(0);
        for(int32_t r = 0; r < a.n; ++r)
        {
            if(r > 0 && !(zero_pivot_row6 && r == 6))
            {
                a.ind.push_back(r - 1);
                a.val.push_back(make_value<T>(-0.01 * (1.0 + 0.1 * (r % 3))));
            }
            a.ind.push_back(r);
            switch(r)
            {
            case 3:
                a.val.push_back(make_value<T>(-0.5, is_cplx<T> ? 0.2 : 0.0));
                break;
            case 6:
                a.val.push_back(make_value<T>(zero_pivot_row6 ? 0.0 : 0.4));
                break;
            case 9:
                a.val.push_back(make_value<T>(2.5));
                break;
            default:
                a.val.push_back(make_value<T>(5.0));
                break;
            }
            if(r + 1 < a.n)
            {
                a.ind.push_back(r + 1);
                a.val.push_back(make_value<T>(-0.01));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        return a;
    }

    // Boost with the given tolerance on the matrix of make_boost_matrix: the factor
    // must equal the host factor with the same boost, the boosted pivots must have
    // the magnitude of the boost value (2) and the sign / phase of the original
    // pivot, the others must be untouched, and the factorization reports no
    // singularity (a boosted pivot is not a zero pivot).
    template <typename T>
    void check_boost(rocsparse_handle handle,
                     bool             zero_pivot_row6,
                     double           boost_tol,
                     bool             boosted3,
                     bool             boosted6,
                     bool             boosted9)
    {
        SCOPED_TRACE("boost_tol=" + std::to_string(boost_tol));
        run_options opt;
        opt.boost     = true;
        opt.boost_tol = boost_tol;
        opt.boost_val = 2.0;
        run_result<T>       out;
        const csr_matrix<T> a = make_boost_matrix<T>(zero_pivot_row6);
        check_csr_matrix<Kind::ilu, T>(
            handle, a, opt, "boost matrix", rocsparse_index_base_zero, &out);
        ASSERT_FALSE(out.values.empty());
        const double kBoost = 2.0;
        const auto   d3     = to_cd(out.values[diagonal_position(a, 3)]);
        const auto   d6     = to_cd(out.values[diagonal_position(a, 6)]);
        const auto   d9     = to_cd(out.values[diagonal_position(a, 9)]);
        if(boosted3)
        {
            EXPECT_NEAR(std::abs(d3), kBoost, 1.0e-5 * kBoost);
            EXPECT_LT(d3.real(), 0.0); // the negative real part of the pivot is kept
            if(is_cplx<T>)
                EXPECT_GT(d3.imag(), 0.0); // and so is its phase
        }
        else
        {
            // |-0.5| or |-0.5 + 0.2i| = 0.539, the weak coupling moves it by about 1e-4.
            EXPECT_NEAR(std::abs(d3), is_cplx<T> ? std::hypot(0.5, 0.2) : 0.5, 5.0e-3);
        }
        if(boosted6)
        {
            EXPECT_NEAR(d6.real(), kBoost, 1.0e-5 * kBoost);
            EXPECT_NEAR(d6.imag(), 0.0, 1.0e-5);
        }
        else
        {
            EXPECT_NEAR(std::abs(d6), 0.4, 5.0e-3);
        }
        if(boosted9)
            EXPECT_NEAR(std::abs(d9), kBoost, 1.0e-5 * kBoost);
        else
            EXPECT_GT(std::abs(d9), 2.4);
    }
}

TEST_F(Cov70PrecondGeneric, spilu0_csr_numeric_boost_partial_and_signed)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        // Pivot magnitudes (about) 5, 0.54 or 0.5, 0.4, 2.5.
        // tol 1.0: the pivots of rows 3 (negative real part) and 6 are boosted.
        check_boost<T>(handle, false, 1.0, true, true, false);
        // tol 0.45 is between the pivot magnitudes: only row 6 is boosted. The row 3
        // pivot (-0.5, or -0.5 + 0.2i with modulus 0.54) stays.
        check_boost<T>(handle, false, 0.45, false, true, false);
        // tol 3: rows 3, 6 and 9 are boosted, the 5s stay.
        check_boost<T>(handle, false, 3.0, true, true, true);
        // An exactly zero pivot is boosted to +2 (and is not reported as singular).
        check_boost<T>(handle, true, 1.0e-3, false, true, false);
    });
}

TEST_F(Cov70PrecondGeneric, spilu0_exact_zero_pivot_without_boost)
{
    // The same matrix without the boost reports the zero pivot of row 6.
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        run_options opt;
        singular_case<Kind::ilu, T>(
            handle, make_boost_matrix<T>(true), opt, rocsparse_singularity_numeric_exact, 6);
    });
}

// spic0 stores the boost parameters but its kernels never apply them: the factor
// is the plain IC(0) factor.
TEST_F(Cov70PrecondGeneric, spic0_csr_boost_inputs_do_not_change_the_factor)
{
    run_options opt;
    opt.boost     = true;
    opt.boost_tol = 1.0e6;
    opt.boost_val = 2.0;
    dev_csr<double, int32_t, int32_t> d;
    d.init(make_grid_matrix<double>(4, false), 1, true);
    ASSERT_NE(d.A, nullptr);
    run_result<double> out;
    run_factorization<Kind::ic, double>(handle, d.A, d.val, 1, opt, out);
    ASSERT_EQ(out.compute_status, rocsparse_status_success);
    const auto want = keep_lower_only(d.base, host_ic0(d.base));
    expect_close(keep_lower_only(d.base, out.values), want);
    EXPECT_EQ(out.singularity[0], rocsparse_singularity_none);
    EXPECT_EQ(out.position[0], -1);
}

// ---------------------------------------------------------------------------
// Singularity reporting
// ---------------------------------------------------------------------------
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
        // The diagonal entry of row 0 is zero, and row 0 has no row above it to
        // change that, so its pivot is exactly 0.
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
        const auto a = make_grid_matrix<double>(4, false);
        // Every pivot is within the tolerance / no pivot is.
        near_singular_case<Kind::ilu>(handle, a, 1.0e3, after, 0);
        near_singular_case<Kind::ic>(handle, a, 1.0e3, after, 0);
        near_singular_case<Kind::ilu>(handle, a, 1.0e-12, after, -1);
        near_singular_case<Kind::ic>(handle, a, 1.0e-12, after, -1);
    }
}

// ILU(0): a pivot is near-singular when |u_ii| <= tol (inclusive). The pivot of
// row 0 is exactly 5 (nothing is subtracted from the diagonal entry of row 0), the
// others are smaller and differ from it by far more than the rounding of the
// factorization.
TEST_F(Cov70PrecondGeneric, spilu0_near_singularity_tolerance_boundaries)
{
    const auto a      = make_grid_matrix<double>(4, false);
    const auto pivots = pivots_of<Kind::ilu>(a);
    ASSERT_EQ(pivots[0], 5.0);
    const double min_pivot = *std::min_element(pivots.begin(), pivots.end());
    ASSERT_LT(min_pivot, 4.9);
    for(const bool after : {false, true})
    {
        // tol == pivot 0: flagged (<=), position 0.
        near_singular_case<Kind::ilu>(handle, a, 5.0, after, 0);
        // The largest tolerance below the row 0 pivot skips row 0: the first row whose
        // pivot is at most that tolerance is reported instead.
        const double  below = std::nextafter(5.0, 0.0);
        const int64_t want  = first_pivot_at_most(pivots, below);
        ASSERT_GT(want, 0);
        near_singular_case<Kind::ilu>(handle, a, below, after, want);
        // Between the smallest pivot and the others, and below all of them.
        const double mid = 0.5 * (min_pivot + 5.0);
        ASSERT_TRUE(clear_of_pivots(pivots, mid));
        near_singular_case<Kind::ilu>(handle, a, mid, after, first_pivot_at_most(pivots, mid));
        ASSERT_TRUE(clear_of_pivots(pivots, 0.5 * min_pivot));
        near_singular_case<Kind::ilu>(handle, a, 0.5 * min_pivot, after, -1);
    }
}

// IC(0): the test is l_ii^2 <= tol^2 on the pivot before the square root (the
// row 0 pivot is 5): tol = 3 flags it (5 <= 9 although 5 > 3), tol = 2 does not
// (5 > 4, and neither does any other pivot, which are all above 4).
TEST_F(Cov70PrecondGeneric, spic0_near_singularity_tolerance_boundaries)
{
    const auto a      = make_grid_matrix<double>(4, false);
    const auto pivots = pivots_of<Kind::ic>(a);
    ASSERT_NEAR(pivots[0], 5.0, 1.0e-12);
    const double min_pivot = *std::min_element(pivots.begin(), pivots.end());
    ASSERT_GT(min_pivot, 4.0);
    ASSERT_LT(min_pivot, 4.9);
    for(const bool after : {false, true})
    {
        near_singular_case<Kind::ic>(handle, a, 3.0, after, 0);
        near_singular_case<Kind::ic>(handle, a, 2.0, after, -1);
        // Between the pivots: the pivot of row 0 is above tol^2 = 4.9, a later one is not.
        const double  tol  = std::sqrt(4.9);
        const int64_t want = first_pivot_at_most(pivots, tol * tol);
        ASSERT_GT(want, 0);
        ASSERT_TRUE(clear_of_pivots(pivots, tol * tol));
        near_singular_case<Kind::ic>(handle, a, tol, after, want);
    }
}

TEST_F(Cov70PrecondGeneric, spilu0_spic0_singularity_one_based)
{
    const rocsparse_index_base one = rocsparse_index_base_one;
    run_options                opt;
    // Symbolic singularity of row 5 (zero based) is reported as 6.
    const csr_matrix<double> symbolic = without_diagonal(make_grid_matrix<double>(4, false), 5);
    singular_case<Kind::ilu, double>(handle, symbolic, opt, rocsparse_singularity_symbolic, 5, one);
    singular_case<Kind::ic, double>(handle, symbolic, opt, rocsparse_singularity_symbolic, 5, one);
    // Exact zero pivot of row 0 is reported as 1.
    auto zero_pivot                   = make_grid_matrix<double>(4, false);
    zero_pivot.val[zero_pivot.ptr[0]] = 0.0;
    singular_case<Kind::ilu, double>(
        handle, zero_pivot, opt, rocsparse_singularity_numeric_exact, 0, one);
    singular_case<Kind::ic, double>(
        handle, zero_pivot, opt, rocsparse_singularity_numeric_exact, 0, one);
    // Near singularity: the first flagged row (1 for row 0), and "none" stays -1.
    const auto a = make_grid_matrix<double>(4, false);
    for(const bool after : {false, true})
    {
        near_singular_case<Kind::ilu>(handle, a, 1.0e3, after, 0, one);
        near_singular_case<Kind::ic>(handle, a, 1.0e3, after, 0, one);
        near_singular_case<Kind::ilu>(handle, a, 1.0e-12, after, -1, one);
        near_singular_case<Kind::ic>(handle, a, 1.0e-12, after, -1, one);
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

// The BSR IC(0) matrices are block tridiagonal with Hermitian diagonal blocks; the
// scalar inner sums are non-zero (a scalar row shares columns with the rows it depends
// on inside a block). The complex types have non-real products here and are compared
// with the host factorization: the conjugation order problem of the csric0 kernels
// (see spic0_csr_nonzero_inner_sums_hash) does not show in the bsric0 kernels.
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

TEST_F(Cov70PrecondGeneric, spilu0_spic0_bsr_one_based)
{
    run_options opt;
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        check_bsr_case<Kind::ilu, double>(handle, 4, 3, dir, opt, true, rocsparse_index_base_one);
        check_bsr_case<Kind::ic, double>(handle, 4, 3, dir, opt, true, rocsparse_index_base_one);
        check_bsr_case<Kind::ilu, float>(handle, 3, 9, dir, opt, true, rocsparse_index_base_one);
        check_bsr_case<Kind::ic, float>(handle, 3, 9, dir, opt, true, rocsparse_index_base_one);
    }
}

// A handle that reports a 64 wide wavefront selects the specialised block
// kernels (2_8 / 9_32 / 33_64 for ILU, 2_8_unrolled / 2_8 / 9_16 / 17_32 for IC).
// They are written for 64 lanes. Natively (a wave64 GPU) every case is checked
// against the host reference. On a 32-lane GPU the override forces the 64-lane
// instantiations: with block_dim <= 4 they still produce the exact factor
// (checked), the wider ones produce non-finite values, so for those only the
// structure is checked (see check_bsr_matrix); the numeric result of every block
// dimension is covered by the wavefront-32 tests above.
namespace
{
    // `native64` must be read from the handle BEFORE the override is applied.
    bool values_checkable(bool native64, int32_t block_dim)
    {
        return native64 || block_dim <= 4;
    }
}

TEST_F(Cov70PrecondGeneric, spilu0_bsr_wave64_kernels)
{
    const bool native64 = (handle->wavefront_size == 64);
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
            check_bsr_case<Kind::ilu, double>(
                handle, 3, block_dim, dir, opt, values_checkable(native64, block_dim));
        }
        check_bsr_case<Kind::ilu, float>(handle, 3, 20, dir, opt, values_checkable(native64, 20));
        check_bsr_case<Kind::ilu, rocsparse_float_complex>(
            handle, 3, 12, dir, opt, values_checkable(native64, 12));
        check_bsr_case<Kind::ilu, rocsparse_double_complex>(
            handle, 3, 20, dir, opt, values_checkable(native64, 20));
    }
}

TEST_F(Cov70PrecondGeneric, spic0_bsr_wave64_kernels)
{
    const bool native64 = (handle->wavefront_size == 64);
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
            check_bsr_case<Kind::ic, double>(
                handle, 3, block_dim, dir, opt, values_checkable(native64, block_dim));
        }
        check_bsr_case<Kind::ic, float>(handle, 3, 24, dir, opt, values_checkable(native64, 24));
        check_bsr_case<Kind::ic, rocsparse_float_complex>(
            handle, 3, 12, dir, opt, values_checkable(native64, 12));
        check_bsr_case<Kind::ic, rocsparse_double_complex>(
            handle, 3, 20, dir, opt, values_checkable(native64, 20));
    }
}

// Block rows of 33 to 128 blocks and block_dim <= 8 select the plain (not
// unrolled) 2_8 bsric0 kernel on a 64 lane wavefront; the lower block pattern has a
// last block row of `fat` blocks. The wavefront-32 (general) kernel is checked on
// the same matrices. The 2_8 kernel forced onto 32 lanes gives wrong values even
// for block_dim 2 (unlike the unrolled kernel for rows of at most 32 blocks), so
// its values are compared only on a native wave64 GPU; elsewhere the structure
// and the launch status are checked (see check_bsr_matrix).
TEST_F(Cov70PrecondGeneric, spic0_bsr_wave64_long_block_rows)
{
    const bool  native64 = (handle->wavefront_size == 64);
    run_options general;
    run_options forced;
    forced.wavefront = 64;
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(const int32_t block_dim : {2, 3, 4, 8})
        {
            const auto m    = make_fat_block_row_matrix<double>(40, block_dim, dir);
            const auto what = "fat block row, block_dim=" + std::to_string(block_dim);
            check_bsr_matrix<Kind::ic, double>(handle, m, general, true, what + " (default)");
            check_bsr_matrix<Kind::ic, double>(handle, m, forced, native64, what + " (wave64)");
        }
        check_bsr_matrix<Kind::ic, float>(handle,
                                          make_fat_block_row_matrix<float>(100, 3, dir),
                                          forced,
                                          native64,
                                          "fat block row 100");
        check_bsr_matrix<Kind::ic, rocsparse_double_complex>(
            handle,
            make_fat_block_row_matrix<rocsparse_double_complex>(40, 2, dir),
            forced,
            native64,
            "fat block row, complex");
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
// Row-length regimes of IC(0), ILU(0) and ILDLT(0)
// ---------------------------------------------------------------------------
TEST_F(Cov70PrecondGeneric, spic0_csr_binsearch_fat_row)
{
    // Rows longer than 512 entries select the binary-search IC(0) kernel. Band 3
    // gives the inner sums of the long row several terms. The complex types use
    // real-valued entries (see spic0_csr_nonzero_inner_sums_hash).
    run_options opt;
    for(const int32_t fat : {520, 600})
    {
        check_csr_matrix<Kind::ic, float>(
            handle, make_fat_row_matrix<float>(fat, false, 3), opt, "fat float");
        check_csr_matrix<Kind::ic, double>(
            handle, make_fat_row_matrix<double>(fat, false, 3), opt, "fat double");
        check_csr_matrix<Kind::ic, rocsparse_float_complex>(
            handle, make_fat_row_matrix<rocsparse_float_complex>(fat, false, 3), opt, "fat cf");
        check_csr_matrix<Kind::ic, rocsparse_double_complex>(
            handle, make_fat_row_matrix<rocsparse_double_complex>(fat, false, 3), opt, "fat cd");
    }
}

// csrilu0 and csrildlt0 switch to their binary-search kernels for rows of 512 (ILU,
// wavefront 32) / more than 512 (ILDLT) entries.
TEST_F(Cov70PrecondGeneric, spilu0_spildlt0_csr_binsearch_fat_row)
{
    run_options opt;
    for(const int32_t fat : {520, 600})
    {
        check_csr_matrix<Kind::ilu, float>(
            handle, make_fat_row_matrix<float>(fat, true, 3), opt, "ilu float");
        check_csr_matrix<Kind::ilu, double>(
            handle, make_fat_row_matrix<double>(fat, true, 3), opt, "ilu double");
        check_csr_matrix<Kind::ilu, rocsparse_double_complex>(
            handle, make_fat_row_matrix<rocsparse_double_complex>(fat, true, 3), opt, "ilu cd");
        check_csr_matrix<Kind::ildlt, float>(
            handle, make_fat_row_matrix<float>(fat, true, 3), opt, "ildlt float");
        check_csr_matrix<Kind::ildlt, double>(
            handle, make_fat_row_matrix<double>(fat, true, 3), opt, "ildlt double");
        check_csr_matrix<Kind::ildlt, rocsparse_double_complex>(
            handle, make_fat_row_matrix<rocsparse_double_complex>(fat, true, 3), opt, "ildlt cd");
    }
}

TEST_F(Cov70PrecondGeneric, spildlt0_csr_hash_row_lengths)
{
    // Row lengths up to 512 select the hash kernel; the table size grows with the length.
    // The lengths stay below the table sizes (32, 64, ..., 512): a row that fills the
    // whole shared-memory table makes the lookup of a missing key probe forever (the
    // generator throws for those lengths).
    run_options opt;
    for(const int32_t fat : {8, 31, 33, 63, 65, 127, 129, 255, 257, 511})
    {
        check_csr_matrix<Kind::ildlt, double>(
            handle, make_fat_row_matrix<double>(fat), opt, "fat=" + std::to_string(fat));
        check_csr_matrix<Kind::ildlt, double>(handle,
                                              make_fat_row_matrix<double>(fat, true, 3),
                                              opt,
                                              "band 3 fat=" + std::to_string(fat));
    }
}

TEST_F(Cov70PrecondGeneric, spildlt0_csr_hash_types_and_indices)
{
    run_options opt;
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        check_csr_matrix<Kind::ildlt, T>(handle, make_fat_row_matrix<T>(40), opt, "types");
    });
    check_csr_matrix<Kind::ildlt, float, int64_t, int32_t>(
        handle, make_fat_row_matrix<float>(20), opt, "i64/i32");
    check_csr_matrix<Kind::ildlt, double, int64_t, int64_t>(
        handle, make_fat_row_matrix<double>(20), opt, "i64/i64");
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

        // Valid inputs: the boost tolerance may be given as a float or a double. The
        // library stores the pointer and records the datatype as f64 whatever the
        // size, so a float tolerance would be read as a double: it is only
        // validated here, no factorization of this test uses it.
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

        descr_guard<K> guard(handle);
        auto&          descr = guard.descr;
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
        descr = nullptr;
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
