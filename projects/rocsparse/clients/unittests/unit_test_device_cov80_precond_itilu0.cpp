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

// Second round of iterative ILU(0) (csritilu0) tests:
//  * the options of the compute call that the first round left out (verbose log,
//    correction norms, stopping criterion with an unreachable tolerance, a zero pivot
//    found by the stopping criterion);
//  * the convergence history without the history option;
//  * empty matrices for the buffer size, analysis and compute calls;
//  * invalid enumerations on the extended (x) interface.
// Matrices are diagonally dominant, the tolerances scale with the row length and the
// wavefront size of the handle is never overridden.

#include "unit_test_device_cov70_precond_common.hpp"

#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"

#include <algorithm>
#include <limits>
#include <string>

using namespace cov70;

namespace
{
    // Diagonally dominant band matrix with half bandwidth `w`.
    template <typename T>
    csr_matrix<T> make_band(int32_t m, int32_t w)
    {
        csr_matrix<T> a;
        a.n = m;
        a.ptr.push_back(0);
        for(int32_t i = 0; i < m; ++i)
        {
            for(int32_t j = std::max(0, i - w); j <= std::min(m - 1, i + w); ++j)
            {
                a.ind.push_back(j);
                if(i == j)
                {
                    a.val.push_back(make_value<T>(4.0 + 1.6 * std::min(w, m)));
                }
                else
                {
                    const double r = -0.3 * (1.0 + 0.1 * ((i + 2 * j) % 3));
                    const double c = is_cplx<T> ? 0.05 * ((i - j) % 4) : 0.0;
                    a.val.push_back(make_value<T>(r * (j < i ? 1.1 : 1.0), c));
                }
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        return a;
    }

    template <typename F>
    void for_all_types(F&& f)
    {
        f(float{});
        f(double{});
        f(rocsparse_float_complex{});
        f(rocsparse_double_complex{});
    }

    // Relative tolerance on the factor, see the first round: it scales with the row
    // length (the rounding of the row sums depends on the summation order of the
    // kernels, which differs between wavefront sizes).
    template <typename T>
    double iterative_tolerance(bool stopping, int32_t row_length)
    {
        const double fixed = is_single_precision<T> ? (stopping ? 2.1e-4 : 1.6e-6)
                                                    : (stopping ? 2.4e-11 : 1.0e-14);
        return std::max(fixed, epsilon_of<T>() * static_cast<double>(row_length));
    }

    template <typename T>
    rocsparse_status compute_ex(rocsparse_handle     handle,
                                rocsparse_itilu0_alg alg,
                                rocsparse_int        options,
                                rocsparse_int*       nmaxiter,
                                rocsparse_int        nfreeiter,
                                real_of<T>           tol,
                                rocsparse_int        m,
                                rocsparse_int        nnz,
                                const rocsparse_int* ptr,
                                const rocsparse_int* ind,
                                const T*             val,
                                T*                   ilu0,
                                rocsparse_index_base base,
                                size_t               buffer_size,
                                void*                buffer)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsritilu0_compute_ex(handle,
                                                   alg,
                                                   options,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   tol,
                                                   m,
                                                   nnz,
                                                   ptr,
                                                   ind,
                                                   val,
                                                   ilu0,
                                                   base,
                                                   buffer_size,
                                                   buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsritilu0_compute_ex(handle,
                                                   alg,
                                                   options,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   tol,
                                                   m,
                                                   nnz,
                                                   ptr,
                                                   ind,
                                                   val,
                                                   ilu0,
                                                   base,
                                                   buffer_size,
                                                   buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsritilu0_compute_ex(handle,
                                                   alg,
                                                   options,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   tol,
                                                   m,
                                                   nnz,
                                                   ptr,
                                                   ind,
                                                   val,
                                                   ilu0,
                                                   base,
                                                   buffer_size,
                                                   buffer);
        else
            return rocsparse_zcsritilu0_compute_ex(handle,
                                                   alg,
                                                   options,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   tol,
                                                   m,
                                                   nnz,
                                                   ptr,
                                                   ind,
                                                   val,
                                                   ilu0,
                                                   base,
                                                   buffer_size,
                                                   buffer);
    }

    template <typename T>
    rocsparse_status history(rocsparse_handle     handle,
                             rocsparse_itilu0_alg alg,
                             rocsparse_int*       niter,
                             real_of<T>*          nrms,
                             size_t               buffer_size,
                             void*                buffer)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsritilu0_history(handle, alg, niter, nrms, buffer_size, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsritilu0_history(handle, alg, niter, nrms, buffer_size, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsritilu0_history(handle, alg, niter, nrms, buffer_size, buffer);
        else
            return rocsparse_zcsritilu0_history(handle, alg, niter, nrms, buffer_size, buffer);
    }

    // The four algorithms of the public interface.
    constexpr rocsparse_itilu0_alg algorithms[] = {rocsparse_itilu0_alg_default,
                                                   rocsparse_itilu0_alg_async_inplace,
                                                   rocsparse_itilu0_alg_async_split,
                                                   rocsparse_itilu0_alg_sync_split};

    struct run_out
    {
        rocsparse_status compute_status = rocsparse_status_internal_error;
        rocsparse_int    nmaxiter       = 0;
        std::string      log;
    };

    // Buffer size, analysis and compute of `a` through the public interface with
    // `options`, keeping the buffer for later history queries. The matrix of the
    // device arrays is `a`; `ilu` receives the factor.
    template <typename T>
    struct itilu0_run
    {
        csr_matrix<T>                a;
        device_vector<rocsparse_int> drow;
        device_vector<rocsparse_int> dcol;
        device_vector<T>             dval;
        device_vector<T>             dilu;
        device_vector<char>          buffer;
        size_t                       buffer_size = 0;
        rocsparse_int                options     = 0;
        rocsparse_int                nmaxiter_in = 0;
        rocsparse_itilu0_alg         alg         = rocsparse_itilu0_alg_default;

        void prepare(rocsparse_handle     handle,
                     rocsparse_itilu0_alg alg_,
                     const csr_matrix<T>& a_,
                     rocsparse_int        options_,
                     rocsparse_int        nmaxiter_)
        {
            a           = a_;
            alg         = alg_;
            options     = options_;
            nmaxiter_in = nmaxiter_;
            upload(drow, a.ptr);
            upload(dcol, a.ind);
            upload(dval, a.val);
            upload(dilu, a.val); // starting iterate
            ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);
            ASSERT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                                      alg,
                                                      options,
                                                      nmaxiter_,
                                                      a.n,
                                                      a.nnz(),
                                                      drow,
                                                      dcol,
                                                      rocsparse_index_base_zero,
                                                      dt_of<T>(),
                                                      &buffer_size),
                      rocsparse_status_success);
            upload(buffer, std::vector<char>(buffer_size ? buffer_size : size_t{1}));
            ASSERT_TRUE(buffer.ptr);
            ASSERT_EQ(rocsparse_csritilu0_preprocess(handle,
                                                     alg,
                                                     options,
                                                     nmaxiter_,
                                                     a.n,
                                                     a.nnz(),
                                                     drow,
                                                     dcol,
                                                     rocsparse_index_base_zero,
                                                     dt_of<T>(),
                                                     buffer_size,
                                                     buffer.ptr),
                      rocsparse_status_success);
        }

        run_out compute(rocsparse_handle handle, real_of<T> tol, rocsparse_int nfreeiter = 0)
        {
            run_out       out;
            rocsparse_int nmaxiter = nmaxiter_in;
            const bool    verbose  = (options & rocsparse_itilu0_option_verbose) != 0;
            if(verbose)
                testing::internal::CaptureStdout();
            out.compute_status = compute_ex<T>(handle,
                                               alg,
                                               options,
                                               &nmaxiter,
                                               nfreeiter,
                                               tol,
                                               a.n,
                                               a.nnz(),
                                               drow,
                                               dcol,
                                               dval,
                                               dilu,
                                               rocsparse_index_base_zero,
                                               buffer_size,
                                               buffer.ptr);
            if(verbose)
                out.log = testing::internal::GetCapturedStdout();
            (void)hipDeviceSynchronize();
            out.nmaxiter = nmaxiter;
            return out;
        }
    };

    constexpr rocsparse_int nrm_residual   = rocsparse_itilu0_option_compute_nrm_residual;
    constexpr rocsparse_int nrm_correction = rocsparse_itilu0_option_compute_nrm_correction;
    constexpr rocsparse_int history_opt    = rocsparse_itilu0_option_convergence_history;
    constexpr rocsparse_int stopping       = rocsparse_itilu0_option_stopping_criteria;
    constexpr rocsparse_int verbose_opt    = rocsparse_itilu0_option_verbose;

    template <typename T>
    real_of<T> default_tol()
    {
        return std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
    }
}

namespace
{
    // L (strictly lower, by rows), U (strictly upper, by columns) and D of a CSR matrix,
    // with the position of every entry in the CSR arrays.
    template <typename T>
    struct split_matrix
    {
        std::vector<int32_t> lptr, lind, lpos;
        std::vector<int32_t> uptr, uind, upos;
        std::vector<int32_t> dpos;

        explicit split_matrix(const csr_matrix<T>& a)
        {
            const int32_t n = a.n;
            lptr.assign(1, 0);
            dpos.assign(static_cast<size_t>(n), -1);
            std::vector<std::vector<std::pair<int32_t, int32_t>>> cols(static_cast<size_t>(n));
            for(int32_t i = 0; i < n; ++i)
            {
                for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                {
                    const int32_t j = a.ind[p];
                    if(j < i)
                    {
                        lind.push_back(j);
                        lpos.push_back(p);
                    }
                    else if(j > i)
                    {
                        cols[static_cast<size_t>(j)].push_back({i, p});
                    }
                    else
                    {
                        dpos[static_cast<size_t>(i)] = p;
                    }
                }
                lptr.push_back(static_cast<int32_t>(lind.size()));
            }
            uptr.assign(1, 0);
            for(int32_t j = 0; j < n; ++j)
            {
                for(const auto& e : cols[static_cast<size_t>(j)])
                {
                    uind.push_back(e.first);
                    upos.push_back(e.second);
                }
                uptr.push_back(static_cast<int32_t>(uind.size()));
            }
        }
    };

    template <typename T>
    std::vector<T> gather(const std::vector<T>& v, const std::vector<int32_t>& pos)
    {
        std::vector<T> out;
        out.reserve(pos.size());
        for(const int32_t p : pos)
            out.push_back(v[static_cast<size_t>(p)]);
        return out;
    }

    // The sync_split algorithm of the extended interface: buffer size, analysis and
    // compute with hand-built L, U and D, checked against a host ILU(0). The residual norm
    // is always requested.
    template <typename T>
    void check_sync_split_x(rocsparse_handle     handle,
                            const csr_matrix<T>& a,
                            rocsparse_int        options,
                            rocsparse_int        nmaxiter,
                            const std::string&   what,
                            rocsparse_itilu0_alg alg      = rocsparse_itilu0_alg_sync_split,
                            bool                 poisoned = false)
    {
        SCOPED_TRACE(what);
        using I = int32_t;
        using J = int32_t;

        const split_matrix<T> split(a);
        const I               m    = a.n;
        const I               nnz  = a.nnz();
        const I               lnnz = static_cast<I>(split.lind.size());
        const I               unnz = static_cast<I>(split.uind.size());

        device_vector<I> dptr{a.ptr};
        device_vector<J> dind{a.ind};
        device_vector<T> dval{a.val};
        device_vector<I> dlptr{split.lptr};
        device_vector<J> dlind{split.lind};
        device_vector<T> dlval{gather(a.val, split.lpos)};
        device_vector<I> duptr{split.uptr};
        device_vector<J> duind{split.uind};
        device_vector<T> duval{gather(a.val, split.upos)};
        device_vector<T> ddval{gather(a.val, split.dpos)};
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dlptr.ptr && dlind.ptr && dlval.ptr
                    && duptr.ptr && duind.ptr && duval.ptr && ddval.ptr);

        // The diagonal is stored apart (D) only when both factors have a unit diagonal.
        const auto ldiag = rocsparse_diag_type_unit;
        const auto udiag = rocsparse_diag_type_unit;
        const auto ldir  = rocsparse_direction_row;
        const auto udir  = rocsparse_direction_column;
        const auto base  = rocsparse_index_base_zero;

        size_t buffer_size = 0;
        ASSERT_EQ((rocsparse::csritilu0x_buffer_size_impl<I, J>(handle,
                                                                alg,
                                                                options,
                                                                nmaxiter,
                                                                m,
                                                                nnz,
                                                                dptr.ptr,
                                                                dptr.ptr + 1,
                                                                dind.ptr,
                                                                base,
                                                                ldiag,
                                                                ldir,
                                                                udiag,
                                                                udir,
                                                                dt_of<T>(),
                                                                &buffer_size)),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ((rocsparse::csritilu0x_preprocess_impl<I, J>(handle,
                                                               alg,
                                                               options,
                                                               nmaxiter,
                                                               m,
                                                               nnz,
                                                               dptr.ptr,
                                                               dptr.ptr + 1,
                                                               dind.ptr,
                                                               base,
                                                               ldiag,
                                                               ldir,
                                                               lnnz,
                                                               dlptr.ptr,
                                                               dlptr.ptr + 1,
                                                               dlind.ptr,
                                                               base,
                                                               udiag,
                                                               udir,
                                                               unnz,
                                                               duptr.ptr,
                                                               duptr.ptr + 1,
                                                               duind.ptr,
                                                               base,
                                                               dt_of<T>(),
                                                               buffer_size,
                                                               buffer.ptr)),
                  rocsparse_status_success);

        rocsparse_int nsweeps = nmaxiter;
        const bool    verbose = (options & verbose_opt) != 0;
        if(verbose)
            testing::internal::CaptureStdout();
        const auto  status = rocsparse::csritilu0x_compute_impl<T, I, J>(handle,
                                                                        alg,
                                                                        options,
                                                                        &nsweeps,
                                                                        0,
                                                                        default_tol<T>(),
                                                                        m,
                                                                        nnz,
                                                                        dptr.ptr,
                                                                        dptr.ptr + 1,
                                                                        dind.ptr,
                                                                        dval.ptr,
                                                                        base,
                                                                        ldiag,
                                                                        ldir,
                                                                        lnnz,
                                                                        dlptr.ptr,
                                                                        dlptr.ptr + 1,
                                                                        dlind.ptr,
                                                                        dlval.ptr,
                                                                        base,
                                                                        udiag,
                                                                        udir,
                                                                        unnz,
                                                                        duptr.ptr,
                                                                        duptr.ptr + 1,
                                                                        duind.ptr,
                                                                        duval.ptr,
                                                                        base,
                                                                        ddval.ptr,
                                                                        buffer_size,
                                                                        buffer.ptr);
        std::string log_text;
        if(verbose)
            log_text = testing::internal::GetCapturedStdout();
        if(poisoned)
        {
            // Non-finite data: the iteration may notice and report a zero pivot, or end at the
            // sweep limit. It must not fault.
            EXPECT_TRUE(status == rocsparse_status_success || status == rocsparse_status_zero_pivot)
                << status;
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            return;
        }
        ASSERT_EQ(status, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_GE(nsweeps, 1);
        EXPECT_LE(nsweeps, nmaxiter);
        if(verbose)
            EXPECT_NE(log_text.find("iter"), std::string::npos);

        const std::vector<T> ref = host_ilu0(a);
        const double rel_tol = iterative_tolerance<T>((options & stopping) != 0, max_row_length(a));
        expect_close(to_host(dlval), gather(ref, split.lpos), rel_tol);
        expect_close(to_host(duval), gather(ref, split.upos), rel_tol);
        expect_close(to_host(ddval), gather(ref, split.dpos), rel_tol);
    }

    void check_extended_interface_arguments(rocsparse_handle handle, rocsparse_itilu0_alg alg)
    {
        using I = int32_t;
        using J = int32_t;

        // Tridiagonal 4 x 4 matrix with its strictly lower part by rows (L) and strictly
        // upper part by columns (U).
        const std::vector<I>      ptr{0, 2, 5, 8, 10};
        const std::vector<J>      ind{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
        const std::vector<I>      lptr{0, 0, 1, 2, 3};
        const std::vector<J>      lind{0, 1, 2};
        const std::vector<I>      uptr{0, 0, 1, 2, 3};
        const std::vector<J>      uind{0, 1, 2};
        const std::vector<double> val{4, -1, -1, 4, -1, -1, 4, -1, -1, 4};
        const std::vector<double> lval{-0.25, -0.25, -0.25};
        const std::vector<double> uval{-1, -1, -1};
        const std::vector<double> dval{4, 4, 4, 4};

        device_vector<I>      dptr{ptr};
        device_vector<J>      dind{ind};
        device_vector<double> dv{val};
        device_vector<I>      dlptr{lptr};
        device_vector<J>      dlind{lind};
        device_vector<double> dlval{lval};
        device_vector<I>      duptr{uptr};
        device_vector<J>      duind{uind};
        device_vector<double> duval{uval};
        device_vector<double> ddval{dval};
        ASSERT_TRUE(dptr.ptr && dind.ptr && dv.ptr && dlptr.ptr && dlind.ptr && dlval.ptr
                    && duptr.ptr && duind.ptr && duval.ptr && ddval.ptr);

        const auto ldiag = rocsparse_diag_type_unit;
        const auto udiag = rocsparse_diag_type_non_unit;
        const auto ldir  = rocsparse_direction_row;
        const auto udir  = rocsparse_direction_column;
        const auto base  = rocsparse_index_base_zero;
        const I    m = 4, nnz = 10, lnnz = 3, unnz = 3;

        const auto bad_diag = static_cast<rocsparse_diag_type>(-1);
        const auto bad_dir  = static_cast<rocsparse_direction>(-1);
        const auto bad_base = static_cast<rocsparse_index_base>(-1);

        auto buffer_size_call = [&](rocsparse_index_base b,
                                    rocsparse_diag_type  ld,
                                    rocsparse_direction  ldr,
                                    rocsparse_diag_type  ud,
                                    rocsparse_direction  udr,
                                    size_t*              size) {
            return rocsparse::csritilu0x_buffer_size_impl<I, J>(handle,
                                                                alg,
                                                                0,
                                                                10,
                                                                m,
                                                                nnz,
                                                                dptr.ptr,
                                                                dptr.ptr + 1,
                                                                dind.ptr,
                                                                b,
                                                                ld,
                                                                ldr,
                                                                ud,
                                                                udr,
                                                                rocsparse_datatype_f64_r,
                                                                size);
        };

        size_t buffer_size = 0;
        ASSERT_EQ(buffer_size_call(base, ldiag, ldir, udiag, udir, &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);

        // Buffer size.
        size_t tmp = 0;
        EXPECT_NE(buffer_size_call(bad_base, ldiag, ldir, udiag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, bad_diag, ldir, udiag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, ldiag, bad_dir, udiag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, ldiag, ldir, bad_diag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, ldiag, ldir, udiag, bad_dir, &tmp),
                  rocsparse_status_success);

        // Analysis.
        auto preprocess_call = [&](rocsparse_index_base b,
                                   rocsparse_index_base lb,
                                   rocsparse_index_base ub,
                                   rocsparse_diag_type  ld,
                                   rocsparse_direction  ldr,
                                   rocsparse_diag_type  ud,
                                   rocsparse_direction  udr,
                                   const J*             lind_ptr,
                                   const J*             uind_ptr) {
            return rocsparse::csritilu0x_preprocess_impl<I, J>(handle,
                                                               alg,
                                                               0,
                                                               10,
                                                               m,
                                                               nnz,
                                                               dptr.ptr,
                                                               dptr.ptr + 1,
                                                               dind.ptr,
                                                               b,
                                                               ld,
                                                               ldr,
                                                               lnnz,
                                                               dlptr.ptr,
                                                               dlptr.ptr + 1,
                                                               lind_ptr,
                                                               lb,
                                                               ud,
                                                               udr,
                                                               unnz,
                                                               duptr.ptr,
                                                               duptr.ptr + 1,
                                                               uind_ptr,
                                                               ub,
                                                               rocsparse_datatype_f64_r,
                                                               buffer_size,
                                                               buffer.ptr);
        };
        EXPECT_NE(
            preprocess_call(bad_base, base, base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, bad_base, base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, bad_base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, bad_diag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, ldiag, bad_dir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, ldiag, ldir, bad_diag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, ldiag, ldir, udiag, bad_dir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        // A non-empty U or L needs its column indices.
        EXPECT_EQ(preprocess_call(base, base, base, ldiag, ldir, udiag, udir, dlind.ptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(preprocess_call(base, base, base, ldiag, ldir, udiag, udir, nullptr, duind.ptr),
                  rocsparse_status_invalid_pointer);

        // Compute (a valid analysis comes first for the calls that get past the checks).
        ASSERT_EQ(preprocess_call(base, base, base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
                  rocsparse_status_success);
        auto compute_call = [&](rocsparse_index_base b,
                                rocsparse_index_base lb,
                                rocsparse_index_base ub,
                                rocsparse_diag_type  ld,
                                rocsparse_direction  ldr,
                                rocsparse_diag_type  ud,
                                rocsparse_direction  udr,
                                I                    nz,
                                const J*             lind_ptr,
                                const J*             uind_ptr) {
            rocsparse_int nsweeps = 5;
            return rocsparse::csritilu0x_compute_impl<double, I, J>(
                handle,
                alg,
                rocsparse_itilu0_option_compute_nrm_residual,
                &nsweeps,
                0,
                1.0e-12,
                m,
                nz,
                dptr.ptr,
                dptr.ptr + 1,
                dind.ptr,
                dv.ptr,
                b,
                ld,
                ldr,
                lnnz,
                dlptr.ptr,
                dlptr.ptr + 1,
                lind_ptr,
                dlval.ptr,
                lb,
                ud,
                udr,
                unnz,
                duptr.ptr,
                duptr.ptr + 1,
                uind_ptr,
                duval.ptr,
                ub,
                ddval.ptr,
                buffer_size,
                buffer.ptr);
        };
        EXPECT_NE(
            compute_call(bad_base, base, base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, bad_base, base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, bad_base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, bad_diag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, ldiag, bad_dir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, ldiag, ldir, bad_diag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, ldiag, ldir, udiag, bad_dir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_EQ(compute_call(base, base, base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(compute_call(base, base, base, ldiag, ldir, udiag, udir, nnz, nullptr, duind.ptr),
                  rocsparse_status_invalid_pointer);
        // Only the row-wise L and the column-wise U are supported.
        EXPECT_NE(compute_call(base,
                               base,
                               base,
                               ldiag,
                               rocsparse_direction_column,
                               udiag,
                               udir,
                               nnz,
                               dlind.ptr,
                               duind.ptr),
                  rocsparse_status_success);
        EXPECT_NE(compute_call(base,
                               base,
                               base,
                               ldiag,
                               ldir,
                               udiag,
                               rocsparse_direction_row,
                               nnz,
                               dlind.ptr,
                               duind.ptr),
                  rocsparse_status_success);
    }
}

class Cov80PrecondItilu0 : public HandleTest
{
};

// The verbose option prints one table per sweep to stdout; with the residual (and the
// correction) norms requested the table has the matching columns. The factor and the
// history are still the ones of a silent run.
TEST_F(Cov80PrecondItilu0, verbose_log_and_norm_history)
{
    for(const auto alg : algorithms)
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            SCOPED_TRACE("alg " + std::to_string(alg));
            const auto    a    = make_band<T>(24, 3);
            const auto    opts = verbose_opt | nrm_residual | nrm_correction | history_opt;
            itilu0_run<T> r;
            r.prepare(handle, alg, a, opts, 30);
            const auto out = r.compute(handle, default_tol<T>());
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            EXPECT_NE(out.log.find("iter"), std::string::npos);
            EXPECT_NE(out.log.find("residual"), std::string::npos);
            EXPECT_GE(out.nmaxiter, 1);
            EXPECT_LE(out.nmaxiter, 30);
            expect_close(to_host(r.dilu), host_ilu0(a), iterative_tolerance<T>(false, 7));

            rocsparse_int           niter = -1;
            std::vector<real_of<T>> log(2 * 30, real_of<T>(-1));
            ASSERT_EQ(history<T>(handle, alg, &niter, log.data(), r.buffer_size, r.buffer.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            ASSERT_GE(niter, 1);
            ASSERT_LE(niter, 30);
            // The residual norms follow the niter entries reserved for the corrections.
            for(rocsparse_int i = 0; i < niter; ++i)
            {
                EXPECT_TRUE(std::isfinite(log[niter + i])) << "iteration " << i;
                EXPECT_GE(log[niter + i], real_of<T>(0)) << "iteration " << i;
            }
        });
    }
}

// The stopping criterion with a tolerance the residual cannot reach in a sensible
// number of sweeps: the iteration ends before the sweep limit only through the
// stagnation test, or runs to the limit. Either way it succeeds with a converged factor.
TEST_F(Cov80PrecondItilu0, stopping_criterion_with_unreachable_tolerance)
{
    for(const auto alg : algorithms)
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            SCOPED_TRACE("alg " + std::to_string(alg));
            const auto a = make_band<T>(40, 2);
            for(const bool verbose : {false, true})
            {
                itilu0_run<T> r;
                r.prepare(handle,
                          alg,
                          a,
                          stopping | nrm_residual | nrm_correction | (verbose ? verbose_opt : 0),
                          200);
                const auto out = r.compute(handle, real_of<T>(1.0e-30));
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                EXPECT_GE(out.nmaxiter, 1);
                EXPECT_LE(out.nmaxiter, 200);
                expect_close(to_host(r.dilu), host_ilu0(a), iterative_tolerance<T>(true, 5));
            }
        });
    }
}

// A zero diagonal entry makes the first pivot numerically zero. The sweeps divide by it
// and the factor is not finite; the stopping criterion may notice (zero pivot) or the
// run ends at the sweep limit. Nothing may hang or fault, and the call must not claim
// anything else.
TEST_F(Cov80PrecondItilu0, numerically_zero_pivot_with_stopping_criterion)
{
    for(const auto alg : algorithms)
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        auto a   = make_band<double>(16, 2);
        a.val[0] = 0.0; // A(0, 0)
        itilu0_run<double> r;
        r.prepare(handle, alg, a, stopping | nrm_residual | nrm_correction, 20);
        const auto out = r.compute(handle, 1.0e-12);
        EXPECT_TRUE(out.compute_status == rocsparse_status_zero_pivot
                    || out.compute_status == rocsparse_status_success)
            << out.compute_status;
        EXPECT_GE(out.nmaxiter, 1);
        EXPECT_LE(out.nmaxiter, 20);
    }
}

// The history can only be queried if the history option was given to the compute
// call.
TEST_F(Cov80PrecondItilu0, history_without_the_history_option)
{
    for(const auto alg : algorithms)
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        const auto         a = make_band<double>(16, 2);
        itilu0_run<double> r;
        r.prepare(handle, alg, a, nrm_residual, 10);
        const auto out = r.compute(handle, 1.0e-12);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);

        rocsparse_int       niter = -1;
        std::vector<double> log(20, -1.0);
        testing::internal::CaptureStderr();
        const rocsparse_status status
            = history<double>(handle, alg, &niter, log.data(), r.buffer_size, r.buffer.ptr);
        (void)testing::internal::GetCapturedStderr();
        // Either the library reports the missing history or it has nothing to report.
        EXPECT_TRUE(status == rocsparse_status_internal_error
                    || (status == rocsparse_status_success && niter == 0))
            << "status " << status << " niter " << niter;
    }
}

// Quick returns on empty problems and argument checks of the buffer size and the
// analysis.
TEST_F(Cov80PrecondItilu0, empty_problems)
{
    const rocsparse_index_base   base = rocsparse_index_base_zero;
    device_vector<rocsparse_int> dptr{std::vector<rocsparse_int>{0, 0, 0, 0}};
    device_vector<rocsparse_int> dind{std::vector<rocsparse_int>{0}};
    ASSERT_TRUE(dptr.ptr && dind.ptr);
    for(const auto alg : algorithms)
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        size_t buffer_size = 12345;
        // No rows: nothing to do.
        EXPECT_EQ(
            rocsparse_csritilu0_buffer_size(
                handle, alg, 0, 10, 0, 0, dptr, dind, base, rocsparse_datatype_f64_r, &buffer_size),
            rocsparse_status_success);
        EXPECT_EQ(buffer_size, 0u);
        EXPECT_EQ(
            rocsparse_csritilu0_preprocess(
                handle, alg, 0, 10, 0, 0, dptr, dind, base, rocsparse_datatype_f64_r, 0, nullptr),
            rocsparse_status_success);

        // Rows but no entries: the diagonal is missing (documented as a zero pivot).
        buffer_size = 12345;
        EXPECT_EQ(
            rocsparse_csritilu0_buffer_size(
                handle, alg, 0, 10, 3, 0, dptr, dind, base, rocsparse_datatype_f64_r, &buffer_size),
            rocsparse_status_zero_pivot);
        EXPECT_EQ(
            rocsparse_csritilu0_preprocess(
                handle, alg, 0, 10, 3, 0, dptr, dind, base, rocsparse_datatype_f64_r, 0, nullptr),
            rocsparse_status_zero_pivot);

        // Invalid enumerations.
        EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                                  alg,
                                                  0,
                                                  10,
                                                  3,
                                                  1,
                                                  dptr,
                                                  dind,
                                                  static_cast<rocsparse_index_base>(-1),
                                                  rocsparse_datatype_f64_r,
                                                  &buffer_size),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_csritilu0_preprocess(handle,
                                                 alg,
                                                 0,
                                                 10,
                                                 3,
                                                 1,
                                                 dptr,
                                                 dind,
                                                 static_cast<rocsparse_index_base>(-1),
                                                 rocsparse_datatype_f64_r,
                                                 0,
                                                 nullptr),
                  rocsparse_status_invalid_value);
    }
}

// The extended interface checks its enumerations (the public interface fixes them).
// A bad enumeration, a missing pointer for a non-empty L or U, or an unsupported
// direction must be rejected before any work is done.
TEST_F(Cov80PrecondItilu0, extended_interface_rejects_invalid_arguments)
{
    for(const auto alg : {rocsparse_itilu0_alg_async_split, rocsparse_itilu0_alg_sync_split})
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        check_extended_interface_arguments(handle, alg);
    }
}

TEST_F(Cov80PrecondItilu0, extended_interface_buffer_size_empty_and_unsupported_types)
{
    using I = int32_t;
    using J = int32_t;
    const std::vector<I> ptr{0, 2, 5, 8, 10};
    device_vector<I>     dptr{ptr};
    device_vector<J>     dind{std::vector<J>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3}};
    ASSERT_TRUE(dptr.ptr && dind.ptr);
    for(const auto alg : {rocsparse_itilu0_alg_async_split, rocsparse_itilu0_alg_sync_split})
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        auto size_of = [&](I m, I nnz, rocsparse_datatype dtype, size_t* size) {
            return rocsparse::csritilu0x_buffer_size_impl<I, J>(
                handle,
                alg,
                rocsparse_itilu0_option_compute_nrm_residual,
                10,
                m,
                nnz,
                dptr.ptr,
                dptr.ptr + 1,
                dind.ptr,
                rocsparse_index_base_zero,
                rocsparse_diag_type_unit,
                rocsparse_direction_row,
                rocsparse_diag_type_non_unit,
                rocsparse_direction_column,
                dtype,
                size);
        };
        size_t size = 777;
        // Nothing to factorize: no buffer is needed.
        EXPECT_EQ(size_of(0, 0, rocsparse_datatype_f64_r, &size), rocsparse_status_success);
        EXPECT_EQ(size, 0u);
        // A non-empty matrix without entries has no diagonal: only require a clean return.
        size = 777;
        (void)size_of(4, 0, rocsparse_datatype_f64_r, &size);
        // The four floating point types need a buffer, the other types are not supported.
        for(const auto dtype : {rocsparse_datatype_f32_r,
                                rocsparse_datatype_f64_r,
                                rocsparse_datatype_f32_c,
                                rocsparse_datatype_f64_c})
        {
            size = 0;
            EXPECT_EQ(size_of(4, 10, dtype, &size), rocsparse_status_success);
            EXPECT_GT(size, 0u);
        }
        for(const auto dtype : {rocsparse_datatype_i8_r,
                                rocsparse_datatype_u8_r,
                                rocsparse_datatype_i32_r,
                                rocsparse_datatype_u32_r,
                                rocsparse_datatype_f16_r,
                                rocsparse_datatype_bf16_r})
        {
            EXPECT_NE(size_of(4, 10, dtype, &size), rocsparse_status_success);
        }
    }
}

TEST_F(Cov80PrecondItilu0, sync_split_extended_interface_sweeps)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        // Fixed sweeps, then the stopping criterion with the verbose table and the
        // correction norms.
        check_sync_split_x<T>(handle, make_band<T>(24, 3), nrm_residual, 60, "fixed");
        check_sync_split_x<T>(handle,
                              make_band<T>(24, 3),
                              stopping | nrm_residual | nrm_correction | history_opt | verbose_opt,
                              60,
                              "stopping");
    });
    // The sweep kernel width follows the mean number of entries per row.
    for(const auto& [w, m] :
        {std::pair<int32_t, int32_t>{1, 64}, {2, 64}, {6, 64}, {12, 64}, {24, 64}, {100, 100}})
    {
        check_sync_split_x<double>(
            handle, make_band<double>(m, w), nrm_residual, 80, "w=" + std::to_string(w));
    }
}

namespace
{
    // Makes the handle report a tiny internal buffer, so that every routine that uses it
    // as temporary storage has to allocate its own.
    struct shrunk_buffer
    {
        rocsparse_handle handle;
        size_t           saved;

        shrunk_buffer(rocsparse_handle h, size_t size)
            : handle(h)
            , saved(h->buffer_size)
        {
            handle->buffer_size = size;
        }

        ~shrunk_buffer()
        {
            handle->buffer_size = saved;
        }

        shrunk_buffer(const shrunk_buffer&) = delete;
        shrunk_buffer& operator=(const shrunk_buffer&) = delete;
    };
}

// The factorization does not depend on where the temporary storage of the library lives.
TEST_F(Cov80PrecondItilu0, temporary_storage_when_the_handle_buffer_is_small)
{
    shrunk_buffer shrink(handle, 1);
    for(const auto alg : algorithms)
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            SCOPED_TRACE("alg " + std::to_string(alg));
            const auto    a = make_band<T>(40, 4);
            itilu0_run<T> r;
            r.prepare(handle, alg, a, nrm_residual, 40);
            const auto out = r.compute(handle, default_tol<T>());
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            expect_close(to_host(r.dilu), host_ilu0(a), iterative_tolerance<T>(false, 9));
        });
    }
}

// The extended interface only implements the split algorithms; the others are refused
// when computing, and the unsupported value types are refused when sizing the buffer.
TEST_F(Cov80PrecondItilu0, extended_interface_with_the_unimplemented_algorithms)
{
    using I = int32_t;
    using J = int32_t;
    const std::vector<I>  ptr{0, 2, 5, 8, 10};
    const std::vector<J>  ind{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    const std::vector<I>  lptr{0, 0, 1, 2, 3};
    const std::vector<J>  lind{0, 1, 2};
    const std::vector<I>  uptr{0, 0, 1, 2, 3};
    const std::vector<J>  uind{0, 1, 2};
    device_vector<I>      dptr{ptr};
    device_vector<J>      dind{ind};
    device_vector<I>      dlptr{lptr};
    device_vector<J>      dlind{lind};
    device_vector<I>      duptr{uptr};
    device_vector<J>      duind{uind};
    device_vector<double> dval{std::vector<double>{4, -1, -1, 4, -1, -1, 4, -1, -1, 4}};
    device_vector<double> dlval{std::vector<double>(3, 0.0)};
    device_vector<double> duval{std::vector<double>(3, 0.0)};
    device_vector<double> ddval{std::vector<double>(4, 0.0)};
    ASSERT_TRUE(dptr.ptr && dind.ptr && dlptr.ptr && dlind.ptr && duptr.ptr && duind.ptr && dval.ptr
                && dlval.ptr && duval.ptr && ddval.ptr);

    for(const auto alg : {rocsparse_itilu0_alg_default, rocsparse_itilu0_alg_async_inplace})
    {
        SCOPED_TRACE("alg " + std::to_string(alg));

        size_t size = 0;
        for(const auto dtype : {rocsparse_datatype_i8_r,
                                rocsparse_datatype_u8_r,
                                rocsparse_datatype_i32_r,
                                rocsparse_datatype_u32_r,
                                rocsparse_datatype_f16_r,
                                rocsparse_datatype_bf16_r})
        {
            EXPECT_NE((rocsparse::csritilu0x_buffer_size_impl<I, J>(
                          handle,
                          alg,
                          rocsparse_itilu0_option_compute_nrm_residual,
                          10,
                          4,
                          10,
                          dptr.ptr,
                          dptr.ptr + 1,
                          dind.ptr,
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          dtype,
                          &size)),
                      rocsparse_status_success);
        }

        rocsparse_int nsweeps = 10;
        EXPECT_NE((rocsparse::csritilu0x_compute_impl<double, I, J>(handle,
                                                                    alg,
                                                                    0,
                                                                    &nsweeps,
                                                                    0,
                                                                    1.0e-12,
                                                                    4,
                                                                    10,
                                                                    dptr.ptr,
                                                                    dptr.ptr + 1,
                                                                    dind.ptr,
                                                                    dval.ptr,
                                                                    rocsparse_index_base_zero,
                                                                    rocsparse_diag_type_unit,
                                                                    rocsparse_direction_row,
                                                                    3,
                                                                    dlptr.ptr,
                                                                    dlptr.ptr + 1,
                                                                    dlind.ptr,
                                                                    dlval.ptr,
                                                                    rocsparse_index_base_zero,
                                                                    rocsparse_diag_type_unit,
                                                                    rocsparse_direction_column,
                                                                    3,
                                                                    duptr.ptr,
                                                                    duptr.ptr + 1,
                                                                    duind.ptr,
                                                                    duval.ptr,
                                                                    rocsparse_index_base_zero,
                                                                    ddval.ptr,
                                                                    0,
                                                                    nullptr)),
                  rocsparse_status_success);
    }
}

// Non-finite entries make the norms used by the stopping criterion infinite or undefined.
// The iteration may report a zero pivot or end at the sweep limit, but it must not fault.
TEST_F(Cov80PrecondItilu0, non_finite_input_with_stopping_criterion)
{
    const double infinity = std::numeric_limits<double>::infinity();
    for(const auto options : {stopping | nrm_residual,
                              stopping | nrm_residual | nrm_correction,
                              stopping | nrm_correction | nrm_residual | history_opt})
    {
        for(const auto alg : algorithms)
        {
            SCOPED_TRACE("alg " + std::to_string(alg) + " options " + std::to_string(options));
            auto a   = make_band<double>(16, 2);
            a.val[5] = infinity;
            itilu0_run<double> r;
            r.prepare(handle, alg, a, options, 20);
            const auto out = r.compute(handle, 1.0e-12);
            EXPECT_TRUE(out.compute_status == rocsparse_status_zero_pivot
                        || out.compute_status == rocsparse_status_success)
                << out.compute_status;
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        for(const auto alg : {rocsparse_itilu0_alg_async_split, rocsparse_itilu0_alg_sync_split})
        {
            auto a   = make_band<double>(24, 3);
            a.val[7] = infinity;
            check_sync_split_x<double>(
                handle, a, options, 20, "x interface with an infinite entry", alg, true);
        }
    }
}
