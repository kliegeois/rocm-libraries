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

// Iterative ILU(0) (csritilu0) coverage:
//  * the async_split driver of the "x" interface (also reached by
//    rocsparse_Xcsritilu0_compute(_ex) with rocsparse_itilu0_alg_async_split),
//    driven through csritilu0x_compute_impl with hand-built L (rows), U
//    (columns) and D structures;
//  * the correction norm of the async_split and sync_split algorithms on matrices
//    with an empty triangle, through the public interface.

#include "unit_test_device_precond_common.hpp"

#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_history.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

using namespace precond_ut;

namespace
{
    // Diagonally dominant band matrix with half bandwidth `w` (w >= m gives a
    // dense matrix). The strong diagonal makes the fixed-point iterations converge fast.
    template <typename T>
    csr_matrix<T> make_band_matrix(int32_t m, int32_t w)
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

    // ------------------------------------------------------------------
    // Public (deprecated) compute entry points
    // ------------------------------------------------------------------
    template <typename T, typename... A>
    rocsparse_status history(A&&... args)
    {
        return PRECOND_CALL(T, csritilu0_history, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status compute_ex(A&&... args)
    {
        return PRECOND_CALL(T, csritilu0_compute_ex, std::forward<A>(args)...);
    }

    // The sweeps stop at a fixed count or on the residual criterion; the error left in
    // the factor differs (see the measured values in the tests).
    constexpr bool is_stopping(rocsparse_int options)
    {
        return (options & rocsparse_itilu0_option_stopping_criteria) != 0;
    }

    // Relative (per entry) tolerance on the factor, about 10x the largest error measured on
    // gfx1201 over every algorithm, type and matrix of this file:
    //  * fixed sweep count (60-80 sweeps, the iteration has converged to rounding):
    //    float/cfloat <= 1.6e-7, double/cdouble <= 8.7e-16 -> 1.6e-6, 1e-14 (the double
    //    bound is only 10x the rounding level of a direct factorization);
    //  * stopping criterion (tol 1e-6 float, 1e-12 double): the error is the residual
    //    left at the stop: float <= 2.1e-5 (async_inplace; <= 5e-7 for the others),
    //    double <= 2.4e-12 -> 2.1e-4, 2.4e-11.
    //
    // The converged factor still differs from the host one by the rounding of the
    // row sums, which grows with the row length and depends on the summation order
    // (wave32 and wave64 kernels, async scheduling), so the bound never goes below
    // eps * row_length (dense 100 x 100 rows: 2.2e-14 in double against a measured
    // 1.5e-15 on gfx1201).
    template <typename T>
    double iterative_tolerance(bool stopping, int32_t row_length)
    {
        const double fixed = is_single_precision<T> ? (stopping ? 2.1e-4 : 1.6e-6)
                                                    : (stopping ? 2.4e-11 : 1.0e-14);
        return std::max(fixed, epsilon_of<T>() * static_cast<double>(row_length));
    }

    // ------------------------------------------------------------------
    // x interface: async_split driven directly
    // ------------------------------------------------------------------
    // L (strictly lower, by rows), U (strictly upper, by columns) and D of a CSR
    // matrix, with the position of every entry in the CSR arrays.
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

    struct x_options
    {
        rocsparse_int options   = rocsparse_itilu0_option_compute_nrm_residual;
        rocsparse_int nmaxiter  = 80;
        rocsparse_int nfreeiter = 0;
    };

    template <typename T>
    void check_async_split_x(rocsparse_handle     handle,
                             const csr_matrix<T>& a,
                             const x_options&     opt,
                             const std::string&   what)
    {
        SCOPED_TRACE(what);
        using I = rocsparse_int;
        using J = rocsparse_int;

        const split_matrix<T> split(a);
        const I               m    = a.n;
        const I               nnz  = a.nnz();
        const I               lnnz = static_cast<I>(split.lind.size());
        const I               unnz = static_cast<I>(split.uind.size());
        // The starting iterate is the matching part of A.
        const std::vector<T> l0 = gather(a.val, split.lpos);
        const std::vector<T> u0 = gather(a.val, split.upos);
        const std::vector<T> d0 = gather(a.val, split.dpos);

        device_vector<I> dptr{as_rocsparse_int(a.ptr)};
        device_vector<J> dind{as_rocsparse_int(a.ind)};
        device_vector<T> dval{a.val};
        device_vector<I> dlptr{as_rocsparse_int(split.lptr)};
        device_vector<J> dlind{as_rocsparse_int(split.lind)};
        device_vector<T> dlval{l0};
        device_vector<I> duptr{as_rocsparse_int(split.uptr)};
        device_vector<J> duind{as_rocsparse_int(split.uind)};
        device_vector<T> duval{u0};
        device_vector<T> ddval{d0};
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dlptr.ptr && dlind.ptr && dlval.ptr
                    && duptr.ptr && duind.ptr && duval.ptr && ddval.ptr);

        const auto ldiag = rocsparse_diag_type_unit;
        const auto udiag = rocsparse_diag_type_non_unit;
        const auto ldir  = rocsparse_direction_row;
        const auto udir  = rocsparse_direction_column;
        const auto base  = rocsparse_index_base_zero;
        const auto alg   = rocsparse_itilu0_alg_async_split;

        size_t buffer_size = 0;
        ASSERT_EQ((rocsparse::csritilu0x_buffer_size_impl<I, J>(handle,
                                                                alg,
                                                                opt.options,
                                                                opt.nmaxiter,
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
                                                               opt.options,
                                                               opt.nmaxiter,
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

        rocsparse_int          nsweeps = opt.nmaxiter;
        const real_of<T>       tol     = std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
        const rocsparse_status status  = rocsparse::csritilu0x_compute_impl<T, I, J>(handle,
                                                                                    alg,
                                                                                    opt.options,
                                                                                    &nsweeps,
                                                                                    opt.nfreeiter,
                                                                                    tol,
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
        ASSERT_EQ(status, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_GE(nsweeps, 1);
        EXPECT_LE(nsweeps, opt.nmaxiter);

        if((opt.options & rocsparse_itilu0_option_convergence_history) != 0)
        {
            // The history reports the sweep count. Without a norm option nothing is
            // logged, so the history array keeps its content.
            rocsparse_int           niter = -1;
            std::vector<real_of<T>> hist(2 * static_cast<size_t>(opt.nmaxiter), real_of<T>(-1));
            ASSERT_EQ((rocsparse::csritilu0x_history_template<real_of<T>, J>(
                          handle, alg, &niter, hist.data(), buffer_size, buffer.ptr)),
                      rocsparse_status_success);
            EXPECT_EQ(niter, nsweeps);
            if((opt.options
                & (rocsparse_itilu0_option_compute_nrm_residual
                   | rocsparse_itilu0_option_compute_nrm_correction))
               == 0)
            {
                for(size_t i = 0; i < hist.size(); ++i)
                    EXPECT_EQ(hist[i], real_of<T>(-1)) << "history[" << i << "]";
            }
        }

        const std::vector<T> ref = host_ilu0(a);
        const double rel_tol = iterative_tolerance<T>(is_stopping(opt.options), max_row_length(a));
        expect_close(to_host(dlval), gather(ref, split.lpos), rel_tol);
        expect_close(to_host(duval), gather(ref, split.upos), rel_tol);
        expect_close(to_host(ddval), gather(ref, split.dpos), rel_tol);
    }

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
            upload(drow, as_rocsparse_int(a.ptr));
            upload(dcol, as_rocsparse_int(a.ind));
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

    // Lower triangle (with the diagonal) of a matrix plus the corner entry A(0, n - 1), so
    // that U is not empty. U(0, n - 1) only meets L(n - 1, 0), which is not in the pattern
    // (half bandwidth < n - 1), so one sweep gives the exact factor (L(i, j) = A(i, j) /
    // A(j, j), U and D unchanged) whatever the update order, and the correction of every
    // sweep is known on the host.
    template <typename T>
    csr_matrix<T> lower_part(const csr_matrix<T>& a)
    {
        csr_matrix<T> l;
        l.n = a.n;
        l.ptr.push_back(0);
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
            {
                if(a.ind[p] <= i)
                {
                    l.ind.push_back(a.ind[p]);
                    l.val.push_back(a.val[p]);
                }
            }
            if(i == 0)
            {
                l.ind.push_back(a.n - 1);
                l.val.push_back(make_value<T>(-0.2));
            }
            l.ptr.push_back(static_cast<int32_t>(l.ind.size()));
        }
        return l;
    }

    // Correction norm of the first sweep on lower_part(): max |L1 - L0| / max |A|, with
    // L0 the lower part of A (U and D do not change).
    template <typename T>
    double host_first_correction(const csr_matrix<T>& a)
    {
        double nrm_a = 0.0, corr = 0.0;
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
            {
                nrm_a = std::max(nrm_a, std::abs(to_cd(a.val[p])));
            }
        }
        std::vector<std::complex<double>> diag(static_cast<size_t>(a.n));
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
            {
                if(a.ind[p] == i)
                    diag[static_cast<size_t>(i)] = to_cd(a.val[p]);
            }
        }
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
            {
                const int32_t j = a.ind[p];
                if(j < i)
                {
                    const std::complex<double> v = to_cd(a.val[p]);
                    corr = std::max(corr, std::abs(v / diag[static_cast<size_t>(j)] - v));
                }
            }
        }
        return corr / nrm_a;
    }

    // Outcome of one async_split compute: sweep count, history (correction norms, then
    // residual norms, nsweeps each) and the factor.
    template <typename T>
    struct x_run
    {
        rocsparse_int           nsweeps = -1;
        std::vector<real_of<T>> corr, residual;
        std::string             log_text;
    };

    // async_split through the x interface with L (unit diagonal, rows), U (columns) and D.
    // The work buffer starts filled with NaN, so a history entry the compute does not
    // write reads back as NaN.
    template <typename T>
    void run_async_split_x(rocsparse_handle     handle,
                           const csr_matrix<T>& a,
                           rocsparse_int        options,
                           rocsparse_int        nmaxiter,
                           x_run<T>&            run)
    {
        using I = rocsparse_int;
        using J = rocsparse_int;

        const split_matrix<T> split(a);
        const I               m    = a.n;
        const I               nnz  = a.nnz();
        const I               lnnz = static_cast<I>(split.lind.size());
        const I               unnz = static_cast<I>(split.uind.size());

        device_vector<I> dptr{as_rocsparse_int(a.ptr)};
        device_vector<J> dind{as_rocsparse_int(a.ind)};
        device_vector<T> dval{a.val};
        device_vector<I> dlptr{as_rocsparse_int(split.lptr)};
        device_vector<J> dlind{as_rocsparse_int(split.lind)};
        device_vector<T> dlval{gather(a.val, split.lpos)};
        device_vector<I> duptr{as_rocsparse_int(split.uptr)};
        device_vector<J> duind{as_rocsparse_int(split.uind)};
        device_vector<T> duval{gather(a.val, split.upos)};
        device_vector<T> ddval{gather(a.val, split.dpos)};
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dlptr.ptr && dlind.ptr && dlval.ptr
                    && duptr.ptr && duind.ptr && duval.ptr && ddval.ptr);

        const auto ldiag = rocsparse_diag_type_unit;
        const auto udiag = rocsparse_diag_type_non_unit;
        const auto ldir  = rocsparse_direction_row;
        const auto udir  = rocsparse_direction_column;
        const auto base  = rocsparse_index_base_zero;
        const auto alg   = rocsparse_itilu0_alg_async_split;

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
        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(hipMemset(buffer.ptr, 0xFF, buffer_size), hipSuccess);

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

        const bool verbose = (options & rocsparse_itilu0_option_verbose) != 0;
        if(verbose)
            testing::internal::CaptureStdout();
        run.nsweeps                   = nmaxiter;
        const real_of<T>       tol    = std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
        const rocsparse_status status = rocsparse::csritilu0x_compute_impl<T, I, J>(handle,
                                                                                    alg,
                                                                                    options,
                                                                                    &run.nsweeps,
                                                                                    0,
                                                                                    tol,
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
        if(verbose)
            run.log_text = testing::internal::GetCapturedStdout();
        ASSERT_EQ(status, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_GE(run.nsweeps, 1);
        ASSERT_LE(run.nsweeps, nmaxiter);

        if((options & rocsparse_itilu0_option_convergence_history) != 0)
        {
            rocsparse_int           niter = -1;
            std::vector<real_of<T>> hist(2 * static_cast<size_t>(nmaxiter), real_of<T>(-1));
            ASSERT_EQ((rocsparse::csritilu0x_history_template<real_of<T>, J>(
                          handle, alg, &niter, hist.data(), buffer_size, buffer.ptr)),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            ASSERT_EQ(niter, run.nsweeps);
            run.corr.assign(hist.begin(), hist.begin() + niter);
            run.residual.assign(hist.begin() + niter, hist.begin() + 2 * niter);
        }

        const std::vector<T> ref     = host_ilu0(a);
        const bool           stops   = (options & rocsparse_itilu0_option_stopping_criteria) != 0;
        const double         rel_tol = iterative_tolerance<T>(stops, max_row_length(a));
        expect_close(to_host(dlval), gather(ref, split.lpos), rel_tol);
        expect_close(to_host(duval), gather(ref, split.upos), rel_tol);
        expect_close(to_host(ddval), gather(ref, split.dpos), rel_tol);
    }

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

    // Relative accuracy of a correction norm computed in T against the host value.
    template <typename T>
    double norm_tolerance()
    {
        return is_single_precision<T> ? 1.0e-5 : 1.0e-13;
    }

    // On lower_part() the first sweep reaches the factor: the correction of the first sweep
    // is host_first_correction() and every later one is zero.
    template <typename T>
    void check_exact_corrections(rocsparse_handle handle, rocsparse_int extra)
    {
        SCOPED_TRACE(std::string(is_cplx<T> ? "complex " : "real ")
                     + (is_single_precision<T> ? "single" : "double"));
        const csr_matrix<T> a     = lower_part(make_band_matrix<T>(40, 3));
        const double        first = host_first_correction(a);
        ASSERT_GT(first, 10.0 * default_tol<T>());

        // Fixed sweep count: the history holds the correction of every sweep.
        x_run<T> fixed;
        run_async_split_x(handle, a, nrm_correction | history_opt | extra, 5, fixed);
        ASSERT_EQ(fixed.nsweeps, 5);
        ASSERT_EQ(fixed.corr.size(), size_t{5});
        EXPECT_NEAR(fixed.corr[0], first, norm_tolerance<T>() * first);
        for(size_t k = 1; k < fixed.corr.size(); ++k)
            EXPECT_EQ(fixed.corr[k], real_of<T>(0)) << "sweep " << k;

        // Stopping on the correction: the second sweep is the first with a zero correction.
        x_run<T> stop;
        run_async_split_x(handle, a, nrm_correction | stopping | history_opt | extra, 40, stop);
        EXPECT_EQ(stop.nsweeps, 2);
        ASSERT_EQ(stop.corr.size(), static_cast<size_t>(stop.nsweeps));
        EXPECT_NEAR(stop.corr[0], first, norm_tolerance<T>() * first);
        EXPECT_EQ(stop.corr[1], real_of<T>(0));
    }

    // Stopping on a band matrix: every sweep before the last has an indicator (the
    // correction norm, or the largest of both norms) above the tolerance, the last one
    // is below it (or below 10x the tolerance for the stagnation test).
    template <typename T>
    void check_stopping_history(rocsparse_handle handle, rocsparse_int extra)
    {
        SCOPED_TRACE(std::string(is_cplx<T> ? "complex " : "real ")
                     + (is_single_precision<T> ? "single" : "double"));
        const csr_matrix<T> a = make_band_matrix<T>(48, 3);
        x_run<T>            run;
        run_async_split_x(handle, a, nrm_correction | stopping | history_opt | extra, 80, run);
        EXPECT_GT(run.nsweeps, 2);
        const size_t n = static_cast<size_t>(run.nsweeps);
        ASSERT_EQ(run.corr.size(), n);
        const bool with_residual = (extra & nrm_residual) != 0;
        for(size_t k = 0; k < n; ++k)
        {
            ASSERT_TRUE(std::isfinite(run.corr[k])) << "sweep " << k;
            EXPECT_GE(run.corr[k], real_of<T>(0)) << "sweep " << k;
            if(with_residual)
            {
                ASSERT_TRUE(std::isfinite(run.residual[k])) << "sweep " << k;
            }
            const real_of<T> indicator
                = with_residual ? std::max(run.corr[k], run.residual[k]) : run.corr[k];
            if(k + 1 < n)
                EXPECT_GT(indicator, default_tol<T>()) << "sweep " << k;
            else
                EXPECT_LE(indicator, 10 * default_tol<T>()) << "sweep " << k;
        }
    }

    // Entries of `a` on or below the diagonal.
    template <typename T>
    csr_matrix<T> lower_triangle(const csr_matrix<T>& a)
    {
        csr_matrix<T> l;
        l.n = a.n;
        l.ptr.push_back(0);
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
            {
                if(a.ind[p] <= i)
                {
                    l.ind.push_back(a.ind[p]);
                    l.val.push_back(a.val[p]);
                }
            }
            l.ptr.push_back(static_cast<int32_t>(l.ind.size()));
        }
        return l;
    }

    // Correction norm with an empty triangle (lower bidiagonal: U empty; diagonal: L and U
    // empty) through the public interface. An empty part launches no kernel (no HIP error
    // is left pending) and contributes zero to the norm. As on lower_part(), one sweep
    // gives the factor, so the norms are known on the host.
    template <typename T>
    void check_empty_triangles(rocsparse_handle handle, rocsparse_itilu0_alg alg, bool diagonal)
    {
        SCOPED_TRACE(std::string(is_single_precision<T> ? "single" : "double") + ", alg "
                     + std::to_string(alg) + (diagonal ? ", diagonal" : ", lower bidiagonal"));
        const csr_matrix<T> a
            = diagonal ? make_band_matrix<T>(16, 0) : lower_triangle(make_band_matrix<T>(16, 1));
        const double first = host_first_correction(a);

        // Fixed sweep count: the history holds the correction of every sweep.
        {
            (void)hipGetLastError();
            itilu0_run<T> r;
            r.prepare(handle, alg, a, nrm_correction | history_opt, 4);
            const run_out out = r.compute(handle, default_tol<T>());
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            EXPECT_EQ(hipGetLastError(), hipSuccess);
            EXPECT_EQ(out.nmaxiter, 4);

            rocsparse_int           niter = -1;
            std::vector<real_of<T>> hist(8, real_of<T>(-1));
            ASSERT_EQ(history<T>(handle, alg, &niter, hist.data(), r.buffer_size, r.buffer.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            ASSERT_EQ(niter, 4);
            EXPECT_NEAR(hist[0], first, norm_tolerance<T>() * first);
            for(size_t k = 1; k < 4; ++k)
                EXPECT_EQ(hist[k], real_of<T>(0)) << "sweep " << k;
        }

        // Stopping on the correction: the first sweep with a zero correction stops.
        {
            (void)hipGetLastError();
            itilu0_run<T> r;
            r.prepare(handle, alg, a, nrm_correction | stopping, 40);
            const run_out out = r.compute(handle, default_tol<T>());
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            EXPECT_EQ(hipGetLastError(), hipSuccess);
            EXPECT_EQ(out.nmaxiter, diagonal ? 1 : 2);
            expect_close(
                to_host(r.dilu), host_ilu0(a), iterative_tolerance<T>(true, max_row_length(a)));
        }
    }
}

class PrecondItilu0 : public HandleTest
{
};

TEST_F(PrecondItilu0, x_async_split_without_residual_norm)
{
    x_options opt; // 80 sweeps
    opt.options = 0;
    check_async_split_x<double>(handle, make_band_matrix<double>(48, 3), opt, "no options");
    check_async_split_x<float>(handle, make_band_matrix<float>(48, 2), opt, "no options float");
    check_async_split_x<double>(handle, make_band_matrix<double>(100, 100), opt, "dense");

    opt.nfreeiter = 2;
    check_async_split_x<double>(handle, make_band_matrix<double>(48, 2), opt, "free sweeps");

    // The history option without the residual norm has nothing to log.
    opt.nfreeiter = 0;
    opt.options   = rocsparse_itilu0_option_convergence_history;
    check_async_split_x<double>(handle, make_band_matrix<double>(48, 3), opt, "history");
}

// The correction norm next to the residual norm: history and stopping indicator.
TEST_F(PrecondItilu0, x_async_split_correction_and_residual_norms)
{
    check_exact_corrections<double>(handle, nrm_residual);
    check_exact_corrections<float>(handle, nrm_residual);
    check_exact_corrections<rocsparse_double_complex>(handle, nrm_residual);
    check_exact_corrections<rocsparse_float_complex>(handle, nrm_residual);
    check_stopping_history<double>(handle, nrm_residual);
    check_stopping_history<float>(handle, nrm_residual);
}

// The correction norm alone: history, stopping criterion, plain sweeps and the verbose
// table (one table per sweep).
TEST_F(PrecondItilu0, x_async_split_correction_norm)
{
    check_exact_corrections<double>(handle, 0);
    check_exact_corrections<float>(handle, 0);
    check_exact_corrections<rocsparse_double_complex>(handle, 0);
    check_exact_corrections<rocsparse_float_complex>(handle, 0);
    check_stopping_history<double>(handle, 0);
    check_stopping_history<float>(handle, 0);

    x_run<double> plain;
    run_async_split_x(handle, make_band_matrix<double>(48, 3), nrm_correction, 80, plain);
    EXPECT_EQ(plain.nsweeps, 80);

    x_run<double> verbose;
    run_async_split_x(
        handle, make_band_matrix<double>(32, 2), nrm_correction | verbose_opt, 30, verbose);
    EXPECT_EQ(verbose.nsweeps, 30);
    size_t tables = 0;
    for(size_t pos = verbose.log_text.find("corr"); pos != std::string::npos;
        pos        = verbose.log_text.find("corr", pos + 1))
        ++tables;
    EXPECT_EQ(tables, size_t{30});
}

// Lower bidiagonal and diagonal matrices with the correction norm, for both split
// algorithms.
TEST_F(PrecondItilu0, correction_norm_with_empty_triangles)
{
    for(const auto alg : {rocsparse_itilu0_alg_async_split, rocsparse_itilu0_alg_sync_split})
    {
        for(const bool diagonal : {false, true})
        {
            check_empty_triangles<double>(handle, alg, diagonal);
            check_empty_triangles<float>(handle, alg, diagonal);
        }
    }
}
