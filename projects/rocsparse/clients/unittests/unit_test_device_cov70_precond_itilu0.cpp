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
//  * the (deprecated) rocsparse_Xcsritilu0_compute entry points, for every
//    algorithm and value type, checked against the exact ILU(0) factor;
//  * the async_split driver of the "x" interface, which is not reachable from
//    the public API (the public async_split path forwards to sync_split), driven
//    through csritilu0x_compute_impl with hand-built L (rows), U (columns) and D
//    structures for every nnz-per-row bucket of the sweep kernels.

#include "unit_test_device_cov70_precond_common.hpp"

#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"

#include <algorithm>
#include <string>
#include <utility>

using namespace cov70;

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

    template <typename F>
    void for_all_types(F&& f)
    {
        f(float{});
        f(double{});
        f(rocsparse_float_complex{});
        f(rocsparse_double_complex{});
    }

    // ------------------------------------------------------------------
    // Public (deprecated) compute entry points
    // ------------------------------------------------------------------
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    template <typename T>
    rocsparse_status plain_compute(rocsparse_handle     handle,
                                   rocsparse_itilu0_alg alg,
                                   rocsparse_int        options,
                                   rocsparse_int*       nmaxiter,
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
            return rocsparse_scsritilu0_compute(handle,
                                                alg,
                                                options,
                                                nmaxiter,
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
            return rocsparse_dcsritilu0_compute(handle,
                                                alg,
                                                options,
                                                nmaxiter,
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
            return rocsparse_ccsritilu0_compute(handle,
                                                alg,
                                                options,
                                                nmaxiter,
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
            return rocsparse_zcsritilu0_compute(handle,
                                                alg,
                                                options,
                                                nmaxiter,
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
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    template <typename T>
    rocsparse_status plain_history(rocsparse_handle     handle,
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

    // Factorization of `a` through rocsparse_Xcsritilu0_compute, compared with
    // the exact ILU(0) factor.
    template <typename T>
    void check_plain_compute(rocsparse_handle     handle,
                             rocsparse_itilu0_alg alg,
                             const csr_matrix<T>& a,
                             rocsparse_int        options,
                             rocsparse_int        nmaxiter_in,
                             const std::string&   what)
    {
        SCOPED_TRACE(what);
        const rocsparse_int        m        = a.n;
        const rocsparse_int        nnz      = a.nnz();
        const rocsparse_index_base base     = rocsparse_index_base_zero;
        rocsparse_int              nmaxiter = nmaxiter_in;

        device_vector<rocsparse_int> drow{a.ptr};
        device_vector<rocsparse_int> dcol{a.ind};
        device_vector<T>             dval{a.val};
        device_vector<T>             dilu{a.val}; // starting iterate
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(
            rocsparse_csritilu0_buffer_size(
                handle, alg, options, nmaxiter, m, nnz, drow, dcol, base, dt_of<T>(), &buffer_size),
            rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_csritilu0_preprocess(handle,
                                                 alg,
                                                 options,
                                                 nmaxiter,
                                                 m,
                                                 nnz,
                                                 drow,
                                                 dcol,
                                                 base,
                                                 dt_of<T>(),
                                                 buffer_size,
                                                 buffer.ptr),
                  rocsparse_status_success);
        const real_of<T> tol = std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
        ASSERT_EQ(plain_compute<T>(handle,
                                   alg,
                                   options,
                                   &nmaxiter,
                                   tol,
                                   m,
                                   nnz,
                                   drow,
                                   dcol,
                                   dval,
                                   dilu,
                                   base,
                                   buffer_size,
                                   buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_GE(nmaxiter, 1);
        EXPECT_LE(nmaxiter, nmaxiter_in);
        expect_close(to_host(dilu),
                     host_ilu0(a),
                     iterative_tolerance<T>(is_stopping(options), max_row_length(a)));

        if((options & rocsparse_itilu0_option_convergence_history) != 0)
        {
            // The log holds `niter` residual norms (after `niter` entries reserved for the
            // correction norms, which were not requested).
            rocsparse_int           niter = -1;
            std::vector<real_of<T>> log(2 * static_cast<size_t>(nmaxiter_in), real_of<T>(-1));
            ASSERT_EQ(plain_history<T>(handle, alg, &niter, log.data(), buffer_size, buffer.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            // The history is what the compute call reported, each residual norm is a finite
            // non-negative number, the iteration reduced the residual, and a stopping run
            // that ended early reached the requested tolerance.
            EXPECT_EQ(niter, nmaxiter);
            ASSERT_GE(niter, 1);
            ASSERT_LE(niter, nmaxiter_in);
            for(rocsparse_int i = 0; i < niter; ++i)
            {
                EXPECT_TRUE(std::isfinite(log[niter + i])) << "iteration " << i;
                EXPECT_GE(log[niter + i], real_of<T>(0)) << "iteration " << i;
            }
            EXPECT_LT(log[2 * niter - 1], log[niter]);
            if(is_stopping(options))
            {
                EXPECT_LT(nmaxiter, nmaxiter_in);
                EXPECT_LE(log[2 * niter - 1], tol);
            }
        }
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

    // The sweep kernels always reduce the residual norm into a device scalar, so
    // every run requests it (without the option that scalar is a null pointer).
    struct x_options
    {
        rocsparse_int options   = rocsparse_itilu0_option_compute_nrm_residual;
        rocsparse_int nmaxiter  = 80;
        rocsparse_int nfreeiter = 0;
        int           wavefront = 0;
    };

    template <typename T>
    void check_async_split_x(rocsparse_handle     handle,
                             const csr_matrix<T>& a,
                             const x_options&     opt,
                             const std::string&   what)
    {
        SCOPED_TRACE(what);
        using I = int32_t;
        using J = int32_t;

        const split_matrix<T> split(a);
        const I               m    = a.n;
        const I               nnz  = a.nnz();
        const I               lnnz = static_cast<I>(split.lind.size());
        const I               unnz = static_cast<I>(split.uind.size());
        // The starting iterate is the matching part of A.
        const std::vector<T> l0 = gather(a.val, split.lpos);
        const std::vector<T> u0 = gather(a.val, split.upos);
        const std::vector<T> d0 = gather(a.val, split.dpos);

        device_vector<I> dptr{a.ptr};
        device_vector<J> dind{a.ind};
        device_vector<T> dval{a.val};
        device_vector<I> dlptr{split.lptr};
        device_vector<J> dlind{split.lind};
        device_vector<T> dlval{l0};
        device_vector<I> duptr{split.uptr};
        device_vector<J> duind{split.uind};
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

        rocsparse_int    nsweeps = opt.nmaxiter;
        const real_of<T> tol     = std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
        rocsparse_status status  = rocsparse_status_internal_error;
        const bool       verbose = (opt.options & rocsparse_itilu0_option_verbose) != 0;
        if(verbose)
        {
            testing::internal::CaptureStdout();
        }
        {
            ScopedWavefront guard(handle,
                                  opt.wavefront > 0 ? opt.wavefront : handle->wavefront_size);
            status = rocsparse::csritilu0x_compute_impl<T, I, J>(handle,
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
        }
        std::string log_text;
        if(verbose)
        {
            log_text = testing::internal::GetCapturedStdout();
        }
        ASSERT_EQ(status, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_GE(nsweeps, 1);
        EXPECT_LE(nsweeps, opt.nmaxiter);

        if(verbose)
        {
            // One header line per sweep. The residual and rate columns of this driver are
            // not meaningful (residual 0, rate nan), so only the table is checked.
            size_t headers = 0;
            for(size_t at = log_text.find("residual"); at != std::string::npos;
                at        = log_text.find("residual", at + 1))
                ++headers;
            EXPECT_EQ(headers, static_cast<size_t>(nsweeps));
        }

        const std::vector<T> ref = host_ilu0(a);
        const double rel_tol = iterative_tolerance<T>(is_stopping(opt.options), max_row_length(a));
        expect_close(to_host(dlval), gather(ref, split.lpos), rel_tol);
        expect_close(to_host(duval), gather(ref, split.upos), rel_tol);
        expect_close(to_host(ddval), gather(ref, split.dpos), rel_tol);
    }
}

class Cov70PrecondItilu0 : public HandleTest
{
};

TEST_F(Cov70PrecondItilu0, compute_algorithms_and_types)
{
    const rocsparse_int nrm  = rocsparse_itilu0_option_compute_nrm_residual;
    const rocsparse_int stop = rocsparse_itilu0_option_stopping_criteria
                               | rocsparse_itilu0_option_compute_nrm_residual
                               | rocsparse_itilu0_option_convergence_history;
    for(const auto alg : {rocsparse_itilu0_alg_default,
                          rocsparse_itilu0_alg_async_inplace,
                          rocsparse_itilu0_alg_async_split,
                          rocsparse_itilu0_alg_sync_split})
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            // Fixed number of sweeps, then with the stopping criterion.
            check_plain_compute<T>(
                handle, alg, make_band_matrix<T>(24, 2), nrm, 60, "alg " + std::to_string(alg));
            check_plain_compute<T>(handle,
                                   alg,
                                   make_band_matrix<T>(24, 3),
                                   stop,
                                   60,
                                   "stop alg " + std::to_string(alg));
        });
    }
}

TEST_F(Cov70PrecondItilu0, compute_nnz_per_row_buckets)
{
    // The sweep kernel width follows the mean number of nonzeros per row (nnz / m, integer
    // division): <= 2, 4, 8, 16, 32, 64 select that width, anything larger the wavefront
    // width. Half bandwidths 1, 2, 6, 12, 24 and a dense matrix give the means 2, 4, 12,
    // 22, 39 and 100, one per kernel.
    const std::pair<int32_t, int32_t> cases[]
        = {{1, 2}, {2, 4}, {6, 12}, {12, 22}, {24, 39}, {100, 100}};
    for(const auto alg : {rocsparse_itilu0_alg_async_inplace, rocsparse_itilu0_alg_sync_split})
    {
        for(const auto& [w, mean] : cases)
        {
            const int32_t m = (w >= 100) ? 100 : 64;
            const auto    a = make_band_matrix<double>(m, w);
            ASSERT_EQ(a.nnz() / a.n, mean);
            check_plain_compute<double>(handle,
                                        alg,
                                        a,
                                        rocsparse_itilu0_option_compute_nrm_residual,
                                        80,
                                        "w=" + std::to_string(w));
        }
    }
}

TEST_F(Cov70PrecondItilu0, compute_quick_returns_and_argument_checks)
{
    const rocsparse_index_base   base     = rocsparse_index_base_zero;
    rocsparse_int                nmaxiter = 5;
    device_vector<rocsparse_int> dptr{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> dind{std::vector<rocsparse_int>{0}};
    device_vector<double>        dval{std::vector<double>{2.0}};
    device_vector<double>        dilu{std::vector<double>{2.0}};
    ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dilu.ptr);

    const auto alg = rocsparse_itilu0_alg_sync_split;
    // m == 0: success and no sweep performed.
    EXPECT_EQ(plain_compute<double>(handle,
                                    alg,
                                    0,
                                    &nmaxiter,
                                    1.0e-10,
                                    0,
                                    0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    base,
                                    0,
                                    nullptr),
              rocsparse_status_success);
    EXPECT_EQ(nmaxiter, 0);
    // No entries in a non-empty matrix: the first pivot is missing.
    nmaxiter = 5;
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 3, 0, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_zero_pivot);
    EXPECT_EQ(nmaxiter, 0);

    // Argument validation.
    nmaxiter = 5;
    EXPECT_EQ(
        plain_compute<double>(
            nullptr, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(plain_compute<double>(handle,
                                    static_cast<rocsparse_itilu0_alg>(-5),
                                    0,
                                    &nmaxiter,
                                    1.0e-10,
                                    1,
                                    1,
                                    dptr,
                                    dind,
                                    dval,
                                    dilu,
                                    base,
                                    0,
                                    nullptr),
              rocsparse_status_invalid_value);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, -1, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_value);
    EXPECT_EQ(plain_compute<double>(
                  handle, alg, 0, nullptr, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(plain_compute<double>(
                  handle, alg, 0, &nmaxiter, -1.0, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
              rocsparse_status_invalid_value);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, -1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_size);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, -1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_size);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, nullptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, nullptr, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, nullptr, dilu, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, nullptr, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(plain_compute<double>(handle,
                                    alg,
                                    0,
                                    &nmaxiter,
                                    1.0e-10,
                                    1,
                                    1,
                                    dptr,
                                    dind,
                                    dval,
                                    dilu,
                                    static_cast<rocsparse_index_base>(-1),
                                    0,
                                    nullptr),
              rocsparse_status_invalid_value);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 16, nullptr),
        rocsparse_status_invalid_pointer);
}

TEST_F(Cov70PrecondItilu0, x_async_split_nnz_per_row_buckets)
{
    // Same selection as the plain driver (mean nnz per row, integer division): half
    // bandwidths 1, 2, 3, 6, 12, 24 and a dense matrix give the means 2, 4, 6, 12, 22, 39
    // and 100, i.e. the widths 2, 4, 8, 16, 32, 64 and the wavefront size.
    const std::pair<int32_t, int32_t> cases[]
        = {{1, 2}, {2, 4}, {3, 6}, {6, 12}, {12, 22}, {24, 39}, {100, 100}};
    x_options opt;
    for(const auto& [w, mean] : cases)
    {
        const int32_t m = (w >= 100) ? 100 : 64;
        const auto    a = make_band_matrix<double>(m, w);
        ASSERT_EQ(a.nnz() / a.n, mean);
        check_async_split_x<double>(handle, a, opt, "w=" + std::to_string(w));
    }
}

TEST_F(Cov70PrecondItilu0, x_async_split_types)
{
    x_options opt;
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        check_async_split_x<T>(handle, make_band_matrix<T>(40, 3), opt, "type");
    });
}

TEST_F(Cov70PrecondItilu0, x_async_split_options)
{
    // Residual norm, stopping criterion, convergence history and free sweeps.
    x_options opt;
    opt.options = rocsparse_itilu0_option_stopping_criteria
                  | rocsparse_itilu0_option_compute_nrm_residual
                  | rocsparse_itilu0_option_convergence_history;
    check_async_split_x<double>(handle, make_band_matrix<double>(48, 3), opt, "stopping");
    check_async_split_x<float>(handle, make_band_matrix<float>(48, 3), opt, "stopping float");

    opt.nfreeiter = 2;
    check_async_split_x<double>(handle, make_band_matrix<double>(48, 2), opt, "free sweeps");

    // Residual norm only, with the iteration log captured and checked (one table per sweep).
    opt.nfreeiter = 0;
    opt.nmaxiter  = 30;
    opt.options   = rocsparse_itilu0_option_compute_nrm_residual | rocsparse_itilu0_option_verbose;
    check_async_split_x<double>(handle, make_band_matrix<double>(32, 2), opt, "verbose");
}

TEST_F(Cov70PrecondItilu0, x_async_split_wavefront_64)
{
    // A 64 lane wavefront selects the widest sweep kernel for dense rows.
    x_options opt;
    opt.wavefront = 64;
    check_async_split_x<double>(handle, make_band_matrix<double>(100, 100), opt, "wave64");
}
