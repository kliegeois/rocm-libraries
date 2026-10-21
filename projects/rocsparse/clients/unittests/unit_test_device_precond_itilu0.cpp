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
//    (columns) and D structures.

#include "unit_test_device_precond_common.hpp"

#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_history.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"

#include <algorithm>
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
