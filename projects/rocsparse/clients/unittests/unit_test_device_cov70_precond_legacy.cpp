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

// Coverage of the mat_info based preconditioner entry points that the generic
// descriptor API does not use:
//  * rocsparse_csric0_set_tolerance / get_tolerance / singular_pivot;
//  * rocsparse_Xbsrilu0_numeric_boost for the four value types, in host and in
//    device pointer mode.
// Every factorization is compared with a host reference.

#include "unit_test_device_cov70_precond_common.hpp"

#include <algorithm>
#include <cmath>
#include <string>

using namespace cov70;

namespace
{
    struct descr_info
    {
        rocsparse_mat_descr descr = nullptr;
        rocsparse_mat_info  info  = nullptr;

        descr_info()
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
        }

        ~descr_info()
        {
            if(info)
                (void)rocsparse_destroy_mat_info(info);
            if(descr)
                (void)rocsparse_destroy_mat_descr(descr);
        }

        descr_info(const descr_info&) = delete;
        descr_info& operator=(const descr_info&) = delete;
    };

    template <typename F>
    void for_all_types(F&& f)
    {
        f(float{});
        f(double{});
        f(rocsparse_float_complex{});
        f(rocsparse_double_complex{});
    }

    // ------------------------------------------------------------------
    // bsrilu0, one wrapper per value type
    // ------------------------------------------------------------------
    template <typename T>
    rocsparse_status bsrilu0_buffer_size(rocsparse_handle     handle,
                                         rocsparse_direction  dir,
                                         rocsparse_int        mb,
                                         rocsparse_int        nnzb,
                                         rocsparse_mat_descr  descr,
                                         T*                   val,
                                         const rocsparse_int* ptr,
                                         const rocsparse_int* ind,
                                         rocsparse_int        bd,
                                         rocsparse_mat_info   info,
                                         size_t*              size)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsrilu0_buffer_size(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, size);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsrilu0_buffer_size(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, size);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsrilu0_buffer_size(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, size);
        else
            return rocsparse_zbsrilu0_buffer_size(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, size);
    }

    template <typename T>
    rocsparse_status bsrilu0_analysis(rocsparse_handle     handle,
                                      rocsparse_direction  dir,
                                      rocsparse_int        mb,
                                      rocsparse_int        nnzb,
                                      rocsparse_mat_descr  descr,
                                      T*                   val,
                                      const rocsparse_int* ptr,
                                      const rocsparse_int* ind,
                                      rocsparse_int        bd,
                                      rocsparse_mat_info   info,
                                      void*                buffer)
    {
        const auto ap = rocsparse_analysis_policy_reuse;
        const auto sp = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsrilu0_analysis(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, ap, sp, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsrilu0_analysis(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, ap, sp, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsrilu0_analysis(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, ap, sp, buffer);
        else
            return rocsparse_zbsrilu0_analysis(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, ap, sp, buffer);
    }

    template <typename T>
    rocsparse_status bsrilu0_compute(rocsparse_handle     handle,
                                     rocsparse_direction  dir,
                                     rocsparse_int        mb,
                                     rocsparse_int        nnzb,
                                     rocsparse_mat_descr  descr,
                                     T*                   val,
                                     const rocsparse_int* ptr,
                                     const rocsparse_int* ind,
                                     rocsparse_int        bd,
                                     rocsparse_mat_info   info,
                                     void*                buffer)
    {
        const auto sp = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsrilu0(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, sp, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsrilu0(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, sp, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsrilu0(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, sp, buffer);
        else
            return rocsparse_zbsrilu0(
                handle, dir, mb, nnzb, descr, val, ptr, ind, bd, info, sp, buffer);
    }

    // Boost the pivots through the legacy call: tol / value are in host memory
    // (`tol`, `val`) or, in device pointer mode, copies of them in device memory.
    template <typename T>
    rocsparse_status bsrilu0_numeric_boost(rocsparse_handle   handle,
                                           rocsparse_mat_info info,
                                           int                enable,
                                           const real_of<T>*  tol,
                                           const T*           val)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsrilu0_numeric_boost(handle, info, enable, tol, val);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsrilu0_numeric_boost(handle, info, enable, tol, val);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsrilu0_numeric_boost(handle, info, enable, tol, val);
        else
            return rocsparse_zbsrilu0_numeric_boost(handle, info, enable, tol, val);
    }

    enum class boost_mode
    {
        off, // boost enabled then disabled again
        host_pointers,
        device_pointers
    };

    // Factorizes a block matrix with the legacy bsrilu0 API and the requested
    // numeric boost and compares the factor with the exact scalar ILU(0).
    template <typename T>
    void check_bsrilu0_boost(rocsparse_handle handle, boost_mode mode, rocsparse_direction dir)
    {
        const int32_t       mb = 4, bd = 3;
        const bsr_matrix<T> bsr = make_block_matrix<T>(mb, bd, dir, true);

        const real_of<T> tol   = 1.0e6; // every pivot is boosted
        const T          boost = make_value<T>(2.0);

        descr_info di;
        ASSERT_NE(di.info, nullptr);
        device_vector<rocsparse_int> drow{
            std::vector<rocsparse_int>(bsr.ptr.begin(), bsr.ptr.end())};
        device_vector<rocsparse_int> dcol{
            std::vector<rocsparse_int>(bsr.ind.begin(), bsr.ind.end())};
        device_vector<T> dval{bsr.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        size_t size = 0;
        ASSERT_EQ(bsrilu0_buffer_size<T>(
                      handle, dir, mb, bsr.nnzb(), di.descr, dval, drow, dcol, bd, di.info, &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(bsrilu0_analysis<T>(
                      handle, dir, mb, bsr.nnzb(), di.descr, dval, drow, dcol, bd, di.info, buffer),
                  rocsparse_status_success);

        device_vector<real_of<T>> dtol{std::vector<real_of<T>>{tol}};
        device_vector<T>          dboost{std::vector<T>{boost}};
        ASSERT_TRUE(dtol.ptr && dboost.ptr);
        if(mode == boost_mode::host_pointers)
        {
            ASSERT_EQ(bsrilu0_numeric_boost<T>(handle, di.info, 1, &tol, &boost),
                      rocsparse_status_success);
        }
        else if(mode == boost_mode::device_pointers)
        {
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
            ASSERT_EQ(bsrilu0_numeric_boost<T>(handle, di.info, 1, dtol.ptr, dboost.ptr),
                      rocsparse_status_success);
        }
        else
        {
            // Enabled, then disabled by the second call: the exact factor results.
            ASSERT_EQ(bsrilu0_numeric_boost<T>(handle, di.info, 1, &tol, &boost),
                      rocsparse_status_success);
            ASSERT_EQ(bsrilu0_numeric_boost<T>(handle, di.info, 0, nullptr, nullptr),
                      rocsparse_status_success);
        }

        const rocsparse_status st = bsrilu0_compute<T>(
            handle, dir, mb, bsr.nnzb(), di.descr, dval, drow, dcol, bd, di.info, buffer);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        ASSERT_EQ(st, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const csr_matrix<T>  scalar  = expand_bsr(bsr);
        const bool           boosted = (mode != boost_mode::off);
        const std::vector<T> want    = host_ilu0(scalar, boosted ? tol : -1.0, 2.0);
        bsr_matrix<T>        got     = bsr;
        got.val                      = to_host(dval);
        // The factor of the nonsingular matrix is the host factor to rounding (the
        // boost replaces pivots by +-2, which keeps the factor well conditioned).
        expect_close(expand_bsr(got).val, want);

        // A boosted (or exactly factorized, nonsingular) run reports no zero pivot.
        rocsparse_int zero_pivot = -7;
        EXPECT_EQ(rocsparse_bsrilu0_zero_pivot(handle, di.info, &zero_pivot),
                  rocsparse_status_success);
        EXPECT_EQ(zero_pivot, -1);
        EXPECT_EQ(rocsparse_bsrilu0_clear(handle, di.info), rocsparse_status_success);
    }

    // One dense 3x3 block (a single block row) whose first pivot is `pivot0`, the rest of
    // the block being well conditioned. Without the boost an exactly zero first pivot is
    // reported as a zero pivot at the block row (in the index base of the matrix); with a
    // boost tolerance of 1e-2 (boost value 2) the pivots that are at most 1e-2 in
    // magnitude (here only the first, of either sign) are replaced by 2 with their sign
    // and the factor matches the host reference, with no zero pivot reported.
    template <typename T>
    void check_bsrilu0_first_pivot(rocsparse_handle     handle,
                                   double               pivot0,
                                   bool                 boost,
                                   rocsparse_index_base base)
    {
        SCOPED_TRACE(::testing::Message() << "pivot0 " << pivot0 << " boost " << boost << " base "
                                          << static_cast<int>(base));
        const int32_t bd = 3;
        bsr_matrix<T> bsr;
        bsr.mb        = 1;
        bsr.block_dim = bd;
        bsr.dir       = rocsparse_direction_row;
        bsr.ptr       = {0, 1};
        bsr.ind       = {0};
        for(const double x : {pivot0, 1.0, 0.5, 1.0, 4.0, 1.0, 0.5, 1.0, 5.0})
            bsr.val.push_back(make_value<T>(x));

        const rocsparse_int shift = (base == rocsparse_index_base_one) ? 1 : 0;
        descr_info          di;
        ASSERT_NE(di.info, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(di.descr, base), rocsparse_status_success);
        device_vector<rocsparse_int> drow{
            std::vector<rocsparse_int>{shift, static_cast<rocsparse_int>(1 + shift)}};
        device_vector<rocsparse_int> dcol{std::vector<rocsparse_int>{shift}};
        device_vector<T>             dval{bsr.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        const rocsparse_direction dir  = bsr.dir;
        size_t                    size = 0;
        ASSERT_EQ(bsrilu0_buffer_size<T>(
                      handle, dir, 1, 1, di.descr, dval, drow, dcol, bd, di.info, &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(
            bsrilu0_analysis<T>(handle, dir, 1, 1, di.descr, dval, drow, dcol, bd, di.info, buffer),
            rocsparse_status_success);

        const real_of<T> tol    = 1.0e-2;
        const T          bvalue = make_value<T>(2.0);
        if(boost)
        {
            ASSERT_EQ(bsrilu0_numeric_boost<T>(handle, di.info, 1, &tol, &bvalue),
                      rocsparse_status_success);
        }
        ASSERT_EQ(
            bsrilu0_compute<T>(handle, dir, 1, 1, di.descr, dval, drow, dcol, bd, di.info, buffer),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocsparse_int zero_pivot = -7;
        const auto    zp_status  = rocsparse_bsrilu0_zero_pivot(handle, di.info, &zero_pivot);
        if(!boost && pivot0 == 0.0)
        {
            EXPECT_EQ(zp_status, rocsparse_status_zero_pivot);
            EXPECT_EQ(zero_pivot, static_cast<rocsparse_int>(shift));
        }
        else
        {
            EXPECT_EQ(zp_status, rocsparse_status_success);
            EXPECT_EQ(zero_pivot, -1);
            bsr_matrix<T> got          = bsr;
            got.val                    = to_host(dval);
            const csr_matrix<T> scalar = expand_bsr(bsr);
            const auto          want   = host_ilu0(scalar, boost ? tol : -1.0, 2.0);
            // The boosted pivot is exactly +-2.
            if(boost)
            {
                const double sign = (pivot0 < 0.0) ? -1.0 : 1.0;
                EXPECT_EQ(to_cd(want[0]), std::complex<double>(2.0 * sign, 0.0));
            }
            expect_close(expand_bsr(got).val, want);
        }
        EXPECT_EQ(rocsparse_bsrilu0_clear(handle, di.info), rocsparse_status_success);
    }
}

class Cov70PrecondLegacy : public HandleTest
{
};

TEST_F(Cov70PrecondLegacy, bsrilu0_numeric_boost_types_and_pointer_modes)
{
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(const auto mode :
            {boost_mode::off, boost_mode::host_pointers, boost_mode::device_pointers})
        {
            check_bsrilu0_boost<float>(handle, mode, dir);
            check_bsrilu0_boost<double>(handle, mode, dir);
            check_bsrilu0_boost<rocsparse_float_complex>(handle, mode, dir);
            check_bsrilu0_boost<rocsparse_double_complex>(handle, mode, dir);
        }
    }
}

TEST_F(Cov70PrecondLegacy, bsrilu0_numeric_boost_first_pivot)
{
    for(const auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            // Exactly zero pivot: reported without the boost, replaced by +2 with it.
            check_bsrilu0_first_pivot<T>(handle, 0.0, false, base);
            check_bsrilu0_first_pivot<T>(handle, 0.0, true, base);
            // Small pivots of either sign are boosted to +-2.
            check_bsrilu0_first_pivot<T>(handle, 1.0e-3, true, base);
            check_bsrilu0_first_pivot<T>(handle, -1.0e-3, true, base);
        });
    }
}

TEST_F(Cov70PrecondLegacy, bsrilu0_numeric_boost_argument_checks)
{
    descr_info   di;
    const double tol   = 1.0;
    const double boost = 2.0;
    EXPECT_EQ(rocsparse_dbsrilu0_numeric_boost(nullptr, di.info, 1, &tol, &boost),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, nullptr, 1, &tol, &boost),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, di.info, 1, nullptr, &boost),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, di.info, 1, &tol, nullptr),
              rocsparse_status_invalid_pointer);
    // Disabling needs no values.
    EXPECT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, di.info, 0, nullptr, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, di.info, 1, &tol, &boost),
              rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// csric0 tolerance and singular pivot
// ---------------------------------------------------------------------------
namespace
{
    struct csric0_run
    {
        std::vector<double> values;
        rocsparse_status    compute_status = rocsparse_status_internal_error;
    };

    // Analysis and compute of the 4x4 grid matrix (kept zero based in `a`, the device copy
    // is shifted for `base`) through the mat_info API. `tolerance` < 0 leaves the default;
    // the value is set after the analysis, once the csric0 data of the mat_info exists.
    // The singular pivot is read back in the pointer mode selected by `device_position`
    // and is expressed in `base`.
    void run_csric0(rocsparse_handle     handle,
                    double               tolerance,
                    bool                 device_position,
                    rocsparse_int&       position,
                    double*              tolerance_back,
                    csr_matrix<double>&  a,
                    csric0_run&          out,
                    rocsparse_index_base base = rocsparse_index_base_zero)
    {
        a = make_grid_matrix<double>(4, false);
        descr_info di;
        ASSERT_NE(di.info, nullptr);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(di.descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_index_base(di.descr, base), rocsparse_status_success);
        const rocsparse_int        shift = (base == rocsparse_index_base_one) ? 1 : 0;
        std::vector<rocsparse_int> hptr(a.ptr.begin(), a.ptr.end());
        std::vector<rocsparse_int> hind(a.ind.begin(), a.ind.end());
        for(auto& v : hptr)
            v += shift;
        for(auto& v : hind)
            v += shift;
        device_vector<rocsparse_int> drow{hptr};
        device_vector<rocsparse_int> dcol{hind};
        device_vector<double>        dval{a.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        size_t size = 0;
        ASSERT_EQ(rocsparse_dcsric0_buffer_size(
                      handle, a.n, a.nnz(), di.descr, dval, drow, dcol, di.info, &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_dcsric0_analysis(handle,
                                             a.n,
                                             a.nnz(),
                                             di.descr,
                                             dval,
                                             drow,
                                             dcol,
                                             di.info,
                                             rocsparse_analysis_policy_reuse,
                                             rocsparse_solve_policy_auto,
                                             buffer),
                  rocsparse_status_success);
        if(tolerance >= 0.0)
        {
            ASSERT_EQ(rocsparse_csric0_set_tolerance(handle, di.info, tolerance),
                      rocsparse_status_success);
        }
        if(tolerance_back)
        {
            ASSERT_EQ(rocsparse_csric0_get_tolerance(handle, di.info, tolerance_back),
                      rocsparse_status_success);
        }
        out.compute_status = rocsparse_dcsric0(handle,
                                               a.n,
                                               a.nnz(),
                                               di.descr,
                                               dval,
                                               drow,
                                               dcol,
                                               di.info,
                                               rocsparse_solve_policy_auto,
                                               buffer);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        out.values = to_host(dval);

        position = -7;
        if(device_position)
        {
            device_vector<rocsparse_int> dpos{std::vector<rocsparse_int>{-7}};
            ASSERT_TRUE(dpos.ptr);
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
            const auto st = rocsparse_csric0_singular_pivot(handle, di.info, dpos.ptr);
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                      rocsparse_status_success);
            ASSERT_EQ(st, rocsparse_status_success);
            position = to_host(dpos)[0];
        }
        else
        {
            ASSERT_EQ(rocsparse_csric0_singular_pivot(handle, di.info, &position),
                      rocsparse_status_success);
        }
        ASSERT_EQ(rocsparse_csric0_clear(handle, di.info), rocsparse_status_success);
    }
}

namespace
{
    // Pre-square-root pivots of the IC(0) factorization of `a`: the diagonal entry of
    // the host factor squared (csric0 compares these with tolerance^2).
    std::vector<double> ic0_pivots(const csr_matrix<double>& a)
    {
        const std::vector<double> factor = host_ic0(a);
        std::vector<double>       pivots(static_cast<size_t>(a.n), 0.0);
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] == i)
                    pivots[i] = factor[p] * factor[p];
        return pivots;
    }

    // First row whose pivot is at most tolerance^2 (-1 if none), the zero based position
    // the library reports for the singular pivot.
    rocsparse_int expected_singular_row(const std::vector<double>& pivots, double tolerance)
    {
        for(size_t i = 0; i < pivots.size(); ++i)
            if(pivots[i] <= tolerance * tolerance)
                return static_cast<rocsparse_int>(i);
        return -1;
    }
}

TEST_F(Cov70PrecondLegacy, csric0_tolerance_and_singular_pivot)
{
    // Default tolerance: no singular pivot, the exact factor.
    {
        rocsparse_int      position = 0;
        double             tol_back = -1.0;
        csr_matrix<double> a;
        csric0_run         out;
        run_csric0(handle, -1.0, false, position, &tol_back, a, out);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        EXPECT_EQ(position, -1);
        EXPECT_GE(tol_back, 0.0);
        EXPECT_LT(ic_pattern_residual(a, keep_lower_only(a, out.values)), 1.0e-9);
        expect_close(keep_lower_only(a, out.values), keep_lower_only(a, host_ic0(a)));
    }

    // The pivot of row 0 is the diagonal entry 5 (it has no inner sum) and the later
    // pivots decrease towards about 4.4, so
    //  * tolerance 3 (3^2 = 9 >= 5)  flags row 0;
    //  * tolerance 2 (2^2 = 4)       flags nothing;
    //  * the tolerance halfway between the first and the smallest pivot flags a later row,
    //    the first one whose pivot is at most tolerance^2 (computed on the host);
    //  * 1e3 flags row 0.
    const csr_matrix<double> grid   = make_grid_matrix<double>(4, false);
    const auto               pivots = ic0_pivots(grid);
    ASSERT_NEAR(pivots[0], 5.0, 1.0e-12); // the squared square root of the diagonal entry
    const double smallest = *std::min_element(pivots.begin(), pivots.end());
    ASSERT_LT(smallest, 5.0);
    const double middle = std::sqrt(0.5 * (5.0 + smallest));
    for(const double pivot : pivots) // no pivot sits on the boundary
        ASSERT_GT(std::abs(pivot - middle * middle), 1.0e-6);
    ASSERT_GT(expected_singular_row(pivots, middle), 0);
    ASSERT_EQ(expected_singular_row(pivots, 3.0), 0);
    ASSERT_EQ(expected_singular_row(pivots, 2.0), -1);

    for(const auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(const bool device_position : {false, true})
        {
            for(const double tolerance : {3.0, 2.0, middle, 1.0e3})
            {
                SCOPED_TRACE(::testing::Message()
                             << "tolerance " << tolerance << " device_position " << device_position
                             << " base " << static_cast<int>(base));
                rocsparse_int      position = 0;
                double             tol_back = -1.0;
                csr_matrix<double> a;
                csric0_run         out;
                run_csric0(handle, tolerance, device_position, position, &tol_back, a, out, base);
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                EXPECT_EQ(tol_back, tolerance);
                // The position is returned in the index base of the matrix (-1: none).
                const rocsparse_int row = expected_singular_row(pivots, tolerance);
                EXPECT_EQ(position,
                          row < 0 ? rocsparse_int(-1)
                                  : row + (base == rocsparse_index_base_one ? 1 : 0));
                // The factor is computed whether or not a pivot is flagged.
                expect_close(keep_lower_only(a, out.values), keep_lower_only(a, host_ic0(a)));
            }
        }
    }
}

TEST_F(Cov70PrecondLegacy, csric0_tolerance_argument_checks)
{
    descr_info    di;
    double        tol      = 0.0;
    rocsparse_int position = 0;
    EXPECT_EQ(rocsparse_csric0_get_tolerance(nullptr, di.info, &tol),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csric0_get_tolerance(handle, nullptr, &tol),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csric0_get_tolerance(handle, di.info, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csric0_set_tolerance(nullptr, di.info, 1.0),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csric0_set_tolerance(handle, di.info, -1.0),
              rocsparse_status_invalid_value);
    // Without an analysis there is no singular pivot.
    EXPECT_EQ(rocsparse_csric0_singular_pivot(handle, di.info, &position),
              rocsparse_status_success);
    EXPECT_EQ(position, -1);
    EXPECT_EQ(rocsparse_csric0_singular_pivot(nullptr, di.info, &position),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csric0_singular_pivot(handle, nullptr, &position),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csric0_singular_pivot(handle, di.info, nullptr),
              rocsparse_status_invalid_pointer);
}
