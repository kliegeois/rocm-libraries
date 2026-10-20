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

// Legacy csrilu0 interface (mat_info based): the numerical tolerance and singular
// pivot queries and the numeric boost of every value type, in both pointer modes and
// both index bases. Reference results come from the host ILU(0) of the cov70 common
// header on a diagonally dominant grid matrix, so every pivot is finite and bounded
// away from zero.

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

    template <typename T>
    rocsparse_status csrilu0_buffer_size(rocsparse_handle     handle,
                                         rocsparse_int        m,
                                         rocsparse_int        nnz,
                                         rocsparse_mat_descr  descr,
                                         const T*             val,
                                         const rocsparse_int* ptr,
                                         const rocsparse_int* ind,
                                         rocsparse_mat_info   info,
                                         size_t*              size)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0_buffer_size(handle, m, nnz, descr, val, ptr, ind, info, size);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0_buffer_size(handle, m, nnz, descr, val, ptr, ind, info, size);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0_buffer_size(handle, m, nnz, descr, val, ptr, ind, info, size);
        else
            return rocsparse_zcsrilu0_buffer_size(handle, m, nnz, descr, val, ptr, ind, info, size);
    }

    template <typename T>
    rocsparse_status csrilu0_analysis(rocsparse_handle     handle,
                                      rocsparse_int        m,
                                      rocsparse_int        nnz,
                                      rocsparse_mat_descr  descr,
                                      const T*             val,
                                      const rocsparse_int* ptr,
                                      const rocsparse_int* ind,
                                      rocsparse_mat_info   info,
                                      void*                buffer)
    {
        const auto reuse = rocsparse_analysis_policy_reuse;
        const auto solve = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0_analysis(
                handle, m, nnz, descr, val, ptr, ind, info, reuse, solve, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0_analysis(
                handle, m, nnz, descr, val, ptr, ind, info, reuse, solve, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0_analysis(
                handle, m, nnz, descr, val, ptr, ind, info, reuse, solve, buffer);
        else
            return rocsparse_zcsrilu0_analysis(
                handle, m, nnz, descr, val, ptr, ind, info, reuse, solve, buffer);
    }

    template <typename T>
    rocsparse_status csrilu0_compute(rocsparse_handle     handle,
                                     rocsparse_int        m,
                                     rocsparse_int        nnz,
                                     rocsparse_mat_descr  descr,
                                     T*                   val,
                                     const rocsparse_int* ptr,
                                     const rocsparse_int* ind,
                                     rocsparse_mat_info   info,
                                     void*                buffer)
    {
        const auto solve = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0(handle, m, nnz, descr, val, ptr, ind, info, solve, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0(handle, m, nnz, descr, val, ptr, ind, info, solve, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0(handle, m, nnz, descr, val, ptr, ind, info, solve, buffer);
        else
            return rocsparse_zcsrilu0(handle, m, nnz, descr, val, ptr, ind, info, solve, buffer);
    }

    // The boost value types of the numeric boost entry points: the standard ones
    // (tolerance and value of the precision of T) and the mixed ones (a double
    // tolerance with single precision values).
    enum class boost_api
    {
        standard,
        mixed
    };

    template <typename T>
    rocsparse_status set_boost(rocsparse_handle   handle,
                               rocsparse_mat_info info,
                               int                enable,
                               boost_api          api,
                               const void*        tol,
                               const void*        val)
    {
        if(api == boost_api::mixed)
        {
            if constexpr(std::is_same_v<T, float>)
                return rocsparse_dscsrilu0_numeric_boost(handle,
                                                         info,
                                                         enable,
                                                         static_cast<const double*>(tol),
                                                         static_cast<const float*>(val));
            else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
                return rocsparse_dccsrilu0_numeric_boost(
                    handle,
                    info,
                    enable,
                    static_cast<const double*>(tol),
                    static_cast<const rocsparse_float_complex*>(val));
            else
                return rocsparse_status_not_implemented;
        }
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0_numeric_boost(handle,
                                                    info,
                                                    enable,
                                                    static_cast<const float*>(tol),
                                                    static_cast<const float*>(val));
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0_numeric_boost(handle,
                                                    info,
                                                    enable,
                                                    static_cast<const double*>(tol),
                                                    static_cast<const double*>(val));
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0_numeric_boost(
                handle,
                info,
                enable,
                static_cast<const float*>(tol),
                static_cast<const rocsparse_float_complex*>(val));
        else
            return rocsparse_zcsrilu0_numeric_boost(
                handle,
                info,
                enable,
                static_cast<const double*>(tol),
                static_cast<const rocsparse_double_complex*>(val));
    }

    struct boost_request
    {
        bool      enable = false;
        double    tol    = 0.0;
        double    val    = 1.0;
        bool      device = false; // pointer mode of the boost parameters
        boost_api api    = boost_api::standard;
    };

    template <typename T>
    struct ilu0_out
    {
        rocsparse_status compute_status = rocsparse_status_internal_error;
        std::vector<T>   values;
        rocsparse_int    singular_pivot = -7;
        double           tolerance_back = -1.0;
    };

    // Analysis and compute of `a` (kept zero based, the device copy is shifted for
    // `base`). `tolerance` < 0 keeps the default; it is set after the analysis, once the
    // csrilu0 data of the mat_info exists. The singular pivot is read back in the
    // pointer mode selected by `device_position` and is expressed in `base`.
    template <typename T>
    void run_csrilu0(rocsparse_handle     handle,
                     const csr_matrix<T>& a,
                     rocsparse_index_base base,
                     double               tolerance,
                     const boost_request& boost,
                     bool                 device_position,
                     ilu0_out<T>&         out)
    {
        descr_info di;
        ASSERT_NE(di.info, nullptr);
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
        device_vector<T>             dval{a.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        size_t size = 0;
        ASSERT_EQ(csrilu0_buffer_size<T>(
                      handle, a.n, a.nnz(), di.descr, dval, drow, dcol, di.info, &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(
            csrilu0_analysis<T>(handle, a.n, a.nnz(), di.descr, dval, drow, dcol, di.info, buffer),
            rocsparse_status_success);
        if(tolerance >= 0.0)
        {
            ASSERT_EQ(rocsparse_csrilu0_set_tolerance(handle, di.info, tolerance),
                      rocsparse_status_success);
        }
        ASSERT_EQ(rocsparse_csrilu0_get_tolerance(handle, di.info, &out.tolerance_back),
                  rocsparse_status_success);

        // The boost parameters, in the requested pointer mode. Device copies stay alive
        // until the factorization is done.
        const real_of<T>              tol_s = static_cast<real_of<T>>(boost.tol);
        const T                       val_s = make_value<T>(boost.val, 0.0);
        const double                  tol_d = boost.tol;
        const float                   valf  = static_cast<float>(boost.val);
        const rocsparse_float_complex valc  = make_value<rocsparse_float_complex>(boost.val, 0.0);
        device_vector<real_of<T>>     dtol_s{std::vector<real_of<T>>{tol_s}};
        device_vector<T>              dval_s{std::vector<T>{val_s}};
        device_vector<double>         dtol_d{std::vector<double>{tol_d}};
        device_vector<float>          dvalf{std::vector<float>{valf}};
        device_vector<rocsparse_float_complex> dvalc{std::vector<rocsparse_float_complex>{valc}};
        ASSERT_TRUE(dtol_s.ptr && dval_s.ptr && dtol_d.ptr && dvalf.ptr && dvalc.ptr);
        if(boost.enable)
        {
            if(boost.device)
            {
                ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                          rocsparse_status_success);
            }
            const void* ptol = nullptr;
            const void* pval = nullptr;
            if(boost.api == boost_api::standard)
            {
                ptol = boost.device ? static_cast<const void*>(dtol_s.ptr)
                                    : static_cast<const void*>(&tol_s);
                pval = boost.device ? static_cast<const void*>(dval_s.ptr)
                                    : static_cast<const void*>(&val_s);
            }
            else
            {
                ptol = boost.device ? static_cast<const void*>(dtol_d.ptr)
                                    : static_cast<const void*>(&tol_d);
                if constexpr(std::is_same_v<T, float>)
                    pval = boost.device ? static_cast<const void*>(dvalf.ptr)
                                        : static_cast<const void*>(&valf);
                else
                    pval = boost.device ? static_cast<const void*>(dvalc.ptr)
                                        : static_cast<const void*>(&valc);
            }
            const auto st = set_boost<T>(handle, di.info, 1, boost.api, ptol, pval);
            if(boost.device)
            {
                ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                          rocsparse_status_success);
            }
            ASSERT_EQ(st, rocsparse_status_success);
            // The pointer mode that counts is the one of the call that defined the boost:
            // the compute below runs in host mode.
        }
        else if(boost.api == boost_api::standard && boost.tol != 0.0)
        {
            // Enabled and then disabled again.
            ASSERT_EQ(set_boost<T>(handle, di.info, 1, boost.api, &tol_s, &val_s),
                      rocsparse_status_success);
            ASSERT_EQ(set_boost<T>(handle, di.info, 0, boost.api, nullptr, nullptr),
                      rocsparse_status_success);
        }

        out.compute_status
            = csrilu0_compute<T>(handle, a.n, a.nnz(), di.descr, dval, drow, dcol, di.info, buffer);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        out.values = to_host(dval);

        out.singular_pivot = -7;
        if(device_position)
        {
            device_vector<rocsparse_int> dpos{std::vector<rocsparse_int>{-7}};
            ASSERT_TRUE(dpos.ptr);
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
            const auto st = rocsparse_csrilu0_singular_pivot(handle, di.info, dpos.ptr);
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                      rocsparse_status_success);
            ASSERT_EQ(st, rocsparse_status_success);
            out.singular_pivot = to_host(dpos)[0];
        }
        else
        {
            ASSERT_EQ(rocsparse_csrilu0_singular_pivot(handle, di.info, &out.singular_pivot),
                      rocsparse_status_success);
        }
        ASSERT_EQ(rocsparse_csrilu0_clear(handle, di.info), rocsparse_status_success);
    }

    // Pivots (diagonal of U) of the host ILU(0) factorization of `a`.
    template <typename T>
    std::vector<double> ilu0_pivots(const csr_matrix<T>& a)
    {
        const std::vector<T> factor = host_ilu0(a);
        std::vector<double>  pivots(static_cast<size_t>(a.n), 0.0);
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] == i)
                    pivots[i] = std::abs(to_cd(factor[p]));
        return pivots;
    }

    // First row whose pivot magnitude is at most `tolerance` (-1 if none).
    rocsparse_int expected_singular_row(const std::vector<double>& pivots, double tolerance)
    {
        for(size_t i = 0; i < pivots.size(); ++i)
            if(pivots[i] <= tolerance)
                return static_cast<rocsparse_int>(i);
        return -1;
    }
}

class Cov80PrecondLegacy : public HandleTest
{
};

TEST_F(Cov80PrecondLegacy, csrilu0_tolerance_and_singular_pivot)
{
    const csr_matrix<double> grid     = make_grid_matrix<double>(4, false);
    const auto               pivots   = ilu0_pivots(grid);
    const double             smallest = *std::min_element(pivots.begin(), pivots.end());
    const double             largest  = *std::max_element(pivots.begin(), pivots.end());
    ASSERT_GT(smallest, 1.0);
    ASSERT_LT(smallest, largest);
    // The tolerances sit away from every pivot: one below all pivots, one between the
    // smallest and the largest, one above all of them.
    const double middle = 0.5 * (smallest + largest);
    for(const double pivot : pivots)
        ASSERT_GT(std::abs(pivot - middle), 1.0e-6);
    ASSERT_EQ(expected_singular_row(pivots, 0.5 * smallest), -1);
    ASSERT_EQ(expected_singular_row(pivots, 2.0 * largest), 0);

    for(const auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(const bool device_position : {false, true})
        {
            for(const double tolerance : {-1.0, 0.5 * smallest, middle, 2.0 * largest})
            {
                SCOPED_TRACE(::testing::Message()
                             << "tolerance " << tolerance << " device_position " << device_position
                             << " base " << static_cast<int>(base));
                ilu0_out<double> out;
                run_csrilu0<double>(
                    handle, grid, base, tolerance, boost_request{}, device_position, out);
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                if(tolerance >= 0.0)
                    EXPECT_EQ(out.tolerance_back, tolerance);
                else
                    EXPECT_GE(out.tolerance_back, 0.0);
                // The position is returned in the index base of the matrix (-1: none).
                const rocsparse_int row = (tolerance < 0.0)
                                              ? rocsparse_int(-1)
                                              : expected_singular_row(pivots, tolerance);
                if(tolerance >= 0.0)
                {
                    EXPECT_EQ(out.singular_pivot,
                              row < 0 ? rocsparse_int(-1)
                                      : row + (base == rocsparse_index_base_one ? 1 : 0));
                }
                // The factor is computed whether or not a pivot is flagged.
                expect_close(
                    out.values, host_ilu0(grid), cov70::tolerance<double>(max_row_length(grid)));
            }
        }
    }
}

TEST_F(Cov80PrecondLegacy, csrilu0_tolerance_argument_checks)
{
    descr_info    di;
    double        tol      = 0.0;
    rocsparse_int position = 0;
    EXPECT_EQ(rocsparse_csrilu0_get_tolerance(nullptr, di.info, &tol),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csrilu0_get_tolerance(handle, nullptr, &tol),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csrilu0_get_tolerance(handle, di.info, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csrilu0_set_tolerance(nullptr, di.info, 1.0),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csrilu0_set_tolerance(handle, nullptr, 1.0),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csrilu0_set_tolerance(handle, di.info, -1.0),
              rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_csrilu0_singular_pivot(nullptr, di.info, &position),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csrilu0_singular_pivot(handle, nullptr, &position),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csrilu0_singular_pivot(handle, di.info, nullptr),
              rocsparse_status_invalid_pointer);
    // Without an analysis there is no singular pivot.
    EXPECT_EQ(rocsparse_csrilu0_singular_pivot(handle, di.info, &position),
              rocsparse_status_success);
    EXPECT_EQ(position, -1);
}

TEST_F(Cov80PrecondLegacy, csrilu0_numeric_boost_types_and_pointer_modes)
{
    for_all_types([&](auto tag) {
        using T               = decltype(tag);
        const auto   grid     = make_grid_matrix<T>(4, true);
        const auto   pivots   = ilu0_pivots(grid);
        const double smallest = *std::min_element(pivots.begin(), pivots.end());
        const double largest  = *std::max_element(pivots.begin(), pivots.end());
        ASSERT_LT(smallest, largest);
        // Every pivot is at most 2 * largest (boosted to 7); only the first pivots of the
        // sweep are at most the middle value, later pivots follow from the boosted rows.
        const double                     middle   = 0.5 * (smallest + largest);
        const std::vector<boost_request> requests = {
            // enabled with a tolerance that catches every pivot
            {true, 2.0 * largest, 7.0, false, boost_api::standard},
            {true, 2.0 * largest, 7.0, true, boost_api::standard},
            // a tolerance in the middle of the pivot range
            {true, middle, 7.0, false, boost_api::standard},
            {true, middle, 7.0, true, boost_api::standard},
            // a tolerance below every pivot: nothing is replaced
            {true, 0.5 * smallest, 7.0, false, boost_api::standard},
            // disabled again after having been enabled
            {false, 2.0 * largest, 7.0, false, boost_api::standard},
        };
        for(const auto& req : requests)
        {
            SCOPED_TRACE(::testing::Message() << "enable " << req.enable << " tol " << req.tol
                                              << " device " << req.device);
            ilu0_out<T> out;
            run_csrilu0<T>(handle, grid, rocsparse_index_base_zero, -1.0, req, false, out);
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            const double boost_tol = (req.enable ? req.tol : -1.0);
            expect_close(out.values,
                         host_ilu0(grid, boost_tol, req.val),
                         tolerance<T>(max_row_length(grid)));
        }
    });

    // Mixed precision entry points: a double tolerance with single precision values.
    for(const bool device : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "mixed, device " << device);
        {
            const auto      grid   = make_grid_matrix<float>(4, true);
            const auto      pivots = ilu0_pivots(grid);
            const double    tol    = 2.0 * *std::max_element(pivots.begin(), pivots.end());
            ilu0_out<float> out;
            run_csrilu0<float>(handle,
                               grid,
                               rocsparse_index_base_zero,
                               -1.0,
                               {true, tol, 7.0, device, boost_api::mixed},
                               false,
                               out);
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            expect_close(
                out.values, host_ilu0(grid, tol, 7.0), tolerance<float>(max_row_length(grid)));
        }
        {
            const auto   grid   = make_grid_matrix<rocsparse_float_complex>(4, true);
            const auto   pivots = ilu0_pivots(grid);
            const double tol    = 2.0 * *std::max_element(pivots.begin(), pivots.end());
            ilu0_out<rocsparse_float_complex> out;
            run_csrilu0<rocsparse_float_complex>(handle,
                                                 grid,
                                                 rocsparse_index_base_zero,
                                                 -1.0,
                                                 {true, tol, 7.0, device, boost_api::mixed},
                                                 false,
                                                 out);
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            expect_close(
                out.values, host_ilu0(grid, tol, 7.0), tolerance<float>(max_row_length(grid)));
        }
    }
}

TEST_F(Cov80PrecondLegacy, csrilu0_numeric_boost_argument_checks)
{
    descr_info                     di;
    const float                    tol_f = 1.0f;
    const float                    val_f = 2.0f;
    const double                   tol_d = 1.0;
    const double                   val_d = 2.0;
    const rocsparse_float_complex  val_c = make_value<rocsparse_float_complex>(2.0, 0.0);
    const rocsparse_double_complex val_z = make_value<rocsparse_double_complex>(2.0, 0.0);

    EXPECT_EQ(rocsparse_scsrilu0_numeric_boost(nullptr, di.info, 1, &tol_f, &val_f),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_dcsrilu0_numeric_boost(handle, nullptr, 1, &tol_d, &val_d),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_ccsrilu0_numeric_boost(handle, di.info, 1, nullptr, &val_c),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_zcsrilu0_numeric_boost(handle, di.info, 1, &tol_d, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dscsrilu0_numeric_boost(handle, di.info, 1, nullptr, &val_f),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dccsrilu0_numeric_boost(handle, di.info, 1, &tol_d, nullptr),
              rocsparse_status_invalid_pointer);
    // Disabling needs no values.
    EXPECT_EQ(rocsparse_scsrilu0_numeric_boost(handle, di.info, 0, nullptr, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_zcsrilu0_numeric_boost(handle, di.info, 1, &tol_d, &val_z),
              rocsparse_status_success);
}
