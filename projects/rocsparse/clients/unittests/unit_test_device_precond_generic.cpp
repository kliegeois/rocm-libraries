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

// Generic incomplete-factorization API (rocsparse_spildlt0): numeric boost. Every
// factorization is compared with a host reference.

#include "unit_test_device_precond_common.hpp"

#include <algorithm>
#include <string>

using namespace precond_ut;

class PrecondGeneric : public HandleTest
{
};

// ---------------------------------------------------------------------------
// Numeric boost
// ---------------------------------------------------------------------------
namespace
{
    // On the grid matrix (positive pivots) a boost tolerance above every pivot replaces
    // each of them by the boost value, one below every pivot replaces none. The pivots
    // are the entries of D.
    template <typename T>
    void check_ildlt_boost(rocsparse_handle handle)
    {
        const auto           a     = make_grid_matrix<T>(4, false);
        const std::vector<T> plain = host_ildlt0(a);
        std::vector<double>  pivots;
        for(int32_t r = 0; r < a.n; ++r)
            pivots.push_back(std::abs(to_cd(plain[diagonal_position(a, r)])));
        const double largest  = *std::max_element(pivots.begin(), pivots.end());
        const double smallest = *std::min_element(pivots.begin(), pivots.end());
        for(const double tol : {2.0 * largest, 0.5 * smallest})
        {
            run_options opt;
            opt.boost     = true;
            opt.boost_tol = tol;
            opt.boost_val = 7.0;
            check_csr_matrix<Kind::ildlt, T>(handle, a, opt, "boost tol " + std::to_string(tol));
        }
    }

    // spildlt0 of `a` with the boost enabled, a boost value of 7 and the boost tolerance
    // `tol` passed as a float (`float_tol`) or a double. With `device`, the tolerance and
    // the value are device pointers, set in rocsparse_pointer_mode_device. Returns the
    // factor.
    template <typename T>
    void run_ildlt_boost(rocsparse_handle     handle,
                         const csr_matrix<T>& a,
                         double               tol,
                         bool                 float_tol,
                         bool                 device,
                         std::vector<T>&      factor)
    {
        dev_csr<T, int32_t, int32_t> d;
        d.init(a, 1, true);
        ASSERT_NE(d.A, nullptr);
        descr_guard<Kind::ildlt> guard(handle);
        auto&                    descr = guard.descr;
        ASSERT_EQ(api<Kind::ildlt>::create(handle, &descr), rocsparse_status_success);
        const rocsparse_datatype dtype = dt_of<T>();
        ASSERT_EQ((set_value<Kind::ildlt>(handle, descr, In::dtype, &dtype, sizeof(dtype))),
                  rocsparse_status_success);
        const int32_t alg = 0;
        ASSERT_EQ((set_value<Kind::ildlt>(handle, descr, In::alg, &alg, sizeof(int32_t))),
                  rocsparse_status_success);
        const run_options opt;
        ASSERT_EQ(
            (set_value<Kind::ildlt>(handle, descr, In::policy, &opt.policy, sizeof(opt.policy))),
            rocsparse_status_success);
        const int32_t enable = 1;
        ASSERT_EQ(
            (set_value<Kind::ildlt>(handle, descr, In::boost_enable, &enable, sizeof(int32_t))),
            rocsparse_status_success);

        const float           tol_s = static_cast<float>(tol);
        const double          tol_d = tol;
        const T               val   = make_value<T>(7.0);
        device_vector<float>  dtol_s(std::vector<float>{tol_s});
        device_vector<double> dtol_d(std::vector<double>{tol_d});
        device_vector<T>      dval(std::vector<T>{val});
        const void*           tol_ptr = device ? (float_tol ? static_cast<const void*>(dtol_s.ptr)
                                                            : static_cast<const void*>(dtol_d.ptr))
                                               : (float_tol ? static_cast<const void*>(&tol_s)
                                                            : static_cast<const void*>(&tol_d));
        const void*           val_ptr
            = device ? static_cast<const void*>(dval.ptr) : static_cast<const void*>(&val);
        ASSERT_TRUE(dtol_s.ptr && dtol_d.ptr && dval.ptr);
        if(device)
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
        const rocsparse_status tol_status = set_value<Kind::ildlt>(
            handle, descr, In::boost_tol, tol_ptr, float_tol ? sizeof(float) : sizeof(double));
        const rocsparse_status val_status
            = set_value<Kind::ildlt>(handle, descr, In::boost_val, val_ptr, sizeof(void*));
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        ASSERT_EQ(tol_status, rocsparse_status_success);
        ASSERT_EQ(val_status, rocsparse_status_success);

        for(const Stage stage : {Stage::analysis, Stage::compute})
        {
            size_t size = 0;
            ASSERT_EQ(api<Kind::ildlt>::buffer_size(
                          handle, descr, d.A, d.A, api<Kind::ildlt>::stage(stage), &size),
                      rocsparse_status_success);
            device_vector<char> buffer{size ? size : size_t{1}};
            ASSERT_TRUE(buffer.ptr);
            ASSERT_EQ(api<Kind::ildlt>::run(handle,
                                            descr,
                                            d.A,
                                            d.A,
                                            api<Kind::ildlt>::stage(stage),
                                            size,
                                            size ? buffer.ptr : nullptr),
                      rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        factor = to_host(d.val);
    }

    // The boost of every pivot or of none, with the tolerance passed as a float or a
    // double, in host or device memory, through the hash kernel (the grid matrix) and the
    // binary-search kernel (a last row of 600 entries).
    template <typename T>
    void check_ildlt_boost_inputs(rocsparse_handle handle)
    {
        for(const auto& a : {make_grid_matrix<T>(4, false), make_fat_row_matrix<T>(600)})
        {
            const std::vector<T> plain = host_ildlt0(a);
            std::vector<double>  pivots;
            for(int32_t r = 0; r < a.n; ++r)
                pivots.push_back(std::abs(to_cd(plain[diagonal_position(a, r)])));
            const double largest  = *std::max_element(pivots.begin(), pivots.end());
            const double smallest = *std::min_element(pivots.begin(), pivots.end());
            for(const double tol : {2.0 * largest, 0.5 * smallest})
            {
                for(const bool float_tol : {false, true})
                {
                    for(const bool device : {false, true})
                    {
                        SCOPED_TRACE(::testing::Message()
                                     << "n " << a.n << " tol " << tol << " float_tol " << float_tol
                                     << " device " << device);
                        std::vector<T> got;
                        run_ildlt_boost(handle, a, tol, float_tol, device, got);
                        expect_close(
                            got, host_ildlt0(a, tol, 7.0), tolerance<T>(max_row_length(a)));
                    }
                }
            }
        }
    }
}

TEST_F(PrecondGeneric, spildlt0_numeric_boost)
{
    for_all_types([&](auto tag) { check_ildlt_boost<decltype(tag)>(handle); });
}

TEST_F(PrecondGeneric, spildlt0_numeric_boost_inputs)
{
    for_all_types([&](auto tag) { check_ildlt_boost_inputs<decltype(tag)>(handle); });
}

// The boost enabled without a boost tolerance or value: the analysis succeeds and the
// compute stage returns rocsparse_status_invalid_pointer.
TEST_F(PrecondGeneric, spildlt0_numeric_boost_unset)
{
    const auto                        a = make_grid_matrix<double>(4, false);
    dev_csr<double, int32_t, int32_t> d;
    d.init(a, 1, true);
    ASSERT_NE(d.A, nullptr);
    descr_guard<Kind::ildlt> guard(handle);
    auto&                    descr = guard.descr;
    ASSERT_EQ(api<Kind::ildlt>::create(handle, &descr), rocsparse_status_success);
    const rocsparse_datatype dtype = dt_of<double>();
    ASSERT_EQ((set_value<Kind::ildlt>(handle, descr, In::dtype, &dtype, sizeof(dtype))),
              rocsparse_status_success);
    const int32_t alg = 0;
    ASSERT_EQ((set_value<Kind::ildlt>(handle, descr, In::alg, &alg, sizeof(int32_t))),
              rocsparse_status_success);
    const run_options opt;
    ASSERT_EQ((set_value<Kind::ildlt>(handle, descr, In::policy, &opt.policy, sizeof(opt.policy))),
              rocsparse_status_success);
    const int32_t enable = 1;
    ASSERT_EQ((set_value<Kind::ildlt>(handle, descr, In::boost_enable, &enable, sizeof(int32_t))),
              rocsparse_status_success);

    for(const Stage stage : {Stage::analysis, Stage::compute})
    {
        size_t size = 0;
        ASSERT_EQ(api<Kind::ildlt>::buffer_size(
                      handle, descr, d.A, d.A, api<Kind::ildlt>::stage(stage), &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        EXPECT_EQ(api<Kind::ildlt>::run(handle,
                                        descr,
                                        d.A,
                                        d.A,
                                        api<Kind::ildlt>::stage(stage),
                                        size,
                                        size ? buffer.ptr : nullptr),
                  stage == Stage::analysis ? rocsparse_status_success
                                           : rocsparse_status_invalid_pointer);
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}
