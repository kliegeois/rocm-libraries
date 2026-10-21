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

//
// Device unit tests that drive the public C API dispatch of the generic sparse
// products and triangular solves: the scalar conversions of SpTRSV.
//
// The matrices are small bidiagonal matrices with distinct entries, the vectors have
// distinct entries and alpha != 1, so a transposed operation or a dropped scalar
// changes the result, which is compared with a host reference.
//
#include "unit_test_device_level2_common.hpp"

using namespace rocsparse_ut;
using namespace level2;

namespace
{
    constexpr auto f32  = rocsparse_datatype_f32_r;
    constexpr auto f64  = rocsparse_datatype_f64_r;
    constexpr auto op_n = rocsparse_operation_none;
    constexpr auto op_t = rocsparse_operation_transpose;
    constexpr auto lo   = rocsparse_fill_mode_lower;
    constexpr auto nu   = rocsparse_diag_type_non_unit;

    // ----- staged calls ------------------------------------------------------------

    // True when the next stage can run. A configuration documented as unsupported
    // may stop with not_implemented, any other status is a failure.
    bool proceed(rocsparse_status status, bool unsupported, const char* what)
    {
        if(status == rocsparse_status_success)
            return true;
        if(unsupported)
            EXPECT_EQ(status, rocsparse_status_not_implemented) << what;
        else
            ADD_FAILURE() << what << " returned status " << status;
        return false;
    }

    // v2 SpMV, SpTRSV and SpTRSM: every stage queries its own buffer size.
    // Returns true when the last stage ran.
    template <typename Stage, typename Size, typename Run>
    bool run_stages(std::initializer_list<Stage> stages, Size size, Run run, bool unsupported)
    {
        for(const Stage stage : stages)
        {
            size_t bytes = 0;
            if(!proceed(size(stage, &bytes), unsupported, "buffer_size"))
                return false;
            device_vector<char> buffer(std::max<size_t>(bytes, 1));
            EXPECT_NE(buffer.ptr, nullptr);
            if(!proceed(run(stage, bytes, (bytes && buffer.ptr) ? buffer.ptr : nullptr),
                        unsupported,
                        "stage"))
                return false;
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        }
        return true;
    }

    // ----- data ----------------------------------------------------------------

    template <typename T>
    std::vector<T> values(size_t count, int64_t seed)
    {
        std::vector<T> v(count);
        for(size_t i = 0; i < count; ++i)
            v[i] = value_of<T>(seed + static_cast<int64_t>(i), true);
        return v;
    }

    rocsparse_status
        set_triangular(rocsparse_spmat_descr A, rocsparse_fill_mode fill, rocsparse_diag_type diag)
    {
        const rocsparse_status st
            = rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill, sizeof(fill));
        return st != rocsparse_status_success
                   ? st
                   : rocsparse_spmat_set_attribute(
                       A, rocsparse_spmat_diag_type, &diag, sizeof(diag));
    }

    template <typename V>
    rocsparse_status trs_input(rocsparse_handle       h,
                               rocsparse_sptrsv_descr d,
                               rocsparse_sptrsv_input input,
                               const V&               value)
    {
        return rocsparse_sptrsv_set_input(h, d, input, &value, sizeof(V), nullptr);
    }

    // ----- SpTRSV ----------------------------------------------------------------------

    struct sptrsv_options
    {
        rocsparse_operation       op          = op_n;
        rocsparse_datatype        scalar      = f32;
        rocsparse_datatype        compute     = f32;
        rocsparse_analysis_policy policy      = rocsparse_analysis_policy_reuse;
        const void*               alpha       = nullptr;
        bool                      diagonal    = false; // diagonal solve instead of triangular
        bool                      unsupported = false;
        rocsparse_singularity     singularity = rocsparse_singularity_none;
        int64_t                   singularity_position = -1;
    };

    bool
        configure_sptrsv(rocsparse_handle handle, rocsparse_sptrsv_descr d, const sptrsv_options& o)
    {
        bool ok  = true;
        auto put = [&](rocsparse_sptrsv_input input, const auto& value) {
            ok = ok && trs_input(handle, d, input, value) == rocsparse_status_success;
        };
        put(rocsparse_sptrsv_input_alg, rocsparse_sptrsv_alg_default);
        put(rocsparse_sptrsv_input_operation, o.op);
        put(rocsparse_sptrsv_input_scalar_datatype, o.scalar);
        put(rocsparse_sptrsv_input_compute_datatype, o.compute);
        put(rocsparse_sptrsv_input_analysis_policy, o.policy);
        // the pointer itself is the input
        ok = ok
             && rocsparse_sptrsv_set_input(handle,
                                           d,
                                           rocsparse_sptrsv_input_scalar_alpha,
                                           o.alpha,
                                           sizeof(const void*),
                                           nullptr)
                    == rocsparse_status_success;
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        put(rocsparse_sptrsv_input_solve_mode,
            o.diagonal ? rocsparse_solve_mode_diagonal : rocsparse_solve_mode_triangular);
        if(o.diagonal)
            put(rocsparse_sptrsv_input_diagonal_modifier, rocsparse_diagonal_modifier_absolute);
#endif
        return ok;
    }

    // Returns true when the solve ran.
    bool run_sptrsv(rocsparse_handle      handle,
                    rocsparse_spmat_descr A,
                    rocsparse_dnvec_descr x,
                    rocsparse_dnvec_descr y,
                    const sptrsv_options& o)
    {
        sptrsv_descr_guard descr;
        EXPECT_NE(descr.d, nullptr);
        if(!descr.d
           || !(configure_sptrsv(handle, descr.d, o) || (ADD_FAILURE() << "configure", false)))
            return false;

        // The outputs are written through host pointers.
        int64_t               position    = -1;
        rocsparse_singularity singularity = rocsparse_singularity_none;
        auto                  query       = [&](rocsparse_sptrsv_output output, auto* value) {
            scoped_pointer_mode host(handle, false);
            return rocsparse_sptrsv_get_output(
                handle, descr.d, output, value, sizeof(*value), nullptr);
        };
        // Before the analysis: only the query path is exercised, the header does not
        // specify the result.
        (void)query(rocsparse_sptrsv_output_singularity_position, &position);
        (void)query(rocsparse_sptrsv_output_singularity, &singularity);

        const bool ran = run_stages(
            {rocsparse_sptrsv_stage_analysis, rocsparse_sptrsv_stage_compute},
            [&](rocsparse_sptrsv_stage stage, size_t* bytes) {
                return rocsparse_sptrsv_buffer_size(
                    handle, descr.d, A, x, y, stage, bytes, nullptr);
            },
            [&](rocsparse_sptrsv_stage stage, size_t bytes, void* buffer) {
                return rocsparse_sptrsv(handle, descr.d, A, x, y, stage, bytes, buffer, nullptr);
            },
            o.unsupported);
        if(!ran)
            return false;

        EXPECT_EQ(query(rocsparse_sptrsv_output_singularity_position, &position),
                  rocsparse_status_success);
        EXPECT_EQ(query(rocsparse_sptrsv_output_singularity, &singularity),
                  rocsparse_status_success);
        EXPECT_EQ(singularity, o.singularity);
        EXPECT_EQ(position, o.singularity_position);
        const rocsparse_status st = query(rocsparse_sptrsv_output_zero_pivot_position, &position);
        EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_zero_pivot);
        return true;
    }

    class L2DeviceSptrsvMisc : public HandleTest
    {
    };

    TEST_F(L2DeviceSptrsvMisc, scalar_type_differs_from_compute_type)
    {
        // The matrix and the vectors are f64, the scalar is f32.
        const int32_t             n  = 3;
        const host_csr<double>    a  = make_triangular_pattern<double>(n, 1, true, false, true);
        const std::vector<double> x0 = values<double>(n, 100);
        const std::vector<double> y0 = values<double>(n, 200);
        device_vector<int32_t>    ptr(a.ptr), ind(a.ind);
        device_vector<double>     val(a.val), dx(x0), dy(y0);
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && dx.ptr && dy.ptr);

        const auto  it = rocsparse_indextype_i32;
        spmat_guard A;
        dnvec_guard x, y;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &A.d, n, n, a.nnz(), ptr, ind, val, it, it, rocsparse_index_base_zero, f64),
                  rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, lo, nu), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&x.d, n, dx, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&y.d, n, dy, f64), rocsparse_status_success);

        // Every solve starts from y0, so each check is independent of the previous run.
        auto reset_y = [&] {
            EXPECT_EQ(hipMemcpy(dy.ptr, y0.data(), n * sizeof(double), hipMemcpyHostToDevice),
                      hipSuccess);
        };

        const float           alpha_host = 2.0f;
        device_vector<float>  alpha_device_f32(std::vector<float>(1, 2.0f));
        device_vector<double> alpha_device(std::vector<double>(1, 3.0));
        ASSERT_TRUE(alpha_device_f32.ptr && alpha_device.ptr);

        sptrsv_options o;
        o.scalar  = f32;
        o.compute = f64;
        o.alpha   = &alpha_host;
        {
            scoped_pointer_mode host(handle, false);
            reset_y();
            if(run_sptrsv(handle, A.d, x.d, y.d, o))
                expect_near(host_triangular_solve(a, lo, nu, op_n, 2.0, x0), to_host(dy), "sptrsv");
        }
        scoped_pointer_mode device(handle, true);
        o.op     = op_t;
        o.policy = rocsparse_analysis_policy_force;
        // An f32 device scalar is converted to the f64 compute type.
        o.alpha = alpha_device_f32.ptr;
        reset_y();
        if(run_sptrsv(handle, A.d, x.d, y.d, o))
            expect_near(host_triangular_solve(a, lo, nu, op_t, 2.0, x0),
                        to_host(dy),
                        "sptrsv device f32 alpha");
        o.scalar = f64;
        o.alpha  = alpha_device.ptr;
        reset_y();
        if(run_sptrsv(handle, A.d, x.d, y.d, o))
            expect_near(host_triangular_solve(a, lo, nu, op_t, 3.0, x0),
                        to_host(dy),
                        "sptrsv device alpha");
    }
} // namespace
