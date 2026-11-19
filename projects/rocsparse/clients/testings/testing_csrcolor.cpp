/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

#include "rocsparse_enum.hpp"

template <typename T>
void testing_csrcolor_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_local_mat_descr local_descr;
    rocsparse_local_mat_info  local_info;

    rocsparse_handle          handle            = local_handle;
    rocsparse_int             m                 = 4;
    rocsparse_int             nnz               = 4;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  csr_val           = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind       = (const rocsparse_int*)0x4;
    const floating_data_t<T>* fraction_to_color = (const floating_data_t<T>*)0x4;
    rocsparse_int*            ncolors           = (rocsparse_int*)0x4;
    rocsparse_int*            coloring          = (rocsparse_int*)0x4;
    rocsparse_int*            reordering        = nullptr;
    rocsparse_mat_info        info              = local_info;

#define PARAMS                                                                            \
    handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, fraction_to_color, ncolors, \
        coloring, reordering, info

    { //
        // Exclude reordering since this is an optional argument.
        //
        static constexpr int nargs_to_exclude                  = 1;
        static constexpr int args_to_exclude[nargs_to_exclude] = {10};

        select_bad_arg_analysis(rocsparse_csrcolor<T>,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                m,
                                nnz,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                fraction_to_color,
                                ncolors,
                                coloring,
                                reordering,
                                info);
    }

    //
    // Not implemented cases.
    //
    for(auto val : rocsparse_matrix_type_t::values)
    {
        if(val != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, val));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(PARAMS),
                                    rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(PARAMS),
                            rocsparse_status_requires_sorted_storage);

#undef PARAMS
}

// Disjoint triangles. The first Jones-Plassmann-Luby round colors exactly two vertices
// of every triangle (the local maximum and the local minimum of the vertex weights)
// and leaves the third, whatever the weights are, so the result does not depend on
// the hash. With Mt = 3 * (M / 3) vertices the number of colors is known in closed
// form: when fraction_to_color allows Mt / 3 uncolored vertices the loop stops after
// one round and every leftover gets its own color (2 + Mt / 3); otherwise a second
// round colors all the leftovers alike (3). This pins the fraction, which the random
// matrix cannot.
template <typename T>
static void testing_csrcolor_triangles(rocsparse_handle         handle,
                                       rocsparse_mat_descr      descr,
                                       rocsparse_mat_info       info,
                                       rocsparse_int            M,
                                       rocsparse_index_base     base,
                                       const floating_data_t<T> fraction_to_color)
{
    const rocsparse_int Mt   = 3 * (M / 3);
    const rocsparse_int ntri = Mt / 3;
    if(ntri < 1)
    {
        return;
    }

    host_csr_matrix<T> hA(Mt, Mt, 2 * Mt, base);
    rocsparse_int      k = 0;
    for(rocsparse_int v = 0; v < Mt; ++v)
    {
        const rocsparse_int t = v / 3;
        hA.ptr[v]             = k + base;
        for(rocsparse_int w = 3 * t; w < 3 * t + 3; ++w)
        {
            if(w != v)
            {
                hA.ind[k] = w + base;
                hA.val[k] = static_cast<T>(1);
                ++k;
            }
        }
    }
    hA.ptr[Mt] = k + base;

    device_csr_matrix<T>               dA(hA);
    device_dense_vector<rocsparse_int> dcoloring(Mt);
    rocsparse_int                      ncolor;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrcolor<T>(handle,
                                                dA.m,
                                                dA.nnz,
                                                descr,
                                                dA.val,
                                                dA.ptr,
                                                dA.ind,
                                                &fraction_to_color,
                                                &ncolor,
                                                dcoloring,
                                                nullptr,
                                                info));

    // Same arithmetic as the library: the loop runs while more than
    // max_num_uncolored vertices are uncolored.
    const rocsparse_int max_num_uncolored = static_cast<rocsparse_int>(Mt - Mt * fraction_to_color);
    const bool          one_round         = (ntri <= max_num_uncolored);
    unit_check_scalar<rocsparse_int>(one_round ? 2 + ntri : 3, ncolor);

    // Every triangle carries the colors 0 and 1 once each; its third vertex has the
    // color 2 + t (the leftovers are numbered in increasing vertex order) after one
    // round and the color 2 after two.
    host_dense_vector<rocsparse_int> hcoloring(dcoloring);
    for(rocsparse_int t = 0; t < ntri; ++t)
    {
        rocsparse_int count0 = 0, count1 = 0, third = -1;
        for(rocsparse_int v = 3 * t; v < 3 * t + 3; ++v)
        {
            if(hcoloring[v] == 0)
            {
                ++count0;
            }
            else if(hcoloring[v] == 1)
            {
                ++count1;
            }
            else
            {
                third = hcoloring[v];
            }
        }
        unit_check_scalar<rocsparse_int>(1, count0);
        unit_check_scalar<rocsparse_int>(1, count1);
        unit_check_scalar<rocsparse_int>(one_round ? 2 + t : 2, third);
    }
}

template <typename T>
void testing_csrcolor(const Arguments& arg)
{
    //
    // Create the matrix factory.
    //
    rocsparse_matrix_factory<T> matrix_factory(arg);

    //
    // Get dimensions.
    //
    rocsparse_int            M                 = arg.M;
    rocsparse_index_base     csr_base          = arg.baseA;
    const floating_data_t<T> fraction_to_color = arg.get_percentage<T>() / 100;

    //
    // Local variables.
    //
    rocsparse_local_handle    handle;
    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_info  mat_info;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));

    rocsparse_int ncolor;

    //
    // Argument sanity check before allocating invalid memory
    //
    if(M == 0)
    {
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(handle,
                                                      0,
                                                      7,
                                                      csr_descr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(handle,
                                                      7,
                                                      0,
                                                      csr_descr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr),
                                rocsparse_status_success);
        return;
    }

    //
    // Init a CSR symmetric matrix.
    //
    host_csr_matrix<T> hA;
    {
        host_csr_matrix<T> nonsymA;
        matrix_factory.init_csr(nonsymA, M, M, csr_base);
        CHECK_ROCSPARSE_ERROR(rocsparse_matrix_utils::host_csrsym(nonsymA, hA));
    }

    //
    // Allocate device memory and transfer data from host.
    //
    device_csr_matrix<T>               dA(hA);
    device_dense_vector<rocsparse_int> dcoloring(hA.m);
    device_dense_vector<rocsparse_int> dreordering(hA.m);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrcolor<T>(handle,
                                                    dA.m,
                                                    dA.nnz,
                                                    csr_descr,
                                                    dA.val,
                                                    dA.ptr,
                                                    dA.ind,
                                                    &fraction_to_color,
                                                    &ncolor,
                                                    dcoloring,
                                                    dreordering,
                                                    mat_info));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dcoloring", dcoloring, "dreordering", dreordering);
        }

        //
        // CHECK CONSISTENCY: COUNT NUMBER OF COLORS IN HCOLORING
        //
        host_dense_vector<rocsparse_int> hcoloring(dcoloring);

        //
        // CHECK CONSISTENCY: CHECK ANY COLOR NOT BEING SHARED BY TWO ADJACENT NODES.
        //
        for(rocsparse_int i = 0; i < M; ++i)
        {
            auto icolor = hcoloring[i];
            for(rocsparse_int at = hA.ptr[i] - hA.base; at < hA.ptr[i + 1] - hA.base; ++at)
            {
                auto j = hA.ind[at] - hA.base;
                if(i != j)
                {
                    auto jcolor = hcoloring[j];
                    EXPECT_ROCSPARSE_STATUS((icolor != jcolor) ? rocsparse_status_success
                                                               : rocsparse_status_internal_error,
                                            rocsparse_status_success);
                }
            }
        }

        //
        // Verification of the number of colors by counting them.
        //

        //
        // Check if colors are contiguous
        //
        {
            rocsparse_int max_value = 0;
            for(rocsparse_int i = 0; i < hcoloring.size(); ++i)
            {
                //
                // Check value is well defined.
                //
                EXPECT_ROCSPARSE_STATUS((hcoloring[i] >= 0 && hcoloring[i] < M)
                                            ? rocsparse_status_success
                                            : rocsparse_status_internal_error,
                                        rocsparse_status_success);

                //
                // Calculate maximum value.
                //
                if(hcoloring[i] > max_value)
                {
                    max_value = hcoloring[i];
                }
            }
            ++max_value;

            bool* marker = new bool[max_value];
            for(rocsparse_int i = 0; i < max_value; ++i)
            {
                marker[i] = false;
            }

            for(rocsparse_int i = 0; i < hcoloring.size(); ++i)
            {
                marker[hcoloring[i]] = true;
            }

            for(rocsparse_int i = 0; i < max_value; ++i)
            {
                EXPECT_ROCSPARSE_STATUS(marker[i] ? rocsparse_status_success
                                                  : rocsparse_status_internal_error,
                                        rocsparse_status_success);
            }

            delete[] marker;

            //
            // Compare the number of colors.
            //
            unit_check_scalar(max_value, ncolor);
        }

        if(dreordering)
        {
            //
            // Need to verify this is a valid permutation array..
            //
            host_dense_vector<rocsparse_int> hreordering(dreordering);
            host_dense_vector<rocsparse_int> cache(M);
            for(rocsparse_int i = 0; i < M; ++i)
            {
                cache[i] = 0;
            }

            for(rocsparse_int i = 0; i < M; ++i)
            {
                EXPECT_ROCSPARSE_STATUS((hreordering[i] >= 0 && hreordering[i] < M)
                                            ? rocsparse_status_success
                                            : rocsparse_status_internal_error,
                                        rocsparse_status_success);

                cache[hreordering[i]] = 1;
            }

            for(rocsparse_int i = 0; i < M; ++i)
            {
                EXPECT_ROCSPARSE_STATUS((cache[i] != 0) ? rocsparse_status_success
                                                        : rocsparse_status_internal_error,
                                        rocsparse_status_success);
            }
        }

        testing_csrcolor_triangles<T>(handle, csr_descr, mat_info, M, csr_base, fraction_to_color);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csrcolor<T>,
                                                                      handle,
                                                                      dA.m,
                                                                      dA.nnz,
                                                                      csr_descr,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      &fraction_to_color,
                                                                      &ncolor,
                                                                      dcoloring,
                                                                      dreordering,
                                                                      mat_info);

        display_timing_info(display_key_t::M,
                            dA.m,
                            display_key_t::nnz,
                            dA.nnz,
                            "frac",
                            fraction_to_color,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                               \
    template void testing_csrcolor_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrcolor<TYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrcolor_extra(const Arguments& arg) {}
