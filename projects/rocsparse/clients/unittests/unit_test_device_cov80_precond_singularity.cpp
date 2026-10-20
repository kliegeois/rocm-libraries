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

// Consistency checks of the singularity read-back (singularity_get_async and
// singularity_get_position_async): the batch counts and the index types recorded by
// the exact and near singularity trackers must agree with the requested batch; the
// call is refused otherwise instead of reading mismatched buffers.

#include "unit_test_device_cov70_precond_common.hpp"

#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_singularity.hpp"

#include <vector>

using namespace cov70;

class Cov80PrecondSingularity : public HandleTest
{
};

namespace
{
    struct trackers
    {
        rocsparse::singular_info_t near4, near3, exact4, exact3;
        rocsparse::singular_info_t exact4_wide, near4_wide;

        explicit trackers(hipStream_t stream)
        {
            EXPECT_EQ(near4.create_singular_pivot_async(4, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(near3.create_singular_pivot_async(3, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(exact4.create_singular_pivot_async(4, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(exact3.create_singular_pivot_async(3, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(exact4_wide.create_singular_pivot_async(4, rocsparse_indextype_i64, stream),
                      rocsparse_status_success);
            EXPECT_EQ(near4_wide.create_singular_pivot_async(4, rocsparse_indextype_i64, stream),
                      rocsparse_status_success);
        }
    };
}

TEST_F(Cov80PrecondSingularity, position_read_back_checks_the_trackers)
{
    trackers t(handle->stream);
    int32_t  host_position[4]   = {-5, -5, -5, -5};
    int64_t  host_position64[4] = {-5, -5, -5, -5};
    auto     read               = [&](int64_t                           batch,
                    const rocsparse::singular_info_t* exact,
                    const rocsparse::singular_info_t* near,
                    bool                              wide) {
        return rocsparse::singularity_get_position_async(
            handle,
            batch,
            nullptr,
            exact,
            near,
            rocsparse_pointer_mode_host,
            wide ? rocsparse_indextype_i64 : rocsparse_indextype_i32,
            wide ? static_cast<void*>(host_position64) : static_cast<void*>(host_position));
    };

    // A tracker of another batch than the requested one.
    EXPECT_NE(read(4, nullptr, &t.near3, false), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, nullptr, false), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, &t.near4, false), rocsparse_status_success);

    // Trackers that disagree on the type of the positions.
    EXPECT_NE(read(4, &t.exact4_wide, &t.near4, false), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact4, &t.near4_wide, true), rocsparse_status_success);

    // Consistent trackers are read back.
    EXPECT_EQ(read(4, &t.exact4, &t.near4, false), rocsparse_status_success);
    EXPECT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
    EXPECT_EQ(read(4, &t.exact4_wide, &t.near4_wide, true), rocsparse_status_success);
    EXPECT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
}

TEST_F(Cov80PrecondSingularity, singularity_read_back_checks_the_trackers)
{
    trackers              t(handle->stream);
    rocsparse_singularity host_singularity[4];
    auto                  read = [&](int64_t                           batch,
                    const rocsparse::singular_info_t* exact,
                    const rocsparse::singular_info_t* near) {
        return rocsparse::singularity_get_async(
            handle, batch, nullptr, exact, near, rocsparse_pointer_mode_host, host_singularity);
    };

    EXPECT_NE(read(4, nullptr, &t.near3), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, nullptr), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, &t.near4), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact4_wide, &t.near4), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact4, &t.near4_wide), rocsparse_status_success);

    EXPECT_EQ(read(4, &t.exact4, &t.near4), rocsparse_status_success);
    EXPECT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
    for(const auto s : host_singularity)
    {
        EXPECT_TRUE(s == rocsparse_singularity_numeric_exact
                    || s == rocsparse_singularity_numeric_near
                    || s == rocsparse_singularity_symbolic || s == rocsparse_singularity_none)
            << "unexpected singularity " << static_cast<int>(s);
    }
}
