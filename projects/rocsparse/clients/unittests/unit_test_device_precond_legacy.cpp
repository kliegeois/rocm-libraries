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

// The mat_info based preconditioner entry points that the generic descriptor API does
// not use: the numerical tolerance and singular pivot of csrilu0 / csric0.

#include "unit_test_device_precond_common.hpp"

using namespace precond_ut;

namespace
{
    template <Kind K>
    rocsparse_status get_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double* tol)
    {
        return (K == Kind::ilu) ? rocsparse_csrilu0_get_tolerance(handle, info, tol)
                                : rocsparse_csric0_get_tolerance(handle, info, tol);
    }

    template <Kind K>
    rocsparse_status set_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double tol)
    {
        return (K == Kind::ilu) ? rocsparse_csrilu0_set_tolerance(handle, info, tol)
                                : rocsparse_csric0_set_tolerance(handle, info, tol);
    }

    template <Kind K>
    rocsparse_status
        singular_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position)
    {
        return (K == Kind::ilu) ? rocsparse_csrilu0_singular_pivot(handle, info, position)
                                : rocsparse_csric0_singular_pivot(handle, info, position);
    }
}

template <typename KindTag>
class PrecondLegacyCsr : public KindFixture<KindTag>
{
};
TYPED_TEST_SUITE(PrecondLegacyCsr, IluIc);

// ---------------------------------------------------------------------------
// csrilu0 / csric0 tolerance and singular pivot
// ---------------------------------------------------------------------------

TYPED_TEST(PrecondLegacyCsr, tolerance_argument_checks)
{
    constexpr Kind K      = TestFixture::K;
    const auto     handle = this->handle;
    descr_info     di;
    double         tol      = 0.0;
    rocsparse_int  position = 0;
    EXPECT_EQ(get_tolerance<K>(nullptr, di.info, &tol), rocsparse_status_invalid_handle);
    EXPECT_EQ(get_tolerance<K>(handle, nullptr, &tol), rocsparse_status_invalid_pointer);
    EXPECT_EQ(get_tolerance<K>(handle, di.info, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(set_tolerance<K>(nullptr, di.info, 1.0), rocsparse_status_invalid_handle);
    EXPECT_EQ(set_tolerance<K>(handle, nullptr, 1.0), rocsparse_status_invalid_pointer);
    EXPECT_EQ(set_tolerance<K>(handle, di.info, 0.5), rocsparse_status_success);
    // A negative tolerance is rejected (the status is not documented).
    EXPECT_NE(set_tolerance<K>(handle, di.info, -1.0), rocsparse_status_success);
    // The rejected tolerance does not replace the stored one.
    EXPECT_EQ(get_tolerance<K>(handle, di.info, &tol), rocsparse_status_success);
    EXPECT_EQ(tol, 0.5);
    EXPECT_EQ(singular_pivot<K>(nullptr, di.info, &position), rocsparse_status_invalid_handle);
    EXPECT_EQ(singular_pivot<K>(handle, nullptr, &position), rocsparse_status_invalid_pointer);
    EXPECT_EQ(singular_pivot<K>(handle, di.info, nullptr), rocsparse_status_invalid_pointer);
    // Without an analysis there is no singular pivot.
    EXPECT_EQ(singular_pivot<K>(handle, di.info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);
}
