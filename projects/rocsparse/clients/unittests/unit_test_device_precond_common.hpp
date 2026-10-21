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

// Shared helpers of the preconditioner device tests.

#pragma once

#include "unit_test_utils.hpp"

#include <type_traits>

namespace precond_ut
{
    using namespace rocsparse_ut;

    // Owns the descriptor and the info of the mat_info based entry points.
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

    enum class Kind
    {
        ilu,
        ic,
        ildlt
    };
} // namespace precond_ut

// The tests that run the same body for ILU(0) and IC(0) are typed tests over these.
template <typename KindTag>
class KindFixture : public precond_ut::HandleTest
{
protected:
    static constexpr precond_ut::Kind K = KindTag::value;
};

using IluIc = ::testing::Types<std::integral_constant<precond_ut::Kind, precond_ut::Kind::ilu>,
                               std::integral_constant<precond_ut::Kind, precond_ut::Kind::ic>>;
