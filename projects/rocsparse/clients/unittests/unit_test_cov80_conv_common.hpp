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

// Helpers shared by the unit_test_device_cov80_conv_*.cpp tests: RAII for device
// buffers and library objects, so that a failing ASSERT_* never leaks or hangs.

#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

// The call must return an error; the status is not documented.
#define COV80_EXPECT_REJECTED(call) EXPECT_NE((call), rocsparse_status_success)

namespace cov80conv
{
    // Device copy of a host vector. At least `min_elems` elements are allocated
    // (zero filled beyond the host data) and the pointer is never null.
    template <typename T>
    class DevBuf
    {
        T* m_ptr = nullptr;

    public:
        explicit DevBuf(const std::vector<T>& host, size_t min_elems = 0)
        {
            const size_t n = std::max(std::max(host.size(), min_elems), size_t(1));
            if(hipMalloc(reinterpret_cast<void**>(&m_ptr), n * sizeof(T)) != hipSuccess)
            {
                ADD_FAILURE() << "hipMalloc failed";
                m_ptr = nullptr;
                return;
            }
            if(hipMemset(m_ptr, 0, n * sizeof(T)) != hipSuccess)
            {
                ADD_FAILURE() << "hipMemset failed";
            }
            if(!host.empty()
               && hipMemcpy(m_ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice)
                      != hipSuccess)
            {
                ADD_FAILURE() << "hipMemcpy failed";
            }
        }
        ~DevBuf()
        {
            if(m_ptr != nullptr)
            {
                EXPECT_EQ(hipFree(m_ptr), hipSuccess);
            }
        }
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;

        T* get() const
        {
            return m_ptr;
        }
        // The first n elements, copied back to the host.
        std::vector<T> read(size_t n) const
        {
            std::vector<T> host(n);
            if(n > 0
               && hipMemcpy(host.data(), m_ptr, n * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess)
            {
                ADD_FAILURE() << "hipMemcpy failed";
            }
            return host;
        }
        operator T*() const
        {
            return m_ptr;
        }
    };

    // Destroys an object owned by a local variable when the scope is left, also when
    // an ASSERT_* returns early. Declare it right after the variable:
    //     rocsparse_spmat_descr mat = nullptr;
    //     SpmatGuard            mat_guard(mat);
    template <auto DESTROY>
    class Guard
    {
        std::function<void()> m_cleanup;

    public:
        template <typename V>
        explicit Guard(V& ref)
            : m_cleanup([&ref]() {
                if(ref != nullptr)
                {
                    EXPECT_EQ(DESTROY(ref), rocsparse_status_success);
                    ref = nullptr;
                }
            })
        {
        }
        ~Guard()
        {
            m_cleanup();
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };

    inline rocsparse_status destroy_spmat(rocsparse_spmat_descr d)
    {
        return rocsparse_destroy_spmat_descr(d);
    }
    inline rocsparse_status destroy_hyb(rocsparse_hyb_mat h)
    {
        return rocsparse_destroy_hyb_mat(h);
    }
    inline rocsparse_status destroy_mat_descr(rocsparse_mat_descr d)
    {
        return rocsparse_destroy_mat_descr(d);
    }
    inline rocsparse_status destroy_mat_info(rocsparse_mat_info i)
    {
        return rocsparse_destroy_mat_info(i);
    }

    using SpmatGuard    = Guard<destroy_spmat>;
    using HybGuard      = Guard<destroy_hyb>;
    using MatDescrGuard = Guard<destroy_mat_descr>;
    using MatInfoGuard  = Guard<destroy_mat_info>;
}
