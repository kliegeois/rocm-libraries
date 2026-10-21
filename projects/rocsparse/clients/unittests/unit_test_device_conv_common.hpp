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

// Helpers shared by the unit_test_device_conv_*.cpp tests: status shortcuts, RAII
// wrappers for device buffers and library objects, and small host references.

#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <vector>

// EXPECT_REJECTED(call): the call returns an error, the status is not documented. The
// call is evaluated exactly once.
#define EXPECT_REJECTED(call) EXPECT_NE((call), rocsparse_status_success)

namespace convut
{
    using namespace rocsparse_ut;

    inline constexpr auto st_ok   = rocsparse_status_success;
    inline constexpr auto st_ptr  = rocsparse_status_invalid_pointer;
    inline constexpr auto st_size = rocsparse_status_invalid_size;

    inline constexpr auto base0 = rocsparse_index_base_zero;
    inline constexpr auto i32   = rocsparse_indextype_i32;
    inline constexpr auto f32   = rocsparse_datatype_f32_r;

    // Fixture owning a handle.
    class ConvTest : public HandleTest
    {
    };

    // Device copy of a host vector. At least `min_elems` elements are allocated (zero
    // filled beyond the host data) and the pointer is never null. Unlike
    // rocsparse_ut::device_vector, a failed allocation or copy fails the test.
    template <typename T>
    class DevBuf
    {
        T*     m_ptr = nullptr;
        size_t m_n;

    public:
        explicit DevBuf(const std::vector<T>& host, size_t min_elems = 0)
            : m_n(std::max({host.size(), min_elems, size_t(1)}))
        {
            if(hipMalloc(&m_ptr, m_n * sizeof(T)) != hipSuccess
               || hipMemset(m_ptr, 0, m_n * sizeof(T)) != hipSuccess
               || (!host.empty()
                   && hipMemcpy(m_ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice)
                          != hipSuccess))
            {
                ADD_FAILURE() << "device allocation or copy failed";
            }
        }
        explicit DevBuf(size_t count)
            : DevBuf(std::vector<T>(), count)
        {
        }
        ~DevBuf()
        {
            EXPECT_EQ(hipFree(m_ptr), hipSuccess);
        }
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;

        T* get() const
        {
            return m_ptr;
        }
        operator T*() const
        {
            return m_ptr;
        }
        // The first `count` elements (all allocated elements by default).
        std::vector<T> read(size_t count = size_t(-1)) const
        {
            std::vector<T> host(std::min(count, m_n));
            if(!host.empty())
            {
                EXPECT_EQ(
                    hipMemcpy(host.data(), m_ptr, host.size() * sizeof(T), hipMemcpyDeviceToHost),
                    hipSuccess);
            }
            return host;
        }
    };

    // Owns a library object and destroys it (checking the status) at the end of the
    // scope. `&owner` is the address of the stored handle, to be passed to a create call.
    template <typename H, auto DESTROY>
    class Owned
    {
        H m_h = nullptr;

    public:
        Owned() = default;
        ~Owned()
        {
            if(m_h != nullptr)
            {
                EXPECT_EQ(DESTROY(m_h), st_ok);
            }
        }
        Owned(const Owned&) = delete;
        Owned& operator=(const Owned&) = delete;

        H* operator&()
        {
            return &m_h;
        }
        H get() const
        {
            return m_h;
        }
        operator H() const
        {
            return m_h;
        }
    };

    using Spmat = Owned<rocsparse_spmat_descr, rocsparse_destroy_spmat_descr>;
    using Dnvec = Owned<rocsparse_dnvec_descr, rocsparse_destroy_dnvec_descr>;
    using S2sDescr
        = Owned<rocsparse_sparse_to_sparse_descr, rocsparse_destroy_sparse_to_sparse_descr>;
} // namespace convut
