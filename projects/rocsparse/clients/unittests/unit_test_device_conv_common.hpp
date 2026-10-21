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

// Helpers shared by the unit_test_device_conv_*.cpp tests: status shortcuts and RAII
// wrappers for device buffers and library objects.

#pragma once

#include "rocsparse_handle.hpp"
#include "unit_test_utils.hpp"

#include <algorithm>
#include <initializer_list>
#include <vector>

namespace convut
{
    using namespace rocsparse_ut;

    using I = rocsparse_int;

    inline constexpr auto st_ok   = rocsparse_status_success;
    inline constexpr auto st_ptr  = rocsparse_status_invalid_pointer;
    inline constexpr auto st_size = rocsparse_status_invalid_size;
    inline constexpr auto st_val  = rocsparse_status_invalid_value;

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

    using Spmat         = Owned<rocsparse_spmat_descr, rocsparse_destroy_spmat_descr>;
    using Dnmat         = Owned<rocsparse_dnmat_descr, rocsparse_destroy_dnmat_descr>;
    using Dnvec         = Owned<rocsparse_dnvec_descr, rocsparse_destroy_dnvec_descr>;
    using MatDescrOwner = Owned<rocsparse_mat_descr, rocsparse_destroy_mat_descr>;
    using MatInfoOwner  = Owned<rocsparse_mat_info, rocsparse_destroy_mat_info>;

    // Matrix descriptor with the given index base.
    struct MatDescr : MatDescrOwner
    {
        explicit MatDescr(rocsparse_index_base base = base0)
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&*this), st_ok);
            EXPECT_EQ(rocsparse_set_mat_index_base(*this, base), st_ok);
        }
    };

    // Created matrix info structure.
    struct MatInfo : MatInfoOwner
    {
        MatInfo()
        {
            EXPECT_EQ(rocsparse_create_mat_info(&*this), st_ok);
        }
    };

    // Sets the pointer mode of a handle for the lifetime of the object.
    class PointerModeScope
    {
        rocsparse_handle       m_h;
        rocsparse_pointer_mode m_saved;

    public:
        PointerModeScope(rocsparse_handle h, rocsparse_pointer_mode mode)
            : m_h(h)
            , m_saved(h->pointer_mode)
        {
            EXPECT_EQ(rocsparse_set_pointer_mode(h, mode), st_ok);
        }
        ~PointerModeScope()
        {
            (void)rocsparse_set_pointer_mode(m_h, m_saved);
        }
    };

    inline const rocsparse_pointer_mode s_pointer_modes[]
        = {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device};

    // The nnz result of a routine, obtained through a host or a device pointer.
    template <typename F>
    rocsparse_int nnz_via(rocsparse_handle handle, bool device_mode, F f)
    {
        PointerModeScope      pm(handle, s_pointer_modes[device_mode]);
        rocsparse_int         host = -1;
        DevBuf<rocsparse_int> dev(std::vector<rocsparse_int>{-1});
        EXPECT_EQ(f(device_mode ? dev.get() : &host), st_ok);
        EXPECT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
        return device_mode ? dev.read(1)[0] : host;
    }

} // namespace convut
