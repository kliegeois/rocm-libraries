/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

// Suppress ASAN on all rocprim GPU kernels/device functions compiled in this TU.
// rocprim's lookback-scan and other device code triggers false positives under
// GPU ASAN (xnack+) because device memory accesses are checked against the host
// shadow memory.
#if defined(__SANITIZE_ADDRESS__)
_Pragma("clang attribute push(__attribute__((no_sanitize(\"address\"))), apply_to = function)")
#endif
#include <rocprim/rocprim.hpp>
#if defined(__SANITIZE_ADDRESS__)
_Pragma("clang attribute pop")
#endif

template <typename I, typename J>
rocsparse_status rocsparse::primitives::exclusive_scan_buffer_size(rocsparse_handle handle,
                                                                   J                initial_value,
                                                                   size_t           length,
                                                                   size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                *buffer_size,
                                                (I*)nullptr,
                                                (J*)nullptr,
                                                initial_value,
                                                length,
                                                rocprim::plus<J>(),
                                                handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::exclusive_scan(rocsparse_handle handle,
                                                       I*               input,
                                                       J*               output,
                                                       J                initial_value,
                                                       size_t           length,
                                                       size_t           buffer_size,
                                                       void*            buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(buffer,
                                                buffer_size,
                                                input,
                                                output,
                                                initial_value,
                                                length,
                                                rocprim::plus<J>(),
                                                handle->stream));

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE)                                                                   \
    template rocsparse_status rocsparse::primitives::exclusive_scan_buffer_size<ITYPE, JTYPE>(      \
        rocsparse_handle handle, JTYPE initial_value, size_t length, size_t * buffer_size);         \
    template rocsparse_status rocsparse::primitives::exclusive_scan(rocsparse_handle handle,        \
                                                                    ITYPE*           input,         \
                                                                    JTYPE*           output,        \
                                                                    JTYPE            initial_value, \
                                                                    size_t           length,        \
                                                                    size_t           buffer_size,   \
                                                                    void*            buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE
