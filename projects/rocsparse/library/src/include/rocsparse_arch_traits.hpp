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

#pragma once

//
// Layer 0 of the shared, architecture-aware launch-configuration substrate
// (see designs/rocsparse/performance/LAUNCH_CONFIG_DISPATCH_DESIGN.md).
//
// A single, normalized source of truth for the handful of device facts every
// launch-tuning decision actually needs. Kernels should read these semantic
// fields instead of re-deriving `multiProcessorCount * maxThreadsPerMultiProcessor`
// or branching on a bare `wavefront_size == 32` in each dispatcher.
//

#include "rocsparse_handle.hpp"

#include <cstdint>
#include <string>

namespace rocsparse
{
    // Coarse architecture family. Enough to gate tuning that was only
    // characterized on one class of hardware; finer decisions should key off
    // the concrete facts below (or, later, an arch_id + autotuned table).
    enum class arch_family
    {
        unknown,
        gcn, // pre-CDNA GCN (gfx8xx and earlier)
        cdna, // gfx9xx datacenter (MI-series)
        rdna // gfx10xx / gfx11xx / gfx12xx client
    };

    // Normalized, semantic device facts. Cheap value type; construct via
    // traits_of(handle).
    struct arch_traits
    {
        uint32_t    wavefront_size        = 0; // 32 (RDNA) or 64 (CDNA/GCN)
        uint32_t    cu_count              = 0; // multiProcessorCount
        uint32_t    max_threads_per_cu    = 0; // maxThreadsPerMultiProcessor
        uint32_t    max_threads_per_block = 0; // maxThreadsPerBlock
        uint32_t    lds_bytes_per_block   = 0; // sharedMemPerBlock
        arch_family family                = arch_family::unknown;

        // True on wave32 (RDNA) hardware. All AMD wave32 parts are RDNA, so this
        // is exactly the condition the hand-written perf-portability pass gated on.
        constexpr bool is_wave32() const
        {
            return wavefront_size == 32;
        }

        // Number of hardware threads that can be simultaneously resident on the
        // whole device (== 57344 on gfx1201). This is the physical quantity the
        // device-capacity crossovers scale against.
        constexpr int64_t resident_thread_capacity() const
        {
            return int64_t(cu_count) * int64_t(max_threads_per_cu);
        }

        // How many thread-blocks of `block` threads fit resident across the whole
        // device (occupancy-limited by threads only). 0 if `block` is 0.
        constexpr int64_t saturation_blocks(uint32_t block) const
        {
            return block ? int64_t(cu_count) * int64_t(max_threads_per_cu / block) : 0;
        }
    };

    namespace detail
    {
        // Map a (stripped) gfx architecture name to a coarse family. Kept as a
        // free function so it is trivially unit-testable without a handle.
        inline arch_family arch_family_from_name(const std::string& gfx)
        {
            const auto starts_with = [&](const char* p) { return gfx.rfind(p, 0) == 0; };

            if(starts_with("gfx12") || starts_with("gfx11") || starts_with("gfx10"))
            {
                return arch_family::rdna;
            }
            if(starts_with("gfx9"))
            {
                // gfx90a/gfx908/gfx94x/gfx95x are CDNA; older gfx9 are GCN but
                // wave64 and untuned, so a single "cdna" bucket is sufficient here.
                return arch_family::cdna;
            }
            if(starts_with("gfx8") || starts_with("gfx7") || starts_with("gfx6"))
            {
                return arch_family::gcn;
            }
            return arch_family::unknown;
        }
    }

    // Derive the normalized traits from a handle. Reads the cached device
    // properties on the handle; does not touch the device. Header-only for now
    // (the design's on-handle cache is a later, purely-performance refinement).
    inline arch_traits traits_of(rocsparse_handle handle)
    {
        arch_traits a;
        a.wavefront_size = static_cast<uint32_t>(handle->wavefront_size);
        a.cu_count       = static_cast<uint32_t>(handle->properties.multiProcessorCount);
        a.max_threads_per_cu
            = static_cast<uint32_t>(handle->properties.maxThreadsPerMultiProcessor);
        a.max_threads_per_block = static_cast<uint32_t>(handle->properties.maxThreadsPerBlock);
        a.lds_bytes_per_block   = static_cast<uint32_t>(handle->properties.sharedMemPerBlock);
        a.family = detail::arch_family_from_name(rocsparse::handle_get_arch_name(handle));
        return a;
    }
}
