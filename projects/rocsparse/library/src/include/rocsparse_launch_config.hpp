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
// Layer 1 of the shared, architecture-aware launch-configuration substrate
// (see designs/rocsparse/performance/LAUNCH_CONFIG_DISPATCH_DESIGN.md).
//
// A small vocabulary of pure launch-math helpers that the perf-portability pass
// previously hand-inlined in coomv/coomv_aos/coomm/sellmv/bellmm/gemvi. They are
// deliberately GPU-free (functions of an arch_traits value only) so the
// arithmetic can be reasoned about and unit-tested on the host.
//

#include "rocsparse_arch_traits.hpp"

#include <cstdint>

namespace rocsparse
{
    namespace launch
    {
        // Threads in `n` whole wavefronts on this arch. 4 waves == 128 threads on
        // wave32, 256 on wave64.
        constexpr uint32_t waves(const arch_traits& a, uint32_t n)
        {
            return n * a.wavefront_size;
        }

        // How many whole wavefronts a thread count spans (floored).
        constexpr uint32_t block_in_waves(const arch_traits& a, uint32_t threads)
        {
            return a.wavefront_size ? threads / a.wavefront_size : 0;
        }

        // Round a desired thread count down to a whole number of wavefronts,
        // never below one wavefront.
        constexpr uint32_t round_down_to_waves(const arch_traits& a, uint32_t threads)
        {
            const uint32_t w = a.wavefront_size;
            if(w == 0)
            {
                return threads;
            }
            const uint32_t floored = (threads / w) * w;
            return floored ? floored : w;
        }

        // Smallest power-of-two block >= rows_of_work, floored to a whole
        // wavefront and capped by the HW max block size. This is bellmm's square
        // per-block-dim tile logic, generalized.
        inline uint32_t fit_block(const arch_traits& a, uint32_t rows_of_work)
        {
            const uint32_t w   = a.wavefront_size ? a.wavefront_size : 32u;
            const uint32_t cap = a.max_threads_per_block ? a.max_threads_per_block : 1024u;

            uint32_t block = w;
            while(block < rows_of_work && block < cap)
            {
                block <<= 1;
            }
            return block > cap ? cap : block;
        }

        // Occupancy-derived crossover: does the problem present enough work to
        // over-subscribe the device by `factor` full resident populations?
        //
        // Number of blocks needed is ceil(work_items / block); saturation happens
        // when that reaches factor * saturation_blocks(block). The block size
        // cancels algebraically, so this reduces to
        //     work_items >= factor * resident_thread_capacity()
        // which is exactly the "nnz >= 52 * device_capacity" knee the coomv family
        // was tuned to, now expressed once and device-relative.
        inline bool saturates(const arch_traits& a, int64_t work_items, double factor)
        {
            const int64_t capacity = a.resident_thread_capacity();
            if(capacity <= 0)
            {
                return false;
            }
            return work_items >= static_cast<int64_t>(factor * static_cast<double>(capacity));
        }
    }
}
