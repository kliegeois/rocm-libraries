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
// Layer 2 (per-kernel policy) for coomv's segmented default path, on top of the
// shared arch/launch substrate. This is the *same* tuning that was validated on
// gfx1201 (309/309 coomv tests, see PERF_PORTABLE_OPTIMIZATION.md); it is simply
// expressed against arch_traits + launch:: helpers instead of open-coded reads
// of handle->properties. Proof-of-concept for the design in
// designs/rocsparse/performance/LAUNCH_CONFIG_DISPATCH_DESIGN.md.
//

#include "rocsparse_launch_config.hpp"

#include <cstdint>

namespace rocsparse
{
    // Runtime matrix signals that drive coomv's block-size choice.
    struct coomv_signals
    {
        int64_t nnz;
    };

    // Resolved launch knobs for the coomv segmented default path.
    struct coomv_params
    {
        uint32_t block_threads;
    };

    // Device-capacity-scaled crossover for the coomv segmented default path.
    //
    //  - Historical default is a 256-thread workgroup (tuned for wave64), kept
    //    unconditionally on wave64 and below the crossover.
    //  - On wave32 (RDNA) hardware a 4-wavefront (128-thread) workgroup wins once
    //    the problem is large enough to over-subscribe the device; the crossover
    //    is 52 resident-thread populations (~3M nnz on gfx1201), device-relative
    //    rather than a fixed nnz magic number.
    //  - Gated to wave32 because the crossover was only characterized there.
    //
    // A future autotuned table (Layer 4) can override this per arch_id without
    // touching the caller.
    inline coomv_params coomv_params_for(const arch_traits& a, const coomv_signals& s)
    {
        // Crossover in units of full device resident-thread populations.
        static constexpr double coomv_segmented_saturation_factor = 52.0;

        coomv_params p{256u};

        if(a.is_wave32() && launch::saturates(a, s.nnz, coomv_segmented_saturation_factor))
        {
            p.block_threads = launch::waves(a, 4); // 4 wavefronts == 128 threads on wave32
        }

        return p;
    }
}
