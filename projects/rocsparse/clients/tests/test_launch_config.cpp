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

//
// Host-only unit tests for the shared launch-configuration substrate
// (Layer 0 rocsparse_arch_traits.hpp + Layer 1 rocsparse_launch_config.hpp,
// see designs/rocsparse/performance/LAUNCH_CONFIG_DISPATCH_DESIGN.md).
//
// These verify the pure launch arithmetic and the per-kernel coomv/coomv_aos
// policies without touching a GPU: arch_traits values are constructed by hand,
// so no rocsparse_handle and no device are required. This is the "pure host
// functions - no GPU needed to verify the arithmetic" step of the design's
// migration plan.
//

#include "rocsparse_arch_traits.hpp"
#include "rocsparse_coomv_tuning.hpp"
#include "rocsparse_launch_config.hpp"

#include "rocsparse_data.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace
{
    // A gfx1201-like wave32 arch: resident_thread_capacity() == 57344, matching
    // the device the coomv/coomv_aos crossovers were measured on. The exact cu /
    // per-cu split is irrelevant to the tuning math (only the product matters), so
    // any factorization of 57344 exercises the same knee.
    rocsparse::arch_traits make_rdna_wave32()
    {
        rocsparse::arch_traits a;
        a.wavefront_size        = 32;
        a.cu_count              = 56;
        a.max_threads_per_cu    = 1024; // 56 * 1024 == 57344
        a.max_threads_per_block = 1024;
        a.lds_bytes_per_block   = 65536;
        a.family                = rocsparse::arch_family::rdna;
        return a;
    }

    // A CDNA-like wave64 arch (untuned path: must never switch off 256).
    rocsparse::arch_traits make_cdna_wave64()
    {
        rocsparse::arch_traits a;
        a.wavefront_size        = 64;
        a.cu_count              = 104;
        a.max_threads_per_cu    = 2048;
        a.max_threads_per_block = 1024;
        a.lds_bytes_per_block   = 65536;
        a.family                = rocsparse::arch_family::cdna;
        return a;
    }

    // Skip these host-only tests when a --yaml data filter is active, matching the
    // convention used by the other standalone tests (e.g. test_atomic_add.cpp).
    class launch_config : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(RocSPARSE_TestData::is_yaml_filter_active())
            {
                GTEST_SKIP() << "Skipping non-yaml test when --yaml filter is active";
            }
        }
    };
}

// -----------------------------------------------------------------------------
// Layer 0: arch_traits facts / derived quantities
// -----------------------------------------------------------------------------
TEST_F(launch_config, arch_family_from_name)
{
    using rocsparse::arch_family;
    using rocsparse::detail::arch_family_from_name;

    EXPECT_EQ(arch_family_from_name("gfx1201"), arch_family::rdna);
    EXPECT_EQ(arch_family_from_name("gfx1100"), arch_family::rdna);
    EXPECT_EQ(arch_family_from_name("gfx1030"), arch_family::rdna);
    EXPECT_EQ(arch_family_from_name("gfx942"), arch_family::cdna);
    EXPECT_EQ(arch_family_from_name("gfx90a"), arch_family::cdna);
    EXPECT_EQ(arch_family_from_name("gfx908"), arch_family::cdna);
    EXPECT_EQ(arch_family_from_name("gfx803"), arch_family::gcn);
    EXPECT_EQ(arch_family_from_name("gfxUNKNOWN"), arch_family::unknown);
    EXPECT_EQ(arch_family_from_name(""), arch_family::unknown);
}

TEST_F(launch_config, arch_traits_derived)
{
    const rocsparse::arch_traits rdna = make_rdna_wave32();
    const rocsparse::arch_traits cdna = make_cdna_wave64();

    EXPECT_TRUE(rdna.is_wave32());
    EXPECT_FALSE(cdna.is_wave32());

    EXPECT_EQ(rdna.resident_thread_capacity(), int64_t(57344));
    EXPECT_EQ(cdna.resident_thread_capacity(), int64_t(104) * 2048);

    // saturation_blocks: cu_count * (max_threads_per_cu / block)
    EXPECT_EQ(rdna.saturation_blocks(128), int64_t(56) * (1024 / 128));
    EXPECT_EQ(rdna.saturation_blocks(256), int64_t(56) * (1024 / 256));
    EXPECT_EQ(rdna.saturation_blocks(0), int64_t(0)); // guard against div-by-zero
}

// -----------------------------------------------------------------------------
// Layer 1: pure launch-math vocabulary
// -----------------------------------------------------------------------------
TEST_F(launch_config, waves_and_block_in_waves)
{
    const rocsparse::arch_traits rdna = make_rdna_wave32();
    const rocsparse::arch_traits cdna = make_cdna_wave64();

    // 4 waves == 128 on wave32, 256 on wave64.
    EXPECT_EQ(rocsparse::launch::waves(rdna, 4), 128u);
    EXPECT_EQ(rocsparse::launch::waves(cdna, 4), 256u);

    EXPECT_EQ(rocsparse::launch::block_in_waves(rdna, 128), 4u);
    EXPECT_EQ(rocsparse::launch::block_in_waves(cdna, 256), 4u);
    EXPECT_EQ(rocsparse::launch::block_in_waves(rdna, 100), 3u); // floored
}

TEST_F(launch_config, round_down_to_waves)
{
    const rocsparse::arch_traits rdna = make_rdna_wave32();

    EXPECT_EQ(rocsparse::launch::round_down_to_waves(rdna, 128), 128u);
    EXPECT_EQ(rocsparse::launch::round_down_to_waves(rdna, 130), 128u);
    EXPECT_EQ(rocsparse::launch::round_down_to_waves(rdna, 159), 128u);
    EXPECT_EQ(rocsparse::launch::round_down_to_waves(rdna, 160), 160u);
    // Never below one wavefront.
    EXPECT_EQ(rocsparse::launch::round_down_to_waves(rdna, 1), 32u);
    EXPECT_EQ(rocsparse::launch::round_down_to_waves(rdna, 0), 32u);
}

TEST_F(launch_config, fit_block)
{
    const rocsparse::arch_traits rdna = make_rdna_wave32();

    // Smallest power-of-two block >= rows, floored to a whole wavefront (>= 32),
    // capped by max_threads_per_block (1024).
    EXPECT_EQ(rocsparse::launch::fit_block(rdna, 1), 32u);
    EXPECT_EQ(rocsparse::launch::fit_block(rdna, 32), 32u);
    EXPECT_EQ(rocsparse::launch::fit_block(rdna, 33), 64u);
    EXPECT_EQ(rocsparse::launch::fit_block(rdna, 200), 256u);
    EXPECT_EQ(rocsparse::launch::fit_block(rdna, 1024), 1024u);
    EXPECT_EQ(rocsparse::launch::fit_block(rdna, 5000), 1024u); // capped
}

TEST_F(launch_config, saturates_reduces_to_capacity_multiple)
{
    const rocsparse::arch_traits rdna = make_rdna_wave32();
    const int64_t                cap  = rdna.resident_thread_capacity(); // 57344

    // saturates(a, work, factor) <=> work >= factor * capacity, block cancels.
    EXPECT_FALSE(rocsparse::launch::saturates(rdna, 52 * cap - 1, 52.0));
    EXPECT_TRUE(rocsparse::launch::saturates(rdna, 52 * cap, 52.0));
    EXPECT_TRUE(rocsparse::launch::saturates(rdna, 52 * cap + 1, 52.0));

    // A zero-capacity arch never saturates (defensive).
    rocsparse::arch_traits degenerate = rdna;
    degenerate.cu_count               = 0;
    EXPECT_FALSE(rocsparse::launch::saturates(degenerate, 1 << 30, 1.0));
}

// -----------------------------------------------------------------------------
// Layer 2: per-kernel policies reproduce the validated gfx1201 crossovers.
// -----------------------------------------------------------------------------
TEST_F(launch_config, coomv_policy_matches_validated_knee)
{
    const rocsparse::arch_traits rdna = make_rdna_wave32();
    const rocsparse::arch_traits cdna = make_cdna_wave64();
    const int64_t                cap  = rdna.resident_thread_capacity();

    // wave32: 256 below the 52x knee, 128 (== 4 waves) at/above it.
    EXPECT_EQ(rocsparse::coomv_params_for(rdna, {52 * cap - 1}).block_threads, 256u);
    EXPECT_EQ(rocsparse::coomv_params_for(rdna, {52 * cap}).block_threads, 128u);
    EXPECT_EQ(rocsparse::coomv_params_for(rdna, {10000000}).block_threads, 128u);

    // 52 * 57344 == 2981888, the ~3M-nnz value measured on gfx1201.
    EXPECT_EQ(52 * cap, int64_t(2981888));

    // wave64 keeps 256 unconditionally (untuned path).
    EXPECT_EQ(rocsparse::coomv_params_for(cdna, {int64_t(1) << 40}).block_threads, 256u);
}

TEST_F(launch_config, coomv_aos_policy_matches_validated_knee)
{
    const rocsparse::arch_traits rdna = make_rdna_wave32();
    const rocsparse::arch_traits cdna = make_cdna_wave64();
    const int64_t                cap  = rdna.resident_thread_capacity();

    // wave32: 256 below the 38x knee, 128 at/above it.
    EXPECT_EQ(rocsparse::coomv_aos_params_for(rdna, {38 * cap - 1}).block_threads, 256u);
    EXPECT_EQ(rocsparse::coomv_aos_params_for(rdna, {38 * cap}).block_threads, 128u);

    // 38 * 57344 == 2179072, the ~2.2M-nnz value measured on gfx1201.
    EXPECT_EQ(38 * cap, int64_t(2179072));

    // wave64 keeps 256 unconditionally.
    EXPECT_EQ(rocsparse::coomv_aos_params_for(cdna, {int64_t(1) << 40}).block_threads, 256u);
}
