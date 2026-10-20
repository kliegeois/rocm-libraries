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

// The triangular-solve data holders behind rocsparse_mat_info, exercised directly:
//  * rocsparse::trm_info_t: every accessor, the copy constructor and the assignment
//    (which refuses a destination of another size);
//  * the per-routine info objects of a mat_info (csrsv, csrsm, csrilu0, csric0,
//    csrildlt0, bsrsv, bsrsm, bsrilu0, bsric0, ellsv): creation, the four
//    (operation, fill mode) slots, sharing and clearing.

#include "unit_test_device_cov70_precond_common.hpp"

#include "rocsparse_mat_info.hpp"
#include "rocsparse_trm_data_t.hpp"
#include "rocsparse_trm_info.hpp"

#include <vector>

using namespace cov70;

namespace
{
    // Owns a device array holding `host`, released with the info that adopts it. The
    // pointer is handed over with release().
    template <typename T>
    T* make_device_array(const std::vector<T>& host)
    {
        T* p = nullptr;
        if(hipMalloc(&p, host.size() * sizeof(T)) != hipSuccess)
            return nullptr;
        if(hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
        {
            (void)hipFree(p);
            return nullptr;
        }
        return p;
    }

    template <typename T>
    std::vector<T> read_device_array(const void* p, size_t n)
    {
        std::vector<T> out(n);
        EXPECT_EQ(hipMemcpy(out.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
        return out;
    }

    // A trm_info_t of order m = 4 with nnz = 6 and 32-bit indices whose five device arrays
    // are filled with `base`, `base + 1`, ...
    void fill_trm_info(rocsparse::trm_info_t& info, int32_t base)
    {
        const int32_t m   = 4;
        const int32_t nnz = 6;
        info.set_m(m);
        info.set_nnz(nnz);
        info.set_max_nnz(3);
        info.set_offset_indextype(rocsparse_indextype_i32);
        info.set_index_indextype(rocsparse_indextype_i32);
        auto fill = [&](void** slot, int32_t count, int32_t first) {
            std::vector<int32_t> host(static_cast<size_t>(count));
            for(int32_t i = 0; i < count; ++i)
                host[i] = first + i;
            *slot = make_device_array(host);
            ASSERT_NE(*slot, nullptr);
        };
        fill(info.get_ref_row_map(), m, base);
        fill(info.get_ref_diag_ind(), m, base + 10);
        fill(info.get_ref_transposed_perm(), nnz, base + 20);
        fill(info.get_ref_transposed_row_ptr(), m + 1, base + 30);
        fill(info.get_ref_transposed_col_ind(), nnz, base + 40);
    }

    void expect_same_arrays(rocsparse::trm_info_t& a, rocsparse::trm_info_t& b)
    {
        const size_t m   = static_cast<size_t>(a.get_m());
        const size_t nnz = static_cast<size_t>(a.get_nnz());
        EXPECT_EQ(read_device_array<int32_t>(a.get_row_map(), m),
                  read_device_array<int32_t>(b.get_row_map(), m));
        EXPECT_EQ(read_device_array<int32_t>(a.get_diag_ind(), m),
                  read_device_array<int32_t>(b.get_diag_ind(), m));
        EXPECT_EQ(read_device_array<int32_t>(a.get_transposed_perm(), nnz),
                  read_device_array<int32_t>(b.get_transposed_perm(), nnz));
        EXPECT_EQ(read_device_array<int32_t>(a.get_transposed_row_ptr(), m + 1),
                  read_device_array<int32_t>(b.get_transposed_row_ptr(), m + 1));
        EXPECT_EQ(read_device_array<int32_t>(a.get_transposed_col_ind(), nnz),
                  read_device_array<int32_t>(b.get_transposed_col_ind(), nnz));
    }

    struct descr_guard
    {
        rocsparse_mat_descr descr = nullptr;
        descr_guard()
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        }
        ~descr_guard()
        {
            if(descr)
                (void)rocsparse_destroy_mat_descr(descr);
        }
        descr_guard(const descr_guard&) = delete;
        descr_guard& operator=(const descr_guard&) = delete;
    };

    struct info_guard
    {
        rocsparse_mat_info info = nullptr;
        info_guard()
        {
            EXPECT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
        }
        ~info_guard()
        {
            if(info)
                (void)rocsparse_destroy_mat_info(info);
        }
        info_guard(const info_guard&) = delete;
        info_guard& operator=(const info_guard&) = delete;
    };

    // The four storage slots of a trm data holder: no-transpose / transpose times lower /
    // upper (conjugate transpose shares the transpose slot).
    struct slot
    {
        rocsparse_operation op;
        rocsparse_fill_mode fill;
    };
    constexpr slot slots[] = {{rocsparse_operation_none, rocsparse_fill_mode_lower},
                              {rocsparse_operation_none, rocsparse_fill_mode_upper},
                              {rocsparse_operation_transpose, rocsparse_fill_mode_lower},
                              {rocsparse_operation_transpose, rocsparse_fill_mode_upper}};

    // Drives the accessors of one family of mat_info: `get()` returns the family object,
    // `get_slot` / `set_slot` its slots.
    template <typename Get, typename GetSlot, typename SetSlot, typename Shared, typename Clear>
    void exercise_family(Get get, GetSlot get_slot, SetSlot set_slot, Shared shared, Clear clear)
    {
        // Created on first use.
        auto* object = get();
        ASSERT_NE(object, nullptr);
        EXPECT_EQ(get(), object);
        EXPECT_NE(shared().get(), nullptr);
        EXPECT_EQ(shared().get(), object);

        // Nothing is analyzed yet.
        for(const auto& s : slots)
            EXPECT_EQ(get_slot(s.op, s.fill), nullptr);

        // Every slot takes (and later releases) its own data.
        rocsparse::trm_info_t* stored[4] = {};
        for(size_t i = 0; i < 4; ++i)
        {
            stored[i] = new rocsparse::trm_info_t();
            set_slot(slots[i].op, slots[i].fill, stored[i]);
        }
        for(size_t i = 0; i < 4; ++i)
            EXPECT_EQ(get_slot(slots[i].op, slots[i].fill), stored[i]);
        // The conjugate transpose shares the slot of the transpose.
        EXPECT_EQ(get_slot(rocsparse_operation_conjugate_transpose, rocsparse_fill_mode_lower),
                  stored[2]);
        EXPECT_EQ(get_slot(rocsparse_operation_conjugate_transpose, rocsparse_fill_mode_upper),
                  stored[3]);

        // Clearing drops the object; the next access creates a new empty one.
        clear();
        for(const auto& s : slots)
            EXPECT_EQ(get_slot(s.op, s.fill), nullptr);
    }
}

class Cov80PrecondTrm : public HandleTest
{
};

TEST_F(Cov80PrecondTrm, trm_info_accessors)
{
    descr_guard d;
    ASSERT_NE(d.descr, nullptr);
    rocsparse::trm_info_t info;
    EXPECT_EQ(info.get_m(), 0);
    EXPECT_EQ(info.get_nnz(), 0);
    EXPECT_EQ(info.get_max_nnz(), 0);
    EXPECT_EQ(info.get_row_map(), nullptr);
    EXPECT_EQ(info.get_diag_ind(), nullptr);
    EXPECT_EQ(info.get_transposed_perm(), nullptr);
    EXPECT_EQ(info.get_transposed_row_ptr(), nullptr);
    EXPECT_EQ(info.get_transposed_col_ind(), nullptr);
    EXPECT_EQ(info.get_row_ptr(), nullptr);
    EXPECT_EQ(info.get_col_ind(), nullptr);
    EXPECT_EQ(info.get_descr(), nullptr);

    const int32_t marker[2] = {0, 0};
    info.set_m(11);
    info.set_nnz(22);
    info.set_max_nnz(33);
    info.set_row_ptr(&marker[0]);
    info.set_col_ind(&marker[1]);
    info.set_descr(d.descr);
    info.set_offset_indextype(rocsparse_indextype_i64);
    info.set_index_indextype(rocsparse_indextype_i32);
    EXPECT_EQ(info.get_m(), 11);
    EXPECT_EQ(info.get_nnz(), 22);
    EXPECT_EQ(info.get_max_nnz(), 33);
    EXPECT_EQ(info.get_row_ptr(), &marker[0]);
    EXPECT_EQ(info.get_col_ind(), &marker[1]);
    EXPECT_EQ(info.get_descr(), d.descr);
    EXPECT_EQ(info.get_offset_indextype(), rocsparse_indextype_i64);
    EXPECT_EQ(info.get_index_indextype(), rocsparse_indextype_i32);

    // The same through a constant reference.
    const rocsparse::trm_info_t& cinfo = info;
    EXPECT_EQ(cinfo.get_m(), 11);
    EXPECT_EQ(cinfo.get_row_map(), nullptr);
    EXPECT_EQ(cinfo.get_diag_ind(), nullptr);
    EXPECT_EQ(cinfo.get_transposed_perm(), nullptr);
    EXPECT_EQ(cinfo.get_transposed_row_ptr(), nullptr);
    EXPECT_EQ(cinfo.get_transposed_col_ind(), nullptr);

    // destroy() accepts both a live object and a null pointer.
    rocsparse::trm_info_t::destroy(new rocsparse::trm_info_t());
    rocsparse::trm_info_t::destroy(nullptr);
}

TEST_F(Cov80PrecondTrm, trm_info_copy_and_assignment)
{
    rocsparse::trm_info_t source;
    fill_trm_info(source, 100);
    ASSERT_NE(source.get_row_map(), nullptr);

    // Copy construction duplicates the device arrays.
    rocsparse::trm_info_t copy(source);
    EXPECT_EQ(copy.get_m(), source.get_m());
    EXPECT_EQ(copy.get_nnz(), source.get_nnz());
    EXPECT_EQ(copy.get_max_nnz(), source.get_max_nnz());
    EXPECT_NE(copy.get_row_map(), source.get_row_map());
    EXPECT_NE(copy.get_transposed_col_ind(), source.get_transposed_col_ind());
    expect_same_arrays(source, copy);

    // Assignment to an info of the same size allocates what the destination lacks ...
    rocsparse::trm_info_t empty;
    empty.set_m(source.get_m());
    empty.set_nnz(source.get_nnz());
    empty.set_max_nnz(source.get_max_nnz());
    empty.set_offset_indextype(rocsparse_indextype_i32);
    empty.set_index_indextype(rocsparse_indextype_i32);
    empty = source;
    ASSERT_NE(empty.get_row_map(), nullptr);
    ASSERT_NE(empty.get_transposed_row_ptr(), nullptr);
    expect_same_arrays(source, empty);

    // ... and overwrites the arrays of a destination that has them.
    rocsparse::trm_info_t other;
    fill_trm_info(other, 500);
    other = source;
    expect_same_arrays(source, other);

    // A destination of another size is refused.
    rocsparse::trm_info_t smaller;
    smaller.set_m(source.get_m() - 1);
    smaller.set_nnz(source.get_nnz());
    smaller.set_max_nnz(source.get_max_nnz());
    smaller.set_offset_indextype(rocsparse_indextype_i32);
    smaller.set_index_indextype(rocsparse_indextype_i32);
    EXPECT_ANY_THROW(smaller = source);
}

TEST_F(Cov80PrecondTrm, mat_info_trm_slots_of_every_routine)
{
    info_guard g;
    ASSERT_NE(g.info, nullptr);
    rocsparse_mat_info info = g.info;

    exercise_family([&] { return info->get_csrsv_info(); },
                    [&](auto op, auto fill) { return info->get_csrsv_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_csrsv_info(op, fill, t); },
                    [&] { return info->get_shared_csrsv_info(); },
                    [&] { info->clear_csrsv_info(); });
    exercise_family([&] { return info->get_csrsm_info(); },
                    [&](auto op, auto fill) { return info->get_csrsm_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_csrsm_info(op, fill, t); },
                    [&] { return info->get_shared_csrsm_info(); },
                    [&] { info->clear_csrsm_info(); });
    exercise_family([&] { return info->get_csrilu0_info(); },
                    [&](auto op, auto fill) { return info->get_csrilu0_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_csrilu0_info(op, fill, t); },
                    [&] { return info->get_shared_csrilu0_info(); },
                    [&] { info->clear_csrilu0_info(); });
    exercise_family([&] { return info->get_csric0_info(); },
                    [&](auto op, auto fill) { return info->get_csric0_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_csric0_info(op, fill, t); },
                    [&] { return info->get_shared_csric0_info(); },
                    [&] { info->clear_csric0_info(); });
    exercise_family([&] { return info->get_csrildlt0_info(); },
                    [&](auto op, auto fill) { return info->get_csrildlt0_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_csrildlt0_info(op, fill, t); },
                    [&] { return info->get_shared_csrildlt0_info(); },
                    [&] { info->clear_csrildlt0_info(); });
    exercise_family([&] { return info->get_bsrsv_info(); },
                    [&](auto op, auto fill) { return info->get_bsrsv_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_bsrsv_info(op, fill, t); },
                    [&] { return info->get_shared_bsrsv_info(); },
                    [&] { info->clear_bsrsv_info(); });
    exercise_family([&] { return info->get_bsrsm_info(); },
                    [&](auto op, auto fill) { return info->get_bsrsm_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_bsrsm_info(op, fill, t); },
                    [&] { return info->get_shared_bsrsm_info(); },
                    [&] { info->clear_bsrsm_info(); });
    exercise_family([&] { return info->get_bsrilu0_info(); },
                    [&](auto op, auto fill) { return info->get_bsrilu0_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_bsrilu0_info(op, fill, t); },
                    [&] { return info->get_shared_bsrilu0_info(); },
                    [&] { info->clear_bsrilu0_info(); });
    exercise_family([&] { return info->get_bsric0_info(); },
                    [&](auto op, auto fill) { return info->get_bsric0_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_bsric0_info(op, fill, t); },
                    [&] { return info->get_shared_bsric0_info(); },
                    [&] { info->clear_bsric0_info(); });
    exercise_family([&] { return info->get_ellsv_info(); },
                    [&](auto op, auto fill) { return info->get_ellsv_info(op, fill); },
                    [&](auto op, auto fill, auto* t) { info->set_ellsv_info(op, fill, t); },
                    [&] { return info->get_shared_ellsv_info(); },
                    [&] { info->clear_ellsv_info(); });

    // The numeric boost holder, through a constant reference too.
    const rocsparse_mat_info cinfo = info;
    EXPECT_NE(cinfo->get_boost(), nullptr);
    EXPECT_EQ(cinfo->get_boost(), info->get_boost());
}

// The four (operation, fill mode) slots of the data of a triangular routine: the index
// type of the slots, copies that duplicate every filled slot, and the release of slots
// that two holders share.
TEST_F(Cov80PrecondTrm, trm_data_slots_copy_and_uncouple)
{
    rocsparse::trm_data_t        empty;
    const rocsparse::trm_data_t& const_empty = empty;
    EXPECT_EQ(empty.get_indextype_J(), rocsparse_indextype_i32);
    EXPECT_EQ(const_empty.get(rocsparse_operation_none, rocsparse_fill_mode_lower), nullptr);
    EXPECT_EQ(const_empty.get(rocsparse_operation_transpose, rocsparse_fill_mode_upper), nullptr);

    // The index type is the one of the first slot that is filled.
    {
        rocsparse::trm_data_t wide;
        auto*                 info = new rocsparse::trm_info_t();
        info->set_index_indextype(rocsparse_indextype_i64);
        wide.set(rocsparse_operation_transpose, rocsparse_fill_mode_upper, info);
        EXPECT_EQ(wide.get_indextype_J(), rocsparse_indextype_i64);
        EXPECT_EQ(wide.get(rocsparse_operation_transpose, rocsparse_fill_mode_upper), info);
    }

    // A copy duplicates the filled slots and leaves the others empty.
    rocsparse::trm_data_t source;
    auto*                 lower = new rocsparse::trm_info_t();
    auto*                 trans = new rocsparse::trm_info_t();
    fill_trm_info(*lower, 10);
    fill_trm_info(*trans, 200);
    source.set(rocsparse_operation_none, rocsparse_fill_mode_lower, lower);
    source.set(rocsparse_operation_transpose, rocsparse_fill_mode_upper, trans);

    rocsparse::trm_data_t copy;
    copy.copy(&source, handle->stream);
    auto* copied_lower = copy.get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    auto* copied_trans = copy.get(rocsparse_operation_transpose, rocsparse_fill_mode_upper);
    ASSERT_NE(copied_lower, nullptr);
    ASSERT_NE(copied_trans, nullptr);
    EXPECT_NE(copied_lower, lower);
    EXPECT_NE(copied_trans, trans);
    EXPECT_EQ(copy.get(rocsparse_operation_none, rocsparse_fill_mode_upper), nullptr);
    EXPECT_EQ(copy.get(rocsparse_operation_transpose, rocsparse_fill_mode_lower), nullptr);
    expect_same_arrays(*lower, *copied_lower);
    expect_same_arrays(*trans, *copied_trans);
    copy.copy(nullptr, handle->stream);

    // Uncoupling releases the slots that are shared with the other holder only.
    rocsparse::trm_data_t holder_a;
    rocsparse::trm_data_t holder_b;
    auto*                 shared = new rocsparse::trm_info_t();
    auto*                 own    = new rocsparse::trm_info_t();
    fill_trm_info(*shared, 30);
    fill_trm_info(*own, 40);
    holder_a.set(rocsparse_operation_none, rocsparse_fill_mode_upper, shared);
    holder_b.set(rocsparse_operation_none, rocsparse_fill_mode_upper, shared);
    holder_a.set(rocsparse_operation_transpose, rocsparse_fill_mode_lower, own);

    holder_a.uncouple(nullptr);
    holder_a.uncouple(&holder_a);
    EXPECT_EQ(holder_a.get(rocsparse_operation_none, rocsparse_fill_mode_upper), shared);
    holder_a.uncouple(&holder_b);
    EXPECT_EQ(holder_a.get(rocsparse_operation_none, rocsparse_fill_mode_upper), nullptr);
    EXPECT_EQ(holder_a.get(rocsparse_operation_transpose, rocsparse_fill_mode_lower), own);
    EXPECT_EQ(holder_b.get(rocsparse_operation_none, rocsparse_fill_mode_upper), shared);
}
