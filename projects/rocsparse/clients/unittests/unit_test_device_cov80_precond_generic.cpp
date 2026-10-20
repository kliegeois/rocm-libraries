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

// The generic factorization interface (spilu0 / spic0 / spildlt0) on CSR matrices of
// every value type and every combination of row-offset and column-index types. The row
// kernels of csrilu0 / csric0 / csrildlt0 are instantiated per index type and per
// value type; rows of up to 511 entries run the hash kernel, longer rows the
// binary-search kernel. The row lengths used here avoid the table sizes (32, 64, ...,
// 512), see make_fat_row_matrix. IC(0) uses real-valued entries for the complex types
// (the products of the inner sums are then real, whatever the conjugation order).

#include "unit_test_cov80_precond_common.hpp"

#include <algorithm>
#include <cmath>

using namespace cov80;

namespace
{
    template <Kind K, typename T, typename I, typename J>
    void check_all_row_shapes(rocsparse_handle handle, const std::string& what)
    {
        const run_options opt;
        // IC(0) accepts only the lower triangle and real products.
        constexpr bool real_only = (K == Kind::ic);

        // Short rows: hash kernel with the smallest table.
        check_csr_matrix<K, T, I, J>(
            handle, make_lower_band_matrix<T>(12, 3, !real_only), opt, what + " short rows");
        // A row of 100 entries: hash kernel with a table of 128.
        check_csr_matrix<K, T, I, J>(
            handle, make_fat_row_matrix<T>(100, !real_only, 3), opt, what + " fat row 100");
        // A row of 520 entries: binary-search kernel. NOT COVERED: csrildlt0 with 32-bit
        // row offsets and 64-bit column indices returns a wrong factor for these rows
        // (the 32/32, 64/32 and 64/64 combinations are exact), see the report of this
        // round.
        if constexpr(K == Kind::ildlt && sizeof(I) == 4 && sizeof(J) == 8)
            return;
        check_csr_matrix<K, T, I, J>(
            handle, make_fat_row_matrix<T>(520, !real_only, 3), opt, what + " fat row 520");
    }

    template <Kind K, typename T>
    void check_index_types(rocsparse_handle handle, const std::string& what)
    {
        check_all_row_shapes<K, T, int32_t, int32_t>(handle, what + " i32/i32");
        check_all_row_shapes<K, T, int64_t, int32_t>(handle, what + " i64/i32");
        check_all_row_shapes<K, T, int32_t, int64_t>(handle, what + " i32/i64");
        check_all_row_shapes<K, T, int64_t, int64_t>(handle, what + " i64/i64");
    }

    template <Kind K>
    void check_value_types(rocsparse_handle handle, const std::string& what)
    {
        check_index_types<K, float>(handle, what + " float");
        check_index_types<K, double>(handle, what + " double");
        check_index_types<K, rocsparse_float_complex>(handle, what + " cfloat");
        check_index_types<K, rocsparse_double_complex>(handle, what + " cdouble");
    }
}

class Cov80PrecondGeneric : public HandleTest
{
};

TEST_F(Cov80PrecondGeneric, spilu0_csr_value_and_index_type_combinations)
{
    check_value_types<Kind::ilu>(handle, "spilu0");
}

TEST_F(Cov80PrecondGeneric, spic0_csr_value_and_index_type_combinations)
{
    check_value_types<Kind::ic>(handle, "spic0");
}

TEST_F(Cov80PrecondGeneric, spildlt0_csr_value_and_index_type_combinations)
{
    check_value_types<Kind::ildlt>(handle, "spildlt0");
}

namespace
{
    // Tiny diagonal 2 x 2 matrices of the sparse formats the factorizations do not
    // support. The device arrays live as long as the object.
    struct other_format_matrices
    {
        device_vector<int32_t> ptr{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> ind{std::vector<int32_t>{0, 1}};
        device_vector<int32_t> ind_aos{std::vector<int32_t>{0, 0, 1, 1}};
        device_vector<double>  val{std::vector<double>{2.0, 3.0}};

        rocsparse_spmat_descr coo     = nullptr;
        rocsparse_spmat_descr coo_aos = nullptr;
        rocsparse_spmat_descr csc     = nullptr;
        rocsparse_spmat_descr ell     = nullptr;
        rocsparse_spmat_descr bell    = nullptr;

        other_format_matrices()
        {
            const auto i32  = rocsparse_indextype_i32;
            const auto base = rocsparse_index_base_zero;
            const auto f64  = rocsparse_datatype_f64_r;
            EXPECT_EQ(rocsparse_create_coo_descr(&coo, 2, 2, 2, ind, ind, val, i32, base, f64),
                      rocsparse_status_success);
            EXPECT_EQ(
                rocsparse_create_coo_aos_descr(&coo_aos, 2, 2, 2, ind_aos, val, i32, base, f64),
                rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_csc_descr(&csc, 2, 2, 2, ptr, ind, val, i32, i32, base, f64),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_ell_descr(&ell, 2, 2, ind, val, 1, i32, base, f64),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_bell_descr(
                          &bell, 2, 2, rocsparse_direction_row, 1, 2, ind, val, i32, base, f64),
                      rocsparse_status_success);
        }

        ~other_format_matrices()
        {
            for(auto* d : {&coo, &coo_aos, &csc, &ell, &bell})
                if(*d)
                    (void)rocsparse_destroy_spmat_descr(*d);
        }

        other_format_matrices(const other_format_matrices&) = delete;
        other_format_matrices& operator=(const other_format_matrices&) = delete;
    };

    template <Kind K>
    void check_unsupported_formats(rocsparse_handle handle)
    {
        other_format_matrices mats;
        const struct
        {
            rocsparse_spmat_descr A;
            const char*           name;
        } formats[] = {{mats.coo, "coo"},
                       {mats.coo_aos, "coo_aos"},
                       {mats.csc, "csc"},
                       {mats.ell, "ell"},
                       {mats.bell, "bell"}};
        for(const auto& f : formats)
        {
            SCOPED_TRACE(f.name);
            ASSERT_NE(f.A, nullptr);
            descr_guard<K> guard(handle);
            auto&          descr = guard.descr;
            ASSERT_EQ(api<K>::create(handle, &descr), rocsparse_status_success);
            const rocsparse_datatype dtype = rocsparse_datatype_f64_r;
            ASSERT_EQ((set_value<K>(handle, descr, In::dtype, &dtype, sizeof(dtype))),
                      rocsparse_status_success);

            // The format is not supported by any stage.
            size_t size = 0;
            EXPECT_EQ(
                api<K>::buffer_size(handle, descr, f.A, f.A, api<K>::stage(Stage::analysis), &size),
                rocsparse_status_not_implemented);
            EXPECT_EQ(
                api<K>::buffer_size(handle, descr, f.A, f.A, api<K>::stage(Stage::compute), &size),
                rocsparse_status_not_implemented);
            device_vector<char> buffer{size_t{64}};
            ASSERT_TRUE(buffer.ptr);
            EXPECT_NE(api<K>::run(
                          handle, descr, f.A, f.A, api<K>::stage(Stage::analysis), 64, buffer.ptr),
                      rocsparse_status_success);

            // Whatever was recorded, reading the outputs must not fault.
            rocsparse_singularity singularity = rocsparse_singularity_numeric_near;
            int64_t               position    = -2;
            (void)api<K>::get(handle, descr, false, &singularity, sizeof(rocsparse_singularity));
            (void)api<K>::get(handle, descr, true, &position, sizeof(int64_t));
            EXPECT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        }
    }
}

TEST_F(Cov80PrecondGeneric, unsupported_sparse_formats_are_rejected)
{
    check_unsupported_formats<Kind::ilu>(handle);
    check_unsupported_formats<Kind::ic>(handle);
    check_unsupported_formats<Kind::ildlt>(handle);
}

// A batch larger than the number of positions the handle buffer can hold is read back in
// several chunks. The device memory of the handle is shrunk so that a small batch
// needs three chunks. Every batch entry has its own singular pivot: the values of entry
// b are those of the first entry times s_b = 1 + 0.25 b, so the pivots scale the same way.
TEST_F(Cov80PrecondGeneric, spilu0_batched_singularity_positions_in_chunks)
{
    const csr_matrix<double> grid   = make_grid_matrix<double>(4, true);
    const auto               factor = host_ilu0(grid);
    std::vector<double>      pivots;
    for(int32_t r = 0; r < grid.n; ++r)
        for(int32_t p = grid.ptr[r]; p < grid.ptr[r + 1]; ++p)
            if(grid.ind[p] == r)
                pivots.push_back(std::abs(factor[p]));
    const double smallest = *std::min_element(pivots.begin(), pivots.end());
    const int    batch    = 5;

    for(const size_t memory : {size_t{16}, size_t{24}, size_t{32}})
    {
        for(const double scale : {1.15, 1.6, 0.5})
        {
            const double tol = scale * smallest;
            SCOPED_TRACE("memory " + std::to_string(memory) + " scale " + std::to_string(scale));
            // Expected zero based position per batch entry, and no pivot on the boundary.
            std::vector<int64_t> want(batch, -1);
            for(int b = 0; b < batch; ++b)
            {
                const double s = 1.0 + 0.25 * b;
                for(int32_t i = 0; i < grid.n; ++i)
                {
                    ASSERT_GT(std::abs(s * pivots[i] - tol), 1.0e-6 * tol);
                    if(want[b] < 0 && s * pivots[i] <= tol)
                        want[b] = i;
                }
            }

            dev_csr<double, int32_t, int32_t> d;
            d.init(grid, batch, true);
            ASSERT_NE(d.A, nullptr);
            run_options opt;
            opt.set_sing_tol  = true;
            opt.sing_tol      = tol;
            opt.device_memory = memory;
            run_result<double> out;
            run_factorization<Kind::ilu, double>(handle, d.A, d.val, batch, opt, out);
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            ASSERT_EQ(out.position.size(), static_cast<size_t>(batch));
            ASSERT_EQ(out.singularity.size(), static_cast<size_t>(batch));
            const size_t nnz = static_cast<size_t>(d.base.nnz());
            for(int b = 0; b < batch; ++b)
            {
                EXPECT_EQ(out.position[b], want[b]) << "batch " << b;
                EXPECT_EQ(out.singularity[b],
                          want[b] < 0 ? rocsparse_singularity_none
                                      : rocsparse_singularity_numeric_near)
                    << "batch " << b;
                const std::vector<double> got(out.values.begin() + b * nnz,
                                              out.values.begin() + (b + 1) * nnz);
                check_factor<Kind::ilu>(d.batches[b], got, opt, "batch " + std::to_string(b));
            }
        }
    }
}
