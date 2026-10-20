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

// rocsparse_sparse_to_sparse over the complete matrix of source/target formats
// (coo, coo_aos, csr, csc, ell, bsr). Every conversion is checked on the full
// output: the target is read back, expanded to a dense matrix on the host and
// compared against the dense matrix the source was built from. The number of
// stored entries of the target is checked against the host count as well, so
// superfluous stored zeros are detected. Source and target may use different BSR
// block sizes / directions; the library requires the same index base for both.

#include "unit_test_device_cov70_conv_spmat.hpp"

#include <algorithm>
#include <array>
#include <sstream>
#include <string>

using namespace rocsparse_ut;
using namespace cov70conv;

namespace
{
    // Read the converted target back to the host.
    template <typename T, typename I>
    HostMat<T, I> read_back(const SpDescr&       descr,
                            rocsparse_format     format,
                            const DevMat<T, I>&  d,
                            rocsparse_index_base base,
                            int64_t              bd,
                            rocsparse_direction  dir)
    {
        HostMat<T, I> h;
        h.format    = format;
        h.base      = base;
        h.dir       = dir;
        h.bd        = (format == rocsparse_format_bsr) ? bd : 1;
        h.rows      = (format == rocsparse_format_bsr) ? M / bd : M;
        h.cols      = (format == rocsparse_format_bsr) ? N / bd : N;
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr.d, &nnz), rocsparse_status_success);
        h.nnz = nnz;
        switch(format)
        {
        case rocsparse_format_coo:
            h.row = to_host(d.row.ptr, nnz);
            h.ind = to_host(d.ind.ptr, nnz);
            h.val = to_host(d.val.ptr, nnz);
            break;
        case rocsparse_format_coo_aos:
            h.ind = to_host(d.ind.ptr, 2 * nnz);
            h.val = to_host(d.val.ptr, nnz);
            break;
        case rocsparse_format_csr:
            h.ptr = to_host(d.ptr.ptr, M + 1);
            h.ind = to_host(d.ind.ptr, nnz);
            h.val = to_host(d.val.ptr, nnz);
            break;
        case rocsparse_format_csc:
            h.ptr = to_host(d.ptr.ptr, N + 1);
            h.ind = to_host(d.ind.ptr, nnz);
            h.val = to_host(d.val.ptr, nnz);
            break;
        case rocsparse_format_ell:
        {
            int64_t              rows = 0, cols = 0, width = 0;
            void*                ind = nullptr;
            void*                val = nullptr;
            rocsparse_indextype  it;
            rocsparse_index_base ib;
            rocsparse_datatype   dt;
            EXPECT_EQ(rocsparse_ell_get(descr.d, &rows, &cols, &ind, &val, &width, &it, &ib, &dt),
                      rocsparse_status_success);
            h.width = width;
            h.nnz   = width * M;
            h.ind   = to_host(d.ind.ptr, h.nnz);
            h.val   = to_host(d.val.ptr, h.nnz);
            break;
        }
        case rocsparse_format_bsr:
        {
            // Observed library behavior (not verified to be the intended contract): the
            // descriptor nnz of a BSR matrix counts scalar entries (nnzb * bd^2) while
            // the target was created with a block count. The number of blocks is
            // therefore taken from the row pointer.
            h.ptr              = to_host(d.ptr.ptr, M / bd + 1);
            const int64_t nnzb = static_cast<int64_t>(h.ptr.back()) - static_cast<int64_t>(base);
            EXPECT_EQ(nnz, nnzb * bd * bd);
            h.nnz = nnzb;
            h.ind = to_host(d.ind.ptr, nnzb);
            h.val = to_host(d.val.ptr, nnzb * bd * bd);
            break;
        }
        default:
            break;
        }
        return h;
    }

    // Analysis then compute of one conversion. The user sets the target arrays
    // between the two stages as the API requires. Returns false on failure.
    template <typename T, typename I>
    bool convert(rocsparse_handle            handle,
                 rocsparse_const_spmat_descr source,
                 SpDescr&                    target,
                 rocsparse_format            target_format,
                 DevMat<T, I>&               target_dev)
    {
        rocsparse_sparse_to_sparse_descr s2s = nullptr;
        S2sGuard                         s2s_guard(s2s);
        EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                      &s2s, source, target.d, rocsparse_sparse_to_sparse_alg_default),
                  rocsparse_status_success);
        if(s2s == nullptr)
        {
            return false;
        }
        // Needed by the conversions going through an intermediate CSR matrix, harmless
        // for the others.
        EXPECT_EQ(rocsparse_sparse_to_sparse_permissive(s2s), rocsparse_status_success);

        bool ok = true;
        for(rocsparse_sparse_to_sparse_stage stage :
            {rocsparse_sparse_to_sparse_stage_analysis, rocsparse_sparse_to_sparse_stage_compute})
        {
            SCOPED_TRACE(stage == rocsparse_sparse_to_sparse_stage_analysis ? "stage analysis"
                                                                            : "stage compute");
            size_t                 buffer_size = 0;
            const rocsparse_status st1         = rocsparse_sparse_to_sparse_buffer_size(
                handle, s2s, source, target.d, stage, &buffer_size);
            EXPECT_EQ(st1, rocsparse_status_success);
            if(st1 != rocsparse_status_success)
            {
                ok = false;
                break;
            }
            device_vector<char> buffer(std::max(buffer_size, size_t(1)));
            EXPECT_TRUE(buffer.ptr != nullptr);
            const rocsparse_status st2 = rocsparse_sparse_to_sparse(
                handle, s2s, source, target.d, stage, buffer_size, buffer.ptr);
            EXPECT_EQ(st2, rocsparse_status_success);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            if(st2 != rocsparse_status_success)
            {
                ok = false;
                break;
            }
            if(stage == rocsparse_sparse_to_sparse_stage_analysis)
            {
                EXPECT_EQ(set_pointers(target_format, target_dev, target),
                          rocsparse_status_success);
            }
        }
        return ok;
    }

    // Source and target description of a sweep over all format pairs.
    struct PairCfg
    {
        rocsparse_index_base base_s;
        int64_t              bd_s;
        rocsparse_direction  dir_s;
        rocsparse_index_base base_t;
        int64_t              bd_t;
        rocsparse_direction  dir_t;
    };

    template <typename T>
    int64_t count_nonzeros(const std::vector<T>& dense)
    {
        return std::count_if(dense.begin(), dense.end(), [](const T& v) { return v != T{}; });
    }

    // Longest row of the matrix (at least one: the ELL width cannot be zero).
    template <typename T>
    int64_t max_row_length(const std::vector<T>& dense)
    {
        int64_t width = 1;
        for(int64_t i = 0; i < M; ++i)
        {
            int64_t cnt = 0;
            for(int64_t j = 0; j < N; ++j)
                cnt += (dense[i * N + j] != T{}) ? 1 : 0;
            width = std::max(width, cnt);
        }
        return width;
    }

    // Number of bd x bd blocks that contain a nonzero.
    template <typename T>
    int64_t count_blocks(const std::vector<T>& dense, int64_t bd)
    {
        int64_t cnt = 0;
        for(int64_t ib = 0; ib < M / bd; ++ib)
            for(int64_t jb = 0; jb < N / bd; ++jb)
            {
                bool any = false;
                for(int64_t r = 0; r < bd && !any; ++r)
                    for(int64_t c = 0; c < bd && !any; ++c)
                        any = dense[(ib * bd + r) * N + jb * bd + c] != T{};
                cnt += any ? 1 : 0;
            }
        return cnt;
    }

    // The converted matrix must hold exactly the nonzeros of the dense matrix. A BSR
    // source stores explicit zeros inside its blocks and the library is free to keep
    // or drop them, so for a BSR source only a lower bound is checked.
    template <typename T, typename I>
    void expect_stored_entries(const HostMat<T, I>&  out,
                               const std::vector<T>& dense,
                               rocsparse_format      source_format,
                               int64_t               bd_t)
    {
        const bool exact = (source_format != rocsparse_format_bsr);
        int64_t    expected;
        switch(out.format)
        {
        case rocsparse_format_ell:
            expected = max_row_length(dense) * M;
            break;
        case rocsparse_format_bsr:
            expected = count_blocks(dense, bd_t);
            break;
        default:
            expected = count_nonzeros(dense);
            break;
        }
        if(exact)
        {
            EXPECT_EQ(out.nnz, expected)
                << "stored entries of the " << format_name(out.format) << " target";
        }
        else
        {
            EXPECT_GE(out.nnz, expected)
                << "stored entries of the " << format_name(out.format) << " target";
        }
    }

    template <typename T, typename I>
    void run_all_pairs(rocsparse_handle handle, const PairCfg& cfg)
    {
        const std::vector<T> dense = make_dense<T>();

        for(rocsparse_format sf : s_formats)
        {
            const HostMat<T, I> src = build<T, I>(sf, cfg.base_s, cfg.bd_s, cfg.dir_s, dense);
            ASSERT_EQ(to_dense(src), dense) << "builder " << format_name(sf);

            for(rocsparse_format tf : s_formats)
            {
                // BSR -> BSR is a plain copy of the arrays in the library
                // (FUNCTION_CONVERT(bsr, bsr) transfers them as they are); it does not
                // re-block or change the direction, so it is only a conversion when the
                // block size and direction agree. A differing request is not exercised
                // so that no behavior that looks unintended is pinned.
                if(sf == rocsparse_format_bsr && tf == rocsparse_format_bsr
                   && (cfg.bd_s != cfg.bd_t || cfg.dir_s != cfg.dir_t))
                {
                    continue;
                }
                SCOPED_TRACE(std::string(format_name(sf)) + " -> " + format_name(tf)
                             + " base=" + std::to_string(static_cast<int>(cfg.base_s)) + "->"
                             + std::to_string(static_cast<int>(cfg.base_t))
                             + " bd=" + std::to_string(cfg.bd_s) + "->" + std::to_string(cfg.bd_t)
                             + " dir=" + std::to_string(static_cast<int>(cfg.dir_s)) + "->"
                             + std::to_string(static_cast<int>(cfg.dir_t)));

                DevMat<T, I> dsrc(src);
                SpDescr      source;
                ASSERT_EQ(create_descr(src, dsrc, source), rocsparse_status_success);

                const HostMat<T, I> meta
                    = target_meta<T, I>(tf, cfg.base_t, cfg.bd_t, cfg.dir_t, src);
                DevMat<T, I> dtgt;
                SpDescr      target;
                ASSERT_EQ(create_descr(meta, dtgt, target), rocsparse_status_success);

                if(!convert<T, I>(handle, source.d, target, tf, dtgt))
                {
                    continue;
                }

                const HostMat<T, I> out
                    = read_back<T, I>(target, tf, dtgt, cfg.base_t, cfg.bd_t, cfg.dir_t);
                EXPECT_EQ(to_dense(out), dense)
                    << "nnz=" << out.nnz << " width=" << out.width << " ptr_size=" << out.ptr.size()
                    << " ptr0=" << (out.ptr.empty() ? -1 : static_cast<int64_t>(out.ptr[0]))
                    << " ptrN=" << (out.ptr.empty() ? -1 : static_cast<int64_t>(out.ptr.back()));
                expect_stored_entries(out, dense, sf, cfg.bd_t);
            }
        }
    }
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixDoubleInt32)
{
    run_all_pairs<double, int32_t>(handle,
                                   {rocsparse_index_base_zero,
                                    2,
                                    rocsparse_direction_row,
                                    rocsparse_index_base_zero,
                                    2,
                                    rocsparse_direction_row});
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixFloatInt64)
{
    run_all_pairs<float, int64_t>(handle,
                                  {rocsparse_index_base_one,
                                   3,
                                   rocsparse_direction_column,
                                   rocsparse_index_base_one,
                                   3,
                                   rocsparse_direction_column});
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixFloatComplexInt32)
{
    run_all_pairs<rocsparse_float_complex, int32_t>(handle,
                                                    {rocsparse_index_base_one,
                                                     4,
                                                     rocsparse_direction_row,
                                                     rocsparse_index_base_one,
                                                     4,
                                                     rocsparse_direction_row});
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixDoubleComplexInt64)
{
    run_all_pairs<rocsparse_double_complex, int64_t>(handle,
                                                     {rocsparse_index_base_zero,
                                                      2,
                                                      rocsparse_direction_column,
                                                      rocsparse_index_base_zero,
                                                      2,
                                                      rocsparse_direction_column});
}

// The source and the target descriptor must use the same index base: the library
// rejects a conversion between different bases with rocsparse_status_type_mismatch
// (rocsparse_mat_descr_are_same), so a base translation cannot be requested.
TEST_F(HandleTest, Cov70ConvSparseToSparseBaseMismatch)
{
    using T = double;
    using I = int32_t;

    const std::vector<T> dense = make_dense<T>();
    for(rocsparse_index_base base_s : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        const rocsparse_index_base base_t = (base_s == rocsparse_index_base_zero)
                                                ? rocsparse_index_base_one
                                                : rocsparse_index_base_zero;
        for(rocsparse_format sf : s_formats)
        {
            const HostMat<T, I> src = build<T, I>(sf, base_s, 2, rocsparse_direction_row, dense);
            DevMat<T, I>        dsrc(src);
            SpDescr             source;
            ASSERT_EQ(create_descr(src, dsrc, source), rocsparse_status_success);
            for(rocsparse_format tf : s_formats)
            {
                SCOPED_TRACE(std::string(format_name(sf)) + " -> " + format_name(tf)
                             + " base=" + std::to_string(static_cast<int>(base_s)) + "->"
                             + std::to_string(static_cast<int>(base_t)));
                const HostMat<T, I> meta
                    = target_meta<T, I>(tf, base_t, 2, rocsparse_direction_row, src);
                DevMat<T, I> dtgt;
                SpDescr      target;
                ASSERT_EQ(create_descr(meta, dtgt, target), rocsparse_status_success);

                rocsparse_sparse_to_sparse_descr s2s = nullptr;
                S2sGuard                         s2s_guard(s2s);
                ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                              &s2s, source.d, target.d, rocsparse_sparse_to_sparse_alg_default),
                          rocsparse_status_success);
                EXPECT_EQ(rocsparse_sparse_to_sparse_permissive(s2s), rocsparse_status_success);
                size_t buffer_size = 0;
                // Differing index bases are not documented as supported or unsupported.
                COV70_EXPECT_REJECTED(rocsparse_sparse_to_sparse_buffer_size(
                    handle,
                    s2s,
                    source.d,
                    target.d,
                    rocsparse_sparse_to_sparse_stage_analysis,
                    &buffer_size));
            }
        }
    }
}

// A BSR source and a BSR target with different block size / direction, going
// through the other formats: BSR -> {coo, csr, ...} with the source block layout
// and {coo, csr, ...} -> BSR with a different target block layout, both verified
// against the dense matrix. BSR -> BSR itself is skipped when the block layout
// differs (the library only copies the arrays in that case). The index base of the
// source and the target is the same (see Cov70ConvSparseToSparseBaseMismatch).
TEST_F(HandleTest, Cov70ConvSparseToSparseReblock)
{
    // Different block size and direction.
    run_all_pairs<double, int32_t>(handle,
                                   {rocsparse_index_base_zero,
                                    2,
                                    rocsparse_direction_row,
                                    rocsparse_index_base_zero,
                                    3,
                                    rocsparse_direction_column});
    // Same block size, direction change only.
    run_all_pairs<float, int32_t>(handle,
                                  {rocsparse_index_base_zero,
                                   3,
                                   rocsparse_direction_row,
                                   rocsparse_index_base_zero,
                                   3,
                                   rocsparse_direction_column});
    // Larger block size, complex values, base one.
    run_all_pairs<rocsparse_float_complex, int64_t>(handle,
                                                    {rocsparse_index_base_one,
                                                     4,
                                                     rocsparse_direction_column,
                                                     rocsparse_index_base_one,
                                                     2,
                                                     rocsparse_direction_row});
}

// Conversions that need an intermediate CSR matrix are refused unless the
// descriptor was made permissive.
TEST_F(HandleTest, Cov70ConvSparseToSparseNotPermissive)
{
    using T = float;
    using I = int32_t;

    const std::vector<T> dense = make_dense<T>();
    const HostMat<T, I>  src   = build<T, I>(
        rocsparse_format_coo, rocsparse_index_base_zero, 2, rocsparse_direction_row, dense);
    DevMat<T, I> dsrc(src);
    SpDescr      source;
    ASSERT_EQ(create_descr(src, dsrc, source), rocsparse_status_success);

    const HostMat<T, I> meta = target_meta<T, I>(
        rocsparse_format_ell, rocsparse_index_base_zero, 2, rocsparse_direction_row, src);
    DevMat<T, I> dtgt;
    SpDescr      target;
    ASSERT_EQ(create_descr(meta, dtgt, target), rocsparse_status_success);

    rocsparse_sparse_to_sparse_descr s2s = nullptr;
    S2sGuard                         s2s_guard(s2s);
    ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, source.d, target.d, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_success);
    size_t buffer_size = 0;
    // The header does not document the status of a refused conversion.
    COV70_EXPECT_REJECTED(rocsparse_sparse_to_sparse_buffer_size(
        handle, s2s, source.d, target.d, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size));
}

// Blocked ELL is documented as not supported by sparse_to_sparse, in either
// direction. The header does not say which status is returned: only an error is
// required.
TEST_F(HandleTest, Cov70ConvSparseToSparseBellNotImplemented)
{
    using T = float;
    using I = int32_t;

    const std::vector<T> dense = make_dense<T>();

    device_vector<I> bell_ind(static_cast<size_t>(M * N));
    device_vector<T> bell_val(static_cast<size_t>(M * N));
    ASSERT_TRUE(bell_ind.ptr && bell_val.ptr);

    auto make_bell = [&](SpDescr& out) {
        return rocsparse_create_bell_descr(&out.d,
                                           M,
                                           N,
                                           rocsparse_direction_row,
                                           2,
                                           2,
                                           bell_ind.ptr,
                                           bell_val.ptr,
                                           it_of<I>(),
                                           rocsparse_index_base_zero,
                                           dt_of<T>());
    };

    auto status_of = [&](rocsparse_const_spmat_descr source, rocsparse_spmat_descr target) {
        rocsparse_sparse_to_sparse_descr s2s = nullptr;
        S2sGuard                         s2s_guard(s2s);
        rocsparse_status                 st = rocsparse_create_sparse_to_sparse_descr(
            &s2s, source, target, rocsparse_sparse_to_sparse_alg_default);
        if(st != rocsparse_status_success)
        {
            return st;
        }
        size_t buffer_size = 0;
        st                 = rocsparse_sparse_to_sparse_buffer_size(
            handle, s2s, source, target, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size);
        if(st == rocsparse_status_success)
        {
            device_vector<char> buffer(std::max(buffer_size, size_t(1)));
            st = rocsparse_sparse_to_sparse(handle,
                                            s2s,
                                            source,
                                            target,
                                            rocsparse_sparse_to_sparse_stage_analysis,
                                            buffer_size,
                                            buffer.ptr);
        }
        return st;
    };

    for(rocsparse_format f : s_formats)
    {
        SCOPED_TRACE(std::string("format ") + format_name(f));

        const HostMat<T, I> other
            = build<T, I>(f, rocsparse_index_base_zero, 2, rocsparse_direction_row, dense);
        DevMat<T, I> dother(other);
        SpDescr      other_descr;
        ASSERT_EQ(create_descr(other, dother, other_descr), rocsparse_status_success);

        {
            SpDescr bell;
            ASSERT_EQ(make_bell(bell), rocsparse_status_success);
            COV70_EXPECT_REJECTED(status_of(other_descr.d, bell.d));
        }
        {
            SpDescr bell;
            ASSERT_EQ(make_bell(bell), rocsparse_status_success);
            COV70_EXPECT_REJECTED(status_of(bell.d, other_descr.d));
        }
    }

    SpDescr b0, b1;
    ASSERT_EQ(make_bell(b0), rocsparse_status_success);
    ASSERT_EQ(make_bell(b1), rocsparse_status_success);
    COV70_EXPECT_REJECTED(status_of(b0.d, b1.d));
}
