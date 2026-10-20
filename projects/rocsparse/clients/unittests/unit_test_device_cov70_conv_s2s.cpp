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
// compared against the dense matrix the source was built from.

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
            // The descriptor nnz of a BSR matrix counts scalar entries, the number of
            // blocks is read from the row pointer.
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
        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
        return ok;
    }

    template <typename T, typename I>
    void run_all_pairs(rocsparse_handle     handle,
                       rocsparse_index_base base,
                       int64_t              bd,
                       rocsparse_direction  dir)
    {
        const std::vector<T> dense = make_dense<T>();

        for(rocsparse_format sf : s_formats)
        {
            const HostMat<T, I> src = build<T, I>(sf, base, bd, dir, dense);
            ASSERT_EQ(to_dense(src), dense) << "builder " << format_name(sf);

            for(rocsparse_format tf : s_formats)
            {
                SCOPED_TRACE(std::string(format_name(sf)) + " -> " + format_name(tf)
                             + " base=" + std::to_string(static_cast<int>(base))
                             + " bd=" + std::to_string(bd));

                DevMat<T, I> dsrc(src);
                SpDescr      source;
                ASSERT_EQ(create_descr(src, dsrc, source), rocsparse_status_success);

                const HostMat<T, I> meta = target_meta<T, I>(tf, base, bd, dir, src);
                DevMat<T, I>        dtgt;
                SpDescr             target;
                ASSERT_EQ(create_descr(meta, dtgt, target), rocsparse_status_success);

                if(!convert<T, I>(handle, source.d, target, tf, dtgt))
                {
                    continue;
                }

                const HostMat<T, I> out = read_back<T, I>(target, tf, dtgt, base, bd, dir);
                EXPECT_EQ(to_dense(out), dense)
                    << "nnz=" << out.nnz << " width=" << out.width << " ptr_size=" << out.ptr.size()
                    << " ptr0=" << (out.ptr.empty() ? -1 : static_cast<int64_t>(out.ptr[0]))
                    << " ptrN=" << (out.ptr.empty() ? -1 : static_cast<int64_t>(out.ptr.back()));
            }
        }
    }
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixDoubleInt32)
{
    run_all_pairs<double, int32_t>(handle, rocsparse_index_base_zero, 2, rocsparse_direction_row);
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixFloatInt64)
{
    run_all_pairs<float, int64_t>(handle, rocsparse_index_base_one, 3, rocsparse_direction_column);
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixFloatComplexInt32)
{
    run_all_pairs<rocsparse_float_complex, int32_t>(
        handle, rocsparse_index_base_one, 4, rocsparse_direction_row);
}

TEST_F(HandleTest, Cov70ConvSparseToSparseMatrixDoubleComplexInt64)
{
    run_all_pairs<rocsparse_double_complex, int64_t>(
        handle, rocsparse_index_base_zero, 2, rocsparse_direction_column);
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
    ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, source.d, target.d, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_success);
    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_sparse_to_sparse_buffer_size(handle,
                                                     s2s,
                                                     source.d,
                                                     target.d,
                                                     rocsparse_sparse_to_sparse_stage_analysis,
                                                     &buffer_size),
              rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
}

// Blocked ELL is not supported by sparse_to_sparse, in either direction.
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
        rocsparse_status                 st  = rocsparse_create_sparse_to_sparse_descr(
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
        (void)rocsparse_destroy_sparse_to_sparse_descr(s2s);
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
            EXPECT_EQ(status_of(other_descr.d, bell.d), rocsparse_status_not_implemented);
        }
        {
            SpDescr bell;
            ASSERT_EQ(make_bell(bell), rocsparse_status_success);
            EXPECT_EQ(status_of(bell.d, other_descr.d), rocsparse_status_not_implemented);
        }
    }

    SpDescr b0, b1;
    ASSERT_EQ(make_bell(b0), rocsparse_status_success);
    ASSERT_EQ(make_bell(b1), rocsparse_status_success);
    EXPECT_EQ(status_of(b0.d, b1.d), rocsparse_status_not_implemented);
}
