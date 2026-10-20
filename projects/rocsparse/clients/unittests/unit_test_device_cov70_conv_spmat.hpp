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

// Host/device sparse matrices of every format (coo, coo_aos, csr, csc, ell, bsr)
// built from a dense matrix, with their dense expansion and descriptor helpers.

#pragma once

#include "unit_test_device_cov70_conv_common.hpp"

#include <algorithm>
#include <array>
#include <string>

namespace cov70conv
{
    using namespace rocsparse_ut;

    inline constexpr int64_t M = 12;
    inline constexpr int64_t N = 12;

    inline const std::array<rocsparse_format, 6> s_formats{rocsparse_format_coo,
                                                           rocsparse_format_coo_aos,
                                                           rocsparse_format_csr,
                                                           rocsparse_format_csc,
                                                           rocsparse_format_ell,
                                                           rocsparse_format_bsr};

    inline const char* format_name(rocsparse_format f)
    {
        switch(f)
        {
        case rocsparse_format_coo:
            return "coo";
        case rocsparse_format_coo_aos:
            return "coo_aos";
        case rocsparse_format_csr:
            return "csr";
        case rocsparse_format_csc:
            return "csc";
        case rocsparse_format_ell:
            return "ell";
        case rocsparse_format_bsr:
            return "bsr";
        case rocsparse_format_bell:
            return "bell";
        default:
            return "other";
        }
    }

    // Host description of a sparse matrix. For a BSR matrix rows/cols are the
    // block dimensions and nnz is the number of blocks; for an ELL matrix nnz is
    // width * rows.
    template <typename T, typename I>
    struct HostMat
    {
        rocsparse_format     format = rocsparse_format_csr;
        rocsparse_index_base base   = rocsparse_index_base_zero;
        rocsparse_direction  dir    = rocsparse_direction_row;
        int64_t              rows   = 0;
        int64_t              cols   = 0;
        int64_t              nnz    = 0;
        int64_t              width  = 0;
        int64_t              bd     = 1;
        std::vector<I>       ptr, ind, row;
        std::vector<T>       val;
    };

    // Dense matrix (row-major) with an irregular pattern: an empty row, rows of
    // different lengths and a full diagonal elsewhere.
    template <typename T>
    std::vector<T> make_dense()
    {
        std::vector<T> dense(M * N, T{});
        for(int64_t i = 0; i < M; ++i)
        {
            if(i == 7)
            {
                continue;
            }
            for(int64_t j = 0; j < N; ++j)
            {
                if((i + 2 * j) % 5 == 0 || i == j || (i == 3 && j == 11))
                {
                    dense[i * N + j] = make_val<T>(static_cast<int>(1 + i * N + j),
                                                   static_cast<int>(i) - static_cast<int>(j));
                }
            }
        }
        return dense;
    }

    template <typename T, typename I>
    HostMat<T, I> build(rocsparse_format      format,
                        rocsparse_index_base  base,
                        int64_t               bd,
                        rocsparse_direction   dir,
                        const std::vector<T>& dense)
    {
        HostMat<T, I> h;
        h.format  = format;
        h.base    = base;
        h.dir     = dir;
        h.bd      = (format == rocsparse_format_bsr) ? bd : 1;
        const I b = static_cast<I>(base);
        const T z = T{};

        switch(format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        {
            h.rows = M;
            h.cols = N;
            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t j = 0; j < N; ++j)
                {
                    if(dense[i * N + j] != z)
                    {
                        if(format == rocsparse_format_coo)
                        {
                            h.row.push_back(static_cast<I>(i) + b);
                            h.ind.push_back(static_cast<I>(j) + b);
                        }
                        else
                        {
                            h.ind.push_back(static_cast<I>(i) + b);
                            h.ind.push_back(static_cast<I>(j) + b);
                        }
                        h.val.push_back(dense[i * N + j]);
                    }
                }
            }
            h.nnz = static_cast<int64_t>(h.val.size());
            break;
        }
        case rocsparse_format_csr:
        {
            h.rows = M;
            h.cols = N;
            h.ptr.push_back(b);
            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t j = 0; j < N; ++j)
                {
                    if(dense[i * N + j] != z)
                    {
                        h.ind.push_back(static_cast<I>(j) + b);
                        h.val.push_back(dense[i * N + j]);
                    }
                }
                h.ptr.push_back(static_cast<I>(h.val.size()) + b);
            }
            h.nnz = static_cast<int64_t>(h.val.size());
            break;
        }
        case rocsparse_format_csc:
        {
            h.rows = M;
            h.cols = N;
            h.ptr.push_back(b);
            for(int64_t j = 0; j < N; ++j)
            {
                for(int64_t i = 0; i < M; ++i)
                {
                    if(dense[i * N + j] != z)
                    {
                        h.ind.push_back(static_cast<I>(i) + b);
                        h.val.push_back(dense[i * N + j]);
                    }
                }
                h.ptr.push_back(static_cast<I>(h.val.size()) + b);
            }
            h.nnz = static_cast<int64_t>(h.val.size());
            break;
        }
        case rocsparse_format_ell:
        {
            h.rows        = M;
            h.cols        = N;
            int64_t width = 1;
            for(int64_t i = 0; i < M; ++i)
            {
                int64_t count = 0;
                for(int64_t j = 0; j < N; ++j)
                {
                    count += (dense[i * N + j] != z) ? 1 : 0;
                }
                width = std::max(width, count);
            }
            h.width = width;
            h.nnz   = width * M;
            // rocSPARSE marks ELL padding with -1 for either index base
            h.ind.assign(static_cast<size_t>(h.nnz), static_cast<I>(-1));
            h.val.assign(static_cast<size_t>(h.nnz), z);
            for(int64_t i = 0; i < M; ++i)
            {
                int64_t k = 0;
                for(int64_t j = 0; j < N; ++j)
                {
                    if(dense[i * N + j] != z)
                    {
                        h.ind[i + k * M] = static_cast<I>(j) + b;
                        h.val[i + k * M] = dense[i * N + j];
                        ++k;
                    }
                }
            }
            break;
        }
        case rocsparse_format_bsr:
        {
            const int64_t mb = M / bd;
            const int64_t nb = N / bd;
            h.rows           = mb;
            h.cols           = nb;
            h.ptr.push_back(b);
            for(int64_t ib = 0; ib < mb; ++ib)
            {
                for(int64_t jb = 0; jb < nb; ++jb)
                {
                    bool any = false;
                    for(int64_t r = 0; r < bd; ++r)
                    {
                        for(int64_t c = 0; c < bd; ++c)
                        {
                            any = any || dense[(ib * bd + r) * N + jb * bd + c] != z;
                        }
                    }
                    if(!any)
                    {
                        continue;
                    }
                    h.ind.push_back(static_cast<I>(jb) + b);
                    const size_t off = h.val.size();
                    h.val.resize(off + static_cast<size_t>(bd * bd), z);
                    for(int64_t r = 0; r < bd; ++r)
                    {
                        for(int64_t c = 0; c < bd; ++c)
                        {
                            const int64_t pos
                                = (dir == rocsparse_direction_row) ? r * bd + c : c * bd + r;
                            h.val[off + pos] = dense[(ib * bd + r) * N + jb * bd + c];
                        }
                    }
                }
                h.ptr.push_back(static_cast<I>(h.ind.size()) + b);
            }
            h.nnz = static_cast<int64_t>(h.ind.size());
            break;
        }
        default:
            break;
        }
        return h;
    }

    // Expand a host matrix to a dense row-major M x N matrix.
    template <typename T, typename I>
    std::vector<T> to_dense(const HostMat<T, I>& h)
    {
        std::vector<T> dense(M * N, T{});
        const int64_t  b = static_cast<int64_t>(h.base);
        switch(h.format)
        {
        case rocsparse_format_coo:
        {
            for(int64_t k = 0; k < h.nnz; ++k)
            {
                dense[(h.row[k] - b) * N + (h.ind[k] - b)] = h.val[k];
            }
            break;
        }
        case rocsparse_format_coo_aos:
        {
            for(int64_t k = 0; k < h.nnz; ++k)
            {
                dense[(h.ind[2 * k] - b) * N + (h.ind[2 * k + 1] - b)] = h.val[k];
            }
            break;
        }
        case rocsparse_format_csr:
        {
            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t k = h.ptr[i] - b; k < h.ptr[i + 1] - b; ++k)
                {
                    dense[i * N + (h.ind[k] - b)] = h.val[k];
                }
            }
            break;
        }
        case rocsparse_format_csc:
        {
            for(int64_t j = 0; j < N; ++j)
            {
                for(int64_t k = h.ptr[j] - b; k < h.ptr[j + 1] - b; ++k)
                {
                    dense[(h.ind[k] - b) * N + j] = h.val[k];
                }
            }
            break;
        }
        case rocsparse_format_ell:
        {
            for(int64_t i = 0; i < M; ++i)
            {
                for(int64_t k = 0; k < h.width; ++k)
                {
                    const int64_t col = h.ind[i + k * M] - b;
                    if(col >= 0)
                    {
                        dense[i * N + col] = h.val[i + k * M];
                    }
                }
            }
            break;
        }
        case rocsparse_format_bsr:
        {
            const int64_t bd = h.bd;
            for(int64_t ib = 0; ib < h.rows; ++ib)
            {
                for(int64_t k = h.ptr[ib] - b; k < h.ptr[ib + 1] - b; ++k)
                {
                    const int64_t jb = h.ind[k] - b;
                    for(int64_t r = 0; r < bd; ++r)
                    {
                        for(int64_t c = 0; c < bd; ++c)
                        {
                            const int64_t pos
                                = (h.dir == rocsparse_direction_row) ? r * bd + c : c * bd + r;
                            dense[(ib * bd + r) * N + jb * bd + c] = h.val[k * bd * bd + pos];
                        }
                    }
                }
            }
            break;
        }
        default:
            break;
        }
        return dense;
    }

    // Device arrays of a matrix.
    template <typename T, typename I>
    struct DevMat
    {
        device_vector<I> ptr, ind, row;
        device_vector<T> val;

        // Source: upload.
        explicit DevMat(const HostMat<T, I>& h)
            : ptr(h.ptr)
            , ind(h.ind)
            , row(h.row)
            , val(h.val)
        {
        }

        // Target: uninitialized storage large enough for any conversion.
        DevMat()
            : ptr(static_cast<size_t>(32))
            , ind(static_cast<size_t>(2 * (M * N + 16)))
            , row(static_cast<size_t>(M * N + 16))
            , val(static_cast<size_t>(M * N + 16))
        {
        }
    };

    struct SpDescr
    {
        rocsparse_spmat_descr d = nullptr;
        SpDescr()               = default;
        SpDescr(const SpDescr&) = delete;
        SpDescr& operator=(const SpDescr&) = delete;
        ~SpDescr()
        {
            if(d != nullptr)
            {
                EXPECT_EQ(rocsparse_destroy_spmat_descr(d), rocsparse_status_success);
            }
        }
    };

    template <typename T, typename I>
    rocsparse_status create_descr(const HostMat<T, I>& m, DevMat<T, I>& d, SpDescr& out)
    {
        const rocsparse_indextype it = it_of<I>();
        const rocsparse_datatype  dt = dt_of<T>();
        switch(m.format)
        {
        case rocsparse_format_coo:
            return rocsparse_create_coo_descr(
                &out.d, m.rows, m.cols, m.nnz, d.row.ptr, d.ind.ptr, d.val.ptr, it, m.base, dt);
        case rocsparse_format_coo_aos:
            return rocsparse_create_coo_aos_descr(
                &out.d, m.rows, m.cols, m.nnz, d.ind.ptr, d.val.ptr, it, m.base, dt);
        case rocsparse_format_csr:
            return rocsparse_create_csr_descr(
                &out.d, m.rows, m.cols, m.nnz, d.ptr.ptr, d.ind.ptr, d.val.ptr, it, it, m.base, dt);
        case rocsparse_format_csc:
            return rocsparse_create_csc_descr(
                &out.d, m.rows, m.cols, m.nnz, d.ptr.ptr, d.ind.ptr, d.val.ptr, it, it, m.base, dt);
        case rocsparse_format_ell:
            return rocsparse_create_ell_descr(
                &out.d, m.rows, m.cols, d.ind.ptr, d.val.ptr, m.width, it, m.base, dt);
        case rocsparse_format_bsr:
            return rocsparse_create_bsr_descr(&out.d,
                                              m.rows,
                                              m.cols,
                                              m.nnz,
                                              m.dir,
                                              m.bd,
                                              d.ptr.ptr,
                                              d.ind.ptr,
                                              d.val.ptr,
                                              it,
                                              it,
                                              m.base,
                                              dt);
        default:
            return rocsparse_status_invalid_value;
        }
    }

    template <typename T, typename I>
    rocsparse_status set_pointers(rocsparse_format format, DevMat<T, I>& d, SpDescr& out)
    {
        switch(format)
        {
        case rocsparse_format_coo:
            return rocsparse_coo_set_pointers(out.d, d.row.ptr, d.ind.ptr, d.val.ptr);
        case rocsparse_format_coo_aos:
            return rocsparse_coo_aos_set_pointers(out.d, d.ind.ptr, d.val.ptr);
        case rocsparse_format_csr:
            return rocsparse_csr_set_pointers(out.d, d.ptr.ptr, d.ind.ptr, d.val.ptr);
        case rocsparse_format_csc:
            return rocsparse_csc_set_pointers(out.d, d.ptr.ptr, d.ind.ptr, d.val.ptr);
        case rocsparse_format_ell:
            return rocsparse_ell_set_pointers(out.d, d.ind.ptr, d.val.ptr);
        case rocsparse_format_bsr:
            return rocsparse_bsr_set_pointers(out.d, d.ptr.ptr, d.ind.ptr, d.val.ptr);
        default:
            return rocsparse_status_invalid_value;
        }
    }

    // Descriptor metadata of an (empty) target of the given format.
    template <typename T, typename I>
    HostMat<T, I> target_meta(rocsparse_format     format,
                              rocsparse_index_base base,
                              int64_t              bd,
                              rocsparse_direction  dir,
                              const HostMat<T, I>& source)
    {
        HostMat<T, I> t;
        t.format = format;
        t.base   = base;
        t.dir    = dir;
        t.bd     = (format == rocsparse_format_bsr) ? bd : 1;
        t.rows   = (format == rocsparse_format_bsr) ? M / bd : M;
        t.cols   = (format == rocsparse_format_bsr) ? N / bd : N;
        switch(format)
        {
        case rocsparse_format_ell:
            t.width = (source.format == rocsparse_format_ell) ? source.width : 1;
            t.nnz   = t.width * M;
            break;
        case rocsparse_format_bsr:
            t.nnz = (source.format == rocsparse_format_bsr) ? source.nnz : 1;
            break;
        default:
            t.nnz = source.nnz;
            break;
        }
        return t;
    }

} // namespace cov70conv
