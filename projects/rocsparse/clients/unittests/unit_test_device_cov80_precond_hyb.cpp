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

// Copies of HYB matrices (rocsparse_copy_hyb_mat) for every value type: into an empty
// destination, into a destination that already holds a matrix of the same shape and
// into one of another shape. Each copy is validated by the product with a vector against
// the host product with the CSR matrix the HYB matrix was converted from.

#include "unit_test_device_cov70_precond_common.hpp"

#include <complex>
#include <string>

using namespace cov70;

namespace
{
    struct hyb_guard
    {
        rocsparse_hyb_mat hyb = nullptr;

        hyb_guard()
        {
            EXPECT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);
        }

        ~hyb_guard()
        {
            if(hyb)
                (void)rocsparse_destroy_hyb_mat(hyb);
        }

        hyb_guard(const hyb_guard&) = delete;
        hyb_guard& operator=(const hyb_guard&) = delete;
    };

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

    template <typename F>
    void for_all_types(F&& f)
    {
        f(float{});
        f(double{});
        f(rocsparse_float_complex{});
        f(rocsparse_double_complex{});
    }

    template <typename T>
    rocsparse_status csr2hyb(rocsparse_handle        handle,
                             rocsparse_int           m,
                             rocsparse_mat_descr     descr,
                             const T*                val,
                             const rocsparse_int*    ptr,
                             const rocsparse_int*    ind,
                             rocsparse_hyb_mat       hyb,
                             rocsparse_int           width,
                             rocsparse_hyb_partition partition)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
        else
            return rocsparse_zcsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
    }

    template <typename T>
    rocsparse_status hybmv(
        rocsparse_handle handle, rocsparse_mat_descr descr, rocsparse_hyb_mat hyb, const T* x, T* y)
    {
        const T    alpha = make_value<T>(1.0);
        const T    beta  = make_value<T>(0.0);
        const auto op    = rocsparse_operation_none;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_shybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dhybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_chybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
        else
            return rocsparse_zhybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
    }

    // y = A x on the host.
    template <typename T>
    std::vector<T> host_product(const csr_matrix<T>& a, const std::vector<T>& x)
    {
        std::vector<T> y(static_cast<size_t>(a.n));
        for(int32_t i = 0; i < a.n; ++i)
        {
            std::complex<double> sum = 0.0;
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                sum += to_cd(a.val[p]) * to_cd(x[a.ind[p]]);
            y[i] = make_value<T>(sum.real(), sum.imag());
        }
        return y;
    }

    // Checks the product of the HYB matrix with a vector.
    template <typename T>
    void expect_product(rocsparse_handle     handle,
                        rocsparse_mat_descr  descr,
                        rocsparse_hyb_mat    hyb,
                        const csr_matrix<T>& a,
                        const std::string&   what)
    {
        SCOPED_TRACE(what);
        std::vector<T> x(static_cast<size_t>(a.n));
        for(int32_t i = 0; i < a.n; ++i)
            x[i] = make_value<T>(1.0 + 0.125 * (i % 7), is_cplx<T> ? 0.25 * (i % 3) : 0.0);
        device_vector<T> dx{x};
        device_vector<T> dy{std::vector<T>(static_cast<size_t>(a.n), make_value<T>(-1.0))};
        ASSERT_TRUE(dx.ptr && dy.ptr);
        ASSERT_EQ(hybmv<T>(handle, descr, hyb, dx, dy), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_close(to_host(dy), host_product(a, x), tolerance<T>(max_row_length(a)));
    }
}

class Cov80PrecondHyb : public HandleTest
{
};

TEST_F(Cov80PrecondHyb, copy_hyb_mat_of_every_type_and_partition)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        descr_guard d;
        ASSERT_NE(d.descr, nullptr);
        const csr_matrix<T>          a = make_grid_matrix<T>(4, true);
        device_vector<rocsparse_int> drow{a.ptr};
        device_vector<rocsparse_int> dcol{a.ind};
        device_vector<T>             dval{a.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        struct partition_case
        {
            rocsparse_hyb_partition partition;
            rocsparse_int           width;
            const char*             name;
        };
        // A narrow user width leaves a COO part, auto picks its own width and max keeps
        // every entry in the ELL part.
        const partition_case cases[] = {{rocsparse_hyb_partition_user, 2, "user width 2"},
                                        {rocsparse_hyb_partition_auto, 0, "auto"},
                                        {rocsparse_hyb_partition_max, 0, "max"}};
        for(const auto& c : cases)
        {
            SCOPED_TRACE(c.name);
            hyb_guard src;
            ASSERT_NE(src.hyb, nullptr);
            ASSERT_EQ(
                csr2hyb<T>(handle, a.n, d.descr, dval, drow, dcol, src.hyb, c.width, c.partition),
                rocsparse_status_success);
            expect_product<T>(handle, d.descr, src.hyb, a, "source");

            // Into an empty destination.
            hyb_guard dest;
            ASSERT_NE(dest.hyb, nullptr);
            ASSERT_EQ(rocsparse_copy_hyb_mat(dest.hyb, src.hyb), rocsparse_status_success);
            expect_product<T>(handle, d.descr, dest.hyb, a, "copy");

            // Into a destination that already holds a matrix of the same shape.
            ASSERT_EQ(rocsparse_copy_hyb_mat(dest.hyb, src.hyb), rocsparse_status_success);
            expect_product<T>(handle, d.descr, dest.hyb, a, "second copy");
        }
    });
}

TEST_F(Cov80PrecondHyb, copy_hyb_mat_rejects_a_destination_of_another_shape)
{
    descr_guard d;
    ASSERT_NE(d.descr, nullptr);
    const csr_matrix<double>     small = make_grid_matrix<double>(3, true);
    const csr_matrix<double>     large = make_grid_matrix<double>(4, true);
    device_vector<rocsparse_int> srow{small.ptr};
    device_vector<rocsparse_int> scol{small.ind};
    device_vector<double>        sval{small.val};
    device_vector<rocsparse_int> lrow{large.ptr};
    device_vector<rocsparse_int> lcol{large.ind};
    device_vector<double>        lval{large.val};
    ASSERT_TRUE(srow.ptr && scol.ptr && sval.ptr && lrow.ptr && lcol.ptr && lval.ptr);

    hyb_guard hs;
    hyb_guard hl;
    ASSERT_EQ(
        csr2hyb<double>(
            handle, small.n, d.descr, sval, srow, scol, hs.hyb, 2, rocsparse_hyb_partition_user),
        rocsparse_status_success);
    ASSERT_EQ(
        csr2hyb<double>(
            handle, large.n, d.descr, lval, lrow, lcol, hl.hyb, 2, rocsparse_hyb_partition_user),
        rocsparse_status_success);
    // The destination holds a matrix of another order: the shapes must match.
    EXPECT_NE(rocsparse_copy_hyb_mat(hl.hyb, hs.hyb), rocsparse_status_success);
    // A failed copy leaves the destination usable.
    expect_product<double>(handle, d.descr, hl.hyb, large, "destination after the failed copy");
}

TEST_F(Cov80PrecondHyb, copy_hyb_mat_argument_checks_and_empty_matrices)
{
    hyb_guard a;
    hyb_guard b;
    ASSERT_NE(a.hyb, nullptr);
    ASSERT_NE(b.hyb, nullptr);
    EXPECT_EQ(rocsparse_copy_hyb_mat(nullptr, a.hyb), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_copy_hyb_mat(a.hyb, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_copy_hyb_mat(a.hyb, a.hyb), rocsparse_status_invalid_pointer);
    // Two matrices that were only created.
    EXPECT_EQ(rocsparse_copy_hyb_mat(b.hyb, a.hyb), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_hyb_mat(nullptr), rocsparse_status_invalid_pointer);
}
