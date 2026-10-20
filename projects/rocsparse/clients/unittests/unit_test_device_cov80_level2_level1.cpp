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
// Device unit tests of the sparse vector Givens rotation (roti) and of the dot
// products with a sparse vector (doti, dotci): host and device pointer modes,
// both index bases, from a single entry to more entries than one thread block of
// the kernels covers. The results are compared with host computations on
// vectors whose products and sums are exact.
//
#include "unit_test_cov80_level2_common.hpp"

#include <numeric>

using namespace cov80l2;

namespace
{
    // Distinct, unordered indices into a vector of `n` entries.
    std::vector<int32_t> make_indices(int32_t nnz, int32_t n, int base)
    {
        std::vector<int32_t> ind(nnz);
        // i * step mod n is a bijection for a step coprime to n.
        int32_t step = 7919 % n;
        if(step == 0)
            step = 1;
        while(std::gcd(step, n) != 1)
            --step;
        for(int32_t i = 0; i < nnz; ++i)
            ind[i] = static_cast<int32_t>((static_cast<int64_t>(i) * step) % n) + base;
        return ind;
    }

    template <typename T>
    struct level1_api;

    template <>
    struct level1_api<float>
    {
        static rocsparse_status roti(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     float*               x_val,
                                     const rocsparse_int* x_ind,
                                     float*               y,
                                     const float*         c,
                                     const float*         s,
                                     rocsparse_index_base base)
        {
            return rocsparse_sroti(handle, nnz, x_val, x_ind, y, c, s, base);
        }
        static rocsparse_status doti(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     const float*         x_val,
                                     const rocsparse_int* x_ind,
                                     const float*         y,
                                     float*               result,
                                     rocsparse_index_base base)
        {
            return rocsparse_sdoti(handle, nnz, x_val, x_ind, y, result, base);
        }
    };
    template <>
    struct level1_api<double>
    {
        static rocsparse_status roti(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     double*              x_val,
                                     const rocsparse_int* x_ind,
                                     double*              y,
                                     const double*        c,
                                     const double*        s,
                                     rocsparse_index_base base)
        {
            return rocsparse_droti(handle, nnz, x_val, x_ind, y, c, s, base);
        }
        static rocsparse_status doti(rocsparse_handle     handle,
                                     rocsparse_int        nnz,
                                     const double*        x_val,
                                     const rocsparse_int* x_ind,
                                     const double*        y,
                                     double*              result,
                                     rocsparse_index_base base)
        {
            return rocsparse_ddoti(handle, nnz, x_val, x_ind, y, result, base);
        }
    };
    template <>
    struct level1_api<rocsparse_float_complex>
    {
        static rocsparse_status doti(rocsparse_handle               handle,
                                     rocsparse_int                  nnz,
                                     const rocsparse_float_complex* x_val,
                                     const rocsparse_int*           x_ind,
                                     const rocsparse_float_complex* y,
                                     rocsparse_float_complex*       result,
                                     rocsparse_index_base           base)
        {
            return rocsparse_cdoti(handle, nnz, x_val, x_ind, y, result, base);
        }
        static rocsparse_status dotci(rocsparse_handle               handle,
                                      rocsparse_int                  nnz,
                                      const rocsparse_float_complex* x_val,
                                      const rocsparse_int*           x_ind,
                                      const rocsparse_float_complex* y,
                                      rocsparse_float_complex*       result,
                                      rocsparse_index_base           base)
        {
            return rocsparse_cdotci(handle, nnz, x_val, x_ind, y, result, base);
        }
    };
    template <>
    struct level1_api<rocsparse_double_complex>
    {
        static rocsparse_status doti(rocsparse_handle                handle,
                                     rocsparse_int                   nnz,
                                     const rocsparse_double_complex* x_val,
                                     const rocsparse_int*            x_ind,
                                     const rocsparse_double_complex* y,
                                     rocsparse_double_complex*       result,
                                     rocsparse_index_base            base)
        {
            return rocsparse_zdoti(handle, nnz, x_val, x_ind, y, result, base);
        }
        static rocsparse_status dotci(rocsparse_handle                handle,
                                      rocsparse_int                   nnz,
                                      const rocsparse_double_complex* x_val,
                                      const rocsparse_int*            x_ind,
                                      const rocsparse_double_complex* y,
                                      rocsparse_double_complex*       result,
                                      rocsparse_index_base            base)
        {
            return rocsparse_zdotci(handle, nnz, x_val, x_ind, y, result, base);
        }
    };

    class Cov80L2Level1 : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void run_roti(int32_t nnz, int32_t n, int base, bool device, float c, float s)
        {
            SCOPED_TRACE(::testing::Message()
                         << "roti nnz=" << nnz << " n=" << n << " base=" << base
                         << " device=" << device << " c=" << c << " s=" << s);
            const std::vector<int32_t> ind = make_indices(nnz, n, base);
            const std::vector<T>       x   = make_vector<T>(nnz, 4);
            const std::vector<T>       y   = make_vector<T>(n, 11);
            const T                    cc  = rocsparse_ut::scalar<T>(c);
            const T                    ss  = rocsparse_ut::scalar<T>(s);
            std::vector<T>             wantx(x), wanty(y);
            for(int32_t i = 0; i < nnz; ++i)
            {
                const int32_t j = ind[i] - base;
                wantx[i]        = cc * x[i] + ss * y[j];
                wanty[j]        = cc * y[j] - ss * x[i];
            }

            device_vector<int32_t> dind(ind);
            device_vector<T> dx(x), dy(y), dc(std::vector<T>(1, cc)), ds(std::vector<T>(1, ss));
            ASSERT_TRUE(dind.ptr && dx.ptr && dy.ptr && dc.ptr && ds.ptr);

            scoped_pointer_mode mode(
                handle, device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
            ASSERT_EQ(level1_api<T>::roti(handle,
                                          nnz,
                                          dx.ptr,
                                          dind.ptr,
                                          dy.ptr,
                                          device ? dc.ptr : &cc,
                                          device ? ds.ptr : &ss,
                                          to_base(base)),
                      rocsparse_status_success);
            EXPECT_TRUE(expect_near(wantx, to_host(dx), "roti x"));
            EXPECT_TRUE(expect_near(wanty, to_host(dy), "roti y"));
        }

        template <typename T, bool CONJ>
        void run_dot(int32_t nnz, int32_t n, int base, bool device)
        {
            SCOPED_TRACE(::testing::Message()
                         << (CONJ ? "dotci" : "doti") << " nnz=" << nnz << " n=" << n
                         << " base=" << base << " device=" << device);
            const std::vector<int32_t> ind = make_indices(nnz, n, base);
            const std::vector<T>       x   = make_vector<T>(std::max(nnz, 1), 4);
            const std::vector<T>       y   = make_vector<T>(n, 11);
            T                          want{};
            for(int32_t i = 0; i < nnz; ++i)
                want = want + (CONJ ? conj_of(x[i]) : x[i]) * y[ind[i] - base];

            // Garbage in the result: it must be overwritten (nnz == 0: set to zero).
            device_vector<int32_t> dind(nnz > 0 ? ind : std::vector<int32_t>(1, base));
            device_vector<T> dx(x), dy(y), dr(std::vector<T>(1, rocsparse_ut::scalar<T>(77.0f)));
            ASSERT_TRUE(dind.ptr && dx.ptr && dy.ptr && dr.ptr);
            T hr = rocsparse_ut::scalar<T>(77.0f);

            scoped_pointer_mode mode(
                handle, device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
            if constexpr(CONJ)
                dotci_call<T>(nnz, dx.ptr, dind.ptr, dy.ptr, device ? dr.ptr : &hr, base);
            else
                ASSERT_EQ(level1_api<T>::doti(handle,
                                              nnz,
                                              dx.ptr,
                                              dind.ptr,
                                              dy.ptr,
                                              device ? dr.ptr : &hr,
                                              to_base(base)),
                          rocsparse_status_success);
            if(HasFailure())
                return;
            const T got = device ? to_host(dr)[0] : hr;
            EXPECT_TRUE(expect_near(std::vector<T>(1, want), std::vector<T>(1, got), "dot"));
        }

        // Only instantiated for the complex types.
        template <typename T>
        void
            dotci_call(int32_t nnz, const T* x, const int32_t* ind, const T* y, T* result, int base)
        {
            ASSERT_EQ(dotci_dispatch(nnz, x, ind, y, result, base), rocsparse_status_success);
        }
        template <typename T>
        rocsparse_status dotci_dispatch(
            int32_t nnz, const T* x, const int32_t* ind, const T* y, T* result, int base)
        {
            return level1_api<T>::dotci(handle, nnz, x, ind, y, result, to_base(base));
        }
    };

    TEST_F(Cov80L2Level1, roti_float)
    {
        int k = 0;
        for(const int32_t nnz : {1, 37, 600, 4100})
            for(const int base : {0, 1})
            {
                run_roti<float>(nnz, nnz + 150, base, (k++) % 2 == 1, 0.5f, -0.75f);
                if(HasFailure())
                    return;
            }
        // The identity rotation changes nothing.
        run_roti<float>(50, 80, 0, false, 1.0f, 0.0f);
        run_roti<float>(50, 80, 1, true, 1.0f, 0.0f);
    }

    TEST_F(Cov80L2Level1, roti_double)
    {
        run_roti<double>(900, 1200, 1, false, 0.25, 2.0);
        run_roti<double>(20, 33, 0, true, -1.0, 0.5);
    }

    TEST_F(Cov80L2Level1, doti_all_precisions)
    {
        int k = 0;
        for(const int32_t nnz : {0, 1, 100, 3000, 10000})
            for(const int base : {0, 1})
            {
                const bool device = (k++) % 2 == 1;
                run_dot<float, false>(nnz, nnz + 300, base, device);
                run_dot<double, false>(nnz, nnz + 300, base, !device);
                if(nnz <= 3000)
                {
                    run_dot<rocsparse_float_complex, false>(nnz, nnz + 300, base, device);
                    run_dot<rocsparse_double_complex, false>(nnz, nnz + 300, base, !device);
                }
                if(HasFailure())
                    return;
            }
    }

    TEST_F(Cov80L2Level1, dotci_complex)
    {
        int k = 0;
        for(const int32_t nnz : {0, 1, 100, 3000})
            for(const int base : {0, 1})
            {
                const bool device = (k++) % 2 == 1;
                run_dot<rocsparse_float_complex, true>(nnz, nnz + 300, base, device);
                run_dot<rocsparse_double_complex, true>(nnz, nnz + 300, base, !device);
                if(HasFailure())
                    return;
            }
    }
}
