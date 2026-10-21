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
// Helpers shared by the level2 device unit tests: deterministic data, host sparse
// containers and formats, RAII descriptors, precision dispatch of the C API and
// host reference products.
//
#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace level2
{
    using rocsparse_ut::device_vector;
    using rocsparse_ut::to_host;

    // ----- scalar traits ----------------------------------------------------
    template <typename T>
    struct real_of
    {
        using type = T;
    };
    template <>
    struct real_of<rocsparse_float_complex>
    {
        using type = float;
    };
    template <>
    struct real_of<rocsparse_double_complex>
    {
        using type = double;
    };

    template <typename T>
    constexpr bool is_single()
    {
        return std::is_same<typename real_of<T>::type, float>::value;
    }

    // Comparison tolerance used for products and solves of the small, exactly
    // representable data generated below.
    template <typename T>
    constexpr double tolerance()
    {
        return is_single<T>() ? 1.0e-4 : 1.0e-11;
    }

    template <typename T>
    inline double magnitude(const T& v)
    {
        return static_cast<double>(std::abs(v));
    }

    // Deterministic values: multiples of 1/16 in [-1.875, 1.875]. The real part has
    // period 61 and the imaginary part period 53 (both prime), so the sequence
    // does not repeat for any block size or row length used by the tests, and
    // an index that is off by a small multiple of a block dimension reads a
    // different value. All sums and products of these values (a few thousand
    // terms at most) are exact in single precision, which keeps the results of
    // atomic reductions independent of the summation order.
    // The value is zero for one `n` in 61 (a deliberate sprinkling of zeros);
    // with `nonzero` set the zero is replaced, so the value can be used for a
    // diagonal entry.
    template <typename T>
    inline T value_of(int64_t n, bool nonzero = false)
    {
        int64_t k = (n * 37 + 11) % 61 - 30;
        if(nonzero && k == 0)
            k = 7;
        return rocsparse_ut::scalar<T>(static_cast<float>(k) / 16.0f);
    }
    template <>
    inline rocsparse_float_complex value_of<rocsparse_float_complex>(int64_t n, bool nonzero)
    {
        const float im = static_cast<float>((n * 23 + 5) % 53 - 26) / 16.0f;
        return rocsparse_float_complex(value_of<float>(n, nonzero), im);
    }
    template <>
    inline rocsparse_double_complex value_of<rocsparse_double_complex>(int64_t n, bool nonzero)
    {
        const double im = static_cast<double>((n * 23 + 5) % 53 - 26) / 16.0;
        return rocsparse_double_complex(value_of<double>(n, nonzero), im);
    }

    // ----- comparison -------------------------------------------------------
    // Compares two vectors element by element. Reports the first mismatch (and
    // the count of mismatches) once, so a wrong kernel does not flood the log.
    template <typename T>
    inline bool expect_near(const std::vector<T>& want, const std::vector<T>& got, const char* what)
    {
        if(want.size() != got.size())
        {
            ADD_FAILURE() << what << ": size " << got.size() << " != " << want.size();
            return false;
        }
        const double tol   = tolerance<T>();
        size_t       bad   = 0;
        size_t       first = 0;
        for(size_t i = 0; i < want.size(); ++i)
        {
            const double err = magnitude(want[i] - got[i]);
            if(!(err <= tol * (1.0 + magnitude(want[i]))))
            {
                if(bad == 0)
                    first = i;
                ++bad;
            }
        }
        if(bad > 0)
        {
            ADD_FAILURE() << what << ": " << bad << " mismatches, first at " << first
                          << " want=" << want[first] << " got=" << got[first];
            return false;
        }
        return true;
    }

    // ----- RAII descriptors -------------------------------------------------
    // Owns a descriptor created by `create(&d)`; a failed creation leaves d null.
    template <typename D, auto create, auto destroy>
    struct created
    {
        D d = nullptr;
        created()
        {
            if(create(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~created()
        {
            if(d)
                (void)destroy(d);
        }
        created(const created&) = delete;
        created& operator=(const created&) = delete;
                 operator D() const
        {
            return d;
        }
    };

    using mat_descr
        = created<rocsparse_mat_descr, rocsparse_create_mat_descr, rocsparse_destroy_mat_descr>;
    using mat_info
        = created<rocsparse_mat_info, rocsparse_create_mat_info, rocsparse_destroy_mat_info>;

    // ----- host containers --------------------------------------------------
    template <typename T>
    struct host_csr
    {
        int32_t              m    = 0;
        int32_t              n    = 0;
        int                  base = 0;
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;

        int32_t nnz() const
        {
            return static_cast<int32_t>(ind.size());
        }
    };

    template <typename J, typename S>
    inline std::vector<J> convert_vector(const std::vector<S>& v)
    {
        return std::vector<J>(v.begin(), v.end());
    }

    // Device copy of column indices with one guard element (value -7, never a
    // valid column) on each side: the triangular-solve analysis kernels read the
    // entry that precedes or follows an empty row. `ptr` skips the first guard.
    template <typename J = rocsparse_int>
    struct guarded_indices
    {
        device_vector<J> buf;
        J*               ptr = nullptr;

        explicit guarded_indices(const std::vector<int32_t>& v)
            : buf(padded(v))
        {
            ptr = buf.ptr ? buf.ptr + 1 : nullptr;
        }
        operator J*() const
        {
            return ptr;
        }

    private:
        static std::vector<J> padded(const std::vector<int32_t>& v)
        {
            std::vector<J> p(v.size() + 2, static_cast<J>(-7));
            std::copy(v.begin(), v.end(), p.begin() + 1);
            return p;
        }
    };

    // Device copy of a host CSR matrix with row pointer type I and column index
    // type J (the columns are guarded, see guarded_indices).
    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    struct device_csr
    {
        device_vector<I>   ptr;
        guarded_indices<J> ind;
        device_vector<T>   val;

        explicit device_csr(const host_csr<T>& a)
            : ptr(convert_vector<I>(a.ptr))
            , ind(a.ind)
            , val(a.val.empty() ? std::vector<T>(1, T{}) : a.val)
        {
        }

        bool ok() const
        {
            return ptr.ptr && ind.ptr && val.ptr;
        }
    };

    template <typename T>
    inline std::vector<T> make_vector(size_t n, int64_t seed)
    {
        std::vector<T> v(n);
        for(size_t i = 0; i < n; ++i)
            v[i] = value_of<T>(static_cast<int64_t>(i) + seed);
        return v;
    }

    // ----- precision dispatch of the C API -----------------------------------
    // NAME<T>(args...) calls rocsparse_{s,d,c,z}NAME. The arguments are forwarded, so
    // device_vector and similar wrappers convert to the pointer type of the routine.
#define LEVEL2_DISPATCH(NAME)                                              \
    template <typename T, typename... Args>                                \
    inline rocsparse_status NAME(Args&&... args)                           \
    {                                                                      \
        if constexpr(std::is_same<T, float>::value)                        \
            return rocsparse_s##NAME(std::forward<Args>(args)...);         \
        else if constexpr(std::is_same<T, double>::value)                  \
            return rocsparse_d##NAME(std::forward<Args>(args)...);         \
        else if constexpr(std::is_same<T, rocsparse_float_complex>::value) \
            return rocsparse_c##NAME(std::forward<Args>(args)...);         \
        else                                                               \
            return rocsparse_z##NAME(std::forward<Args>(args)...);         \
    }

    LEVEL2_DISPATCH(csritsv_analysis)
    LEVEL2_DISPATCH(csritsv_buffer_size)
    LEVEL2_DISPATCH(csritsv_solve)
    LEVEL2_DISPATCH(csritsv_solve_ex)
#undef LEVEL2_DISPATCH

} // namespace level2
