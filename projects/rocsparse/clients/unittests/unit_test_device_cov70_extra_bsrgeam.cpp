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
// Device unit tests for library/src/extra/rocsparse_bsrgeam.cpp.
//
// rocsparse_bsrgeam_nnzb + rocsparse_?bsrgeam are run for every block
// dimension class handled by the dispatch (1, 2, 3, 4, <= 8, <= 16, <= 32),
// both block directions, the zero/one index bases of A, B and C, host and
// device scalars. The complete C (row pointer, column indices, all block
// values) is compared against a host reference. The argument checks are
// exercised with one bad argument at a time.
//
#include "unit_test_cov70_extra_common.hpp"

using namespace cov70;

namespace
{
    struct GeamCase
    {
        int64_t              mb           = 6;
        int64_t              nb           = 5;
        int64_t              bd           = 2;
        rocsparse_direction  dir          = rocsparse_direction_row;
        rocsparse_index_base base_A       = rocsparse_index_base_zero;
        rocsparse_index_base base_B       = rocsparse_index_base_zero;
        rocsparse_index_base base_C       = rocsparse_index_base_zero;
        bool                 device       = false;
        cplx                 alpha        = cplx(2.0, 1.0);
        cplx                 beta         = cplx(-1.0, 0.5);
        int64_t              max_row_nnzb = 3;
        bool                 empty_A      = false;
        bool                 empty_B      = false;
    };

    std::string describe(const GeamCase& c)
    {
        return "mb=" + std::to_string(c.mb) + " nb=" + std::to_string(c.nb)
               + " bd=" + std::to_string(c.bd) + " dir=" + std::to_string(c.dir)
               + " bases=" + std::to_string(c.base_A) + std::to_string(c.base_B)
               + std::to_string(c.base_C) + " dev=" + std::to_string(c.device);
    }

    template <typename T>
    void run_bsrgeam(rocsparse_handle handle, const GeamCase& c)
    {
        SCOPED_TRACE(describe(c));
        Rng rng(static_cast<uint64_t>(c.mb * 7 + c.bd * 31 + c.nb) + 100);

        Bsr A = random_bsr<T>(c.mb, c.nb, c.bd, c.max_row_nnzb, rng);
        Bsr B = random_bsr<T>(c.mb, c.nb, c.bd, c.max_row_nnzb, rng);
        if(c.empty_A)
            A = Bsr{
                c.mb, c.nb, c.bd, std::vector<int64_t>(static_cast<size_t>(c.mb) + 1, 0), {}, {}};
        if(c.empty_B)
            B = Bsr{
                c.mb, c.nb, c.bd, std::vector<int64_t>(static_cast<size_t>(c.mb) + 1, 0), {}, {}};

        const Scalar<T> alpha(c.alpha, c.device), beta(c.beta, c.device);
        const Bsr       want = bsr_geam(to_c<T>(alpha.h), A, to_c<T>(beta.h), B);

        MatDescrH dA(c.base_A), dB(c.base_B), dC(c.base_C);

        DevBuf<rocsparse_int> pA = up_idx<rocsparse_int>(A.ptr, c.base_A);
        DevBuf<rocsparse_int> iA = up_idx<rocsparse_int>(A.col, c.base_A);
        DevBuf<T>             vA = up_raw<T>(bsr_pack<T>(A, c.dir));
        DevBuf<rocsparse_int> pB = up_idx<rocsparse_int>(B.ptr, c.base_B);
        DevBuf<rocsparse_int> iB = up_idx<rocsparse_int>(B.col, c.base_B);
        DevBuf<T>             vB = up_raw<T>(bsr_pack<T>(B, c.dir));
        DevBuf<rocsparse_int> pC(static_cast<size_t>(c.mb) + 1);
        ASSERT_TRUE(pA.ptr && iA.ptr && vA.ptr && pB.ptr && iB.ptr && vB.ptr && pC.ptr);

        PointerModeGuard mode(handle, mode_of(c.device));

        // nnzb of C (host or device pointer, following the pointer mode)
        rocsparse_int         nnzb_host = -1;
        DevBuf<rocsparse_int> nnzb_dev(size_t{1});
        ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                         c.dir,
                                         static_cast<rocsparse_int>(c.mb),
                                         static_cast<rocsparse_int>(c.nb),
                                         static_cast<rocsparse_int>(c.bd),
                                         dA.d,
                                         static_cast<rocsparse_int>(A.nnzb()),
                                         pA.ptr,
                                         iA.ptr,
                                         dB.d,
                                         static_cast<rocsparse_int>(B.nnzb()),
                                         pB.ptr,
                                         iB.ptr,
                                         dC.d,
                                         pC.ptr,
                                         c.device ? nnzb_dev.ptr : &nnzb_host),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const int64_t nnzb_C = c.device ? nnzb_dev.host()[0] : nnzb_host;
        ASSERT_EQ(nnzb_C, want.nnzb());
        expect_idx_equal<rocsparse_int>(pC.host(), want.ptr, c.base_C, "row_ptr_C");

        DevBuf<rocsparse_int> iC(static_cast<size_t>(nnzb_C));
        DevBuf<T>             vC(static_cast<size_t>(nnzb_C * c.bd * c.bd));
        ASSERT_TRUE(iC.ptr && vC.ptr);
        ASSERT_EQ(Api<T>::bsrgeam(handle,
                                  c.dir,
                                  static_cast<rocsparse_int>(c.mb),
                                  static_cast<rocsparse_int>(c.nb),
                                  static_cast<rocsparse_int>(c.bd),
                                  alpha.typed(),
                                  dA.d,
                                  static_cast<rocsparse_int>(A.nnzb()),
                                  vA.ptr,
                                  pA.ptr,
                                  iA.ptr,
                                  beta.typed(),
                                  dB.d,
                                  static_cast<rocsparse_int>(B.nnzb()),
                                  vB.ptr,
                                  pB.ptr,
                                  iB.ptr,
                                  dC.d,
                                  vC.ptr,
                                  pC.ptr,
                                  iC.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(nnzb_C > 0)
        {
            expect_idx_equal<rocsparse_int>(iC.host(), want.col, c.base_C, "col_ind_C");
            // expected values in the device layout of the block direction
            const std::vector<T> packed = bsr_pack<T>(want, c.dir);
            std::vector<cplx>    want_dev(packed.size());
            for(size_t i = 0; i < packed.size(); ++i)
                want_dev[i] = to_c<T>(packed[i]);
            expect_vec_equal<T>(vC.host(), want_dev, "val_C");
        }
        // inputs are unchanged
        expect_idx_equal<rocsparse_int>(pA.host(), A.ptr, c.base_A, "row_ptr_A");
        expect_idx_equal<rocsparse_int>(iB.host(), B.col, c.base_B, "col_ind_B");
    }

    template <typename T>
    void run_dims(rocsparse_handle handle, const std::vector<int64_t>& dims, bool all_options)
    {
        for(int64_t bd : dims)
            for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
            {
                GeamCase c;
                c.bd  = bd;
                c.dir = dir;
                run_bsrgeam<T>(handle, c);
                if(::testing::Test::HasFatalFailure())
                    return;
                if(!all_options)
                    continue;
                c.device = true;
                run_bsrgeam<T>(handle, c);
                c.device = false;
                c.base_A = c.base_B = c.base_C = rocsparse_index_base_one;
                run_bsrgeam<T>(handle, c);
                c.base_A = rocsparse_index_base_one;
                c.base_B = rocsparse_index_base_zero;
                c.base_C = rocsparse_index_base_one;
                run_bsrgeam<T>(handle, c);
                c.base_C  = rocsparse_index_base_zero;
                c.empty_B = true;
                run_bsrgeam<T>(handle, c);
                c.empty_B = false;
                c.empty_A = true;
                run_bsrgeam<T>(handle, c);
                if(::testing::Test::HasFatalFailure())
                    return;
            }
    }

    struct BsrgeamCov70 : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(BsrgeamCov70, block_dims_float)
{
    run_dims<float>(handle, {1, 2, 3, 4, 5, 8, 9, 16, 17, 32}, true);
}

TEST_F(BsrgeamCov70, block_dims_double)
{
    run_dims<double>(handle, {2, 3, 4, 7, 12, 24}, false);
}

TEST_F(BsrgeamCov70, block_dims_complex)
{
    run_dims<rocsparse_float_complex>(handle, {2, 3, 6, 11}, false);
    run_dims<rocsparse_double_complex>(handle, {3, 4, 14}, false);
}

TEST_F(BsrgeamCov70, wide_rows)
{
    for(int64_t bd : {2, 4, 8})
    {
        GeamCase c;
        c.bd           = bd;
        c.mb           = 40;
        c.nb           = 70;
        c.max_row_nnzb = 60; // rows with many block entries
        run_bsrgeam<float>(handle, c);
        if(HasFatalFailure())
            return;
    }
}

// Quick returns: C is empty, nothing to compute.
//  - rocsparse_bsrgeam_nnzb reports nnzb_C = 0 (host or device pointer) and
//    fills the row pointer of C with the index base of C (the row pointer has
//    mb + 1 entries, the sentinel value is overwritten),
//  - rocsparse_sbsrgeam returns success and does not touch C at all (row
//    pointer, column indices and values keep their sentinel values).
TEST_F(BsrgeamCov70, quick_returns)
{
    struct Dim
    {
        rocsparse_int mb, nb;
    };
    // both inputs empty, all rows empty; mb == 0 and nb == 0 (an empty
    // matrix has no entries).
    const Dim dims[] = {{2, 2}, {3, 0}, {0, 2}};

    for(bool device : {false, true})
        for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            for(const Dim& d : dims)
            {
                SCOPED_TRACE(testing::Message() << "device=" << device << " base=" << base
                                                << " mb=" << d.mb << " nb=" << d.nb);
                const rocsparse_int sentinel = 7;
                MatDescrH           descr(base);
                // A and B are empty: row pointers hold the index base.
                DevBuf<rocsparse_int> ptr
                    = up_idx<rocsparse_int>(std::vector<int64_t>(size_t(d.mb) + 1, 0), base);
                DevBuf<rocsparse_int> col(std::vector<rocsparse_int>{-5});
                DevBuf<float>         val(std::vector<float>{-5.f});
                DevBuf<rocsparse_int> pC(std::vector<rocsparse_int>(size_t(d.mb) + 1, sentinel));
                DevBuf<rocsparse_int> cC(std::vector<rocsparse_int>{sentinel});
                DevBuf<float>         vC(std::vector<float>{-7.f});
                PointerModeGuard      mode(handle, mode_of(device));
                const Scalar<float>   alpha(cplx(1.0, 0.0), device), beta(cplx(1.0, 0.0), device);

                rocsparse_int         nnzb_host = -1;
                DevBuf<rocsparse_int> nnzb_dev(std::vector<rocsparse_int>{-1});
                ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                                 rocsparse_direction_row,
                                                 d.mb,
                                                 d.nb,
                                                 2,
                                                 descr.d,
                                                 0,
                                                 ptr.ptr,
                                                 col.ptr,
                                                 descr.d,
                                                 0,
                                                 ptr.ptr,
                                                 col.ptr,
                                                 descr.d,
                                                 pC.ptr,
                                                 device ? nnzb_dev.ptr : &nnzb_host),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                EXPECT_EQ(device ? nnzb_dev.host()[0] : nnzb_host, 0);
                expect_idx_equal<rocsparse_int>(pC.host(),
                                                std::vector<int64_t>(size_t(d.mb) + 1, 0),
                                                base,
                                                "row_ptr_C after nnzb");

                // The compute routine has nothing to do and leaves C alone.
                EXPECT_EQ(rocsparse_sbsrgeam(handle,
                                             rocsparse_direction_row,
                                             d.mb,
                                             d.nb,
                                             2,
                                             alpha.typed(),
                                             descr.d,
                                             0,
                                             val.ptr,
                                             ptr.ptr,
                                             col.ptr,
                                             beta.typed(),
                                             descr.d,
                                             0,
                                             val.ptr,
                                             ptr.ptr,
                                             col.ptr,
                                             descr.d,
                                             vC.ptr,
                                             pC.ptr,
                                             cC.ptr),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                expect_idx_equal<rocsparse_int>(pC.host(),
                                                std::vector<int64_t>(size_t(d.mb) + 1, 0),
                                                base,
                                                "row_ptr_C after bsrgeam");
                EXPECT_EQ(cC.host()[0], sentinel);
                EXPECT_EQ(vC.host()[0], -7.f);
            }
}

// One invalid argument at a time.
TEST_F(BsrgeamCov70, argument_errors)
{
    MatDescrH             ok, general_b, sym, unsorted;
    DevBuf<rocsparse_int> ptr(std::vector<rocsparse_int>{0, 1, 2});
    DevBuf<rocsparse_int> col(std::vector<rocsparse_int>{0, 1});
    DevBuf<float>         val(std::vector<float>(8, 1.f));
    DevBuf<rocsparse_int> pC(size_t{3});
    DevBuf<rocsparse_int> cC(size_t{8});
    DevBuf<float>         vC(size_t{32});
    ASSERT_EQ(rocsparse_set_mat_type(sym.d, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(unsorted.d, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    const float alpha = 1.f, beta = 1.f;

    struct Args
    {
        rocsparse_handle     handle;
        rocsparse_direction  dir = rocsparse_direction_row;
        rocsparse_int        mb = 2, nb = 2, bd = 2;
        const float*         alpha = nullptr;
        rocsparse_mat_descr  dA, dB, dC;
        rocsparse_int        nnzb_A = 2, nnzb_B = 2;
        const float*         vA   = nullptr;
        const rocsparse_int* pA   = nullptr;
        const rocsparse_int* iA   = nullptr;
        const float*         beta = nullptr;
        const float*         vB   = nullptr;
        const rocsparse_int* pB   = nullptr;
        const rocsparse_int* iB   = nullptr;
        float*               vC   = nullptr;
        const rocsparse_int* pC   = nullptr;
        rocsparse_int*       iC   = nullptr;
    };
    auto good = [&] {
        Args a;
        a.handle = handle;
        a.alpha  = &alpha;
        a.beta   = &beta;
        a.dA = a.dB = a.dC = ok.d;
        a.vA = a.vB = val.ptr;
        a.pA = a.pB = ptr.ptr;
        a.iA = a.iB = col.ptr;
        a.vC        = vC.ptr;
        a.pC        = pC.ptr;
        a.iC        = cC.ptr;
        return a;
    };
    auto call = [&](const Args& a) {
        return rocsparse_sbsrgeam(a.handle,
                                  a.dir,
                                  a.mb,
                                  a.nb,
                                  a.bd,
                                  a.alpha,
                                  a.dA,
                                  a.nnzb_A,
                                  a.vA,
                                  a.pA,
                                  a.iA,
                                  a.beta,
                                  a.dB,
                                  a.nnzb_B,
                                  a.vB,
                                  a.pB,
                                  a.iB,
                                  a.dC,
                                  a.vC,
                                  a.pC,
                                  a.iC);
    };

    // The setup itself is valid: build C row pointers with the nnz routine.
    rocsparse_int nnzb = 0;
    ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                     rocsparse_direction_row,
                                     2,
                                     2,
                                     2,
                                     ok.d,
                                     2,
                                     ptr.ptr,
                                     col.ptr,
                                     ok.d,
                                     2,
                                     ptr.ptr,
                                     col.ptr,
                                     ok.d,
                                     pC.ptr,
                                     &nnzb),
              rocsparse_status_success);
    ASSERT_EQ(nnzb, 2);
    EXPECT_EQ(call(good()), rocsparse_status_success);

    {
        Args a   = good();
        a.handle = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_handle);
    }
    {
        Args a = good();
        a.dA   = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.dB = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.dC = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
    }
    {
        Args a = good();
        a.dA   = sym.d;
        EXPECT_EQ(call(a), rocsparse_status_not_implemented);
        a    = good();
        a.dB = sym.d;
        EXPECT_EQ(call(a), rocsparse_status_not_implemented);
        a    = good();
        a.dC = sym.d;
        EXPECT_EQ(call(a), rocsparse_status_not_implemented);
        a    = good();
        a.dA = unsorted.d;
        EXPECT_EQ(call(a), rocsparse_status_requires_sorted_storage);
        a    = good();
        a.dB = unsorted.d;
        EXPECT_EQ(call(a), rocsparse_status_requires_sorted_storage);
        a    = good();
        a.dC = unsorted.d;
        EXPECT_EQ(call(a), rocsparse_status_requires_sorted_storage);
    }
    {
        Args a = good();
        a.dir  = static_cast<rocsparse_direction>(77);
        EXPECT_EQ(call(a), rocsparse_status_invalid_value);
        a    = good();
        a.mb = -1;
        EXPECT_EQ(call(a), rocsparse_status_invalid_size);
        a    = good();
        a.nb = -1;
        EXPECT_EQ(call(a), rocsparse_status_invalid_size);
        a    = good();
        a.bd = 0;
        EXPECT_EQ(call(a), rocsparse_status_invalid_size);
        a        = good();
        a.nnzb_A = -1;
        EXPECT_EQ(call(a), rocsparse_status_invalid_size);
        a        = good();
        a.nnzb_B = -1;
        EXPECT_EQ(call(a), rocsparse_status_invalid_size);
    }
    {
        Args a  = good();
        a.alpha = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a      = good();
        a.beta = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.vA = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.pA = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.iA = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.vB = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.pB = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.iB = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.pC = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        // C values / columns are required once the row pointer says nnzb_C > 0
        a    = good();
        a.vC = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
        a    = good();
        a.iC = nullptr;
        EXPECT_EQ(call(a), rocsparse_status_invalid_pointer);
    }
}
