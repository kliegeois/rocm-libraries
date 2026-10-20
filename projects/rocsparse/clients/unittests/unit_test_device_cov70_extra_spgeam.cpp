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
// Device unit tests for library/src/extra/rocsparse_spgeam.cpp.
//
// The generic rocsparse_spgeam API is driven through
//  - the combined analysis/compute flow and the split symbolic/numeric flow,
//  - the output matrix C created before or after the analysis,
//  - host and device scalars, scalar types different from the compute type,
//  - the complete stage state machine (every previous stage against every
//    next stage), and
//  - the argument checks.
// Every successful flow compares the whole C (row pointer, column indices and
// values) against a host reference C = alpha * A + beta * B.
//
#include "unit_test_cov70_extra_common.hpp"

#include <memory>

using namespace cov70;

namespace
{
    using stage_t = rocsparse_spgeam_stage;

    const char* stage_name(stage_t s)
    {
        switch(s)
        {
        case rocsparse_spgeam_stage_analysis:
            return "analysis";
        case rocsparse_spgeam_stage_compute:
            return "compute";
        case rocsparse_spgeam_stage_symbolic_analysis:
            return "symbolic_analysis";
        case rocsparse_spgeam_stage_symbolic_compute:
            return "symbolic_compute";
        case rocsparse_spgeam_stage_numeric_analysis:
            return "numeric_analysis";
        case rocsparse_spgeam_stage_numeric_compute:
            return "numeric_compute";
        }
        return "invalid";
    }

    constexpr stage_t SA = rocsparse_spgeam_stage_symbolic_analysis;
    constexpr stage_t SC = rocsparse_spgeam_stage_symbolic_compute;
    constexpr stage_t NA = rocsparse_spgeam_stage_numeric_analysis;
    constexpr stage_t NC = rocsparse_spgeam_stage_numeric_compute;
    constexpr stage_t AN = rocsparse_spgeam_stage_analysis;
    constexpr stage_t CO = rocsparse_spgeam_stage_compute;

    bool is_analysis_stage(stage_t s)
    {
        return s == SA || s == AN;
    }

    // One SpGEAM problem: A, B (CSR), optional C (CSR) and the descriptor.
    template <typename T, typename S, typename I, typename J>
    struct Geam
    {
        rocsparse_handle       handle = nullptr;
        rocsparse_spgeam_descr descr  = nullptr;
        Csr                    A, B;
        Bsr                    want;
        DevBuf<I>              pA, pB, pC;
        DevBuf<J>              cA, cB, cC;
        DevBuf<T>              vA, vB, vC;
        SpmatH                 sA, sB, sC;
        DevBuf<char>           tmp;

        std::unique_ptr<Scalar<S>>        alpha, beta;
        std::unique_ptr<PointerModeGuard> mode;

        int64_t nnz_C         = -1;
        bool    verify_values = true;

        Geam()            = default;
        Geam(const Geam&) = delete;
        Geam& operator=(const Geam&) = delete;
        ~Geam()
        {
            if(descr != nullptr)
                (void)rocsparse_destroy_spgeam_descr(descr);
        }

        void init(rocsparse_handle h, bool device, uint64_t seed, int64_t m = 9, int64_t n = 7)
        {
            handle = h;
            Rng rng(seed);
            A = random_csr<T>(m, n, 4, rng);
            B = random_csr<T>(m, n, 4, rng);
            ASSERT_GT(A.nnz(), 0);
            ASSERT_GT(B.nnz(), 0);

            mode = std::make_unique<PointerModeGuard>(h, mode_of(device));
            alpha.reset(
                new Scalar<S>(is_complex_t<S>::value ? cplx(2.0, 1.0) : cplx(2.0, 0.0), device));
            beta.reset(
                new Scalar<S>(is_complex_t<S>::value ? cplx(-1.0, 0.5) : cplx(-1.0, 0.0), device));
            want = bsr_geam(to_c<S>(alpha->h), bsr_from_csr(A), to_c<S>(beta->h), bsr_from_csr(B));

            ASSERT_NO_FATAL_FAILURE((make_csr_descr<T, I, J>(sA, A, pA, cA, vA)));
            ASSERT_NO_FATAL_FAILURE((make_csr_descr<T, I, J>(sB, B, pB, cB, vB)));

            ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);
            set_inputs(descr);
        }

        void set_inputs(rocsparse_spgeam_descr d,
                        rocsparse_datatype     scalar_type  = rocsparse_ut::dt_of<S>(),
                        rocsparse_datatype     compute_type = rocsparse_ut::dt_of<T>())
        {
            const rocsparse_spgeam_alg alg  = rocsparse_spgeam_alg_default;
            const rocsparse_operation  none = rocsparse_operation_none;
            ASSERT_EQ(rocsparse_spgeam_set_input(
                          handle, d, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                                 d,
                                                 rocsparse_spgeam_input_scalar_datatype,
                                                 &scalar_type,
                                                 sizeof(scalar_type),
                                                 nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                                 d,
                                                 rocsparse_spgeam_input_compute_datatype,
                                                 &compute_type,
                                                 sizeof(compute_type),
                                                 nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(
                rocsparse_spgeam_set_input(
                    handle, d, rocsparse_spgeam_input_operation_A, &none, sizeof(none), nullptr),
                rocsparse_status_success);
            ASSERT_EQ(
                rocsparse_spgeam_set_input(
                    handle, d, rocsparse_spgeam_input_operation_B, &none, sizeof(none), nullptr),
                rocsparse_status_success);
            ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                                 d,
                                                 rocsparse_spgeam_input_scalar_alpha,
                                                 alpha->get(),
                                                 sizeof(void*),
                                                 nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                                 d,
                                                 rocsparse_spgeam_input_scalar_beta,
                                                 beta->get(),
                                                 sizeof(void*),
                                                 nullptr),
                      rocsparse_status_success);
        }

        // C with room for nnz entries (the library fills the row pointer).
        void create_C(int64_t nnz)
        {
            ASSERT_EQ(sC.p, nullptr);
            pC = DevBuf<I>(static_cast<size_t>(A.m) + 1);
            cC = DevBuf<J>(static_cast<size_t>(nnz));
            vC = DevBuf<T>(static_cast<size_t>(nnz));
            ASSERT_TRUE(pC.ptr && cC.ptr && vC.ptr);
            ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                                 A.m,
                                                 A.n,
                                                 nnz,
                                                 pC.ptr,
                                                 cC.ptr,
                                                 vC.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 rocsparse_index_base_zero,
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }

        void clear_values()
        {
            if(vC.ptr != nullptr)
                ASSERT_EQ(hipMemset(vC.ptr, 0, vC.n * sizeof(T)), hipSuccess);
        }

        // Buffer size + execution of one stage. C is only passed if requested
        // and available.
        rocsparse_status run(stage_t stage, bool pass_C)
        {
            rocsparse_spmat_descr C  = (pass_C && sC.p != nullptr) ? sC.p : nullptr;
            size_t                bs = 0;
            rocsparse_status      st
                = rocsparse_spgeam_buffer_size(handle, descr, sA.p, sB.p, C, stage, &bs, nullptr);
            if(st != rocsparse_status_success)
                return st;
            tmp = DevBuf<char>(bs);
            st  = rocsparse_spgeam(handle, descr, sA.p, sB.p, C, stage, bs, tmp.ptr, nullptr);
            if(st == rocsparse_status_success && is_analysis_stage(stage))
            {
                // Without C the number of non-zeros is an output of the
                // descriptor, with C it is written into the matrix itself.
                int64_t nnz = -1;
                if(C == nullptr)
                {
                    EXPECT_EQ(
                        rocsparse_spgeam_get_output(
                            handle, descr, rocsparse_spgeam_output_nnz, &nnz, sizeof(nnz), nullptr),
                        rocsparse_status_success);
                }
                else
                {
                    int64_t rows = 0, cols = 0;
                    EXPECT_EQ(rocsparse_spmat_get_size(C, &rows, &cols, &nnz),
                              rocsparse_status_success);
                }
                nnz_C = nnz;
            }
            return st;
        }

        // Run a stage that must succeed. The first analysis creates C if it
        // does not exist yet.
        void ok(stage_t stage, bool pass_C)
        {
            if(stage == NC || stage == CO || stage == NA || stage == SC)
                clear_values();
            ASSERT_EQ(run(stage, pass_C), rocsparse_status_success) << stage_name(stage);
            if(is_analysis_stage(stage))
            {
                ASSERT_EQ(nnz_C, want.nnzb()) << stage_name(stage);
                if(sC.p == nullptr)
                    ASSERT_NO_FATAL_FAILURE(create_C(nnz_C));
            }
        }

        void check_pattern()
        {
            ASSERT_NE(sC.p, nullptr);
            const std::vector<I> hp = pC.host();
            const std::vector<J> hc = cC.host();
            expect_idx_equal<I>(hp, want.ptr, 0, "C row_ptr");
            expect_idx_equal<J>(hc, want.col, 0, "C col_ind");
        }

        void check_C()
        {
            ASSERT_NO_FATAL_FAILURE(check_pattern());
            if(!verify_values)
                return;
            const std::vector<T> hv = vC.host();
            expect_vec_equal<T>(hv, want.val, "C val");
        }
    };

    // kind 0: analysis (C created afterwards) + compute
    // kind 1: symbolic / numeric split flow with repeated numeric stages
    // kind 2: C created first, analysis + compute
    // kind 3: C created first, symbolic / numeric split flow
    template <typename T, typename S, typename I, typename J>
    void flow(rocsparse_handle handle, bool device, int kind, uint64_t seed, bool verify_values)
    {
        SCOPED_TRACE(testing::Message()
                     << "device=" << device << " kind=" << kind
                     << " compute_type=" << static_cast<int>(rocsparse_ut::dt_of<T>())
                     << " scalar_type=" << static_cast<int>(rocsparse_ut::dt_of<S>()));
        Geam<T, S, I, J> g;
        g.verify_values = verify_values;
        ASSERT_NO_FATAL_FAILURE(g.init(handle, device, seed));
        const bool c_first = kind >= 2;
        if(c_first)
            ASSERT_NO_FATAL_FAILURE(g.create_C(g.want.nnzb()));

        if(kind % 2 == 0)
        {
            ASSERT_NO_FATAL_FAILURE(g.ok(AN, c_first));
            ASSERT_NO_FATAL_FAILURE(g.ok(CO, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
        }
        else
        {
            ASSERT_NO_FATAL_FAILURE(g.ok(SA, c_first));
            ASSERT_NO_FATAL_FAILURE(g.ok(SC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_pattern());
            ASSERT_NO_FATAL_FAILURE(g.ok(NA, true));
            ASSERT_NO_FATAL_FAILURE(g.ok(NC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
            // The numeric stages can be repeated.
            ASSERT_NO_FATAL_FAILURE(g.ok(NC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
            ASSERT_NO_FATAL_FAILURE(g.ok(NA, true));
            ASSERT_NO_FATAL_FAILURE(g.ok(NC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
        }
    }

    template <typename T, typename S, typename I, typename J>
    void flows(rocsparse_handle handle, uint64_t seed, bool verify_values = true)
    {
        for(int device = 0; device < 2; ++device)
            for(int kind = 0; kind < 4; ++kind)
            {
                ASSERT_NO_FATAL_FAILURE(
                    (flow<T, S, I, J>(handle, device != 0, kind, seed, verify_values)));
                ++seed;
            }
    }

    struct SpgeamCov70 : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(SpgeamCov70, flows_float_int32)
{
    flows<float, float, int32_t, int32_t>(handle, 1);
}

TEST_F(SpgeamCov70, flows_double_int64)
{
    flows<double, double, int64_t, int32_t>(handle, 21);
    flows<double, double, int64_t, int64_t>(handle, 41);
}

TEST_F(SpgeamCov70, flows_complex)
{
    flows<rocsparse_float_complex, rocsparse_float_complex, int32_t, int32_t>(handle, 61);
    flows<rocsparse_double_complex, rocsparse_double_complex, int64_t, int64_t>(handle, 81);
}

// The scalars are converted from the scalar type to the compute type.
// The library hands the converted scalars (compute type) to csrgeam together
// with the scalar type, so the numerical values of C are not meaningful when
// the two types differ. These flows therefore verify the status of every
// stage and the complete sparsity pattern (row pointer and column indices)
// of C, and deliberately do not verify the values.
TEST_F(SpgeamCov70, flows_scalar_type_differs_from_compute_type)
{
    flows<double, float, int32_t, int32_t>(handle, 101, false);
    flows<float, double, int32_t, int32_t>(handle, 121, false);
}

// Every previous stage against every next stage. The legal transitions must
// succeed (and compute the right C when they produce values), all others must
// report rocsparse_status_invalid_value.
TEST_F(SpgeamCov70, stage_state_machine)
{
    using G = Geam<float, float, int32_t, int32_t>;

    const std::vector<std::vector<stage_t>> paths
        = {{}, {SA}, {AN}, {AN, CO}, {SA, SC}, {SA, SC, NA}, {SA, SC, NA, NC}};
    const std::vector<stage_t> nexts = {SA, SC, NA, NC, AN, CO};

    auto legal = [](const std::vector<stage_t>& path, stage_t next) {
        if(path.empty())
            return next == SA || next == AN;
        switch(path.back())
        {
        case SA:
            return next == SC;
        case AN:
        case CO:
            return next == CO;
        case SC:
            return next == NA || next == NC;
        case NA:
            return next == NC;
        case NC:
            return next == NA || next == NC;
        }
        return false;
    };

    for(const auto& path : paths)
        for(stage_t next : nexts)
        {
            // Not covered by the state machine: a numeric stage without any
            // symbolic stage before it is not a meaningful request.
            if(path.empty() && (next == NA || next == NC))
                continue;

            std::string desc = "after [";
            for(stage_t s : path)
                desc += std::string(stage_name(s)) + " ";
            desc += std::string("] next=") + stage_name(next);
            SCOPED_TRACE(desc);

            G g;
            ASSERT_NO_FATAL_FAILURE(g.init(handle, false, 7));
            // The output matrix has to exist for the stages that need it.
            if(path.empty())
                ASSERT_NO_FATAL_FAILURE(g.create_C(g.want.nnzb()));
            for(stage_t s : path)
                ASSERT_NO_FATAL_FAILURE(g.ok(s, s != SA && s != AN));

            const bool       pass_C   = true;
            const bool       need     = next == SC || next == NC || next == CO || next == NA;
            const bool       is_legal = legal(path, next);
            rocsparse_status st;
            if(need)
                g.clear_values();
            st = g.run(next, pass_C);
            if(is_legal)
            {
                ASSERT_EQ(st, rocsparse_status_success);
                if(next == NC || next == CO)
                    ASSERT_NO_FATAL_FAILURE(g.check_C());
                if(is_analysis_stage(next))
                    ASSERT_EQ(g.nnz_C, g.want.nnzb());
            }
            else
            {
                ASSERT_EQ(st, rocsparse_status_invalid_value);
            }
        }
}

TEST_F(SpgeamCov70, argument_errors)
{
    using G = Geam<float, float, int32_t, int32_t>;
    G g;
    ASSERT_NO_FATAL_FAILURE(g.init(handle, false, 3));
    ASSERT_NO_FATAL_FAILURE(g.create_C(g.want.nnzb()));

    size_t bs = 0;

    // ---- buffer size ----
    EXPECT_EQ(
        rocsparse_spgeam_buffer_size(nullptr, g.descr, g.sA.p, g.sB.p, g.sC.p, CO, &bs, nullptr),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_spgeam_buffer_size(handle, nullptr, g.sA.p, g.sB.p, g.sC.p, CO, &bs, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_spgeam_buffer_size(handle, g.descr, nullptr, g.sB.p, g.sC.p, CO, &bs, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_spgeam_buffer_size(handle, g.descr, g.sA.p, nullptr, g.sC.p, CO, &bs, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_spgeam_buffer_size(handle, g.descr, g.sA.p, g.sB.p, nullptr, CO, &bs, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spgeam_buffer_size(
                  handle, g.descr, g.sA.p, g.sB.p, g.sC.p, (stage_t)99, &bs, nullptr),
              rocsparse_status_invalid_value);

    // ---- execution ----
    DevBuf<char> buf(size_t(4096));
    auto         call
        = [&](rocsparse_handle            h,
              rocsparse_spgeam_descr      d,
              rocsparse_const_spmat_descr a,
              rocsparse_const_spmat_descr b,
              rocsparse_spmat_descr       c,
              stage_t                     stage,
              size_t                      size,
              void* temp) { return rocsparse_spgeam(h, d, a, b, c, stage, size, temp, nullptr); };
    EXPECT_EQ(call(nullptr, g.descr, g.sA.p, g.sB.p, g.sC.p, AN, 0, nullptr),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(call(handle, nullptr, g.sA.p, g.sB.p, g.sC.p, AN, 0, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle, g.descr, nullptr, g.sB.p, g.sC.p, AN, 0, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle, g.descr, g.sA.p, nullptr, g.sC.p, AN, 0, nullptr),
              rocsparse_status_invalid_pointer);
    for(stage_t s : {SC, NC, CO})
        EXPECT_EQ(call(handle, g.descr, g.sA.p, g.sB.p, nullptr, s, 0, nullptr),
                  rocsparse_status_invalid_pointer)
            << stage_name(s);
    EXPECT_EQ(call(handle, g.descr, g.sA.p, g.sB.p, g.sC.p, AN, 16, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call(handle, g.descr, g.sA.p, g.sB.p, g.sC.p, (stage_t)99, 16, buf.ptr),
              rocsparse_status_invalid_value);

    // ---- mismatching operands ----
    {
        // B stored as COO while A is CSR.
        std::vector<int64_t> rows;
        for(int64_t i = 0; i < g.B.m; ++i)
            for(int64_t p = g.B.ptr[i]; p < g.B.ptr[i + 1]; ++p)
                rows.push_back(i);
        DevBuf<int32_t> coo_row = up_idx<int32_t>(rows);
        SpmatH          coo;
        ASSERT_EQ(rocsparse_create_coo_descr(&coo.p,
                                             g.B.m,
                                             g.B.n,
                                             g.B.nnz(),
                                             coo_row.ptr,
                                             g.cB.ptr,
                                             g.vB.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(
            rocsparse_spgeam_buffer_size(handle, g.descr, g.sA.p, coo.p, g.sC.p, AN, &bs, nullptr),
            rocsparse_status_not_implemented);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, coo.p, g.sC.p, AN, 0, nullptr),
                  rocsparse_status_not_implemented);

        // A and B both COO: the format has no SpGEAM implementation.
        SpmatH coo_a;
        ASSERT_EQ(rocsparse_create_coo_descr(&coo_a.p,
                                             g.B.m,
                                             g.B.n,
                                             g.B.nnz(),
                                             coo_row.ptr,
                                             g.cB.ptr,
                                             g.vB.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spgeam_buffer_size(
                      handle, g.descr, coo_a.p, coo.p, nullptr, AN, &bs, nullptr),
                  rocsparse_status_not_implemented);

        // C stored as COO.
        EXPECT_EQ(
            rocsparse_spgeam_buffer_size(handle, g.descr, g.sA.p, g.sB.p, coo.p, CO, &bs, nullptr),
            rocsparse_status_not_implemented);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, g.sB.p, coo.p, CO, 0, nullptr),
                  rocsparse_status_not_implemented);
    }
    {
        // Value type of B (and of C) different from the compute type.
        Csr             B = g.B;
        DevBuf<int32_t> p;
        DevBuf<int32_t> c;
        DevBuf<double>  v;
        SpmatH          bd;
        ASSERT_NO_FATAL_FAILURE((make_csr_descr<double, int32_t, int32_t>(bd, B, p, c, v)));
        EXPECT_EQ(
            rocsparse_spgeam_buffer_size(handle, g.descr, g.sA.p, bd.p, g.sC.p, CO, &bs, nullptr),
            rocsparse_status_not_implemented);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, bd.p, g.sC.p, CO, 0, nullptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(
            rocsparse_spgeam_buffer_size(handle, g.descr, bd.p, g.sB.p, g.sC.p, CO, &bs, nullptr),
            rocsparse_status_not_implemented);
        EXPECT_EQ(call(handle, g.descr, bd.p, g.sB.p, g.sC.p, CO, 0, nullptr),
                  rocsparse_status_not_implemented);

        EXPECT_EQ(
            rocsparse_spgeam_buffer_size(handle, g.descr, g.sA.p, g.sB.p, bd.p, CO, &bs, nullptr),
            rocsparse_status_not_implemented);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, g.sB.p, bd.p, CO, 0, nullptr),
                  rocsparse_status_not_implemented);
    }
    {
        // Index types of B (and of C) different from the ones of A.
        DevBuf<int64_t> p64;
        DevBuf<int32_t> c32;
        DevBuf<float>   v;
        SpmatH          b_row64;
        ASSERT_NO_FATAL_FAILURE(
            (make_csr_descr<float, int64_t, int32_t>(b_row64, g.B, p64, c32, v)));
        EXPECT_EQ(rocsparse_spgeam_buffer_size(
                      handle, g.descr, g.sA.p, b_row64.p, g.sC.p, CO, &bs, nullptr),
                  rocsparse_status_type_mismatch);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, b_row64.p, g.sC.p, CO, 0, nullptr),
                  rocsparse_status_type_mismatch);
        EXPECT_EQ(rocsparse_spgeam_buffer_size(
                      handle, g.descr, g.sA.p, g.sB.p, b_row64.p, CO, &bs, nullptr),
                  rocsparse_status_type_mismatch);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, g.sB.p, b_row64.p, CO, 0, nullptr),
                  rocsparse_status_type_mismatch);

        DevBuf<int32_t> p32;
        DevBuf<int64_t> c64;
        SpmatH          b_col64;
        ASSERT_NO_FATAL_FAILURE(
            (make_csr_descr<float, int32_t, int64_t>(b_col64, g.B, p32, c64, v)));
        EXPECT_EQ(rocsparse_spgeam_buffer_size(
                      handle, g.descr, g.sA.p, b_col64.p, g.sC.p, CO, &bs, nullptr),
                  rocsparse_status_type_mismatch);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, b_col64.p, g.sC.p, CO, 0, nullptr),
                  rocsparse_status_type_mismatch);
        EXPECT_EQ(rocsparse_spgeam_buffer_size(
                      handle, g.descr, g.sA.p, g.sB.p, b_col64.p, CO, &bs, nullptr),
                  rocsparse_status_type_mismatch);
        EXPECT_EQ(call(handle, g.descr, g.sA.p, g.sB.p, b_col64.p, CO, 0, nullptr),
                  rocsparse_status_type_mismatch);
    }

    // ---- invalid descriptor inputs ----
    {
        auto expect_with = [&](rocsparse_spgeam_input input,
                               const void*            data,
                               size_t                 size,
                               rocsparse_status       want,
                               const char*            what) {
            rocsparse_spgeam_descr d = nullptr;
            ASSERT_EQ(rocsparse_create_spgeam_descr(&d), rocsparse_status_success);
            g.set_inputs(d);
            ASSERT_EQ(rocsparse_spgeam_set_input(handle, d, input, data, size, nullptr),
                      rocsparse_status_success)
                << what;
            const rocsparse_status st
                = rocsparse_spgeam(handle, d, g.sA.p, g.sB.p, g.sC.p, AN, 0, nullptr, nullptr);
            EXPECT_EQ(st, want) << what;
            EXPECT_EQ(rocsparse_destroy_spgeam_descr(d), rocsparse_status_success);
        };
        const rocsparse_spgeam_alg bad_alg = (rocsparse_spgeam_alg)99;
        expect_with(rocsparse_spgeam_input_alg,
                    &bad_alg,
                    sizeof(bad_alg),
                    rocsparse_status_invalid_value,
                    "alg");
        const rocsparse_operation bad_op = (rocsparse_operation)99;
        expect_with(rocsparse_spgeam_input_operation_A,
                    &bad_op,
                    sizeof(bad_op),
                    rocsparse_status_invalid_value,
                    "operation A");
        expect_with(rocsparse_spgeam_input_operation_B,
                    &bad_op,
                    sizeof(bad_op),
                    rocsparse_status_invalid_value,
                    "operation B");
        const rocsparse_datatype bad_type = (rocsparse_datatype)99;
        expect_with(rocsparse_spgeam_input_scalar_datatype,
                    &bad_type,
                    sizeof(bad_type),
                    rocsparse_status_invalid_value,
                    "scalar datatype");
        // The matrices do not have the requested compute type.
        const rocsparse_datatype f64 = rocsparse_datatype_f64_r;
        expect_with(rocsparse_spgeam_input_compute_datatype,
                    &f64,
                    sizeof(f64),
                    rocsparse_status_not_implemented,
                    "compute datatype");
    }

    // ---- set_input / get_output sizes ----
    {
        rocsparse_spgeam_descr d = nullptr;
        ASSERT_EQ(rocsparse_create_spgeam_descr(&d), rocsparse_status_success);
        float x = 1.0f;
        EXPECT_EQ(rocsparse_spgeam_set_input(
                      handle, d, rocsparse_spgeam_input_scalar_alpha, &x, 1, nullptr),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_spgeam_set_input(
                      handle, d, rocsparse_spgeam_input_scalar_beta, &x, 1, nullptr),
                  rocsparse_status_invalid_size);
        int32_t nnz32 = 0;
        EXPECT_EQ(rocsparse_spgeam_get_output(
                      handle, d, rocsparse_spgeam_output_nnz, &nnz32, sizeof(nnz32), nullptr),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_destroy_spgeam_descr(d), rocsparse_status_success);
    }

    // The session is still usable after the failed calls.
    ASSERT_NO_FATAL_FAILURE(g.ok(AN, true));
    ASSERT_NO_FATAL_FAILURE(g.ok(CO, true));
    ASSERT_NO_FATAL_FAILURE(g.check_C());
}
