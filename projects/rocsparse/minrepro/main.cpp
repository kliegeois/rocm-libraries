#include <hip/hip_runtime.h>
#include <iostream>
#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)                                                             \
    {                                                                                    \
        hipError_t error = condition;                                                    \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            std::cerr << "HIP error: " << error << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                      \
            exit(error);                                                                 \
        }                                                                                \
    }

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__ void identity_kernel(int n, int* __restrict__ p)
{
    int gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    if(gid < n)
        p[gid] = gid;
}

constexpr int NNZ = 9529407;
constexpr int M   = 148770;

bool validate_identity(int* data, size_t size)
{
    int* host_data = (int*)malloc(size * sizeof(int));
    HIP_CHECK(hipMemcpy(host_data, data, size * sizeof(int), hipMemcpyDefault));
    for(size_t i = 0; i < size; ++i)
    {
        if(host_data[i] != (int)i)
        {
            std::cout << "FAIL at index " << i << ": " << host_data[i] << " != " << i << "\n";
            free(host_data);
            return false;
        }
    }
    free(host_data);
    return true;
}

bool test_trm_analysis(bool use_async)
{
    int *d_keys_in, *d_keys_out, *d_vals_in, *d_vals_out;

    rocprim::double_buffer<int> keys(d_keys_in, d_keys_out);
    rocprim::double_buffer<int> vals(d_vals_in, d_vals_out);

    size_t temp_size = 0;
    HIP_CHECK(rocprim::radix_sort_pairs(nullptr, temp_size, keys, vals, NNZ, 0, 18, 0));

    void* temp_buffer = nullptr;
    HIP_CHECK(hipMallocAsync(&temp_buffer, temp_size, 0));

    for(int iter = 0; iter < 2; iter++)
    {
        if(use_async)
        {
            HIP_CHECK(hipMallocAsync(&d_keys_in, sizeof(int) * NNZ, 0));
            HIP_CHECK(hipMallocAsync(&d_keys_out, sizeof(int) * NNZ, 0));
            HIP_CHECK(hipMallocAsync(&d_vals_in, sizeof(int) * NNZ, 0));
            HIP_CHECK(hipMallocAsync(&d_vals_out, sizeof(int) * NNZ, 0));
        }
        else
        {
            HIP_CHECK(hipMalloc(&d_keys_in, sizeof(int) * NNZ));
            HIP_CHECK(hipMalloc(&d_keys_out, sizeof(int) * NNZ));
            HIP_CHECK(hipMalloc(&d_vals_in, sizeof(int) * NNZ));
            HIP_CHECK(hipMalloc(&d_vals_out, sizeof(int) * NNZ));
        }

        hipLaunchKernelGGL(
            identity_kernel<256>, dim3((NNZ - 1) / 256 + 1), dim3(256), 0, 0, NNZ, d_vals_in);

        if(!validate_identity(d_vals_in, NNZ))
            return false;

        rocprim::double_buffer<int> keys2(d_keys_in, d_keys_out);
        rocprim::double_buffer<int> vals2(d_vals_in, d_vals_out);

        HIP_CHECK(rocprim::radix_sort_pairs(temp_buffer, temp_size, keys2, vals2, NNZ, 0, 18, 0));

        HIP_CHECK(hipFreeAsync(d_keys_in, 0));
        HIP_CHECK(hipFreeAsync(d_keys_out, 0));
        HIP_CHECK(hipFreeAsync(d_vals_in, 0));
        HIP_CHECK(hipFreeAsync(d_vals_out, 0));
        HIP_CHECK(hipDeviceSynchronize());
    }

    HIP_CHECK(hipFreeAsync(temp_buffer, 0));
    HIP_CHECK(hipDeviceSynchronize());
    return true;
}

int main(int argc, char** argv)
{
    bool use_async = (argc == 1 || strcmp(argv[1], "0") == 0);
    std::cout << (use_async ? "ASYNC" : "SYNC") << " mode\n";
    bool passed = test_trm_analysis(use_async);
    std::cout << (passed ? "PASSED" : "FAILED") << "\n";
    return passed ? 0 : 1;
}
