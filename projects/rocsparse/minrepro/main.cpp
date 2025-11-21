#include <hip/hip_runtime.h>
#include <iostream>
#include <rocprim/rocprim.hpp>
#include <vector>

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

// Kernel to initialize identity permutation
template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__ void identity_kernel(int n, int* __restrict__ p)
{
    int gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    if(gid >= n)
    {
        return;
    }

    p[gid] = gid;
}

// Test configuration matching the failing rocsparse test
constexpr int NNZ = 9529407; // From bmwcra_1 matrix
constexpr int M   = 148770; // Number of rows/columns

// Structure to manage device arrays
struct DeviceArrays
{
    int* d_keys_input  = nullptr;
    int* d_keys_output = nullptr;
    int* d_vals_input  = nullptr;
    int* d_vals_output = nullptr;

    void allocate_async(hipStream_t stream)
    {
        HIP_CHECK(hipMallocAsync(&d_keys_input, sizeof(int) * NNZ, stream));
        HIP_CHECK(hipMallocAsync(&d_keys_output, sizeof(int) * NNZ, stream));
        HIP_CHECK(hipMallocAsync(&d_vals_input, sizeof(int) * NNZ, stream));
        HIP_CHECK(hipMallocAsync(&d_vals_output, sizeof(int) * NNZ, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    void allocate()
    {
        HIP_CHECK(hipMalloc(&d_keys_input, sizeof(int) * NNZ));
        HIP_CHECK(hipMalloc(&d_keys_output, sizeof(int) * NNZ));
        HIP_CHECK(hipMalloc(&d_vals_input, sizeof(int) * NNZ));
        HIP_CHECK(hipMalloc(&d_vals_output, sizeof(int) * NNZ));
    }

    void reset_async(hipStream_t stream)
    {
        HIP_CHECK(hipFreeAsync(d_keys_input, stream));
        d_keys_input = nullptr;
        HIP_CHECK(hipFreeAsync(d_keys_output, stream));
        d_keys_output = nullptr;
        HIP_CHECK(hipFreeAsync(d_vals_input, stream));
        d_vals_input = nullptr;
        HIP_CHECK(hipFreeAsync(d_vals_output, stream));
        d_vals_output = nullptr;
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    ~DeviceArrays()
    {
        // Synchronous cleanup in destructor
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_vals_input));
        HIP_CHECK(hipFree(d_vals_output));
    }
};

uint32_t getEndbit(int m)
{
    uint32_t endbit = 32;
    for(uint32_t i = 0; i < 32; i++)
    {
        if((1u << (31 - i)) >= static_cast<uint32_t>(M))
        {
            endbit = 32 - i;
            break;
        }
    }
    return endbit;
}

size_t get_buffer_size(DeviceArrays& arrays, hipStream_t stream)
{
    // Setup double buffers
    rocprim::double_buffer<int> keys(arrays.d_keys_input, arrays.d_keys_output);
    rocprim::double_buffer<int> vals(arrays.d_vals_input, arrays.d_vals_output);

    // Calculate bit range
    uint32_t startbit = 0;
    uint32_t endbit   = getEndbit(M);

    size_t temp_storage_bytes = 0;
    HIP_CHECK(rocprim::radix_sort_pairs(
        nullptr, temp_storage_bytes, keys, vals, NNZ, startbit, endbit, stream));

    return temp_storage_bytes;
}

template <typename T>
bool validate_array_identity(const T* data, size_t size)
{
    void* buffer = malloc(size * sizeof(T));

    HIP_CHECK(hipMemcpy(buffer, data, size * sizeof(T), hipMemcpyDefault));
    const T* host_data = static_cast<const T*>(buffer);
    for(size_t i = 0; i < size; ++i)
    {
        if(host_data[i] != i)
        {
            std::cout << "--------- Validation failed at index " << i << ": " << host_data[i]
                      << " != " << i << "\n";
            free(buffer);
            return false;
        }
    }
    free(buffer);
    return true;
}

bool trm_analysis(DeviceArrays& arrays,
                  hipStream_t   stream,
                  void*         rocprim_buffer,
                  size_t        rocprim_size,
                  bool          use_async_allocation = true)
{
    // Allocate arrays (simulating rocsparse_hipMallocAsync calls)
    arrays.reset_async(stream);
    if(use_async_allocation)
    {
        arrays.allocate_async(stream);
    }
    else
    {
        arrays.allocate();
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    // Create identity permutation (matching rocsparse::create_identity_permutation_template)
    constexpr unsigned int IDENTITY_DIM = 256;
    dim3                   identity_blocks((NNZ - 1) / IDENTITY_DIM + 1);
    dim3                   identity_threads(IDENTITY_DIM);

    hipLaunchKernelGGL(identity_kernel<IDENTITY_DIM>,
                       identity_blocks,
                       identity_threads,
                       0,
                       stream,
                       NNZ,
                       arrays.d_vals_input);

    if(!validate_array_identity(arrays.d_vals_input, NNZ))
    {
        std::cout << "--------- Validation failed for identity permutation\n";
        return false;
    }
    std::cout << "--------- Validation succeeded for identity permutation\n";

    HIP_CHECK(hipStreamSynchronize(stream));

    // Setup double buffers (matching keys(tmp_work1, transposed_col_ind) and vals(transposed_perm, tmp_work2))
    rocprim::double_buffer<int> keys(arrays.d_keys_input, arrays.d_keys_output);
    rocprim::double_buffer<int> vals(arrays.d_vals_input, arrays.d_vals_output);

    // Calculate bit range (matching rocsparse::clz(m))
    uint32_t startbit = 0;
    uint32_t endbit   = getEndbit(M);

    // Perform radix sort (matching rocsparse::primitives::radix_sort_pairs)
    HIP_CHECK(rocprim::radix_sort_pairs(
        rocprim_buffer, rocprim_size, keys, vals, NNZ, startbit, endbit, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipMemcpyAsync(
        arrays.d_vals_input, vals.current(), sizeof(int) * NNZ, hipMemcpyDeviceToDevice, stream));

    return true;
}

void sync_print(const char* file, int line)
{
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "--- " << file << ":" << line << "\n";
}

#define SYNC_PRINT() sync_print(__FILE__, __LINE__)

bool test_trm_analysis(bool use_async_allocation)
{
    DeviceArrays arrays;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    arrays.allocate_async(stream);

    size_t rocprim_size = get_buffer_size(arrays, stream);

    void* rocprim_buffer = nullptr;
    HIP_CHECK(hipMallocAsync(&rocprim_buffer, rocprim_size, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    arrays.reset_async(stream);

    SYNC_PRINT();
    if(!trm_analysis(arrays, stream, rocprim_buffer, rocprim_size, use_async_allocation))
    {
        std::cerr << "trm_analysis failed (iteration 1)" << std::endl;
        return false;
    }
    SYNC_PRINT();
    if(!trm_analysis(arrays, stream, rocprim_buffer, rocprim_size, use_async_allocation))
    {
        std::cerr << "trm_analysis failed (iteration 2)" << std::endl;
        return false;
    }
    SYNC_PRINT();

    arrays.reset_async(stream);

    SYNC_PRINT();
    if(!trm_analysis(arrays, stream, rocprim_buffer, rocprim_size, use_async_allocation))
    {
        std::cerr << "trm_analysis failed (iteration 3)" << std::endl;
        return false;
    }
    SYNC_PRINT();
    if(!trm_analysis(arrays, stream, rocprim_buffer, rocprim_size, use_async_allocation))
    {
        std::cerr << "trm_analysis failed (iteration 4)" << std::endl;
        return false;
    }
    SYNC_PRINT();

    HIP_CHECK(hipFreeAsync(rocprim_buffer, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));

    return true;
}

int main(int argc, char** argv)
{
    // If argument is "0", use async allocation; otherwise use sync
    bool use_async = (argc > 1 && strcmp(argv[1], "0") == 0);

    std::cout << "Running test with " << (use_async ? "ASYNC" : "SYNC") << " allocation\n";

    bool passed = test_trm_analysis(use_async);

    std::cout << "\nTest " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed ? 0 : 1;
}
