// main.cu - Demo for linear_gemm_tl_1_6144_1024 kernel (simplified A-only version)
// Directly includes the generated kernel and calls linear_kernel with TMA descriptors

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "linear_gemm_tl_1_6144_1024_kernel.cuh"

namespace {

#define CHECK_CU(result)                                                       \
  do {                                                                         \
    CUresult _result = (result);                                               \
    if (_result != CUDA_SUCCESS) {                                              \
      const char *errstr;                                                       \
      cuGetErrorString(_result, &errstr);                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errstr);\
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_RT(result)                                                        \
  do {                                                                         \
    cudaError_t _result = (result);                                             \
    if (_result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorName(_result));                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

}  // anonymous namespace

// Create TMA descriptor for 2D tensor (row-major)
static CUresult CreateTMA2DDesc(
    CUtensorMap *desc,
    void *gmem_ptr,
    uint32_t tensor_rows,
    uint32_t tensor_cols,
    uint32_t box_rows,
    uint32_t box_cols,
    uint32_t row_stride
) {
    CUtensorMap tensor_map;

    uint64_t global_dim[] = {
        static_cast<uint64_t>(tensor_cols),
        static_cast<uint64_t>(tensor_rows),
        1ULL, 1ULL, 1ULL
    };
    uint64_t global_stride[] = {
        sizeof(bfloat16_t),
        static_cast<uint64_t>(row_stride) * sizeof(bfloat16_t),
        0ULL, 0ULL, 0ULL
    };
    uint32_t box_dim[] = {box_cols, box_rows, 1, 1, 1};
    uint32_t element_strides[] = {1, 1, 1, 1, 1};

    printf("CreateTMA2DDesc: tensor=[%u,%u] box=[%u,%u] stride=%u\n",
           tensor_rows, tensor_cols, box_rows, box_cols, row_stride);
    printf("  global_dim: {%lu, %lu}\n", global_dim[0], global_dim[1]);
    printf("  global_stride: {%lu, %lu}\n", global_stride[0], global_stride[1]);
    printf("  box_dim: {%u, %u}\n", box_dim[0], box_dim[1]);
    printf("  gmem_ptr: %p\n", gmem_ptr);

    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        5,
        gmem_ptr,
        global_dim,
        global_stride + 1,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    printf("  cuTensorMapEncodeTiled result: %d\n", result);

    // Copy to output
    *desc = tensor_map;

    return result;
}

int main(int argc, char **argv) {
    constexpr int M = 1;
    constexpr int K = 1024;
    constexpr int THREAD_NUM = 128;
    constexpr int GRID_X = 1;
    constexpr size_t DYNAMIC_SMEM_SIZE = 30720;

    printf("=== Linear GEMM TMA Demo (A-only) ===\n");
    printf("Tensor A: [%d, %d], Grid: (%d,1,1), Block: (%d,1,1), Smem: %zu\n\n",
           M, K, GRID_X, THREAD_NUM, DYNAMIC_SMEM_SIZE);

    // Initialize runtime API
    CHECK_RT(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("ERROR: This kernel requires Hopper (sm_90a)\n");
        return 1;
    }

    // Allocate device memory for tensor A
    bfloat16_t *d_A = nullptr;
    CHECK_RT(cudaMalloc(&d_A, M * K * sizeof(bfloat16_t)));

    // Initialize and copy input data
    std::vector<bfloat16_t> h_A(M * K);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = bfloat16_t((float(i % K) / K) * 0.1f);
    }
    CHECK_RT(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(bfloat16_t), cudaMemcpyHostToDevice));

    // Create A_desc on host (stack)
    CUtensorMap A_desc;
    uint64_t global_dim[] = {
        static_cast<uint64_t>(K),    // dim0 = K = 1024
        static_cast<uint64_t>(M),     // dim1 = M = 1
        1ULL, 1ULL, 1ULL
    };
    uint64_t global_stride[] = {
        sizeof(bfloat16_t),           // dim0 stride = 2 bytes
        static_cast<uint64_t>(K) * sizeof(bfloat16_t),  // dim1 stride = 2048 bytes
        0ULL, 0ULL, 0ULL
    };
    uint32_t box_dim[] = {64, 1, 1, 1, 1};  // [64, 1]
    uint32_t element_strides[] = {1, 1, 1, 1, 1};

    printf("Creating A_desc...\n");
    CUresult res = cuTensorMapEncodeTiled(
        &A_desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        5,
        d_A,
        global_dim,
        global_stride + 1,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    printf("  cuTensorMapEncodeTiled result: %d\n", res);
    if (res != CUDA_SUCCESS) {
        printf("ERROR: cuTensorMapEncodeTiled failed\n");
        return 1;
    }

    // Create stream and launch
    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    dim3 grid_dim(GRID_X, 1, 1);
    dim3 block_dim(THREAD_NUM, 1, 1);

    printf("\nLaunching linear_kernel with A_desc...\n");
    linear_kernel<<<grid_dim, block_dim, DYNAMIC_SMEM_SIZE, stream>>>(A_desc);

    CHECK_RT(cudaStreamSynchronize(stream));
    printf("Kernel completed!\n");

    // Cleanup
    CHECK_RT(cudaFree(d_A));
    CHECK_RT(cudaStreamDestroy(stream));

    printf("\nDone!\n");
    return 0;
}