// main.cu - Demo for linear_gemm_tl_1_6144_1024 kernel
// Directly includes the generated kernel and calls linear_kernel with TMA descriptors
//
// Kernel info from comment:
// - Strategy: linear_gemm_tl_1_6144_1024
// - smem: 30720 bytes
// - layout: (96, 1, 1), (64, 16, 64)
// - block_dim: (128, 1, 1)
// - M=1, N=6144, K=1024

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

// using bfloat16_t = __nv_bfloat16;

// Create TMA descriptor for 2D tensor (row-major)
// This follows the pattern from megakernel/persistent_kernel/tma.cuh
static CUresult CreateTMA2DDesc(
    CUtensorMap *desc,
    void *gmem_ptr,
    uint32_t global_dim_y,   // first dimension (rows / N)
    uint32_t global_dim_x,   // second dimension (cols / M)
    uint32_t box_dim_y,       // TMA box height
    uint32_t box_dim_x,       // TMA box width
    uint32_t stride_x         // stride in first dimension (row stride in bytes / sizeof(bfloat16))
) {
    CUtensorMap tensor_map;

    uint64_t global_dim[] = {
        static_cast<uint64_t>(global_dim_x),
        static_cast<uint64_t>(global_dim_y),
        1ULL, 1ULL, 1ULL
    };
    uint64_t global_stride[] = {
        1ULL,
        static_cast<uint64_t>(stride_x),
        0ULL, 0ULL, 0ULL
    };
    uint32_t box_dim[] = {box_dim_x, box_dim_y, 1, 1, 1};
    uint32_t element_strides[] = {1, 1, 1, 1, 1};

    return cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        5,  // tensor rank
        gmem_ptr,
        global_dim,
        global_stride + 1,  // skip first stride (always 1 in bytes)
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,  // B=3 -> 128B swizzle
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

int main(int argc, char **argv) {
    // Linear layer dimensions: M=1, N=6144, K=1024
    // A: [1, K] = [1, 1024], B: [N, K] = [6144, 1024], C: [M, N] = [1, 6144]
    constexpr int M = 1;
    constexpr int N = 6144;
    constexpr int K = 1024;

    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_K = 64;
    constexpr int THREAD_NUM = 128;
    constexpr int NUM_STAGES = 3;

    // Grid dims from kernel comment: (96, 1, 1)
    constexpr int GRID_X = 96;
    constexpr int GRID_Y = 1;
    constexpr int GRID_Z = 1;

    printf("=== Linear GEMM TMA Demo ===\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block config: block_dim=(%d,1,1), grid=(%d,%d,%d)\n",
           THREAD_NUM, GRID_X, GRID_Y, GRID_Z);
    printf("TILE: (%d, %d, %d)\n\n", BLOCK_N, BLOCK_M, BLOCK_K);

    // Initialize CUDA
    CHECK_RT(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("ERROR: This kernel requires Hopper (sm_90a)\n");
        return 1;
    }

    // Initialize Driver API
    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext context;
    CHECK_CU(cuCtxSetCurrent(context));

    // Allocate device memory
    // A: [1, K] = [1, 1024]
    // B: [N, K] = [6144, 1024]
    // C: [M, N] = [1, 6144]
    bfloat16_t *d_A = nullptr;
    bfloat16_t *d_B = nullptr;
    bfloat16_t *d_C = nullptr;

    CHECK_RT(cudaMalloc(&d_A, M * K * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_B, N * K * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_C, M * N * sizeof(bfloat16_t)));

    // Initialize input data on host and copy to device
    std::vector<bfloat16_t> h_A(M * K);
    std::vector<bfloat16_t> h_B(N * K);
    std::vector<bfloat16_t> h_C(M * N);

    // Initialize A with random values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = bfloat16_t((float(i % K) / K) * 0.1f);
    }
    // Initialize B with random values
    for (int i = 0; i < N * K; i++) {
        h_B[i] = bfloat16_t((float(i % K) / K) * 0.1f);
    }
    // Initialize C to zero
    for (int i = 0; i < M * N; i++) {
        h_C[i] = bfloat16_t(0.0f);
    }

    CHECK_RT(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(bfloat16_t), cudaMemcpyHostToDevice));

    // Create TMA descriptors
    // A_desc: input tensor [1, 1024], loaded as [1, 64] boxes
    // gmem_shape = (K, M) = (1024, 1) in TMA order (x, y)
    // gmem_stride = (1, K) in elements = (1, 1024) in bytes
    // smem_box = (64, 1) -> [cp_async_size, batch_size]
    CUtensorMap A_desc, B_desc, C_desc;

    printf("Creating TMA descriptors...\n");

    // A: [1, 1024], accessed as [1, 64] boxes, repeated 16 times
    // global_dim = (1024, 1), stride = (1, 1024), box_dim = (64, 1)
    CUresult res = CreateTMA2DDesc(
        &A_desc,
        d_A,
        M,               // global_dim_y = 1
        K,               // global_dim_x = 1024
        1,               // box_dim_y = 1
        64,              // box_dim_x = 64 (cp_async_size)
        K                // stride_x = K = 1024 (row stride)
    );
    if (res != CUDA_SUCCESS) {
        printf("CreateTMA2DDesc A failed: %d\n", res);
        return 1;
    }

    // B: [6144, 1024], accessed as [256, 64] boxes
    // global_dim = (1024, 6144), stride = (1, 1024), box_dim = (64, 256)
    // output_atom_size = 256 (since N=6144 >= 256)
    res = CreateTMA2DDesc(
        &B_desc,
        d_B,
        K,               // global_dim_y = 1024
        N,               // global_dim_x = 6144
        256,             // box_dim_y = 256 (output_atom_size)
        64,              // box_dim_x = 64 (cp_async_size)
        N                // stride_x = N = 6144 (row stride)
    );
    if (res != CUDA_SUCCESS) {
        printf("CreateTMA2DDesc B failed: %d\n", res);
        return 1;
    }

    // C: [1, 6144], accessed as [1, 64] boxes
    // global_dim = (6144, 1), stride = (1, 6144), box_dim = (64, 1)
    res = CreateTMA2DDesc(
        &C_desc,
        d_C,
        M,               // global_dim_y = 1
        N,               // global_dim_x = 6144
        1,               // box_dim_y = 1
        64,              // box_dim_x = 64
        N                // stride_x = N = 6144 (row stride)
    );
    if (res != CUDA_SUCCESS) {
        printf("CreateTMA2DDesc C failed: %d\n", res);
        return 1;
    }

    printf("TMA descriptors created successfully\n");

    // Calculate dynamic shared memory size (30720 bytes from kernel comment)
    // This includes the ping-pong buffers for A, B, and C
    constexpr size_t DYNAMIC_SMEM_SIZE = 30720;

    // Create CUDA stream
    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    // Launch kernel
    dim3 grid_dim(GRID_X, GRID_Y, GRID_Z);
    dim3 block_dim(THREAD_NUM, 1, 1);

    printf("\nLaunching linear_kernel...\n");
    printf("Grid: (%d,%d,%d), Block: (%d,1,1), Smem: %zu bytes\n",
           GRID_X, GRID_Y, GRID_Z, THREAD_NUM, DYNAMIC_SMEM_SIZE);

    linear_kernel<<<grid_dim, block_dim, DYNAMIC_SMEM_SIZE, stream>>>(
        A_desc, B_desc, C_desc);

    CHECK_RT(cudaStreamSynchronize(stream));
    printf("Kernel completed!\n");

    // Copy result back to host
    CHECK_RT(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(bfloat16_t), cudaMemcpyDeviceToHost));

    // Verify result using PyTorch reference computation
    printf("\nVerification:\n");

    // Compute reference with PyTorch-style gemm: C = A @ B^T
    // A: [1, 1024], B: [6144, 1024], C: [1, 6144] = A @ B^T
    std::vector<float> h_C_ref(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (float)h_A[i * K + k] * (float)h_B[j * K + k];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    // Compare results
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float expected = h_C_ref[i];
        float actual = (float)h_C[i];
        float diff = fabsf(expected - actual);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) {
            errors++;
            if (errors <= 5) {
                printf("  ERROR at index %d: expected=%f, actual=%f, diff=%f\n",
                       i, expected, actual, diff);
            }
        }
    }

    if (errors == 0) {
        printf("  PASS! All %d elements match. Max diff: %f\n", M * N, max_diff);
        printf("\nFirst 10 elements:\n");
        for (int i = 0; i < 10 && i < M * N; i++) {
            printf("  C[%d] = %f (expected %f)\n", i, (float)h_C[i], h_C_ref[i]);
        }
    } else {
        printf("  FAIL! %d errors out of %d elements. Max diff: %f\n",
               errors, M * N, max_diff);
    }

    // Cleanup
    CHECK_RT(cudaFree(d_A));
    CHECK_RT(cudaFree(d_B));
    CHECK_RT(cudaFree(d_C));
    CHECK_RT(cudaStreamDestroy(stream));

    printf("\nDone!\n");
    return errors > 0 ? 1 : 0;
}