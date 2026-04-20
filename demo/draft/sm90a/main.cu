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

// Create TMA descriptor for 2D tensor (row-major)
// Follows fill_tma_desc pattern from megakernel/persistent_kernel/tma.cuh
// For 2D tensor [rows, cols], TMA expects [cols, rows] in global_dim
static CUresult CreateTMA2DDesc(
    CUtensorMap *desc,
    void *gmem_ptr,
    uint32_t tensor_rows,     // first dim of row-major tensor (N for weight [N,K])
    uint32_t tensor_cols,     // second dim of row-major tensor (K for weight [N,K])
    uint32_t box_rows,        // TMA box height
    uint32_t box_cols,        // TMA box width
    uint32_t row_stride       // stride in elements between rows = tensor_cols
) {
    CUtensorMap tensor_map;

    // TMA expects global_dim in [dim0, dim1, ...] order where dim0 is fastest varying
    // For row-major tensor [tensor_rows, tensor_cols]:
    // - global_dim[0] = tensor_cols (dim1 of tensor, K for weight)
    // - global_dim[1] = tensor_rows (dim0 of tensor, N for weight)
    uint64_t global_dim[] = {
        static_cast<uint64_t>(tensor_cols),  // dim0 = tensor_cols (K for A, N for B/C)
        static_cast<uint64_t>(tensor_rows),  // dim1 = tensor_rows (M for A, K for B, M for C)
        1ULL, 1ULL, 1ULL
    };
    // TMA expects byte strides, not element strides!
    // global_stride[d] = byte offset to reach dim d+1 from element [0,0,...]
    uint64_t global_stride[] = {
        sizeof(bfloat16_t),                                          // dim0 stride = sizeof(bfloat16_t) = 2 bytes
        static_cast<uint64_t>(row_stride) * sizeof(bfloat16_t),      // dim1 stride = row_stride * sizeof(bfloat16)
        0ULL, 0ULL, 0ULL
    };
    uint32_t box_dim[] = {box_cols, box_rows, 1, 1, 1};
    uint32_t element_strides[] = {1, 1, 1, 1, 1};

    return cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        5,
        gmem_ptr,
        global_dim,
        global_stride + 1,  // skip first stride
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
    // A: [M, K] = [1, 1024], B: [N, K] = [6144, 1024], C: [M, N] = [1, 6144]
    constexpr int M = 1;
    constexpr int N = 6144;
    constexpr int K = 1024;

    constexpr int THREAD_NUM = 128;

    // Grid dims from kernel comment: (96, 1, 1)
    constexpr int GRID_X = 96;
    constexpr int GRID_Y = 1;
    constexpr int GRID_Z = 1;

    // Dynamic smem size from kernel comment: 30720 bytes
    constexpr size_t DYNAMIC_SMEM_SIZE = 30720;

    printf("=== Linear GEMM TMA Demo ===\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block config: block_dim=(%d,1,1), grid=(%d,%d,%d)\n",
           THREAD_NUM, GRID_X, GRID_Y, GRID_Z);
    printf("Dynamic smem: %zu bytes\n\n", DYNAMIC_SMEM_SIZE);

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
    bfloat16_t *d_A = nullptr;
    bfloat16_t *d_B = nullptr;
    bfloat16_t *d_C = nullptr;

    CHECK_RT(cudaMalloc(&d_A, M * K * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_B, N * K * sizeof(bfloat16_t)));
    CHECK_RT(cudaMalloc(&d_C, M * N * sizeof(bfloat16_t)));

    // Initialize input data
    std::vector<bfloat16_t> h_A(M * K);
    std::vector<bfloat16_t> h_B(N * K);
    std::vector<bfloat16_t> h_C(M * N);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = bfloat16_t((float(i % K) / K) * 0.1f);
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = bfloat16_t((float(i % K) / K) * 0.1f);
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = bfloat16_t(0.0f);
    }

    CHECK_RT(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(bfloat16_t), cudaMemcpyHostToDevice));

    // Create TMA descriptors
    // Kernel access pattern (from linear_gemm_tl_1_6144_1024_kernel.cuh):
    // - A: tma_load(A_desc, ..., (k * 64), 0) -> 1D access, coord is K dimension
    // - B: tma_load(B_desc, ..., (k * 64), (blockIdx.x * 64)) -> 2D access (k, blockIdx.x)
    // - C: tma_store(C_desc, ..., (blockIdx.x * 64), 0) -> 1D access
    //
    // This means:
    // - A tensor is [M, K] = [1, 1024], accessed by K coordinate -> TMA needs [K, M]
    // - B tensor is [N, K] = [6144, 1024], accessed by (K, N) -> TMA needs [K, N]
    // - C tensor is [M, N] = [1, 6144], accessed by N coordinate -> TMA needs [N, M]
    CUtensorMap A_desc, B_desc, C_desc;

    printf("Creating TMA descriptors...\n");

    // A: [1, 1024] -> TMA needs [K=1024, M=1]
    // box: [64, 1] (cp_async_size=64, batch_size=1)
    CUresult res = CreateTMA2DDesc(
        &A_desc,
        d_A,
        M,                // tensor_rows = M = 1
        K,                // tensor_cols = K = 1024
        1,                // box_rows = 1
        64,               // box_cols = 64 (cp_async_size)
        K                 // row_stride = K = 1024
    );
    if (res != CUDA_SUCCESS) {
        printf("CreateTMA2DDesc A failed: %d\n", res);
        return 1;
    }

    // B: [6144, 1024] -> TMA needs [K=1024, N=6144]
    // box: [64, 256] (cp_async_size=64, output_atom_size=256 since N>=256)
    res = CreateTMA2DDesc(
        &B_desc,
        d_B,
        K,                // tensor_rows = K = 1024
        N,                // tensor_cols = N = 6144
        256,              // box_rows = 256 (output_atom_size)
        64,               // box_cols = 64 (cp_async_size)
        N                 // row_stride = N = 6144
    );
    if (res != CUDA_SUCCESS) {
        printf("CreateTMA2DDesc B failed: %d\n", res);
        return 1;
    }

    // C: [1, 6144] -> TMA needs [N=6144, M=1]
    // box: [64, 1] (cp_async_size=64, batch_size=1)
    res = CreateTMA2DDesc(
        &C_desc,
        d_C,
        M,                // tensor_rows = M = 1
        N,                // tensor_cols = N = 6144
        1,                // box_rows = 1
        64,               // box_cols = 64 (cp_async_size)
        N                 // row_stride = N = 6144
    );
    if (res != CUDA_SUCCESS) {
        printf("CreateTMA2DDesc C failed: %d\n", res);
        return 1;
    }

    printf("TMA descriptors created successfully\n");

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

    // Copy result back
    CHECK_RT(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(bfloat16_t), cudaMemcpyDeviceToHost));

    // Verify result
    printf("\nVerification:\n");

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