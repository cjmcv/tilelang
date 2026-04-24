// main.cu - Demo for linear_gemm_tl_1_6144_1024 kernel
// Includes the generated .cuh and calls both kernel + create function

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "linear_gemm_tl_1_6144_1024.cuh"

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


int main(int argc, char **argv) {
    // M=1, N=6144, K=1024, block_dim=128, grid_dim=96
    constexpr int M = 1;
    constexpr int N = 6144;
    constexpr int K = 1024;
    constexpr int THREAD_NUM = 256;
    constexpr int GRID_X = 96;
    constexpr size_t DYNAMIC_SMEM_SIZE = 30720;

    printf("=== Linear GEMM TMA Demo ===\n");
    printf("Tensor A: [%d, %d], B: [%d, %d], C: [%d, %d]\n", M, K, N, K, M, N);
    printf("Grid: (%d,1,1), Block: (%d,1,1), Smem: %zu bytes\n\n",
           GRID_X, THREAD_NUM, DYNAMIC_SMEM_SIZE);

    // Initialize runtime API
    CHECK_RT(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("ERROR: This kernel requires Hopper (sm_90a)\n");
        return 1;
    }

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
    for (int i = 0; i < M * K; i++) {
        h_A[i] = bfloat16_t((float(i % K) / K) * 0.01f + 0.001f);
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = bfloat16_t((float(i % K) / K) * 0.01f + 0.001f);
    }

    CHECK_RT(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemset(d_C, 0, M * N * sizeof(bfloat16_t)));

    // Create TMA descriptors using the create function
    CUtensorMap A_desc, B_desc, C_desc;
    printf("\nCalling create_linear_gemm_tl_1_6144_1024...\n");
    int ret = create_linear_gemm_tl_1_6144_1024(d_A, d_B, d_C, &A_desc, &B_desc, &C_desc, false);
    if (ret != 0) {
        printf("ERROR: create_linear_gemm_tl_1_6144_1024 failed with code %d\n", ret);
        // const char* err = get_last_error();
        // if (err) printf("  Error message: %s\n", err);
        return 1;
    }
    printf("TMA descriptors created successfully!\n");

    // Create stream and launch
    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    dim3 grid_dim(GRID_X, 1, 1);
    dim3 block_dim(THREAD_NUM, 1, 1);

    // Set max dynamic shared memory
    CHECK_RT(cudaFuncSetAttribute(
        linear_gemm_tl,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        DYNAMIC_SMEM_SIZE));

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    printf("\nLaunching linear_gemm_tl_1_6144_1024 kernel...\n");
    CHECK_RT(cudaEventRecord(start, stream));
    linear_gemm_tl<<<grid_dim, block_dim, DYNAMIC_SMEM_SIZE, stream>>>(
        A_desc, B_desc, C_desc);
    CHECK_RT(cudaEventRecord(stop, stream));
    CHECK_RT(cudaStreamSynchronize(stream));

    float elapsed_ms = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Kernel completed! Time: %.3f ms\n", elapsed_ms);

    // Copy result back and verify
    std::vector<bfloat16_t> h_C(M * N);
    CHECK_RT(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(bfloat16_t), cudaMemcpyDeviceToHost));

    // Compute reference: C = A @ B^T
    // A: [1, 1024], B: [6144, 1024], C: [1, 6144]
    // C[0,j] = sum_k A[0,k] * B[j,k]
    printf("\nVerifying results...\n");
    double max_abs_diff = 0.0;
    double ref_value = 0.0;
    int max_diff_idx = 0;

    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)h_A[k] * (float)h_B[j * K + k];
        }
        float diff = fabs((float)h_C[j] - sum);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            ref_value = sum;
            max_diff_idx = j;
        }
    }

    printf("Max absolute diff: %.6f (at index %d)\n", max_abs_diff, max_diff_idx);
    printf("  Computed: %.6f, Reference: %.6f\n", (float)h_C[max_diff_idx], ref_value);

    // Show first 10 results
    printf("\nFirst 10 results:\n");
    for (int i = 0; i < 10 && i < N; i++) {
        float ref = 0.0f;
        for (int k = 0; k < K; k++) {
            ref += (float)h_A[k] * (float)h_B[i * K + k];
        }
        printf("  C[%d]: kernel=%.6f ref=%.6f diff=%.6f\n",
               i, (float)h_C[i], ref, fabs((float)h_C[i] - ref));
    }

    if (max_abs_diff < 0.1f) {
        printf("\nPASS: Results match! (max_diff=%.6f)\n", max_abs_diff);
    } else {
        printf("\nWARNING: Results differ more than expected (max_diff=%.6f)\n", max_abs_diff);
    }

    // Cleanup
    CHECK_RT(cudaFree(d_A));
    CHECK_RT(cudaFree(d_B));
    CHECK_RT(cudaFree(d_C));
    CHECK_RT(cudaStreamDestroy(stream));

    printf("\nDone!\n");
    return 0;
}