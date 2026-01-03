// ===============================================================
//  main.cpp
//  compile & run:
//  nvcc -std=c++17 -O3 -arch=sm_80 -o gemm_test \
//       gemm_kernel.cu main.cpp
//  ./gemm_test
// ===============================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

using bfloat16_t = __nv_bfloat16;

/* 外部 kernel 声明，与 gemm_kernel.cu 保持一致 */
extern "C" __global__ void __launch_bounds__(128, 1)
gemm_kernel(const bfloat16_t* __restrict__ A,
            const bfloat16_t* __restrict__ B,
            bfloat16_t*       __restrict__ C);

/* 工具宏 */
#define CHECK_CUDA(err) do {                                           \
    cudaError_t e = (err);                                             \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d  code=%d(%s)\n",             \
                __FILE__, __LINE__, e, cudaGetErrorName(e));           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

int main() {
    /* 1. 矩阵规模 --------------------------------------------------- */
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    /* 2. grid 划分：kernel 每个 block 负责 128×128 的 C tile -------- */
    constexpr int BM = 128;
    constexpr int BN = 128;
    const int grid_x = (N + BN - 1) / BN;
    const int grid_y = (M + BM - 1) / BM;
    dim3  grid(grid_x, grid_y);
    dim3  block(128);                 // 128 线程 / block

    /* 3. 显存大小 --------------------------------------------------- */
    size_t bytes_A = size_t(M) * K * sizeof(bfloat16_t);
    size_t bytes_B = size_t(K) * N * sizeof(bfloat16_t);
    size_t bytes_C = size_t(M) * N * sizeof(bfloat16_t);

    /* 4. 申请显存 --------------------------------------------------- */
    bfloat16_t *d_A, *d_B, *d_C;
    CHECK_CUDA( cudaMalloc(&d_A, bytes_A) );
    CHECK_CUDA( cudaMalloc(&d_B, bytes_B) );
    CHECK_CUDA( cudaMalloc(&d_C, bytes_C) );

    /* 5. 主机端随机初始化 A、B 并拷到设备 ------------------------- */
    std::vector<bfloat16_t> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i)
        h_A[i] = __float2bfloat16( 1 ); // float(rand()) / RAND_MAX
    for (int i = 0; i < K * N; ++i)
        h_B[i] = __float2bfloat16( 2 );

    CHECK_CUDA( cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice) );

    /* 6. 共享内存需求计算 ------------------------------------------ */
    /* kernel 里用了 3 个 8192-byte 缓冲区轮播，共 24576 B */ // 49152
    constexpr int shmem_per_block = 128*32*2 * 2 * 3; 
    CHECK_CUDA( cudaFuncSetAttribute(
                    gemm_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shmem_per_block) );

    /* 7. 预热一次 --------------------------------------------------- */
    gemm_kernel<<<grid, block, shmem_per_block>>>(d_A, d_B, d_C);
    CHECK_CUDA( cudaDeviceSynchronize() );

    /* 8. 正式计时 --------------------------------------------------- */
    cudaEvent_t start, stop;
    CHECK_CUDA( cudaEventCreate(&start) );
    CHECK_CUDA( cudaEventCreate(&stop)  );
    CHECK_CUDA( cudaEventRecord(start) );
    gemm_kernel<<<grid, block, shmem_per_block>>>(d_A, d_B, d_C);
    CHECK_CUDA( cudaEventRecord(stop) );
    CHECK_CUDA( cudaEventSynchronize(stop) );

    float millis = 0;
    CHECK_CUDA( cudaEventElapsedTime(&millis, start, stop) );
    double flop = 2.0 * M * N * K;
    printf("GEMM kernel 耗时: %.3f ms  (%.2f TFlop/s)\n",
           millis, flop / millis / 1e9);

    /* 9. 可选：把结果拷回主机，打印前 10 个元素 --------------------- */
    std::vector<bfloat16_t> h_C(M * N);
    CHECK_CUDA( cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost) );
    printf("C[0..9]: ");
    for (int i = 0; i < 10; ++i)
        printf("%f ", __bfloat162float(h_C[i]));
    printf("\n");

    /* 10. 释放资源 -------------------------------------------------- */
    CHECK_CUDA( cudaFree(d_A) );
    CHECK_CUDA( cudaFree(d_B) );
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUDA( cudaEventDestroy(start) );
    CHECK_CUDA( cudaEventDestroy(stop)  );

    return 0;
}