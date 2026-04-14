// host.cu - Host 端代码
// 直接 include TileLang 模板和生成的 kernel

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TileLang 模板库
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>

// TileLang 生成的 kernel
#include "silu_mul_tl_32_9728.cuh"

namespace {

#define CHECK_CU(result)                                                       \
  do {                                                                         \
    CUresult _result = (result);                                                \
    if (_result != CUDA_SUCCESS) {                                              \
      const char *errstr;                                                      \
      cuGetErrorString(_result, &errstr);                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              errstr);                                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_RT(result)                                                       \
  do {                                                                         \
    cudaError_t _result = (result);                                            \
    if (_result != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorName(_result));                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

}  // anonymous namespace

// 创建 2D Tiled TMA 描述符
static CUresult CreateTMA2DDesc(
    CUtensorMap *desc,
    void *gmem_ptr,
    uint32_t dim_x,
    uint32_t dim_y,
    uint32_t box_dim_x,
    uint32_t box_dim_y,
    uint32_t stride_y
) {
    CUtensorMap tensor_map;

    // global_dim: {y_size, x_size}, 坐标约定: crd0=y, crd1=x
    uint64_t global_dim[] = {static_cast<uint64_t>(dim_y), static_cast<uint64_t>(dim_x)};
    uint64_t global_stride[] = {static_cast<uint64_t>(stride_y), 1ULL};
    uint32_t box_dim[] = {box_dim_y, box_dim_x};
    uint32_t element_strides[] = {1, 1};

    return cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        gmem_ptr,
        global_dim,
        global_stride,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// 创建 silu_mul kernel 的 TMA 描述符
static void CreateSiluMulTMADescs(
    CUtensorMap *A_desc,
    CUtensorMap *C_desc,
    void *A_gmem,
    void *C_gmem,
    int M,
    int N
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 64;

    // A: [M, N*2] bfloat16
    // kernel 里 tma_load(..., x, y) 其中 x=blockIdx.x*64, y=0
    // TMA global_dim = {y_size, x_size}，所以要交换：
    // y_size = N*2, x_size = M, stride_y = M (沿 y 方向的跨行 stride)
    CUresult res = CreateTMA2DDesc(
        A_desc,
        A_gmem,
        M,              // dim_x = M (x_size)
        N * 2,          // dim_y = N*2 (y_size)
        BLOCK_M,        // box_dim_x (因为 x_size=M)
        BLOCK_N,        // box_dim_y (因为 y_size=N*2)
        M               // stride_y = M (元素个数，跨行 stride)
    );
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "CreateTMA2DDesc A failed: %d\n", res);
        exit(1);
    }

    // C: [M, N] bfloat16
    res = CreateTMA2DDesc(
        C_desc,
        C_gmem,
        M,              // dim_x = M (x_size)
        N,              // dim_y = N (y_size)
        BLOCK_M,        // box_dim_x
        BLOCK_N,        // box_dim_y
        M               // stride_y = M (元素个数)
    );
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "CreateTMA2DDesc C failed: %d\n", res);
        exit(1);
    }
}

// ============================================
// 主函数
// ============================================
int main(int argc, char **argv) {
    const int M = 32;
    const int N = 9728;
    const int N2 = N * 2;

    printf("=== silu_mul Kernel Launch Test ===\n");
    printf("M=%d, N=%d, Grid=(%d,%d,%d), Block=(%d,%d,%d)\n",
           M, N, 152, 1, 1, 256, 1, 1);
    printf("Grid: (N/BLOCK_N=%d, M/BLOCK_M=%d)\n", 152, 1);
    printf("Shared memory: 16384 bytes\n\n");

    // 使用 Runtime API 初始化
    CHECK_RT(cudaSetDevice(0));

    // 检查架构
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("ERROR: This kernel requires Hopper (sm_90a)\n");
        return 1;
    }

    // 分配内存 (使用 Runtime API)
    bfloat16_t *d_A = nullptr;
    bfloat16_t *d_C = nullptr;

    size_t A_size = M * N2 * sizeof(bfloat16_t);
    size_t C_size = M * N * sizeof(bfloat16_t);

    CHECK_RT(cudaMalloc(&d_A, A_size));
    CHECK_RT(cudaMalloc(&d_C, C_size));

    printf("d_A = %p, size = %zu\n", (void*)d_A, A_size);
    printf("d_C = %p, size = %zu\n", (void*)d_C, C_size);

    // 初始化输入
    std::vector<bfloat16_t> h_A(M * N2);
    for (int i = 0; i < M * N2; i++) {
        h_A[i] = bfloat16_t((float(i % 100) / 100.0f) * 2.0f - 1.0f);
    }
    CHECK_RT(cudaMemcpy(d_A, h_A.data(), A_size, cudaMemcpyHostToDevice));

    // 创建 TMA 描述符 (需要 Driver API)
    // 注意: cuTensorMapEncodeTiled 需要正确的 CUDA context
    CUdevice device;
    CHECK_CU(cuInit(0));
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, device));

    CUtensorMap A_desc, C_desc;
    CreateSiluMulTMADescs(&A_desc, &C_desc, d_A, d_C, M, N);
    printf("TMA descriptors created successfully\n");

    // Grid/Block/Shared memory 维度
    dim3 grid_dim(152, 1, 1);
    dim3 block_dim(256, 1, 1);
    size_t smem_size = 16384;

    // Stream
    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    // 启动 kernel
    printf("Launching silu_mul_kernel...\n");
    silu_mul_kernel<<<grid_dim, block_dim, smem_size, stream>>>(A_desc, C_desc);

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorName(err));
        return 1;
    }
    printf("Kernel completed!\n");

    // 验证结果
    std::vector<bfloat16_t> h_C(M * N);
    CHECK_RT(cudaMemcpy(h_C.data(), d_C, C_size, cudaMemcpyDeviceToHost));

    printf("\nOutput (first 8):\n");
    for (int i = 0; i < 8; i++) {
        printf("  C[%d] = %f\n", i, (float)h_C[i]);
    }

    // 清理
    CHECK_RT(cudaFree(d_A));
    CHECK_RT(cudaFree(d_C));
    CHECK_RT(cudaStreamDestroy(stream));
    CHECK_CU(cuCtxDestroy(ctx));

    printf("\nDone!\n");
    return 0;
}