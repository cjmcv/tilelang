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

    uint64_t global_dim[] = {static_cast<uint64_t>(dim_y), static_cast<uint64_t>(dim_x)};
    uint64_t global_stride[] = {static_cast<uint64_t>(stride_y), 1ULL};
    uint32_t box_dim[] = {box_dim_y, box_dim_x};
    uint32_t element_strides[] = {1, 1};

    return cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BF16,
        2,
        gmem_ptr,
        global_dim,
        global_stride,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_OOB_FILL_NONE
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

    // A: [M, N*2] bfloat16, 前 N 列是 silu 输入，后 N 列是 mul 输入
    CHECK_CU(CreateTMA2DDesc(
        A_desc,
        A_gmem,
        N * 2,          // dim_x = N*2
        M,              // dim_y = M
        BLOCK_N,        // box_dim_x
        BLOCK_M,        // box_dim_y
        N * 2 * 2       // stride_y = N*2 * sizeof(bfloat16)
    ));

    // C: [M, N] bfloat16
    CHECK_CU(CreateTMA2DDesc(
        C_desc,
        C_gmem,
        N,              // dim_x = N
        M,              // dim_y = M
        BLOCK_N,        // box_dim_x
        BLOCK_M,        // box_dim_y
        N * 2           // stride_y = N * sizeof(bfloat16)
    ));
}

// 使用 cuLaunchKernelEx 启动带 __grid_constant__ 的 kernel
static cudaError_t LaunchSiluMulKernel(
    void *A_gmem,
    void *C_gmem,
    int M,
    int N,
    cudaStream_t stream
) {
    // 创建 TMA 描述符
    CUtensorMap A_desc, C_desc;
    CreateSiluMulTMADescs(&A_desc, &C_desc, A_gmem, C_gmem, M, N);

    // Grid/Block 维度 (来自生成代码)
    // grid: (152, 1, 1) = (N/BLOCK_N, M/BLOCK_M, 1)
    // block: (256, 1, 1) = threads
    dim3 grid_dim(152, 1, 1);
    dim3 block_dim(256, 1, 1);
    size_t smem_size = 12288;

    // 启动配置
    CUstream custream = (CUstream)stream;

    CUlaunchAttribute launch_attrs[1];
    launch_attrs[0].id = CU_LAUNCH_ATTRIBUTE_GRID_CONSTANT;
    launch_attrs[0].value.gridConstantDesc.hostBuffer = 1;

    CUlaunchConfig config;
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.numAttrs = 1;
    config.attrs = launch_attrs;
    config.stream = custream;

    // 获取 kernel 函数 (需要先加载编译好的 cubin)
    CUmodule module;
    CUfunction function;
    CHECK_CU(cuModuleGetCurrent(&module));
    CHECK_CU(cuModuleGetFunction(&function, module, "silu_mul_kernel"));

    // 设置共享内存
    CHECK_CU(cuFuncSetAttribute(
        function,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        (int)smem_size
    ));

    // 准备参数
    void *args[] = {&A_desc, &C_desc};

    // 启动
    CHECK_CU(cuLaunchKernelEx(&config, function, args));

    return cudaSuccess;
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
    printf("Shared memory: 12288 bytes\n\n");

    // 初始化 CUDA Driver API
    CHECK_CU(cuInit(0));

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));

    // 检查架构
    int sm_version;
    CHECK_CU(cuDeviceGetAttribute(&sm_version, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, device));
    printf("Device: sm_%d.%d\n", sm_version / 10, sm_version % 10);

    if (sm_version < 90) {
        printf("ERROR: This kernel requires Hopper (sm_90a)\n");
        return 1;
    }

    // 分配内存
    bfloat16_t *d_A = nullptr;
    bfloat16_t *d_C = nullptr;

    size_t A_size = M * N2 * sizeof(bfloat16_t);
    size_t C_size = M * N * sizeof(bfloat16_t);

    CHECK_RT(cudaMalloc(&d_A, A_size));
    CHECK_RT(cudaMalloc(&d_C, C_size));

    // 初始化输入
    std::vector<bfloat16_t> h_A(M * N2);
    for (int i = 0; i < M * N2; i++) {
        h_A[i] = bfloat16_t((float(i % 100) / 100.0f) * 2.0f - 1.0f);
    }
    CHECK_RT(cudaMemcpy(d_A, h_A.data(), A_size, cudaMemcpyHostToDevice));

    // Stream
    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    // 启动 kernel
    printf("Launching silu_mul_kernel...\n");
    cudaError_t err = LaunchSiluMulKernel(d_A, d_C, M, N, stream);
    if (err != cudaSuccess) {
        printf("Launch failed: %s\n", cudaGetErrorName(err));
        return 1;
    }

    CHECK_RT(cudaStreamSynchronize(stream));
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
    CHECK_CU(cuCtxDestroy(context));

    printf("\nDone!\n");
    return 0;
}