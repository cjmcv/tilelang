// test_tma.cu - 简单的 TMA 数据拷贝测试
// 从全局内存 A 拷贝到全局内存 B

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/debug.h>
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>

namespace {

#define CHECK_CU(result)                                                       \
  do {                                                                         \
    CUresult _result = (result);                                                \
    if (_result != CUDA_SUCCESS) {                                              \
      const char *errstr;                                                      \
      cuGetErrorString(_result, &errstr);                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
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

// TMA 2D copy kernel
extern "C" __global__ void __launch_bounds__(256, 1)
tma_copy_kernel(__grid_constant__ const CUtensorMap src_desc,
                __grid_constant__ const CUtensorMap dst_desc) {
  extern __shared__ __align__(1024) uchar smem_buf[];
  __shared__ uint64_t mbarrier_mem[2];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);

  // 只用 thread 0 初始化 barrier 和 prefetch descriptor
  if (threadIdx.x == 0) {
    mbarrier[0].init(128);  // 等待 128 threads
    tl::prefetch_tma_descriptor(src_desc);
    tl::prefetch_tma_descriptor(dst_desc);
  }
  __syncthreads();

  // Warpgroup 1: thread 0-127 做 TMA load
  if (threadIdx.x < 128) {
    // 设置期望的 transaction 数量
    mbarrier[0].expect_transaction(4096);  // 64 * 64 * 2 bytes = 8192 bytes? 但这里用 4096

    // TMA load: 从 A_desc 加载数据到 smem
    // 注意：这里用 2D TMA，box 维度是 64x64
    tl::tma_load(src_desc, mbarrier[0],
                 &(((bfloat16_t*)smem_buf)[0]),
                 (int)blockIdx.x * 64, 0);
  } else {
    // Warpgroup 2: thread 128-255 等待
    mbarrier[0].wait(0);
  }

  __syncthreads();

  // Warpgroup 1: thread 0-127 做 TMA store
  if (threadIdx.x < 128) {
    // 设置期望的 transaction
    mbarrier[0].expect_transaction(4096);

    // TMA store: 从 smem 存储数据到 C_desc
    tl::tma_store(dst_desc,
                  &(((bfloat16_t*)smem_buf)[0]),
                  (int)blockIdx.x * 64, 0);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
}

// 创建 2D TMA 描述符
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

    // TileLang 约定: coord[0]=y, coord[1]=x
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

int main(int argc, char **argv) {
    const int M = 64;    // 行数
    const int N = 256;   // 列数
    const int BLOCK_N = 64;
    const int BLOCK_M = 64;

    printf("=== TMA Copy Test ===\n");
    printf("Matrix size: M=%d, N=%d\n", M, N);
    printf("Block size: %d x %d\n", BLOCK_M, BLOCK_N);
    printf("Grid: (%d, 1, 1)\n\n", (N + BLOCK_N - 1) / BLOCK_N);

    // 初始化
    CHECK_RT(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("ERROR: This kernel requires Hopper (sm_90a)\n");
        return 1;
    }

    // 分配内存
    bfloat16_t *d_A = nullptr;
    bfloat16_t *d_B = nullptr;
    size_t size = M * N * sizeof(bfloat16_t);

    CHECK_RT(cudaMalloc(&d_A, size));
    CHECK_RT(cudaMalloc(&d_B, size));
    printf("d_A = %p\n", (void*)d_A);
    printf("d_B = %p\n", (void*)d_B);

    // 初始化输入数据
    std::vector<bfloat16_t> h_A(M * N);
    for (int i = 0; i < M * N; i++) {
        h_A[i] = bfloat16_t((float)i * 0.001f);
    }
    CHECK_RT(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));

    // 创建 TMA 描述符
    // 矩阵 A 和 B 都是 [M, N]，TMA 坐标 (x, y) 其中 x=blockIdx.x*64, y=0
    // TileLang 中 coord[0]=y, coord[1]=x，所以 dim_y=N, dim_x=M
    CUtensorMap A_desc, B_desc;
    CHECK_CU(CreateTMA2DDesc(
        &A_desc, d_A,
        M, N,              // dim_x=M, dim_y=N (因为 coord[0]=y, coord[1]=x)
        BLOCK_M, BLOCK_N,  // box_dim
        M                   // stride_y = M (跨行 stride，元素个数)
    ));
    CHECK_CU(CreateTMA2DDesc(
        &B_desc, d_B,
        M, N,
        BLOCK_M, BLOCK_N,
        M
    ));
    printf("TMA descriptors created\n");

    // Launch 配置
    int grid_x = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid_dim(grid_x, 1, 1);
    dim3 block_dim(256, 1, 1);
    size_t smem_size = 64 * 64 * sizeof(bfloat16_t);  // 8192 bytes

    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    printf("Launching tma_copy_kernel...\n");
    printf("Grid: (%d,1,1), Block: (256,1,1), Smem: %zu\n\n",
           grid_x, smem_size);

    tma_copy_kernel<<<grid_dim, block_dim, smem_size, stream>>>(A_desc, B_desc);

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorName(err));
        return 1;
    }
    printf("Kernel completed!\n");

    // 验证结果
    std::vector<bfloat16_t> h_B(M * N);
    CHECK_RT(cudaMemcpy(h_B.data(), d_B, size, cudaMemcpyDeviceToHost));

    // 检查结果
    int errors = 0;
    printf("\nVerification:\n");
    for (int i = 0; i < M * N; i++) {
        float expected = (float)h_A[i];
        float actual = (float)h_B[i];
        if (fabsf(expected - actual) > 0.001f) {
            errors++;
            if (errors <= 5) {
                printf("  ERROR at index %d: expected=%f, actual=%f\n",
                       i, expected, actual);
            }
        }
    }

    if (errors == 0) {
        printf("  PASS! All %d elements match.\n", M * N);
        printf("\nFirst 10 elements:\n");
        for (int i = 0; i < 10; i++) {
            printf("  B[%d] = %f (expected %f)\n", i, (float)h_B[i], (float)h_A[i]);
        }
    } else {
        printf("  FAIL! %d errors out of %d elements\n", errors, M * N);
    }

    // 清理
    CHECK_RT(cudaFree(d_A));
    CHECK_RT(cudaFree(d_B));
    CHECK_RT(cudaStreamDestroy(stream));

    printf("\nDone!\n");
    return errors > 0 ? 1 : 0;
}