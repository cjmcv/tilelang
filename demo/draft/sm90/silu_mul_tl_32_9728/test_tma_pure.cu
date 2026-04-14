// test_tma_pure.cu - 纯 CUDA TMA 拷贝示例
// 使用 CUDA Driver API 创建 TMA，用 Runtime API launch

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// =============================================================================
// Device 端 TMA Copy Kernel
// =============================================================================
__device__ void tma_load_2d(void *smem_ptr, const CUtensorMap &desc,
                             uint32_t mbarrier, int32_t crd0, int32_t crd1) {
  uint64_t smem_addr = (uint64_t)__cvta_shared_to_shared(smem_ptr);
  uint64_t desc_addr = (uint64_t)&desc;

  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
    "[%0], [%1, {%3, %4}], [%2], 0;"
    : : "r"(smem_addr), "l"(desc_addr), "r"(mbarrier), "r"(crd0), "r"(crd1)
    : "memory");
}

__device__ void tma_store_2d(void *smem_ptr, const CUtensorMap &desc,
                              uint32_t mbarrier, int32_t crd0, int32_t crd1) {
  uint64_t smem_addr = (uint64_t)__cvta_shared_to_shared(smem_ptr);
  uint64_t desc_addr = (uint64_t)&desc;

  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
    "[%0], [%1, {%3, %4}], [%2], 0;"
    : : "r"(smem_addr), "l"(desc_addr), "r"(mbarrier), "r"(crd0), "r"(crd1)
    : "memory");
}

__device__ void mbarrier_init(uint64_t *barrier, int thread_count) {
  uint32_t barrier_addr = (uint32_t)__cvta_generic_to_shared(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(barrier_addr), "r"(thread_count));
}

__device__ void mbarrier_arrive(uint64_t *barrier) {
  uint32_t barrier_addr = (uint32_t)__cvta_generic_to_shared(barrier);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(barrier_addr));
}

__device__ void mbarrier_wait(uint64_t *barrier, int phase) {
  uint32_t barrier_addr = (uint32_t)__cvta_generic_to_shared(barrier);
  asm volatile(
    "{ .reg .pred P1; "
    "LAB_WAIT: "
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1; "
    "@P1 bra.uni DONE; "
    "bra.uni LAB_WAIT; "
    "DONE: }"
    :: "r"(barrier_addr), "r"(phase));
}

// TMA Copy Kernel
extern "C" __global__ void __launch_bounds__(256, 1)
tma_copy_kernel(CUtensorMap src_desc, CUtensorMap dst_desc) {
  extern __shared__ __align__(16) uchar smem_buf[];
  __shared__ uint64_t mbarrier_mem[1];

  uint64_t *mbarrier = mbarrier_mem;
  uint32_t mbarrier_addr = (uint32_t)__cvta_generic_to_shared(mbarrier);

  // Thread 0 初始化 mbarrier
  if (threadIdx.x == 0) {
    mbarrier_init(mbarrier, 128);
  }
  __syncthreads();

  // Warpgroup 1: threads 0-127 执行 TMA load
  if (threadIdx.x < 128) {
    // 设置期望的 transaction
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mbarrier_addr), "r"(4096));  // 4096 bytes

    // TMA load: 从 src_desc 加载到 smem_buf[0]
    tma_load_2d(&smem_buf[0], src_desc, mbarrier_addr,
                 blockIdx.x * 64, 0);  // (y, x) 坐标
  }

  // Warpgroup 2: threads 128-255 等待
  if (threadIdx.x >= 128) {
    mbarrier_wait(mbarrier, 0);
  }
  __syncthreads();

  // Warpgroup 1: threads 0-127 执行 TMA store
  if (threadIdx.x < 128) {
    // 设置期望的 transaction
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mbarrier_addr), "r"(4096));

    // TMA store: 从 smem_buf[0] 存储到 dst_desc
    tma_store_2d(&smem_buf[0], dst_desc, mbarrier_addr,
                  blockIdx.x * 64, 0);

    // TMA store 完成信号
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbarrier_addr));
    mbarrier_wait(mbarrier, 0);
  }
}

// =============================================================================
// Host 端: TMA Descriptor 创建
// =============================================================================
static CUresult create_tma_2d_desc(
    CUtensorMap *tensor_map,
    void *global_addr,
    uint32_t dim_outer,   // 外层维度 (y)
    uint32_t dim_inner,    // 内层维度 (x)
    uint32_t box_dim_outer,
    uint32_t box_dim_inner,
    uint32_t stride_outer  // 外层 stride (元素个数)
) {
  uint64_t global_dim[] = {dim_outer, dim_inner};
  uint64_t global_stride[] = {stride_outer, 1ULL};
  uint32_t box_dim[] = {box_dim_outer, box_dim_inner};
  uint32_t element_strides[] = {1, 1};

  return cuTensorMapEncodeTiled(
    tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    2,
    global_addr,
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
  const int M = 32;    // 行数 (y)
  const int N = 64;    // 列数 (x)
  const int BLOCK_X = 64;
  const int BLOCK_Y = 32;

  printf("=== Pure CUDA TMA Copy Test ===\n");
  printf("Matrix: M=%d (y), N=%d (x)\n", M, N);
  printf("Block: %d x %d, Grid: %d\n\n", BLOCK_Y, BLOCK_X, (N + BLOCK_X - 1) / BLOCK_X);

  // 初始化 Driver API
  CHECK_CU(cuInit(0));
  CUdevice device;
  CHECK_CU(cuDeviceGet(&device, 0));

  // 创建 context 并设置到当前 device
  CUcontext context;
  CHECK_CU(cuCtxCreate(&context, 0, device));
  CHECK_RT(cudaSetDevice(0));

  // 获取 device 信息
  cudaDeviceProp prop;
  CHECK_RT(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  // 分配内存
  bfloat16_t *d_A = nullptr;
  bfloat16_t *d_B = nullptr;
  size_t size = M * N * sizeof(bfloat16_t);

  CHECK_RT(cudaMalloc(&d_A, size));
  CHECK_RT(cudaMalloc(&d_B, size));
  printf("d_A=%p, d_B=%p, size=%zu\n\n", (void*)d_A, (void*)d_B, size);

  // 初始化输入
  std::vector<bfloat16_t> h_A(M * N);
  for (int i = 0; i < M * N; i++) {
    h_A[i] = bfloat16_t((float)i * 0.01f);
  }
  CHECK_RT(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));

  // 创建 TMA descriptors
  // 矩阵是 [M, N] = [32, 64]，row-major
  // TMA 坐标 (y, x) 其中 y=0..M-1, x=0..N-1
  // stride = N (每行 N 个元素)
  CUtensorMap A_desc, B_desc;

  CUresult res = create_tma_2d_desc(
    &A_desc,
    d_A,
    M, N,           // dim_outer=M=32, dim_inner=N=64
    BLOCK_Y, BLOCK_X,  // box_dim_outer=32, box_dim_inner=64
    N              // stride_outer=N=64 (元素个数)
  );
  if (res != CUDA_SUCCESS) {
    printf("create_tma_2d_desc A failed: %d\n", res);
    return 1;
  }

  res = create_tma_2d_desc(
    &B_desc,
    d_B,
    M, N,
    BLOCK_Y, BLOCK_X,
    N
  );
  if (res != CUDA_SUCCESS) {
    printf("create_tma_2d_desc B failed: %d\n", res);
    return 1;
  }
  printf("TMA descriptors created\n");

  // Launch
  int grid_x = (N + BLOCK_X - 1) / BLOCK_X;
  dim3 grid(grid_x, 1, 1);
  dim3 block(256, 1, 1);
  size_t smem = BLOCK_Y * BLOCK_X * sizeof(bfloat16_t);  // 4096 bytes

  cudaStream_t stream;
  CHECK_RT(cudaStreamCreate(&stream));

  printf("Launching kernel: grid=(%d,1,1), block=(256,1,1), smem=%zu\n",
         grid_x, smem);

  tma_copy_kernel<<<grid, block, smem, stream>>>(A_desc, B_desc);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    printf("Kernel failed: %s\n", cudaGetErrorName(err));
    return 1;
  }
  printf("Kernel completed!\n");

  // 验证
  std::vector<bfloat16_t> h_B(M * N);
  CHECK_RT(cudaMemcpy(h_B.data(), d_B, size, cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < M * N; i++) {
    float expected = (float)h_A[i];
    float actual = (float)h_B[i];
    if (fabsf(expected - actual) > 0.001f) {
      errors++;
      if (errors <= 5) {
        printf("ERROR[%d]: expected=%f, actual=%f\n", i, expected, actual);
      }
    }
  }

  if (errors == 0) {
    printf("\nPASS! All %d elements match.\n", M * N);
    printf("First 8 elements: ");
    for (int i = 0; i < 8; i++) printf("%.3f ", (float)h_B[i]);
    printf("\n");
  } else {
    printf("\nFAIL! %d errors\n", errors);
  }

  // 清理
  CHECK_RT(cudaFree(d_A));
  CHECK_RT(cudaFree(d_B));
  CHECK_RT(cudaStreamDestroy(stream));
  CHECK_CU(cuCtxDestroy(context));

  printf("Done!\n");
  return errors;
}