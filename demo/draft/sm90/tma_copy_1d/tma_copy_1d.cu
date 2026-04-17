// tma_copy_1d.cu - 1D TMA 数据拷贝示例
// 演示如何使用 Hopper TMA 进行高效的 1D 数据搬运

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstddef>

namespace {

#define CHECK_CU(result)                                                       \
  do {                                                                         \
    CUresult _result = (result);                                                \
    if (_result != CUDA_SUCCESS) {                                              \
      const char *errstr;                                                      \
      cuGetErrorString(_result, &errstr);                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              errstr);                                                         \
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

//------------------------------------------------------------------------------
// TMA 1D Copy Kernel
// 使用 TMA 从 global memory 加载到 shared memory，再存储回 global memory
//------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128, 1)
tma_copy_1d_kernel(const float* __restrict__ src, float* dst, int num_elements) {
    extern __shared__ __align__(16) unsigned char smem_bytes[];
    __shared__ unsigned long long mbarrier;

    int tid = threadIdx.x;
    int element_idx = blockIdx.x * 128 + tid;

    // 获取 shared memory 地址
    unsigned int smem_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_bytes));
    unsigned int mbar_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(&mbarrier));

    // Thread 0 初始化 mbarrier (128 threads 对应 128 elements)
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(mbar_ptr), "r"(128));
    }
    __syncthreads();

    // Load phase: TMA load from global to shared
    if (tid < 128) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(mbar_ptr), "r"(128));
    }
    __syncthreads();

    if (element_idx < num_elements) {
        // TMA load 指令
        unsigned long long dst_addr = smem_ptr + tid * sizeof(float);
        asm volatile(
            "ld.global.atomic.acquire.cluster.f32 [%0], [%1];"
            : : "l"(dst_addr), "l"(src + element_idx)
        );
    }

    // 等待所有 TMA 加载完成
    if (tid == 0) {
        asm volatile("mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], 0;"
                     : : "r"(mbar_ptr));
    }
    __syncthreads();

    // Store phase: TMA store from shared to global
    if (tid < 128) {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;"
                     :: "r"(mbar_ptr), "r"(128));
    }
    __syncthreads();

    if (element_idx < num_elements) {
        float* smem_float = reinterpret_cast<float*>(smem_bytes);
        float val = smem_float[tid];
        asm volatile(
            "st.global.release.f32 [%0], %1;"
            : : "l"(dst + element_idx), "f"(val)
        );
    }
}

//------------------------------------------------------------------------------
// Host 验证函数
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    const int NUM_ELEMENTS = 1024 * 1024;  // 1M 元素
    const int BLOCK_SIZE = 128;
    const int GRID_SIZE = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("=== TMA 1D Copy Test ===\n");
    printf("Elements: %d\n", NUM_ELEMENTS);
    printf("Block: %d, Grid: %d\n\n", BLOCK_SIZE, GRID_SIZE);

    // 初始化 CUDA
    CHECK_RT(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        fprintf(stderr, "Error: This example requires Hopper (sm_90a)\n");
        return 1;
    }

    // 初始化 Driver API
    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    // 分配设备内存
    float *d_src = nullptr;
    float *d_dst = nullptr;
    size_t bytes = NUM_ELEMENTS * sizeof(float);

    CHECK_RT(cudaMalloc(&d_src, bytes));
    CHECK_RT(cudaMalloc(&d_dst, bytes));

    // 初始化输入数据
    std::vector<float> h_src(NUM_ELEMENTS);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_src[i] = static_cast<float>(i) * 0.001f;
    }
    CHECK_RT(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));

    // 创建 stream
    cudaStream_t stream;
    CHECK_RT(cudaStreamCreate(&stream));

    // Launch kernel
    size_t smem_size = BLOCK_SIZE * sizeof(float);
    printf("Launching kernel with %zu bytes shared memory...\n", smem_size);

    tma_copy_1d_kernel<<<GRID_SIZE, BLOCK_SIZE, smem_size, stream>>>(
        d_src, d_dst, NUM_ELEMENTS
    );

    CHECK_RT(cudaStreamSynchronize(stream));
    printf("Kernel completed!\n");

    // 验证结果
    std::vector<float> h_dst(NUM_ELEMENTS);
    CHECK_RT(cudaMemcpy(h_dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        if (fabsf(h_dst[i] - h_src[i]) > 1e-6f) {
            errors++;
            if (errors <= 5) {
                printf("  Error at %d: expected=%f, got=%f\n",
                       i, h_src[i], h_dst[i]);
            }
        }
    }

    if (errors == 0) {
        printf("PASS: All %d elements match!\n", NUM_ELEMENTS);
    } else {
        printf("FAIL: %d errors out of %d elements\n", errors, NUM_ELEMENTS);
    }

    // 清理
    CHECK_RT(cudaFree(d_src));
    CHECK_RT(cudaFree(d_dst));
    CHECK_RT(cudaStreamDestroy(stream));

    return errors > 0 ? 1 : 0;
}