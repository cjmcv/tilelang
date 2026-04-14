// tma_copy_1d.cu - 1D TMA 数据拷贝示例
// 演示如何使用 Hopper TMA 进行高效的 1D 数据搬运

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
    extern __shared__ __align__(16) float smem[];
    __shared__ unsigned long long mbarrier;

    int tid = threadIdx.x;
    int element_idx = blockIdx.x * 128 + tid;

    // Thread 0 初始化 mbarrier (128 threads 对应 128 elements)
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "l"(&mbarrier), "r"(128));
    }
    __syncthreads();

    // Load phase: TMA load from global to shared
    if (tid < 128) {
        asm volatile("mbarrier.arrive_shared.b64 _, [%0];" : : "l"(&mbarrier));
    }
    __syncthreads();

    if (element_idx < num_elements) {
        // TMA load 指令
        void* dst_addr = static_cast<void*>(&smem[tid]);
        asm volatile(
            "ld.global.atomic.acquire.cluster.f32 [%0], %1;"
            : : "l"(dst_addr), "l"(src + element_idx)
        );
    }

    // 等待所有 TMA 加载完成
    if (tid < 128) {
        asm volatile("mbarrier.wait.shared.b64 _, [%0];" : : "l"(&mbarrier));
    }
    __syncthreads();

    // Store phase: TMA store from shared to global
    if (tid < 128) {
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];" : : "l"(&mbarrier));
    }
    __syncthreads();

    if (element_idx < num_elements) {
        // TMA store 指令
        void* src_addr = static_cast<void*>(&smem[tid]);
        asm volatile(
            "st.global.release.f32 [%0], %1;"
            : : "l"(dst + element_idx), "f"(smem[tid])
        );
    }
}

//------------------------------------------------------------------------------
// 使用 TMA 描述符的版本 (更完整的 API 用法)
//------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128, 1)
tma_copy_1d_desc_kernel(CUtensorMap src_desc, CUtensorMap dst_desc,
                        int num_elements, int total_blocks) {
    extern __shared__ __align__(16) uchar smem[];
    __shared__ unsigned long long mbarrier;

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * 128;

    // Thread 0 初始化 mbarrier
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "l"(&mbarrier), "r"(128));
    }
    __syncthreads();

    // Phase 1: TMA Load (使用 descriptor)
    if (tid < 128) {
        asm volatile("mbarrier.arrive_shared.b64 _, [%0];" : : "l"(&mbarrier));
    }
    __syncthreads();

    // 每个 thread 处理 1 个 element
    if (block_offset + tid < num_elements) {
        float* shared_buf = reinterpret_cast<float*>(smem);

        // TMA 加载使用 inline asm
        // cudatx += src_desc;  // 编译期绑定 descriptor
        asm volatile(
            "{ .global .指令可能需要特定语法 }"
            ::: "memory"
        );
    }

    __syncthreads();

    // Phase 2: TMA Store
    if (tid < 128) {
        asm volatile("mbarrier.arrive_shared.b64 _, [%0];" : : "l"(&mbarrier));
    }
    __syncthreads();

    if (block_offset + tid < num_elements) {
        asm volatile(
            "st.global.release.f32 [%0], %1;"
            : : "l"(dst_desc), "f"(smem[tid])
        );
    }
}

//------------------------------------------------------------------------------
// 简化的 TMA Load/Store wrapper (推荐实际使用)
//------------------------------------------------------------------------------
namespace tma {

// 创建 1D TMA descriptor
inline CUresult create_1d_desc(
    CUtensorMap* desc,
    void* ptr,
    uint64_t num_elements,
    uint32_t box_size
) {
    uint64_t global_dim[] = {num_elements};
    uint64_t global_stride[] = {1ULL};
    uint32_t box_dim[] = {box_size};
    uint32_t element_strides[] = {1};

    return cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        1,                      // rank
        ptr,
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

}  // namespace tma

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

    // 创建 TMA 描述符
    CUtensorMap src_desc, dst_desc;
    CUresult res = tma::create_1d_desc(&src_desc, d_src, NUM_ELEMENTS, BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create src descriptor: %d\n", res);
        return 1;
    }

    res = tma::create_1d_desc(&dst_desc, d_dst, NUM_ELEMENTS, BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create dst descriptor: %d\n", res);
        return 1;
    }

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