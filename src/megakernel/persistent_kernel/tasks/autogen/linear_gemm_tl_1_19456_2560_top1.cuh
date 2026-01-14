#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif


namespace kernel {

template <typename T,
    int THREAD_NUM,
    int TILE_DIM_X, 
    int TILE_DIM_Y, 
    int TILE_DIM_Z,
    int M,
    int N,
    int K,
    int O_STRIDE = N,
    int PIPE_MAX = 3,
    bool FUSE_RES = false>
    __device__ __forceinline__ void linear_kernel(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr,
                                                const void* __restrict__ weight_ptr,
                                                const void* __restrict__ residual_ptr,
                                                void* __restrict__ output_ptr,
                                                int num_active_tokens,
                                                bool residual) {
  assert(THREAD_NUM==128);
  assert(TILE_DIM_X==256); assert(TILE_DIM_Y==64); assert(TILE_DIM_Z==128);
  assert(M==1); assert(N==19456); assert(K==2560);
  
  const bfloat16* __restrict__ A = static_cast<const bfloat16* __restrict__>(input_ptr);
  const bfloat16* __restrict__ B = static_cast<const bfloat16* __restrict__>(weight_ptr);
  const bfloat16* __restrict__ C = static_cast<const bfloat16* __restrict__>(output_ptr);
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  bfloat16_t A_local[8];
  bfloat16_t B_local[128];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+(((i_1 * 20480) + ((((int)threadIdx.x) >> 4) * 2560)) + ((((int)threadIdx.x) & 15) * 8)), ((((i_1 * 8) + (((int)threadIdx.x) >> 4)) < 1) && (((i_1 * 8) + (((int)threadIdx.x) >> 4)) < 1)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 32768) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), B+((((((int)bx) * 655360) + (i_2 * 20480)) + ((((int)threadIdx.x) >> 4) * 2560)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 19; ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki >> 2) * 16384) + (i_3 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_3 * 8));
      }
      for (int j = 0; j < 16; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+(((((i_4 * 20480) + ((((int)threadIdx.x) >> 4) * 2560)) + (k * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128), ((((i_4 * 8) + (((int)threadIdx.x) >> 4)) < 1) && (((i_4 * 8) + (((int)threadIdx.x) >> 4)) < 1)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 32768) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), B+((((((((int)bx) * 655360) + (i_5 * 20480)) + ((((int)threadIdx.x) >> 4) * 2560)) + (k * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_1 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_1 >> 2) * 16384) + (i_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_6 * 8));
    }
    for (int j_1 = 0; j_1 < 16; ++j_1) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 64; ++i_7) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_7 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 4096) + ((i_7 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((i_7 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    if (((i_8 * 4) + (((int)threadIdx.x) >> 5)) < 1) {
      *(uint4*)(C + ((((i_8 * 77824) + ((((int)threadIdx.x) >> 5) * 19456)) + (((int)bx) * 256)) + ((((int)threadIdx.x) & 31) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((i_8 * 1024) + (((int)threadIdx.x) * 8)));
    }
  }
}


} // kernel
// Strategy: linear_gemm_tl_1_19456_2560
// selected_hparams: [64, 256, 128, 1, 128, <GemmWarpPolicy.FullRow: 1>, True].
// grid_dim: {'blockIdx.x': 76, 'blockIdx.y': 1, 'blockIdx.z': 1}.
// block_dim: {'threadIdx.x': 128, 'threadIdx.y': 1, 'threadIdx.z': 1}.
// smem: 81920 bytes.
// use_cooperative_groups: 0.
// latency: 0.63077, idx: 54