
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
    __device__ __forceinline__ void linear_kernel_1_2560_9728(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr,
                                                const void* __restrict__ weight_ptr,
                                                const void* __restrict__ residual_ptr,
                                                void* __restrict__ output_ptr,
                                                int num_active_tokens,
                                                bool residual) {
  static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==64); static_assert(TILE_DIM_Y==64); static_assert(TILE_DIM_Z==64);
  static_assert(M==1); static_assert(N==2560); static_assert(K==9728);
  
  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  const bfloat16_t* __restrict__ B = static_cast<const bfloat16_t*>(weight_ptr);
  const bfloat16_t* __restrict__ C = static_cast<const bfloat16_t*>(output_ptr);
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[32];
  bfloat16_t A_local[16];
  bfloat16_t B_local[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((i_1 * 155648) + ((((int)threadIdx.x) >> 3) * 9728)) + (((int)bz) * 2432)) + ((((int)threadIdx.x) & 7) * 8)), ((((i_1 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_1 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), B+(((((((int)bx) * 622592) + (i_2 * 155648)) + ((((int)threadIdx.x) >> 3) * 9728)) + (((int)bz) * 2432)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 37; ++ko) {
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + (i_3 * 8));
      }
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 2048) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])) + 0, B_local + (i_4 * 8));
      }
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((i_6 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((i_6 * 155648) + ((((int)threadIdx.x) >> 3) * 9728)) + (((int)bz) * 2432)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64), ((((i_6 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_6 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_7 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), B+(((((((((int)bx) * 622592) + (i_7 * 155648)) + ((((int)threadIdx.x) >> 3) * 9728)) + (((int)bz) * 2432)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_8 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + (i_8 * 8));
    }
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 2048) + (i_9 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])) + 0, B_local + (i_9 * 8));
    }
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      for (int j_1 = 0; j_1 < 2; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_10 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_10 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 16; ++i_11) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_11 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_11 >> 3) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_11 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 32; ++i_12) {
    if (((i_12 * 2) + (((int)threadIdx.x) >> 6)) < 1) {
      AtomicAdd((&(C[((((i_12 * 5120) + ((((int)threadIdx.x) >> 6) * 2560)) + (((int)bx) * 64)) + (((int)threadIdx.x) & 63))])), ((bfloat16_t*)buf_dyn_shmem)[((i_12 * 128) + ((int)threadIdx.x))], 0);
    }
  }
}


} // kernel
// Strategy: linear_gemm_tl_1_2560_9728
// selected_hparams: [64, 64, 64, 4, 1, 128, <GemmWarpPolicy.Square: 0>, False].
// smem: 16384 bytes.
// use_cooperative_groups: 0.
// grid_dim=(40, 1, 4), tile_dim=(64, 64, 64),
// block_dim=(128, 1, 1).
// latency: 0.36214, idx: 37