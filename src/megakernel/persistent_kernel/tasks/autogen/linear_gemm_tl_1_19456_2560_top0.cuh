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
  assert(TILE_DIM_X==128); assert(TILE_DIM_Y==64); assert(TILE_DIM_Z==128);
  assert(M==1); assert(N==19456); assert(K==2560);
  
  const bfloat16* __restrict__ A = static_cast<const bfloat16* __restrict__>(input_ptr);
  const bfloat16* __restrict__ B = static_cast<const bfloat16* __restrict__>(weight_ptr);
  const bfloat16* __restrict__ C = static_cast<const bfloat16* __restrict__>(output_ptr);
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[64];
  bfloat16_t A_local[16];
  bfloat16_t B_local[32];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+(((i_1 * 20480) + ((((int)threadIdx.x) >> 4) * 2560)) + ((((int)threadIdx.x) & 15) * 8)), ((((i_1 * 8) + (((int)threadIdx.x) >> 4)) < 1) && (((i_1 * 8) + (((int)threadIdx.x) >> 4)) < 1)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+((((((int)bx) * 327680) + (i_2 * 20480)) + ((((int)threadIdx.x) >> 4) * 2560)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 19; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k + 1) & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+(((((i_3 * 20480) + ((((int)threadIdx.x) >> 4) * 2560)) + (k * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128), ((((i_3 * 8) + (((int)threadIdx.x) >> 4)) < 1) && (((i_3 * 8) + (((int)threadIdx.x) >> 4)) < 1)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((k + 1) & 1) * 32768) + (((((int)threadIdx.x) & 15) >> 3) * 16384)) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+((((((((int)bx) * 327680) + (i_4 * 20480)) + ((((int)threadIdx.x) >> 4) * 2560)) + (k * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((k & 1) * 8192) + ((ki >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + (i_5 * 8));
      }
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((k & 1) * 16384) + ((ki >> 2) * 8192)) + ((((int)threadIdx.x) >> 6) * 4096)) + (i_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])) + 0, B_local + (i_6 * 8));
      }
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_7 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_7 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_7 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((ki_1 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_8 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])) + 0, A_local + (i_8 * 8));
    }
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((ki_1 >> 2) * 8192) + ((((int)threadIdx.x) >> 6) * 4096)) + (i_9 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])) + 0, B_local + (i_9 * 8));
    }
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_10 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_10 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 32; ++i_11) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_11 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_11 >> 4) * 2048)) + ((i_11 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_11 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 8; ++i_12) {
    if (((i_12 * 8) + (((int)threadIdx.x) >> 4)) < 1) {
      *(uint4*)(C + ((((i_12 * 155648) + ((((int)threadIdx.x) >> 4) * 19456)) + (((int)bx) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((i_12 * 1024) + (((int)threadIdx.x) * 8)));
    }
  }
}


} // kernel
// Strategy: linear_gemm_tl_1_19456_2560
// selected_hparams: [64, 128, 128, 2, 128, <GemmWarpPolicy.Square: 0>, True].
// grid_dim: {'blockIdx.x': 152, 'blockIdx.y': 1, 'blockIdx.z': 1}.
// block_dim: {'threadIdx.x': 128, 'threadIdx.y': 1, 'threadIdx.z': 1}.
// smem: 98304 bytes.
// use_cooperative_groups: 0.
// latency: 0.62914, idx: 24