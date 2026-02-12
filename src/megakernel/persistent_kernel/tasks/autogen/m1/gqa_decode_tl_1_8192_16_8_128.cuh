#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
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
          int SUB_KERNEL_ID,
          int M, 
          int HEAD,
          int GROUPS,
          int DIM>
__device__ __forceinline__ void flashattn_kernel_1_8192_16_8_128__0(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ v,
                                                   const void* __restrict__ mask_ptr, 
                                                   void* __restrict__ output_ptr,
                                                   void* __restrict__ glse_ptr,
                                                   void* __restrict__ output_partial_ptr) {
  static_assert(THREAD_NUM==128);
  static_assert(M==1); static_assert(HEAD==16); static_assert(GROUPS==8); static_assert(DIM==128);
  
  const bfloat16_t* __restrict__ Q = static_cast<const bfloat16_t*>(q);
  const bfloat16_t* __restrict__ K = static_cast<const bfloat16_t*>(k);
  const bfloat16_t* __restrict__ V = static_cast<const bfloat16_t*>(v);
  const uchar* __restrict__ mask = static_cast<const uchar*>(mask_ptr);
  bfloat16_t* __restrict__ Output = static_cast<bfloat16_t*>(output_ptr);
  bfloat16_t* __restrict__ glse = static_cast<bfloat16_t*>(glse_ptr);
  bfloat16_t* __restrict__ Output_partial = static_cast<bfloat16_t*>(output_partial_ptr);

  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  bfloat16_t acc_s_cast[32];
  bfloat16_t A_local[8];
  bfloat16_t B_local[32];
  bfloat16_t B_local_1[64];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint4 condval;
    if (((((((((int)threadIdx.x) >> 5) + ((int)by)) >> 2) + i) < 2) && (((((((int)threadIdx.x) >> 5) + ((int)by)) >> 2) + i) < 2))) {
      condval = *(uint4*)(Q + (((i * 1024) + (((int)by) * 256)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(float2*)(acc_o + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    logsum[i_2] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    scores_max[i_3] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((int)bz) * 2097152) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), V+(((((((int)bz) * 2097152) + (i_5 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 31; ++k) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      *(float2*)(acc_s + (i_6 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki >> 2) * 4096) + (i_7 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_7 * 8));
      }
      for (int j = 0; j < 4; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_8 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((((int)bz) * 2097152) + (k * 65536)) + (i_8 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_max_prev[i_9] = scores_max[i_9];
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      scores_max[i_10] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max[i_11] = max(scores_max[i_11], acc_s[((((rv & 7) * 4) + (i_11 * 2)) + (rv >> 3))]);
      }
      scores_max[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_max[i_12] = max(scores_max[i_12], scores_max_prev[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      scores_scale[i_13] = exp2f(((scores_max_prev[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      acc_s[i_14] = exp2f(((acc_s[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_14 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_sum[i_15] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_15] = (scores_sum[i_15] + acc_s[((((rv_1 & 7) * 4) + (i_15 * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      logsum[i_16] = ((logsum[i_16] * scores_scale[i_16]) + scores_sum[i_16]);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 16; ++i_17) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_17 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
      *(uint1*)(acc_s_cast + (i_17 * 2)) = __1;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 64; ++i_18) {
      acc_o[i_18] = (acc_o[i_18] * scores_scale[((i_18 & 3) >> 1)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_19 >> 2) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_19 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_1 + (i_19 * 8));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_20 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), V+(((((((((int)bz) * 2097152) + (k * 65536)) + (i_20 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536));
    }
    tl::cp_async_commit();
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 16; ++i_21) {
    *(float2*)(acc_s + (i_21 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
    for (int i_22 = 0; i_22 < 4; ++i_22) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_2 >> 2) * 4096) + (i_22 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_22 * 8));
    }
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
    }
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 2; ++i_23) {
    scores_max_prev[i_23] = scores_max[i_23];
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 2; ++i_24) {
    scores_max[i_24] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      scores_max[i_25] = max(scores_max[i_25], acc_s[((((rv_2 & 7) * 4) + (i_25 * 2)) + (rv_2 >> 3))]);
    }
    scores_max[i_25] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_25]);
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    scores_max[i_26] = max(scores_max[i_26], scores_max_prev[i_26]);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    scores_scale[i_27] = exp2f(((scores_max_prev[i_27] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_27] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 32; ++i_28) {
    acc_s[i_28] = exp2f(((acc_s[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_28 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 2; ++i_29) {
    scores_sum[i_29] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      scores_sum[i_29] = (scores_sum[i_29] + acc_s[((((rv_3 & 7) * 4) + (i_29 * 2)) + (rv_3 >> 3))]);
    }
    scores_sum[i_29] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_29]);
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 2; ++i_30) {
    logsum[i_30] = ((logsum[i_30] * scores_scale[i_30]) + scores_sum[i_30]);
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 16; ++i_31) {
    uint1 __2;
    float2 v__1 = *(float2*)(acc_s + (i_31 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__2)) = __float22bfloat162_rn(*(float2*)(&(v__1)));
    *(uint1*)(acc_s_cast + (i_31 * 2)) = __2;
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 64; ++i_32) {
    acc_o[i_32] = (acc_o[i_32] * scores_scale[((i_32 & 3) >> 1)]);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
    for (int i_33 = 0; i_33 < 8; ++i_33) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_33 >> 2) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_33 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_1 + (i_33 * 8));
    }
    for (int j_3 = 0; j_3 < 8; ++j_3) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 8) + 4)));
    }
  }
  #pragma unroll
  for (int i_34 = 0; i_34 < 64; ++i_34) {
    acc_o[i_34] = (acc_o[i_34] / logsum[((i_34 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_35 = 0; i_35 < 2; ++i_35) {
    logsum[i_35] = (log2f(logsum[i_35]) + (scores_max[i_35] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_36 = 0; i_36 < 2; ++i_36) {
      if (((((((int)threadIdx.x) >> 5) * 8) + (i_36 * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) {
        glse[((((((((int)threadIdx.x) >> 5) * 64) + (i_36 * 32)) + (((int)by) * 8)) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + ((int)bz))] = ((bfloat16_t)logsum[i_36]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_37 = 0; i_37 < 32; ++i_37) {
    if ((((((((int)threadIdx.x) >> 5) * 8) + ((i_37 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) && (((((((int)threadIdx.x) >> 5) * 8) + ((i_37 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1)) {
      uint1 __3;
      float2 v__2 = *(float2*)(acc_o + (i_37 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__3)) = __float22bfloat162_rn(*(float2*)(&(v__2)));
      *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_37 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_37 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __3;
    }
  }
  __syncthreads();
  *(uint1*)(Output_partial + ((((((int)by) * 1024) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)bz) * 128)) + ((((int)threadIdx.x) & 63) * 2))) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
}


template <typename T,
          int THREAD_NUM,
          int SUB_KERNEL_ID,
          int M, 
          int HEAD,
          int GROUPS,
          int DIM>
__device__ __forceinline__ void flashattn_kernel_1_8192_16_8_128__1(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ v,
                                                   const void* __restrict__ mask_ptr, 
                                                   void* __restrict__ output_ptr,
                                                   void* __restrict__ glse_ptr,
                                                   void* __restrict__ output_partial_ptr) {
  static_assert(THREAD_NUM==128);
  static_assert(M==1); static_assert(HEAD==16); static_assert(GROUPS==8); static_assert(DIM==128);
  
  const bfloat16_t* __restrict__ Q = static_cast<const bfloat16_t*>(q);
  const bfloat16_t* __restrict__ K = static_cast<const bfloat16_t*>(k);
  const bfloat16_t* __restrict__ V = static_cast<const bfloat16_t*>(v);
  const uchar* __restrict__ mask = static_cast<const uchar*>(mask_ptr);
  bfloat16_t* __restrict__ Output = static_cast<bfloat16_t*>(output_ptr);
  bfloat16_t* __restrict__ glse = static_cast<bfloat16_t*>(glse_ptr);
  bfloat16_t* __restrict__ Output_partial = static_cast<bfloat16_t*>(output_partial_ptr);

  float lse_logsum_local[1];
  float o_accum_local[1];
  bfloat16_t lse_local[4];
  float lse_max_local[1];
  bfloat16_t po_local[1];
  float scale_local[1];
  lse_logsum_local[0] = 0x0p+0f/*0.000000e+00*/;
  o_accum_local[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    lse_local[i] = glse[((((int)bx) * 4) + i)];
  }
  lse_max_local[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    lse_max_local[0] = max(lse_max_local[0], ((float)lse_local[rv]));
  }
  for (int k = 0; k < 4; ++k) {
    lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((((float)lse_local[k]) - lse_max_local[0])));
  }
  lse_logsum_local[0] = (log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_1 = 0; k_1 < 4; ++k_1) {
    po_local[0] = Output_partial[(((((int)bx) * 512) + (k_1 * 128)) + ((int)threadIdx.x))];
    scale_local[0] = exp2f((((float)lse_local[k_1]) - lse_logsum_local[0]));
    o_accum_local[0] = (o_accum_local[0] + (((float)po_local[0]) * scale_local[0]));
  }
  Output[((((int)bx) * 128) + ((int)threadIdx.x))] = ((bfloat16_t)o_accum_local[0]);
}


} // kernel
// Strategy: gqa_decode_tl_1_8192_16_8_128
// selected_hparams: [64, 64, 4, 1, 128].
// smem: 0 bytes.
// use_cooperative_groups: 0.
// layout: (1, 8, 4), (64, 64, 4), (16, 1, 1), (64, 64, 4)
// block_dim=(128, 1, 1).
// latency: 0.26164