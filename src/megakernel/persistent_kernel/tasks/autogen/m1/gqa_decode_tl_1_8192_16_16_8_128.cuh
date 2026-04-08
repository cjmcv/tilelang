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
__device__ __forceinline__ void flashattn_kernel_1_8192_16_16_8_128(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ v,
                                                   const void* __restrict__ edge_ptr, 
                                                   const void* __restrict__ mask_ptr, 
                                                   void* __restrict__ output_ptr,
                                                   void* __restrict__ glse_ptr,
                                                   void* __restrict__ output_partial_ptr) {
  static_assert(THREAD_NUM==128);
  static_assert(M==1); static_assert(HEAD==16); static_assert(GROUPS==8); static_assert(DIM==128);
  if constexpr (SUB_KERNEL_ID == 0) { if (bx >= 1 || by >= 8 || bz >= 1) { return; } }
  if constexpr (SUB_KERNEL_ID == 1) { if (bx >= 0 || by >= 0 || bz >= 0) { return; } }
  const bfloat16_t* __restrict__ Q = static_cast<const bfloat16_t*>(q);
  const bfloat16_t* __restrict__ K = static_cast<const bfloat16_t*>(k);
  const bfloat16_t* __restrict__ V = static_cast<const bfloat16_t*>(v);
  const int* __restrict__ edge = static_cast<const int*>(edge_ptr);
  const uchar* __restrict__ mask = static_cast<const uchar*>(mask_ptr);
  bfloat16_t* __restrict__ Output = static_cast<bfloat16_t*>(output_ptr);
  bfloat16_t* __restrict__ glse = static_cast<bfloat16_t*>(glse_ptr);
  bfloat16_t* __restrict__ Output_partial = static_cast<bfloat16_t*>(output_partial_ptr);

  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[8];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  bfloat16_t acc_s_cast[8];
  bfloat16_t A_local[8];
  bfloat16_t B_local[8];
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
  int valid_kv_seqlen = (edge[0] + 1);
  if (0 < valid_kv_seqlen) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+((((i_4 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), V+((((i_5 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    tl::cp_async_commit();
  }
  if (16 < valid_kv_seqlen) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 20480), K+(((((i_6 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 16384));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 28672), V+(((((i_7 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 16384));
    }
    tl::cp_async_commit();
  }
  for (int k = 0; k < (((valid_kv_seqlen + 15) >> 4) - 2); ++k) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      *(float2*)(acc_s + (i_8 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((k & 1) * 2048) + ((ki >> 2) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + 0);
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_9 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+((((((((int64_t)k) * (int64_t)16384) + (((int64_t)i_9) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)32768), (k < 510));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      float condval_1;
      if ((((((k * 16) + ((i_10 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_10 & 1)) < valid_kv_seqlen)) {
        condval_1 = acc_s[i_10];
      } else {
        condval_1 = -CUDART_INF_F;
      }
      acc_s[i_10] = condval_1;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      scores_max_prev[i_11] = scores_max[i_11];
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_max[i_12] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        scores_max[i_13] = max(scores_max[i_13], acc_s[((((rv & 1) * 4) + (i_13 * 2)) + (rv >> 1))]);
      }
      scores_max[i_13] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      scores_max[i_14] = max(scores_max[i_14], scores_max_prev[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_scale[i_15] = exp2f(((scores_max_prev[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      acc_s[i_16] = exp2f(((acc_s[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_16 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      scores_sum[i_17] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        scores_sum[i_17] = (scores_sum[i_17] + acc_s[((((rv_1 & 1) * 4) + (i_17 * 2)) + (rv_1 >> 1))]);
      }
      scores_sum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      logsum[i_18] = ((logsum[i_18] * scores_scale[i_18]) + scores_sum[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 4; ++i_19) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_19 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
      *(uint1*)(acc_s_cast + (i_19 * 2)) = __1;
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 64; ++i_20) {
      acc_o[i_20] = (acc_o[i_20] * scores_scale[((i_20 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int i_21 = 0; i_21 < 8; ++i_21) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((k & 1) * 2048) + ((i_21 >> 2) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local_1 + (i_21 * 8));
    }
    for (int j = 0; j < 8; ++j) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_22 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), V+((((((((int64_t)k) * (int64_t)16384) + (((int64_t)i_22) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)32768), (k < 510));
    }
    tl::cp_async_commit();
  }
  if (17 <= valid_kv_seqlen) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 4; ++i_23) {
      *(float2*)(acc_s + (i_23 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_1 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((valid_kv_seqlen + 15) & 31) >> 4) * 2048) + ((ki_1 >> 2) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + 0);
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      float condval_2;
      if ((((((((valid_kv_seqlen + 15) >> 4) * 16) + ((i_24 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1)) < (valid_kv_seqlen + 32))) {
        condval_2 = acc_s[i_24];
      } else {
        condval_2 = -CUDART_INF_F;
      }
      acc_s[i_24] = condval_2;
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_max_prev[i_25] = scores_max[i_25];
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      scores_max[i_26] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
        scores_max[i_27] = max(scores_max[i_27], acc_s[((((rv_2 & 1) * 4) + (i_27 * 2)) + (rv_2 >> 1))]);
      }
      scores_max[i_27] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      scores_max[i_28] = max(scores_max[i_28], scores_max_prev[i_28]);
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      scores_scale[i_29] = exp2f(((scores_max_prev[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 8; ++i_30) {
      acc_s[i_30] = exp2f(((acc_s[i_30] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_30 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      scores_sum[i_31] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
        scores_sum[i_31] = (scores_sum[i_31] + acc_s[((((rv_3 & 1) * 4) + (i_31 * 2)) + (rv_3 >> 1))]);
      }
      scores_sum[i_31] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_31]);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      logsum[i_32] = ((logsum[i_32] * scores_scale[i_32]) + scores_sum[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 4; ++i_33) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_s + (i_33 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__2)) = __float22bfloat162_rn(*(float2*)(&(v__1)));
      *(uint1*)(acc_s_cast + (i_33 * 2)) = __2;
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 64; ++i_34) {
      acc_o[i_34] = (acc_o[i_34] * scores_scale[((i_34 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int i_35 = 0; i_35 < 8; ++i_35) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((valid_kv_seqlen + 15) & 31) >> 4) * 2048) + ((i_35 >> 2) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_35 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local_1 + (i_35 * 8));
    }
    for (int j_1 = 0; j_1 < 8; ++j_1) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
    }
  }
  if (1 <= valid_kv_seqlen) {
    #pragma unroll
    for (int i_36 = 0; i_36 < 4; ++i_36) {
      *(float2*)(acc_s + (i_36 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((valid_kv_seqlen + 15) >> 4) + 1) & 1) * 2048) + ((ki_2 >> 2) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + 0);
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 8; ++i_37) {
      float condval_3;
      if ((((((((valid_kv_seqlen + 15) >> 4) * 16) + ((i_37 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_37 & 1)) < (valid_kv_seqlen + 16))) {
        condval_3 = acc_s[i_37];
      } else {
        condval_3 = -CUDART_INF_F;
      }
      acc_s[i_37] = condval_3;
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 2; ++i_38) {
      scores_max_prev[i_38] = scores_max[i_38];
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 2; ++i_39) {
      scores_max[i_39] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 2; ++i_40) {
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 4; ++rv_4) {
        scores_max[i_40] = max(scores_max[i_40], acc_s[((((rv_4 & 1) * 4) + (i_40 * 2)) + (rv_4 >> 1))]);
      }
      scores_max[i_40] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_40]);
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 2; ++i_41) {
      scores_max[i_41] = max(scores_max[i_41], scores_max_prev[i_41]);
    }
    #pragma unroll
    for (int i_42 = 0; i_42 < 2; ++i_42) {
      scores_scale[i_42] = exp2f(((scores_max_prev[i_42] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_42] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 8; ++i_43) {
      acc_s[i_43] = exp2f(((acc_s[i_43] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_43 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_44 = 0; i_44 < 2; ++i_44) {
      scores_sum[i_44] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 4; ++rv_5) {
        scores_sum[i_44] = (scores_sum[i_44] + acc_s[((((rv_5 & 1) * 4) + (i_44 * 2)) + (rv_5 >> 1))]);
      }
      scores_sum[i_44] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_44]);
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      logsum[i_45] = ((logsum[i_45] * scores_scale[i_45]) + scores_sum[i_45]);
    }
    #pragma unroll
    for (int i_46 = 0; i_46 < 4; ++i_46) {
      uint1 __3;
      float2 v__2 = *(float2*)(acc_s + (i_46 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__3)) = __float22bfloat162_rn(*(float2*)(&(v__2)));
      *(uint1*)(acc_s_cast + (i_46 * 2)) = __3;
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 64; ++i_47) {
      acc_o[i_47] = (acc_o[i_47] * scores_scale[((i_47 & 3) >> 1)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int i_48 = 0; i_48 < 8; ++i_48) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((valid_kv_seqlen + 15) >> 4) + 1) & 1) * 2048) + ((i_48 >> 2) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_48 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_48 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local_1 + (i_48 * 8));
    }
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_2 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_2 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_2 * 8) + 4)));
    }
  }
  #pragma unroll
  for (int i_49 = 0; i_49 < 64; ++i_49) {
    acc_o[i_49] = (acc_o[i_49] / logsum[((i_49 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_50 = 0; i_50 < 2; ++i_50) {
    logsum[i_50] = (log2f(logsum[i_50]) + (scores_max[i_50] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  __syncthreads();
  #pragma unroll
  for (int i_51 = 0; i_51 < 32; ++i_51) {
    if ((((((((int)threadIdx.x) >> 5) * 8) + ((i_51 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) && (((((((int)threadIdx.x) >> 5) * 8) + ((i_51 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1)) {
      uint1 __4;
      float2 v__3 = *(float2*)(acc_o + (i_51 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__4)) = __float22bfloat162_rn(*(float2*)(&(v__3)));
      *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_51 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_51 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __4;
    }
  }
  __syncthreads();
  *(uint1*)(Output + ((((int)by) * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
}


} // kernel
// Strategy: gqa_decode_tl_1_8192_16_16_8_128
// selected_hparams: [16, 64, 1, 2, 128].
// smem: 32768 bytes.
// use_cooperative_groups: 0.
// layout: (1, 8, 1), (16, 64, 1)
// block_dim=(128, 1, 1).
// latency: 0.0088 ms vs [ref-0.00768 sim-0.98777], idx: 1