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
__device__ __forceinline__ void flashattn_kernel_1_8192_2_16_8_128(const int bx, const int by, const int bz,
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
  float acc_s[8];
  bfloat16_t A_local[8];
  bfloat16_t B_local[8];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  bfloat16_t acc_s_cast[8];
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
  *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) >> 6) * 64)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)) = *(uint1*)(K + ((((((int)threadIdx.x) >> 6) * 1024) + (((int)by) * 128)) + ((((int)threadIdx.x) & 63) * 2)));
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(float2*)(acc_s + (i_4 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  __syncthreads();
  for (int ki = 0; ki < 8; ++ki) {
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + 0);
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    float condval_1;
    if (((((i_5 >> 2) * 4) + (((int)threadIdx.x) & 3)) < 1)) {
      condval_1 = acc_s[i_5];
    } else {
      condval_1 = -CUDART_INF_F;
    }
    acc_s[i_5] = condval_1;
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    scores_max_prev[i_6] = scores_max[i_6];
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    scores_max[i_7] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 2; ++i_8) {
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      scores_max[i_8] = max(scores_max[i_8], acc_s[((((rv & 1) * 4) + (i_8 * 2)) + (rv >> 1))]);
    }
    scores_max[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_8]);
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 2; ++i_9) {
    scores_max[i_9] = max(scores_max[i_9], scores_max_prev[i_9]);
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 2; ++i_10) {
    scores_scale[i_10] = exp2f(((scores_max_prev[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 8; ++i_11) {
    acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_11 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 2; ++i_12) {
    scores_sum[i_12] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      scores_sum[i_12] = (scores_sum[i_12] + acc_s[((((rv_1 & 1) * 4) + (i_12 * 2)) + (rv_1 >> 1))]);
    }
    scores_sum[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_12]);
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    logsum[i_13] = ((logsum[i_13] * scores_scale[i_13]) + scores_sum[i_13]);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 4; ++i_14) {
    uint1 __1;
    float2 v_ = *(float2*)(acc_s + (i_14 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
    *(uint1*)(acc_s_cast + (i_14 * 2)) = __1;
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 64; ++i_15) {
    acc_o[i_15] = (acc_o[i_15] * scores_scale[((i_15 & 3) >> 1)]);
  }
  __syncthreads();
  *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) >> 6) * 64)) + (((((int)threadIdx.x) & 31) >> 3) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(V + ((((((int)threadIdx.x) >> 6) * 1024) + (((int)by) * 128)) + ((((int)threadIdx.x) & 63) * 2)));
  __syncthreads();
  for (int i_16 = 0; i_16 < 8; ++i_16) {
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((i_16 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, B_local_1 + (i_16 * 8));
  }
  for (int j = 0; j < 8; ++j) {
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 64; ++i_17) {
    acc_o[i_17] = (acc_o[i_17] / logsum[((i_17 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    logsum[i_18] = (log2f(logsum[i_18]) + (scores_max[i_18] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  __syncthreads();
  #pragma unroll
  for (int i_19 = 0; i_19 < 32; ++i_19) {
    if ((((((((int)threadIdx.x) >> 5) * 8) + ((i_19 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) && (((((((int)threadIdx.x) >> 5) * 8) + ((i_19 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1)) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_o + (i_19 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__2)) = __float22bfloat162_rn(*(float2*)(&(v__1)));
      *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_19 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_19 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
    }
  }
  __syncthreads();
  *(uint1*)(Output + ((((int)by) * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
}


} // kernel
// Strategy: gqa_decode_tl_1_8192_2_16_8_128
// selected_hparams: [16, 64, 1, 1, 128].
// smem: 20480 bytes.
// use_cooperative_groups: 0.
// layout: (1, 8, 1), (16, 64, 1)
// block_dim=(128, 1, 1).
// latency: 0.00797 ms vs [ref-0.00797 sim-1.0], idx: -1