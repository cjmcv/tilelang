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

extern "C" __global__ void flashattn_gqa_decode_no_split_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V, const uchar* __restrict__ mask);
extern "C" __global__ void __launch_bounds__(128, 1) flashattn_gqa_decode_no_split_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V, const uchar* __restrict__ mask) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  uchar mask_local[32];
  float acc_s[64];
  half_t A_local[8];
  half_t B_local[64];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t acc_s_cast[64];
  half_t B_local_1[64];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint4 condval;
    if ((((((((int)threadIdx.x) >> 6) + ((int)blockIdx.y)) >> 1) + i) < 4)) {
      condval = *(uint4*)(Q + (((i * 1024) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
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
  for (int k = 0; k < 64; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = *(uint4*)(K + (((((k * 131072) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      mask_local[i_5] = mask[(((((k * 1024) + ((i_5 >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((i_5 & 1) * 8)) + ((int)blockIdx.y))];
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      *(float2*)(acc_s + (i_6 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((((ki >> 2) * 8192) + (i_7 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_7 * 8));
      }
      for (int j = 0; j < 8; ++j) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 64; ++i_8) {
      float condval_1;
      if ((0 < ((int)mask_local[(((i_8 >> 2) * 2) + (i_8 & 1))]))) {
        condval_1 = acc_s[i_8];
      } else {
        condval_1 = -CUDART_INF_F;
      }
      acc_s[i_8] = condval_1;
    }
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
      for (int rv = 0; rv < 32; ++rv) {
        scores_max[i_11] = max(scores_max[i_11], acc_s[((((rv & 15) * 4) + (i_11 * 2)) + (rv >> 4))]);
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
    for (int i_14 = 0; i_14 < 64; ++i_14) {
      acc_s[i_14] = exp2f(((acc_s[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_14 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_sum[i_15] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        scores_sum[i_15] = (scores_sum[i_15] + acc_s[((((rv_1 & 15) * 4) + (i_15 * 2)) + (rv_1 >> 4))]);
      }
      scores_sum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      logsum[i_16] = ((logsum[i_16] * scores_scale[i_16]) + scores_sum[i_16]);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 32; ++i_17) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_17 * 2));
      *(half2*)(&(__1)) = __float22half2_rn(*(float2*)(&(v_)));
      *(uint1*)(acc_s_cast + (i_17 * 2)) = __1;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 64; ++i_18) {
      acc_o[i_18] = (acc_o[i_18] * scores_scale[((i_18 & 3) >> 1)]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_19 = 0; i_19 < 16; ++i_19) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_19 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = *(uint4*)(V + (((((k * 131072) + (i_19 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      for (int i_20 = 0; i_20 < 8; ++i_20) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[((((((i_20 >> 2) * 8192) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])) + 0, B_local_1 + (i_20 * 8));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 64; ++i_21) {
    acc_o[i_21] = (acc_o[i_21] / logsum[((i_21 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 2; ++i_22) {
    logsum[i_22] = (log2f(logsum[i_22]) + (scores_max[i_22] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 32; ++i_23) {
    if ((((((((int)threadIdx.x) >> 5) * 4) + ((i_23 & 1) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < 1) && (((((((int)threadIdx.x) >> 5) * 4) + ((i_23 & 1) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < 1)) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_o + (i_23 * 2));
      *(half2*)(&(__2)) = __float22half2_rn(*(float2*)(&(v__1)));
      *(uint1*)(((half_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_23 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_23 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
    }
  }
  __syncthreads();
  *(uint2*)(Output + ((((int)blockIdx.y) * 512) + (((int)threadIdx.x) * 4))) = *(uint2*)(((half_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 4));
}

