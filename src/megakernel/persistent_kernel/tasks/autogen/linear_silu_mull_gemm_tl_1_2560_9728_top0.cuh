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
    __device__ __forceinline__ void silu_mul_linear_kernel_1_2560_9728(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr,
                                                const void* __restrict__ weight_ptr,
                                                const void* __restrict__ residual_ptr,
                                                void* __restrict__ output_ptr,
                                                int num_active_tokens,
                                                bool residual) {
  // printf("hello silu_mul_linear_kernel_1_2560_9728.\n");
  static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==128); static_assert(TILE_DIM_Y==64); static_assert(TILE_DIM_Z==64);
  static_assert(M==1); static_assert(N==2560); static_assert(K==9728);
  
  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  const bfloat16_t* __restrict__ B = static_cast<const bfloat16_t*>(weight_ptr);
  bfloat16_t* __restrict__ C = static_cast<bfloat16_t*>(output_ptr);
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[64];
  bfloat16_t A_local[16];
  bfloat16_t B_local[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((i_1 * 2048) + (((int)threadIdx.x) * 16)), A+(((i_1 * 311296) + ((((int)threadIdx.x) >> 3) * 19456)) + ((((int)threadIdx.x) & 7) * 8)), ((((i_1 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_1 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_2 * 2048) + (((int)threadIdx.x) * 16)) + 16384), A+((((i_2 * 311296) + ((((int)threadIdx.x) >> 3) * 19456)) + ((((int)threadIdx.x) & 7) * 8)) + 9728), ((((i_2 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_2 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_3 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+((((((int)bx) * 1245184) + (i_3 * 155648)) + ((((int)threadIdx.x) >> 3) * 9728)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_4 * 2048) + (((int)threadIdx.x) * 16)) + 8192), A+((((i_4 * 311296) + ((((int)threadIdx.x) >> 3) * 19456)) + ((((int)threadIdx.x) & 7) * 8)) + 64), ((((i_4 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_4 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_5 * 2048) + (((int)threadIdx.x) * 16)) + 24576), A+((((i_5 * 311296) + ((((int)threadIdx.x) >> 3) * 19456)) + ((((int)threadIdx.x) & 7) * 8)) + 9792), ((((i_5 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_5 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_6 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 49152), B+(((((((int)bx) * 1245184) + (i_6 * 155648)) + ((((int)threadIdx.x) >> 3) * 9728)) + ((((int)threadIdx.x) & 7) * 8)) + 64));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 150; ++k) {
    tl::cp_async_wait<1>();
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      float4 __1;
      uint2 v_ = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((k & 1) * 4096) + (i_7 * 512)) + (((int)threadIdx.x) * 4)));
      ((float2*)(&__1))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v_)));
      ((float2*)(&__1))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v_))+1));
      float4 x = __1;
      float4 __2;
        float4 v__1 = make_float4(0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/);
        float4 __3;
          float4 __4;
          float4 __5;
            float4 v__2 = make_float4(-0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/);
            __5.x = (x.x*v__2.x);
            __5.y = (x.y*v__2.y);
            __5.z = (x.z*v__2.z);
            __5.w = (x.w*v__2.w);
          __4.x = expf(__5.x);
          __4.y = expf(__5.y);
          __4.z = expf(__5.z);
          __4.w = expf(__5.w);
          __3.x = (v__1.x+__4.x);
          __3.y = (v__1.y+__4.y);
          __3.z = (v__1.z+__4.z);
          __3.w = (v__1.w+__4.w);
        __2.x = (v__1.x/__3.x);
        __2.y = (v__1.y/__3.y);
        __2.z = (v__1.z/__3.z);
        __2.w = (v__1.w/__3.w);
      float4 sig = __2;
      uint2 __6;
      float4 __7;
        float4 __8;
          __8.x = (x.x*sig.x);
          __8.y = (x.y*sig.y);
          __8.z = (x.z*sig.z);
          __8.w = (x.w*sig.w);
        float4 __9;
        uint2 v__3 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((k & 1) * 4096) + (i_7 * 512)) + (((int)threadIdx.x) * 4)) + 8192));
        ((float2*)(&__9))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__3)));
        ((float2*)(&__9))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__3))+1));
        __7.x = (__8.x*__9.x);
        __7.y = (__8.y*__9.y);
        __7.z = (__8.z*__9.z);
        __7.w = (__8.w*__9.w);
      (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(*(float2*)(&(__7)));
      (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(*((float2*)(&(__7))+1));
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_7 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 32768)) = __6;
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((k & 1) * 8192) + (i_8 * 2048)) + (((int)threadIdx.x) * 16)), A+(((((i_8 * 311296) + ((((int)threadIdx.x) >> 3) * 19456)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128), ((((i_8 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_8 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((k & 1) * 8192) + (i_9 * 2048)) + (((int)threadIdx.x) * 16)) + 16384), A+(((((i_9 * 311296) + ((((int)threadIdx.x) >> 3) * 19456)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 9856), ((((i_9 * 16) + (((int)threadIdx.x) >> 3)) < 1) && (((i_9 * 16) + (((int)threadIdx.x) >> 3)) < 1)));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_10 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])) + 0, A_local + (i_10 * 8));
      }
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((k & 1) * 8192) + ((((int)threadIdx.x) >> 6) * 4096)) + (i_11 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])) + 0, B_local + (i_11 * 8));
      }
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_12 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_12 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((k & 1) * 16384) + (i_13 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+((((((((int)bx) * 1245184) + (i_13 * 155648)) + ((((int)threadIdx.x) >> 3) * 9728)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_14 = 0; i_14 < 8; ++i_14) {
    float4 __10;
    uint2 v__4 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((i_14 * 512) + (((int)threadIdx.x) * 4)));
    ((float2*)(&__10))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__4)));
    ((float2*)(&__10))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__4))+1));
    float4 x_1 = __10;
    float4 __11;
      float4 v__5 = make_float4(0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/);
      float4 __12;
        float4 __13;
        float4 __14;
          float4 v__6 = make_float4(-0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/);
          __14.x = (x_1.x*v__6.x);
          __14.y = (x_1.y*v__6.y);
          __14.z = (x_1.z*v__6.z);
          __14.w = (x_1.w*v__6.w);
        __13.x = expf(__14.x);
        __13.y = expf(__14.y);
        __13.z = expf(__14.z);
        __13.w = expf(__14.w);
        __12.x = (v__5.x+__13.x);
        __12.y = (v__5.y+__13.y);
        __12.z = (v__5.z+__13.z);
        __12.w = (v__5.w+__13.w);
      __11.x = (v__5.x/__12.x);
      __11.y = (v__5.y/__12.y);
      __11.z = (v__5.z/__12.z);
      __11.w = (v__5.w/__12.w);
    float4 sig_1 = __11;
    uint2 __15;
    float4 __16;
      float4 __17;
        __17.x = (x_1.x*sig_1.x);
        __17.y = (x_1.y*sig_1.y);
        __17.z = (x_1.z*sig_1.z);
        __17.w = (x_1.w*sig_1.w);
      float4 __18;
      uint2 v__7 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_14 * 512) + (((int)threadIdx.x) * 4)) + 8192));
      ((float2*)(&__18))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__7)));
      ((float2*)(&__18))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__7))+1));
      __16.x = (__17.x*__18.x);
      __16.y = (__17.y*__18.y);
      __16.z = (__17.z*__18.z);
      __16.w = (__17.w*__18.w);
    (reinterpret_cast<__nv_bfloat162*>(&__15))[0] = __float22bfloat162_rn(*(float2*)(&(__16)));
    (reinterpret_cast<__nv_bfloat162*>(&__15))[1] = __float22bfloat162_rn(*((float2*)(&(__16))+1));
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_14 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 32768)) = __15;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_15 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])) + 0, A_local + (i_15 * 8));
    }
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 4096) + (i_16 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])) + 0, B_local + (i_16 * 8));
    }
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_17 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_17 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 8; ++i_18) {
    float4 __19;
    uint2 v__8 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_18 * 512) + (((int)threadIdx.x) * 4)) + 4096));
    ((float2*)(&__19))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__8)));
    ((float2*)(&__19))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__8))+1));
    float4 x_2 = __19;
    float4 __20;
      float4 v__9 = make_float4(0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/);
      float4 __21;
        float4 __22;
        float4 __23;
          float4 v__10 = make_float4(-0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/);
          __23.x = (x_2.x*v__10.x);
          __23.y = (x_2.y*v__10.y);
          __23.z = (x_2.z*v__10.z);
          __23.w = (x_2.w*v__10.w);
        __22.x = expf(__23.x);
        __22.y = expf(__23.y);
        __22.z = expf(__23.z);
        __22.w = expf(__23.w);
        __21.x = (v__9.x+__22.x);
        __21.y = (v__9.y+__22.y);
        __21.z = (v__9.z+__22.z);
        __21.w = (v__9.w+__22.w);
      __20.x = (v__9.x/__21.x);
      __20.y = (v__9.y/__21.y);
      __20.z = (v__9.z/__21.z);
      __20.w = (v__9.w/__21.w);
    float4 sig_2 = __20;
    uint2 __24;
    float4 __25;
      float4 __26;
        __26.x = (x_2.x*sig_2.x);
        __26.y = (x_2.y*sig_2.y);
        __26.z = (x_2.z*sig_2.z);
        __26.w = (x_2.w*sig_2.w);
      float4 __27;
      uint2 v__11 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_18 * 512) + (((int)threadIdx.x) * 4)) + 12288));
      ((float2*)(&__27))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__11)));
      ((float2*)(&__27))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__11))+1));
      __25.x = (__26.x*__27.x);
      __25.y = (__26.y*__27.y);
      __25.z = (__26.z*__27.z);
      __25.w = (__26.w*__27.w);
    (reinterpret_cast<__nv_bfloat162*>(&__24))[0] = __float22bfloat162_rn(*(float2*)(&(__25)));
    (reinterpret_cast<__nv_bfloat162*>(&__24))[1] = __float22bfloat162_rn(*((float2*)(&(__25))+1));
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_18 * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 32768)) = __24;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_19 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])) + 0, A_local + (i_19 * 8));
    }
    for (int i_20 = 0; i_20 < 4; ++i_20) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 4096) + (i_20 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])) + 0, B_local + (i_20 * 8));
    }
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_21 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_21 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_22 = 0; i_22 < 32; ++i_22) {
    uint1 __28;
    float2 v__12 = *(float2*)(C_local + (i_22 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__28)) = __float22bfloat162_rn(*(float2*)(&(v__12)));
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_22 >> 4) * 2048)) + ((i_22 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_22 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __28;
  }
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 8; ++i_23) {
    if (((i_23 * 8) + (((int)threadIdx.x) >> 4)) < 1) {
      *(uint4*)(C + ((((i_23 * 20480) + ((((int)threadIdx.x) >> 4) * 2560)) + (((int)bx) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((i_23 * 1024) + (((int)threadIdx.x) * 8)));
    }
  }
}


} // kernel
// Strategy: linear_silu_mull_gemm_tl_1_2560_9728
// selected_hparams: [64, 128, 64, 1, 2, 128, <GemmWarpPolicy.Square: 0>, False].
// smem: 73728 bytes.
// use_cooperative_groups: 0.
// grid_dim=(20, 1, 1), tile_dim=(128, 64, 64),
// block_dim=(128, 1, 1).
// latency: 0.45269, idx: 41