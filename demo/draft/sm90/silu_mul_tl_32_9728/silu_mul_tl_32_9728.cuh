#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void silu_mul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(256, 1) silu_mul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ uint64_t mbarrier_mem[1];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(C_desc);
    mbarrier[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[0].expect_transaction(4096);
      tl::fence_proxy_async();
      tl::tma_load(A_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 64), 0);
      mbarrier[0].expect_transaction(4096);
      tl::fence_proxy_async();
      tl::tma_load(A_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[2048])), ((((int)blockIdx.x) * 64) + 9728), 0);
    }
    mbarrier[0].arrive();
  } else {
    tl::warpgroup_reg_alloc<240>();
    mbarrier[0].wait(0);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float4 __1;
      uint2 v_ = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((i * 512) + (((int)threadIdx.x) * 4)));
      ((float2*)(&__1))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v_)));
      ((float2*)(&__1))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v_))+1));
      float4 xi = __1;
      float4 __2;
        float4 v__1 = make_float4(0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/);
        float4 __3;
          float4 __4;
          float4 __5;
            float4 v__2 = make_float4(-0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/);
            __5.x = (xi.x*v__2.x);
            __5.y = (xi.y*v__2.y);
            __5.z = (xi.z*v__2.z);
            __5.w = (xi.w*v__2.w);
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
          __8.x = (xi.x*sig.x);
          __8.y = (xi.y*sig.y);
          __8.z = (xi.z*sig.z);
          __8.w = (xi.w*sig.w);
        float4 __9;
        uint2 v__3 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 2048));
        ((float2*)(&__9))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__3)));
        ((float2*)(&__9))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__3))+1));
        __7.x = (__8.x*__9.x);
        __7.y = (__8.y*__9.y);
        __7.z = (__8.z*__9.z);
        __7.w = (__8.w*__9.w);
      (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(*(float2*)(&(__7)));
      (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(*((float2*)(&(__7))+1));
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 4096)) = __6;
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[4096])), (((int)blockIdx.x) * 64), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


// Strategy: silu_mul_tl_32_9728
// selected_hparams: [32, 64, 128].
// smem: 12288 bytes.
// use_cooperative_groups: 0.
// layout: (152, 1, 1), (64, 32, 1)
// block_dim=(256, 1, 1).
// latency: 0 ms vs [ref-0 sim-0], idx: -1