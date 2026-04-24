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
        int N>
__device__ __forceinline__ void rms_norm_kernel_1_1024(const int bx, const int by, const int bz,
                                                            const CUtensorMap *A_desc, const CUtensorMap *B_desc, const CUtensorMap *C_desc, ,
                                                            float eps) {
  static_assert(THREAD_NUM==256);
  static_assert(TILE_DIM_X==1); static_assert(TILE_DIM_Y==1); static_assert(TILE_DIM_Z==1);
  static_assert(M==1); static_assert(N==1024);
  if (bx >= 1 || by >= 1 || bz >= 1) { return; }
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float A_local[4];
  float B_local[4];
  float A_pow_local[4];
  float A_powsum[1];
  __shared__ uint64_t mbarrier_mem[2];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[0].arrive_and_expect_tx(2048);
      tl::fence_proxy_async();
      tl::tma_load((&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(A[0])), mbarrier[0], 2048);
      mbarrier[1].arrive_and_expect_tx(2048);
      tl::fence_proxy_async();
      tl::tma_load((&(((bfloat16_t*)buf_dyn_shmem)[1024])), (&(B[0])), mbarrier[1], 2048);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    mbarrier[0].wait(0);
    float4 __1;
    uint2 v_ = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 4));
    ((float2*)(&__1))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v_)));
    ((float2*)(&__1))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v_))+1));
    *(float4*)(A_local + 0) = __1;
    mbarrier[1].wait(0);
    float4 __2;
    uint2 v__1 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 4) + 1024));
    ((float2*)(&__2))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__1)));
    ((float2*)(&__2))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__1))+1));
    *(float4*)(B_local + 0) = __2;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      A_pow_local[i] = (A_local[i] * A_local[i]);
    }
    A_powsum[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      A_powsum[0] = (A_powsum[0] + A_pow_local[rv]);
    }
    tl::__sync_thread_partial<3, 256>();
    A_powsum[0] = tl::AllReduce<tl::SumOp, 256, 1, 0>::run(A_powsum[0], (&(((float*)buf_dyn_shmem)[0])));
    A_powsum[0] = rsqrtf(((A_powsum[0] / 0x1p+10f/*1.024000e+03*/) + 0x1.19799812dea11p-40f/*1.000000e-12*/));
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      A_local[i_1] = (A_local[i_1] * (A_powsum[0] * B_local[i_1]));
    }
    uint2 __3;
    float4 v__2 = *(float4*)(A_local + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(*(float2*)(&(v__2)));
    (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(*((float2*)(&(v__2))+1));
    *(uint2*)(C + (((int)threadIdx.x) * 4)) = __3;
  }
}


} // kernel
} // kernel
// Strategy: rms_norm_tl_1_1024
// selected_hparams: [1, 1, 256].
// smem: 4096 bytes.
// use_cooperative_groups: 0.
// layout: (1, 1, 1), (1, 1, 1)
// block_dim=(384, 1, 1).
// latency: 0 ms vs [ref-0 sim-0], idx: -1