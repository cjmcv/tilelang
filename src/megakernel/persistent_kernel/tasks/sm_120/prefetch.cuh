
#pragma once

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

template <typename T, int THREAD_NUM>
__device__ __forceinline__ void prefetch_kernel(const int bx, const int by, const int bz,
                                                uint64_t* mbarrier_mem, const CUtensorMap *A_desc, const CUtensorMap *B_desc, 
                                                const void* __restrict__ residual_ptr, const CUtensorMap *C_desc, 
                                                int num_active_tokens,
                                                bool residual) {
  // if (threadIdx.x == 0)
  //   printf("prefetch_kernel(%d): %lld, %lld, %lld.\n", bx, A_desc, B_desc, C_desc);
  // const bfloat16_t* __restrict__ R = static_cast<const bfloat16_t*>(residual_ptr);
  // extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  // float C_local[8];
  // bfloat16_t A_local[8];
  // bfloat16_t B_local[8];
  // // __shared__ uint64_t mbarrier_mem[6];
  // auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  // if (tl::tl_shuffle_elect<0>()) {
  //   tl::prefetch_tma_descriptor(*A_desc);
  //   tl::prefetch_tma_descriptor(*B_desc);
  //   tl::prefetch_tma_descriptor(*C_desc);
  //   mbarrier[0].init(128);
  //   mbarrier[1].init(128);
  //   mbarrier[2].init(128);
  //   mbarrier[3].init(128);
  //   mbarrier[4].init(128);
  //   mbarrier[5].init(128);
  // }
  // tl::fence_barrier_init();
  // __syncthreads();
  // if (128 <= ((int)threadIdx.x)) {
  //   tl::warpgroup_reg_dealloc<24>();
  //   for (int k = 0; k < 1; ++k) {
  //     mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
  //     if (tl::tl_shuffle_elect<128>()) {
  //       mbarrier[(k % 3)].expect_transaction(4096);
  //       tl::fence_proxy_async();
  //       tl::tma_load(*A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 2048) + 24576)])), (k * 128), 0);
  //       tl::tma_load(*A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 2048) + 25600)])), ((k * 128) + 64), 0);
  //       mbarrier[(k % 3)].expect_transaction(16384);
  //       tl::fence_proxy_async();
  //       tl::tma_load(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 8192)])), (k * 128), (((int)bx) * 64));
  //       tl::tma_load(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 4096)])), ((k * 128) + 64), (((int)bx) * 64));
  //     }
  //     mbarrier[(k % 3)].arrive();
  //   }
  // }
  // printf("1");
}

} // namespace kernel
