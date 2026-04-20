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

extern "C" __global__ void __launch_bounds__(128, 1) 
linear_kernel(__grid_constant__ const CUtensorMap A_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ uint64_t mbarrier_mem[1];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  
  if (threadIdx.x == 0) {
    mbarrier[0].init(128);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    mbarrier[0].expect_transaction(2048);
    tl::fence_proxy_async();
    tl::tma_load(A_desc, mbarrier[0], &(((bfloat16_t*)buf_dyn_shmem)[0]), 0, 0);
  }
}
// extern "C" __global__ void linear_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
// extern "C" __global__ void __launch_bounds__(128, 1) linear_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
//   extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
//   float C_local[8];
//   bfloat16_t A_local[8];
//   bfloat16_t B_local[8];
//   __shared__ uint64_t mbarrier_mem[6];
//   auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
//   if (tl::tl_shuffle_elect<0>()) {
//     tl::prefetch_tma_descriptor(A_desc);
//     // tl::prefetch_tma_descriptor(B_desc);
//     // tl::prefetch_tma_descriptor(C_desc);
//     mbarrier[0].init(128);
//     mbarrier[1].init(128);
//     mbarrier[2].init(128);
//     mbarrier[3].init(128);
//     mbarrier[4].init(128);
//     mbarrier[5].init(128);
//   }
//   tl::fence_barrier_init();
//   __syncthreads();
//   const dim3 blockIdx = tl::rasterization2DRow<10>();
//   #pragma unroll
//   for (int i = 0; i < 4; ++i) {
//     *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
//   }
//   int k=0;
//   mbarrier[(k % 3)].expect_transaction(2048);
//   tl::fence_proxy_async();
//   tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 1024) + 12288)])), (k * 64), 0);

//   // for (int k = 0; k < 16; ++k) {
//   //   mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
//   //   __syncthreads();
//   //   if (((int)threadIdx.x) == 0) {
//   //     mbarrier[(k % 3)].expect_transaction(2048);
//   //     tl::fence_proxy_async();
//   //     tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 1024) + 12288)])), (k * 64), 0);
//   //     mbarrier[(k % 3)].expect_transaction(8192);
//   //     tl::fence_proxy_async();
//   //     tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 4096)])), (k * 64), (((int)blockIdx.x) * 64));
//   //   }
//   //   mbarrier[(k % 3)].arrive();
//   //   mbarrier[(k % 3)].wait(((k % 6) / 3));
//   //   __syncthreads();
//   //   for (int ki = 0; ki < 4; ++ki) {
//   //     tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((k % 3) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, A_local + 0);
//   //     tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((k % 3) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + 0);
//   //     tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
//   //     tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
//   //   }
//   //   tl::mbarrier_cp_async_arrive(mbarrier[((k % 3) + 3)]);
//   //   mbarrier[((k % 3) + 3)].arrive();
//   // }
//   // __syncthreads();
//   // tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)C_local[0]), ((bfloat16_t)C_local[1])), __pack_half2(((bfloat16_t)C_local[2]), ((bfloat16_t)C_local[3])), __pack_half2(((bfloat16_t)C_local[4]), ((bfloat16_t)C_local[5])), __pack_half2(((bfloat16_t)C_local[6]), ((bfloat16_t)C_local[7])));
//   // tl::fence_proxy_async();
//   // __syncthreads();
//   // if (((int)threadIdx.x) == 0) {
//   //   tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 64), 0);
//   //   tl::tma_store_arrive();
//   //   tl::tma_store_wait<0>();
//   // }
// }


// Strategy: linear_gemm_tl_1_6144_1024
// selected_hparams: [16, 64, 64, 1, 3, 128, 0, True].
// smem: 30720 bytes.
// use_cooperative_groups: 0.
// layout: (96, 1, 1), (64, 16, 64)
// block_dim=(128, 1, 1).
// latency: 0 ms vs [ref-0 sim-0], idx: 28