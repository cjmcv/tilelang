
#pragma once

#include "m1/rms_norm_tl_1_1024.cuh"
#include "m1/rms_norm_tl_24_128.cuh"

#ifdef ENABLE_QWEN3_4B
#include "m1/rms_norm_tl_1_2560.cuh"
#include "m32/rms_norm_tl_32_2560.cuh"
#endif
namespace kernel {

template <typename T,
  int THREAD_NUM,
  int TILE_DIM_X, 
  int TILE_DIM_Y, 
  int TILE_DIM_Z,
  int M,
  int N>
__device__ __forceinline__ void rms_norm_kernel(const int bx, const int by, const int bz,
                                              void const *input_ptr,
                                              void const *weight_ptr,
                                              void *output_ptr,
                                              float eps) {
#ifdef ENABLE_QWEN3_06B
  if constexpr (M == 1) { 
    if constexpr (N == 1024) {
      rms_norm_kernel_1_1024<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N>(bx, by, bz, input_ptr, weight_ptr, output_ptr, eps); return;
    }
  }
  else if constexpr (M == 24) { 
    if constexpr (N == 128) {
      rms_norm_kernel_16_8_128<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N>(bx, by, bz, input_ptr, weight_ptr, output_ptr, eps); return;
    }
  }
  // else if constexpr (M == 8) { 
  //   if constexpr (N == 128) {
  //     rms_norm_kernel_8_128<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N>(bx, by, bz, input_ptr, weight_ptr, output_ptr, eps); return;
  //   }
  // }
  // else if constexpr (M == 16) { 
  //   if constexpr (N == 128) {
  //     rms_norm_kernel_16_128<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N>(bx, by, bz, input_ptr, weight_ptr, output_ptr, eps); return;
  //   }
  // }
#endif
  
#ifdef ENABLE_QWEN3_4B
  if constexpr (M == 1) { 
    if constexpr (N == 2560) {
      rms_norm_kernel_1_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N>(bx, by, bz, input_ptr, weight_ptr, output_ptr, eps); return;
    }
  }
  else if constexpr (M == 32) { 
    if constexpr (N == 2560) {
      rms_norm_kernel_32_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N>(bx, by, bz, input_ptr, weight_ptr, output_ptr, eps);
    }
  }
#endif

  printf("Error: [rms_norm_kernel_%d_%d] There is no suitable microkernel!\n", M,N);
}

} // namespace kernel
