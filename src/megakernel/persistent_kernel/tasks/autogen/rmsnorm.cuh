
#pragma once

#include "rms_norm_tl_1_2560.cuh"

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
  if constexpr (M == 1 && N == 2560) {                                              
    rms_norm_kernel_1_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N>(bx, by, bz, input_ptr, weight_ptr, output_ptr, eps);
  }
  else {
    printf("Error: [rms_norm_kernel_%d_%d] There is no suitable microkernel!\n", M,N);
  }
}

} // namespace kernel
