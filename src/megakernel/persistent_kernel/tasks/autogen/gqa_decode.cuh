
#pragma once

#include "m1/gqa_decode_tl_1_8192_16_8_128.cuh"

namespace kernel {

template <typename T,
  int THREAD_NUM,
  int M, 
  int HEAD,
  int GROUPS,
  int DIM>
__device__ __forceinline__ void gqa_decode_kernel(const int bx, const int by, const int bz,
                                           const void* __restrict__ q, 
                                           const void* __restrict__ k, 
                                           const void* __restrict__ v,
                                           const void* __restrict__ mask_ptr, 
                                           const void* __restrict__ glse_ptr,
                                           void* __restrict__ output_partial_ptr,
                                           void* __restrict__ output_ptr) {
  if constexpr (M == 1) { 
    if constexpr (HEAD == 16) {
      flashattn_kernel_1_8192_16_8_128<T, THREAD_NUM, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, mask_ptr, glse_ptr, output_partial_ptr, output_ptr);      
    }
  }
  else {
    printf("Error: [gqa_decode_kernel_%d_%d_%d_%d] There is no suitable microkernel!\n", M, HEAD, GROUPS, DIM);
  }
}

} // namespace kernel
