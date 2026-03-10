
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

template <typename T>
__device__ __forceinline__ void copy_kernel(const int bx, const int by, const int bz,
                                            const int step, const int onestep_size,
                                            const void* __restrict__ input_ptr1, 
                                            const void* __restrict__ input_ptr2, 
                                            void* __restrict__ output_ptr1, 
                                            void* __restrict__ output_ptr2) {
  // (bx=2, by=1, bz=1)
  if (threadIdx.x == 0) {
    printf("copy_kernel: (%d,%d,%d): %d, %d\n", bx, by, bz, step, onestep_size);    
  }
  const bfloat16_t* __restrict__ input1 = static_cast<const bfloat16_t*>(input_ptr1);
  const bfloat16_t* __restrict__ input2 = static_cast<const bfloat16_t*>(input_ptr2);
  bfloat16_t* __restrict__ output1 = static_cast<bfloat16_t*>(output_ptr1) + step * onestep_size;
  bfloat16_t* __restrict__ output2 = static_cast<bfloat16_t*>(output_ptr2) + step * onestep_size;
  if (bx == 0) {
    // onestep_size == 1024
    const int tid = threadIdx.x * 8;
    for (int i=0; i<8; i++)
      output1[tid*8+i] = input1[tid*8+i];
  } else if (bx == 1) {
    const int tid = threadIdx.x * 8;
    for (int i=0; i<8; i++)
      output2[tid*8+i] = input2[tid*8+i];
  }
}

} // namespace kernel
