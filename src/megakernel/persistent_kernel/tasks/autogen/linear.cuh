#include "linear_gemm_tl_1_19456_2560_top0.cuh"
#include "linear_gemm_tl_1_2560_9728_top9.cuh"

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
    __device__ __forceinline__ void linear_kernel(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr,
                                                const void* __restrict__ weight_ptr,
                                                const void* __restrict__ residual_ptr,
                                                void* __restrict__ output_ptr,
                                                int num_active_tokens,
                                                bool residual) {
  if constexpr (M == 1 && N == 19456 && K == 2560) {
    linear_kernel_1_19456_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
        bx, by, bz,
        input_ptr, weight_ptr, residual_ptr, output_ptr,
        num_active_tokens, residual
    );
  }
  else if (M == 1 && N == 2560 && K == 9728) {
    linear_kernel_1_2560_9728<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
      bx, by, bz,
      input_ptr, weight_ptr, residual_ptr, output_ptr,
      num_active_tokens, residual
  );
  } 
  else {
    printf("Error: [linear_kernel_%d_%d_%d] There is no suitable microkernel!\n", M,N,K);
  } 
}

} // kernel
