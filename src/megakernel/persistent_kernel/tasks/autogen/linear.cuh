#pragma once



#include "m1/linear_gemm_tl_1_19456_2560.cuh"
#include "m1/linear_gemm_add_tl_1_2560_9728.cuh"

#include "m32/linear_gemm_tl_32_19456_2560.cuh"
#include "m32/linear_gemm_add_tl_32_2560_9728.cuh"

#include "m128/linear_gemm_tl_128_19456_2560.cuh"
#include "m128/linear_gemm_add_tl_128_2560_9728.cuh"

#include "m1/linear_gemm_tl_1_6144_1024.cuh"
#include "m1/linear_gemm_add_tl_1_1024_3072.cuh"

#include "m32/linear_gemm_tl_32_6144_1024.cuh"
#include "m32/linear_gemm_add_tl_32_1024_3072.cuh"

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
    bool FUSE_RES = false,
    bool FUSE_SILU_MUL = false>
    __device__ __forceinline__ void linear_kernel(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr,
                                                const void* __restrict__ weight_ptr,
                                                const void* __restrict__ residual_ptr,
                                                void* __restrict__ output_ptr,
                                                int num_active_tokens,
                                                bool residual) {
  if constexpr (FUSE_RES == true) {
    if constexpr (M == 1) {
      if constexpr (N == 2560 && K == 9728) {
        linear_gemm_add_tl_1_2560_9728<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);        
      }
      else if constexpr (N == 1024 && K == 3072) {
        linear_gemm_add_tl_1_1024_3072<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);  
      }
    } 
    else if constexpr (M == 32) {
      if constexpr (N == 2560 && K == 9728) {
        linear_gemm_add_tl_32_2560_9728<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);
      }
      else if constexpr (N == 1024 && K == 3072) {
        linear_gemm_add_tl_32_1024_3072<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);  
      }
    } 
    else if constexpr (M == 128 && N == 2560 && K == 9728) {
      linear_gemm_add_tl_128_2560_9728<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
        bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);
    } 
    else { printf("Error: [linear_kernel_%d_%d_%d] There is no suitable microkernel!\n", M,N,K); }
  }
  else {
    if constexpr (M == 1) {
      if constexpr (N == 19456 && K == 2560) {
        linear_gemm_tl_1_19456_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);        
      }
      else if constexpr (N == 6144 && K == 1024) {
        linear_gemm_tl_1_6144_1024<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);  
      }
    }
    else if constexpr (M == 32) {
      if constexpr (N == 19456 && K == 2560) {
        linear_gemm_tl_32_19456_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);
      }
      else if constexpr (N == 6144 && K == 1024) {
        linear_gemm_tl_32_6144_1024<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);
      }
    }
    else if constexpr (M == 128 && N == 19456 && K == 2560) {
      linear_gemm_tl_128_19456_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
        bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual);
    }
    else { printf("Error: [linear_kernel_%d_%d_%d] There is no suitable microkernel!\n", M,N,K); }     
  }
}

} // kernel
