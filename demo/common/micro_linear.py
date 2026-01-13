import os
import math
from enum import Enum
import itertools
import json
from pathlib import Path

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor
import torch
from tvm import DataType
from tvm.tir import stmt_functor, Block, For, PrimFunc
import tilelang
import tilelang.language as T

from common.pkt_util import TestUtil, TorchRef
from common.micro_kernel_base import BaseMicroKernel, LaunchInfoAnalyzer, HparamSelectMode


class _GemvStrategy:
    def __init__(self, M, N, K, dtype, accum_dtype):
        self.name = "linear_gemv_tl"+f"_{M}_{N}_{K}"
        
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        assert(self.M == 1)
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        BLOCK_N = [2, 4, 8, 32, 64, 128]
        reduce_threads = [4, 8, 32]
        
        res = []
        for n, reduce_thread in itertools.product(BLOCK_N, reduce_threads):
            res.append([n, reduce_thread])
        # print(res)
        return res
    
    def get_tuning_params(self):
        return self.name, self.hparam_space, self.get_kernel
    
    def get_heuristic_hparams(self):
        # [BLOCK_N, reduce_threads]
        return [2, 32]
        
    def get_kernel(self, selected_hparams):
        return self.kernel_main(self.N, self.K, *selected_hparams, self.dtype, self.accum_dtype) 
    
    @tilelang.jit(out_idx=[-1])
    def kernel_main(
        N: int,
        K: int,
        BLOCK_N: int,
        reduce_threads: int,
        dtype: T.dtype = T.bfloat16,
        accum_dtype: T.dtype = T.float,
    ):
        # splitk_gemv_vectorized_tvm
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_K = reduce_threads * TILE_K

        @T.prim_func
        def linear(
            A: T.Tensor((1, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((1, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                tn = T.get_thread_binding(0)
                tk = T.get_thread_binding(1)
                A_local = T.alloc_local((TILE_K,), dtype)
                B_local = T.alloc_local((TILE_K,), dtype)
                C_accum = T.alloc_local((1,), accum_dtype)

                T.clear(C_accum)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for k in T.vectorized(TILE_K):
                        A_local[k] = A[0, bk * BLOCK_K + tk * TILE_K + k]
                        B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                    for k in T.serial(TILE_K):
                        C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                C_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_accum[0],
                            True,
                            C_reduced[0],
                            tk,
                            dtype="handle",
                        )
                    )

                C[0, bn * BLOCK_N + tn] = C_reduced[0]

        return linear
    
    
class _GemmStrategy:
    def __init__(self, M, N, K, dtype, accum_dtype):
        self.name = "linear_gemm_tl"+f"_{M}_{N}_{K}"
        
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        BLOCK_M=[64, 128] #, 256
        BLOCK_N=[64, 128] #, 256
        BLOCK_K=[32, 64]
        num_stages=[0]#, 1, 2, 3
        thread_nums=[128]#, 256
        policies=[T.GemmWarpPolicy.Square]
        enable_rasterations=[True, False]
        
        res = []
        for m, n, k, num_stage, thread_num, policy, enable_rasteration in itertools.product(
            BLOCK_M, BLOCK_N, BLOCK_K, num_stages, thread_nums, policies, enable_rasterations):
            res.append([m, n, k, num_stage, thread_num, policy, enable_rasteration])

        return res 
    
    def get_tuning_params(self):
        return self.name, self.hparam_space, self.get_kernel
    
    def get_heuristic_hparams(self):
        # [BLOCK_M, BLOCK_N, BLOCK_K, num_stages, threads, policy, enable_rasteration]
        return [64, 64, 64, 3, 128, 0, False]
        
    def get_kernel(self, selected_hparams):
        kernel = self.kernel_main(self.M, self.N, self.K, *selected_hparams, self.dtype, self.accum_dtype) 
        return kernel
    
    # @tilelang.jit(out_idx=[-1])
    # def kernel_main(M, N, K, girdDim_x, girdDim_y, BLOCK_M, BLOCK_N, BLOCK_K, threads, dtype, accum_dtype):
    #     @T.prim_func
    #     def linear(
    #         A: T.Tensor((M, K), dtype),
    #         B: T.Tensor((N, K), dtype),
    #         C: T.Tensor((M, N), dtype),
    #     ):
    #         with T.Kernel(girdDim_x, girdDim_y, threads=threads) as (bx, by): # tilelang.next_power_of_2(128)
    #             A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
    #             B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
    #             C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

    #             T.clear(C_local)
    #             for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
    #                 T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
    #                 T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_shared)
    #                 T.gemm(A_shared, B_shared, C_local, transpose_B=True)

    #             T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    #     return linear
    
    @tilelang.jit(out_idx=[-1])
    def kernel_main(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, thread_num, policy, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):

        @T.prim_func
        def linear(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                    T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                    T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                        policy=policy,
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * BLOCK_M, bx * BLOCK_N])

        return linear
    
class MicroLinear(BaseMicroKernel):
    def __init__(self, M, N, K, dtype=T.bfloat16, accum_dtype=T.float32):
        super().__init__()
        self.kernel_name = "linear_tl"+f"_{M}_{N}_{K}"
        
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.strategy = _GemvStrategy(M, N, K, dtype, accum_dtype)
        
    def get_source(self, kernel, selected_hparams):
        head_str = \
'''
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
  assert(THREAD_NUM==<threads>)
  assert(TILE_DIM_X==<BLOCK_N>); assert(TILE_DIM_Y==<BLOCK_M>); assert(TILE_DIM_Z==<BLOCK_K>);
  assert(M==<M>); assert(N==<N>); assert(K==<K>);
  
  const <dtype>* __restrict__ A = static_cast<const <dtype>* __restrict__>(input_ptr);
  const <dtype>* __restrict__ B = static_cast<const <dtype>* __restrict__>(weight_ptr);
  const <dtype>* __restrict__ C = static_cast<const <dtype>* __restrict__>(output_ptr);
'''
        if (isinstance(self.strategy, _GemvStrategy)):
            BLOCK_N, reduce_threads = selected_hparams
            BLOCK_M, BLOCK_K, threads = 1, 1, 128 # todo
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, num_stages, threads, policy, enable_rasteration = selected_hparams
        # gridDim_x, gridDim_y = self.get_grid_dims(self.M, self.N, BLOCK_M, BLOCK_N)
        # print(f"gridDim: ({gridDim_x}, {gridDim_y})")
        
        head_str = head_str.replace('<threads>', str(threads))
        head_str = head_str.replace('<BLOCK_M>', str(BLOCK_M))
        head_str = head_str.replace('<BLOCK_N>', str(BLOCK_N)) 
        head_str = head_str.replace('<BLOCK_K>', str(BLOCK_K)) 
        head_str = head_str.replace('<M>', str(self.M))
        head_str = head_str.replace('<N>', str(self.N)) 
        head_str = head_str.replace('<K>', str(self.K)) 
        if self.dtype == T.bfloat16:
            dtype = "bfloat16"
        else:
            dtype = "float16"
        head_str = head_str.replace('<dtype>', str(dtype))
                
        origin_source = kernel.get_kernel_source()
        source = origin_source.replace("blockIdx.x", "bx")
        source = source.replace("blockIdx.y", "by")
        source = source.replace("blockIdx.z", "bz")
        source = self.replace_line(source, "extern \"C\" __global__", 1, head_str)
        source += "\n} // kernel"
        return source
    
    def get_kernel(self, mode: HparamSelectMode):
        file_path = self.save_path+self.kernel_name
        
        if (mode == HparamSelectMode.TUNING):
            best_latency, selected_hparams = self.run_tuning(*self.strategy.get_tuning_params())
        elif (mode == HparamSelectMode.TUNED):
            best_latency, selected_hparams = self.read_tuned_hparams_from_json(self.strategy.name)
            print("[Tuned] selected_hparams: ", selected_hparams)
        else:
            selected_hparams = self.strategy.get_heuristic_hparams()
            print("[Heuristic] selected_hparams: ", selected_hparams)
        
        kernel = self.strategy.get_kernel(selected_hparams)
        kernel.export_sources(kernel_path=file_path+"_src.cuh", host_path=file_path+"_src.cpp")
        
        analyzer = LaunchInfoAnalyzer(kernel.prim_func)
        analyzer.get_threads_layout()
        extra_attr = f"\n// Strategy: {self.strategy.name}"
        extra_attr += f"\n// grid_dim: {analyzer.grid_dim}."
        extra_attr += f"\n// block_dim: {analyzer.block_dim}."
        extra_attr += f"\n// Smem: {analyzer.get_smem_bytes()} bytes."
        
        with open(file_path+".cuh", "w", encoding="utf-8") as f:
            f.write(self.get_source(kernel, selected_hparams) + extra_attr)
            
        return kernel