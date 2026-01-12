import os
import math
import itertools
import json
# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

import util
from common.pkt_util import TestUtil, TorchRef
  
# 传保存地址，提供代码生成的基础函数  
class BaseLtOps:
    def __init__(self):
        self.megakernel_home = os.getenv("MEGAKERNEL_HOME", default=None)
        if self.megakernel_home is None:
            raise EnvironmentError("The environment variable MEGAKERNEL_HOME is not set.")
        self.save_path = self.megakernel_home + "/demo/gen/"
        
    def replace_line(self, text: str, src_target: str, skip_count: int, dst_target: str) -> str:
        lines = text.splitlines(True)
        processed_lines = []
        target_count = 0
        
        for line in lines:
            if src_target in line:
                target_count += 1
                if target_count == skip_count:
                    continue
                processed_lines.append(dst_target)
            else:
                processed_lines.append(line)
        return "".join(processed_lines)

    def write_tuned_hparams_to_file(self, hparams_time_list, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for time_cost, hparams in hparams_time_list:
                single_config = {
                    "time_cost": time_cost,
                    "hparams": hparams
                }
                json_line = json.dumps(single_config, ensure_ascii=False, separators=(",", ":"))
                f.write(json_line + "\n")
        
        print(f"Save: {file_path}")


class LinearLt(BaseLtOps):
    def __init__(self, M, N, K, block_M, block_N, block_K, num_stages, threads, policy, enable_rasteration, dtype=T.bfloat16, accum_dtype=T.float32):
        super().__init__()
        self.kerne_name = "linear_gemm_tl"
        
        self.M = M
        self.N = N
        self.K = K
        self.girdDim_x = T.ceildiv(N, block_N)
        self.girdDim_y = T.ceildiv(M, block_M)
        self.block_M = block_M
        self.block_N = block_N
        self.block_K = block_K
        self.num_stages = num_stages
        self.threads = threads
        self.policy = policy
        self.enable_rasteration = enable_rasteration
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.hparam_space = self.get_hparam_space()
        print(len(self.hparam_space))

    def get_hparam_space(self):
        block_M=[64, 128, 256]
        block_N=[64, 128, 256]
        block_K=[32, 64]
        num_stages=[0, 1, 2, 3]
        thread_nums=[128, 256]
        policies=[T.GemmWarpPolicy.Square]
        enable_rasterations=[True, False]
        
        res = []
        for m, n, k, num_stage, thread_num, policy, enable_rasteration in itertools.product(
            block_M, block_N, block_K, num_stages, thread_nums, policies, enable_rasterations):
            res.append([m, n, k, num_stage, thread_num, policy, enable_rasteration])

        return res
    
    def _tune_single(self, idx, hparams):
        # idx, hparams = tasks
        try:
            kernel = LinearLt.matmul(
                self.M, self.N, self.K,
                self.girdDim_x, self.girdDim_y,
                *hparams, self.dtype, self.accum_dtype
            )
            profiler = kernel.get_profiler()
            latency = profiler.do_bench(backend="cupti")
            return idx, hparams, round(latency, 5), "success"
        except Exception as e:
            return idx, hparams, None, f"{e}"

    # def _tune_parallel(self):
    #     thread_size = min(8, multiprocessing.cpu_count() // 4)
    #     print(f"Start tuning with {len(self.hparam_space)} schemes (thread_num: {thread_size})")
    #     tasks = [(idx, hparams) for idx, hparams in enumerate(self.hparam_space)]
        
    #     with ThreadPoolExecutor(max_workers=thread_size) as executor:
    #         all_results = executor.map(self._tune_single, tasks)

    #     results = []
    #     for idx, hparams, latency, status in all_results:
    #         if status == "success":
    #             print(f">>>>> tuning({idx}): {latency} -> {hparams}")
    #             results.append({"idx": idx, "latency": latency, "hparams": hparams})
    #         else:
    #             print(f"Failed: Schema {idx+1}: {status}")
    #     return results

                    
    def get_kernel(self, is_tune = False):
        file_path = self.save_path+self.kerne_name
        
        if (is_tune):
            self.hparam_space
            print(f"Start tuning with a total of {len(self.hparam_space)} schemes.")
            
            hparams_time_list = []
            for idx, hparams in enumerate(self.hparam_space):
                _, _, latency, status = self._tune_single(idx, hparams)
                if status == "success":
                    hparams_time_list.append((latency, hparams))
                print(f">>>>> tuning({idx}-{status}): {latency} -> {hparams}")
                
            # hparams_time_list = self._tune_parallel()
            hparams_time_list.sort(key=lambda x: x[0])
            best_latency, best_hparams = hparams_time_list[0]
            print(f"Finish tuning, the best result: {best_latency} ms -> {best_hparams}")
            self.kernel = LinearLt.matmul(self.M, self.N, self.K, 
                                          self.girdDim_x, self.girdDim_y, 
                                          *best_hparams, self.dtype, self.accum_dtype)
            self.write_tuned_hparams_to_file(hparams_time_list, file_path+"_tuned.json")
        else:
            self.kernel = LinearLt.matmul(self.M, self.N, self.K, 
                                        self.girdDim_x, self.girdDim_y, 
                                        self.block_M, self.block_N, self.block_K, 
                                        self.num_stages, self.threads, self.policy, self.enable_rasteration, 
                                        self.dtype, self.accum_dtype)
            
        self.kernel.export_sources(kernel_path=file_path+"_src.cuh", host_path=file_path+"_src.cpp")
        util.save_to_file(self.get_source(), file_path+".cuh")
        return self.kernel

    def get_source(self):
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

        head_str = head_str.replace('<threads>', str(self.threads))
        head_str = head_str.replace('<BLOCK_M>', str(self.block_M))
        head_str = head_str.replace('<BLOCK_N>', str(self.block_N)) 
        head_str = head_str.replace('<BLOCK_K>', str(self.block_K)) 
        head_str = head_str.replace('<M>', str(self.M))
        head_str = head_str.replace('<N>', str(self.N)) 
        head_str = head_str.replace('<K>', str(self.K)) 
        if self.dtype == T.bfloat16:
            dtype = "bfloat16"
        else:
            dtype = "float16"
        head_str = head_str.replace('<dtype>', str(dtype))
                
        origin_source = self.kernel.get_kernel_source()
        source = origin_source.replace("blockIdx.x", "bx")
        source = source.replace("blockIdx.y", "by")
        source = source.replace("blockIdx.z", "bz")
        source = self.replace_line(source, "extern \"C\" __global__", 1, head_str)
        source += "\n} // kernel"
        return source
        
    # @tilelang.jit(out_idx=[-1])
    # def matmul(M, N, K, girdDim_x, girdDim_y, block_M, block_N, block_K, threads, dtype, accum_dtype):
    #     @T.prim_func
    #     def linear(
    #         A: T.Tensor((M, K), dtype),
    #         B: T.Tensor((N, K), dtype),
    #         C: T.Tensor((M, N), dtype),
    #     ):
    #         with T.Kernel(girdDim_x, girdDim_y, threads=threads) as (bx, by): # tilelang.next_power_of_2(128)
    #             A_shared = T.alloc_shared((block_M, block_K), dtype)
    #             B_shared = T.alloc_shared((block_N, block_K), dtype)
    #             C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

    #             T.clear(C_local)
    #             for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    #                 T.copy(A[by * block_M, k * block_K], A_shared)
    #                 T.copy(B[bx * block_N, k * block_K], B_shared)
    #                 T.gemm(A_shared, B_shared, C_local, transpose_B=True)

    #             T.copy(C_local, C[by * block_M, bx * block_N])

    #     return linear

    @tilelang.jit(out_idx=[-1])
    def matmul(M, N, K, girdDim_x, girdDim_y, block_M, block_N, block_K, num_stages, thread_num, policy, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
        @T.prim_func
        def linear(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(girdDim_x, girdDim_y, threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                        policy=policy,
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return linear

def main():
    M = 1
    N = 19456
    K = 2560
    linear = LinearLt(M,N,K, block_M=64, block_N=64, block_K=64, num_stages=3, threads=128, policy=T.GemmWarpPolicy.Square, enable_rasteration=False, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = linear.get_kernel(True)

    import torch

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    c = kernel(a, b)
    ref_c = TorchRef.linear(a, b)

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    # print("All check passed.")

    # Get CUDA Source
    # util.save_to_file(linear.get_source(), cur_path+"/../gen/linear_gemm_tl.cuh")
    # util.save_to_file(kernel.get_host_source(), "../gen/gemm.cpp")
    # kernel.export_sources(kernel_path=cur_path+"/../gen/linear_gemm_tl2.cuh", host_path=cur_path+"/../gen/linear_gemm_tl.cpp")
    
    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    
    print("0:", kernel.prim_func.attrs)
    print("1:", kernel.adapter.params)
    print("2:", kernel.adapter.func)
    print("3:", kernel.config)
    print("4:", kernel.prim_func)
    # print("5:", kernel.prim_func.body)
    
    smem_bytes = util.get_smem_bytes(kernel.prim_func)
    print("shared memory =", smem_bytes, "B")

    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


if __name__ == "__main__":
    main()