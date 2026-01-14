import os
import math
import itertools
import json

# 
import tilelang
import tilelang.language as T
import tvm
from tvm.tir.stmt_functor import ir_transform

from common.pkt_util import TestUtil, TorchRef
from common.micro_base import HparamSelectMode
from common.micro_linear import MicroLinearStrategy, MicroLinear

def main():
    M = 1
    N = 19456
    K = 2560
    # config = [64,64,64,2,128,0,true]
    linear = MicroLinear(MicroLinearStrategy.GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = linear.get_kernel(HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED

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

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    
    # analyzer = Analyzer(kernel.prim_func)
    # analyzer.ir_pass()
    # print("analyzer: ", analyzer.block_counts, analyzer.thread_counts, analyzer.dyn_shared_memory_buf, analyzer.loop_stack)
    # print("a:", kernel.artifact)
    print("0:", kernel.prim_func.attrs)
    print("1:", kernel.adapter.params)
    print("2:", kernel.adapter.func)
    print("3:", kernel.config)
    print("4:", kernel.prim_func)
    # print("5:", kernel.prim_func.body)
    # print("5:", kernel.prim_func.metadata)
    
    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")

if __name__ == "__main__":
    main()