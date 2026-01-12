import os
import math
import itertools
import json

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

import util
from common.pkt_util import TestUtil, TorchRef
from common.tl_ops import HparamSelectMode, LinearTL

def main():
    M = 1
    N = 19456
    K = 2560
    # config = [64,64,64,2,128,0,true]
    linear = LinearTL(M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = linear.get_kernel(HparamSelectMode.HEURISTIC)

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