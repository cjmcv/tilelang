import os
import math
import itertools
import json
import torch
# 
import tilelang
import tilelang.language as T
import tvm
from tvm.tir.stmt_functor import ir_transform

from common.pkt_util import TestUtil, TorchRef
from common.micro_base import HparamSelectMode
from common.micro_linear import MicroLinearStrategy, MicroLinear

@tilelang.jit(out_idx=[-1])
def silu_mul(M, N, BLOCK_M, BLOCK_N, dtype="bfloat16"):
    @T.prim_func
    def main(
        X: T.Tensor((M, N*2), dtype),   # 主输入
        Z: T.Tensor((M, N), dtype),   # 输出
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
            n_block_num = T.ceildiv(N, BLOCK_N)
            # shared tile
            x_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            y_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            z_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

            # gmem -> smem
            T.copy(X[by * BLOCK_M:(by + 1) * BLOCK_M,
                     bx * BLOCK_N:(bx + 1) * BLOCK_N], x_sh)
            T.copy(X[by * BLOCK_M:(by + 1) * BLOCK_M,
                     (bx + n_block_num) * BLOCK_N:(bx + n_block_num + 1) * BLOCK_N], y_sh)

            # silu_mul for each element
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                xi = x_sh[i, j].astype("float32")      # 先转 fp32 求 sigmoid 更稳
                sig = 1.0 / (1.0 + T.exp(-xi))         # sigmoid
                z_sh[i, j] = (xi * sig * y_sh[i, j]).astype(dtype)

            # smem -> gmem
            T.copy(z_sh, Z[by * BLOCK_M:(by + 1) * BLOCK_M,
                           bx * BLOCK_N:(bx + 1) * BLOCK_N])

    return main

def test_silu_mul():
    M, N = 1, 9728
    BLOCK_M, BLOCK_N = 64, 64
    kernel = silu_mul(M, N, BLOCK_M, BLOCK_N, dtype="bfloat16")
    a = torch.randn(M, N*2, dtype=torch.bfloat16, device="cuda")
    c = kernel(a)
    print("c:")
    print(c)
    ref_c = TorchRef.silu_and_mul(a)
    print("ref_c:")
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-1, atol=1e-1)
    
    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    print(f"tilelang Latency: {latency}ms")
    
def test_gemm():
    M = 64
    # N = 19456
    # K = 2560
    N = 2560
    K = 9728
    # config = [64,64,64,2,128,0,true]
    linear = MicroLinear(MicroLinearStrategy.GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = linear.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    c = kernel(a, b)
    ref_c = TorchRef.linear(a, b)

    print("c:")
    print(c)

    print("ref_c:")
    print(ref_c)
    
    torch.testing.assert_close(c, ref_c, rtol=1e-1, atol=1e-1)

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    print(f"tilelang Latency: {latency}ms")

def test_silu_mul_gemm():
    M = 64
    # N = 19456
    # K = 2560
    N = 2560
    K = 9728
    # config = [64,64,64,2,128,0,true]
    linear = MicroLinear(MicroLinearStrategy.SILU_MUL_GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = linear.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    a = torch.randn((M, K*2), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    c = kernel(a, b)
    
    a2 = TorchRef.silu_and_mul(a)
    ref_c = TorchRef.linear(a2, b)

    print("c:")
    print(c)

    print("ref_c:")
    print(ref_c)
    
    # torch.testing.assert_close(c, ref_c, rtol=1e-1, atol=1e-1)

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    print(f"tilelang Latency: {latency}ms")
    
if __name__ == "__main__":
    # test_silu_mul()
    # test_gemm()
    test_silu_mul_gemm()