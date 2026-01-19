import os
import math
import itertools
import json
import torch
# 
import tilelang
import tilelang.language as T
from tilelang.utils.profiler import do_bench
import tvm
from tvm.tir.stmt_functor import ir_transform

from common.pkt_util import TestUtil, TorchRef

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
    latency = do_bench(lambda: kernel(a), warmup=500, backend="cupti")
    print(f"tilelang Latency: {latency}ms")
    
if __name__ == "__main__":
    test_silu_mul()