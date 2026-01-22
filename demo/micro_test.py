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
from common.micro_base import HparamSelectMode
from common.micro_linear import MicroLinearStrategy, MicroLinear
from common.micro_rms_norm import MicroRmsNorm
from common.micro_silu_mul import MicroSiluMul
from common.micro_autogen import MicroAutoGen

def profile(target_func, torch_ref_func):
    c = target_func()
    ref_c = torch_ref_func()
    
    print("c:\n", c, "\nref_c:\n", ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-1, atol=1e-1)
    
    # benchmark
    latency = do_bench(lambda: target_func(), warmup=500, backend="cupti")
    torch_latency = do_bench(lambda: torch_ref_func(), warmup=500, backend="cupti")
    print(f"tilelang Latency: {latency}ms vs {torch_latency}(torch) ms")
    
def test_silu_mul():
    M, N = 32, 9728
    micro = MicroSiluMul(M,N, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    a = torch.randn(M, N*2, dtype=torch.bfloat16, device="cuda")
    
    def target_func():
        return kernel(a)
    def torch_ref():
        return TorchRef.silu_and_mul(a)
    profile(target_func, torch_ref)
    
def test_rms_norm():
    M = 32
    N = 2560
    micro = MicroRmsNorm(M,N, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, fn, info = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    a = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, N, dtype=torch.bfloat16, device="cuda")

    def target_func():
        return kernel(a, b)
    def torch_ref():
        return TorchRef.rms_norm(a, b) 
    profile(target_func, torch_ref)
    
def test_gemm():
    M = 32
    N = 19456
    K = 2560
    # N = 2560
    # K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    
    def target_func():
        return kernel(a, b)
    def torch_ref():
        return TorchRef.linear(a, b) 
    profile(target_func, torch_ref)

# def test_silu_mul_gemm():
#     M = 32
#     # N = 19456
#     # K = 2560
#     N = 2560
#     K = 9728
#     # config = [64,64,64,2,128,0,true]
#     micro = MicroLinear(MicroLinearStrategy.SILU_MUL_GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
#     kernel = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

#     a = torch.randn((M, K*2), dtype=torch.bfloat16, device="cuda")
#     b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

#     def target_func():
#         return kernel(a, b)
#     def torch_ref():
#         a2 = TorchRef.silu_and_mul(a)
#         return TorchRef.linear(a2, b)
#     profile(target_func, torch_ref)
    
def test_gemm_add():
    M = 32
    # N = 19456
    # K = 2560
    N = 2560
    K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM_ADD, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    r = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    
    def target_func():
        return kernel(a, b, r)
    def torch_ref():
        return TorchRef.linear(a, b) + r
    profile(target_func, torch_ref)

if __name__ == "__main__":
    # test_silu_mul()
    # test_rms_norm()
    # test_gemm()
    ## test_silu_mul_gemm() # 逻辑有误，silu_mul被重复计算
    # test_gemm_add()
    
    gen = MicroAutoGen(128, 2560, 9728)
    gen.gen_qwen_mlp(HparamSelectMode.TUNING, 2)
    
    # 整合该文件，形成一个class，通过统一参数，和选定算子，自动生成代码并拷贝到对应路径，打印出grid信息以一键验证。
    # 分析：gemm1的4block -> silu_mul的2block，02->0, 13->1
    #      能否只写回gemm1的后两个block 23，前两个block 01保留在smem，延递silu_mul上。