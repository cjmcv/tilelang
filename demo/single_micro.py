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

from common.pkt_util import TorchRef, PerfReporter
from common.micro_base import HparamSelectMode
from common.micro_linear import MicroLinearStrategy, MicroLinear
from common.micro_rmsnorm import MicroRmsNorm
from common.micro_silu_mul import MicroSiluMul
from common.micro_gqa_decode import MicroGqaDecode
from common.micro_rope import MicroRope

from common.micro_autogen import MicroAutoGen

def profile(target_func, torch_ref_func):
    reporter = PerfReporter() 
    reporter.generate_report(target_func, torch_ref_func,
                            warnup_iter=100, test_iter=500, 
                            allclose_iter=5, print_mode=0)
    
def test_silu_mul():
    M, N = 32, 9728
    micro = MicroSiluMul(M,N, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    test_data = micro.gen_test_data(kernel.config)
    
    def target_func():
        return kernel(*test_data)
    def torch_ref():
        return TorchRef.silu_and_mul(*test_data)
    profile(target_func, torch_ref)
    
def test_rms_norm():
    M = 32
    N = 2560
    micro = MicroRmsNorm(M,N, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, fn, info = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    
    test_data = micro.gen_test_data(kernel.config)
    def target_func():
        return kernel(*test_data)
    def torch_ref():
        return TorchRef.rms_norm(*test_data) 
    profile(target_func, torch_ref)
    
def test_gemm():
    M = 1
    N = 6144
    K = 1024
    # N = 2560
    # K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info = micro.get_kernel(HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED

    test_data = micro.gen_test_data(kernel.config)
    def target_func():
        return kernel(*test_data)
    def torch_ref():
        return TorchRef.linear(*test_data) 
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
    M = 1
    N = 1024
    K = 3072
    # N = 2560
    # K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM_ADD, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED

    test_data = micro.gen_test_data(kernel.config)
    a, b, r = test_data

    def target_func():
        return kernel(a, b, r)
    def torch_ref():
        return TorchRef.linear(a, b) + r
    
    profile(target_func, torch_ref)

def test_gqa_decode():
    batch = 1
    heads = 16
    groups = 8
    max_kv_seqlen = 8192
    target_kv_seqlen = 1024
    valid_kv_seqlen = 1024
    dim = 32
    is_causal = False
    
    # config = [64,64,64,2,128,0,true]
    micro = MicroGqaDecode(batch, max_kv_seqlen, target_kv_seqlen, heads, groups, dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.TUNING) # HEURISTIC, TUNING, TUNED
    
    test_data = micro.gen_test_data(kernel.config)
    q, k, v, edge, mask, glse, Output_partial = test_data
    
    def target_func():
        return kernel(*test_data)
    
    k_slice = k[:, :valid_kv_seqlen, :, :]
    v_slice = v[:, :valid_kv_seqlen, :, :]
    def torch_ref():
        return TorchRef.attention_sdpa(q, k_slice, v_slice, is_causal)
        # return TorchRef.attention(q, k, v, mask, glse, Output_partial)
        # return TorchRef.attention_split(q, k, v, mask, glse, Output_partial)
    print("hello profile")    
    profile(target_func, torch_ref)

def test_rope():
    batch = 1
    seqlen = 1
    heads = 16
    groups = 8
    dim = 128
    
    micro = MicroRope(batch, seqlen, heads, groups, dim, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    # kernel.export_sources(kernel_path="demo/gen/single_micro.cu")
    
    test_data = micro.gen_test_data(kernel.config)
    q, k, cos, sin = test_data
    
    def triton_ref():
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)

    def torch_ref():
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)

    def target_func():
        q_emb, k_emb = kernel(q, k, cos, sin)
        return torch.cat((q_emb, k_emb), dim=-2)
      
    profile(target_func, triton_ref)

if __name__ == "__main__":
    # test_silu_mul()
    # test_rms_norm()
    # test_gemm()
    ## test_silu_mul_gemm() # 逻辑有误，silu_mul被重复计算
    # test_gemm_add()
    # test_gqa_decode()
    # test_rope()

    # # # gen = MicroAutoGen(1, 2560, 9728)
    gen = MicroAutoGen(batch_size=1, hidden_size=1024, intermediate_size=3072, 
                       max_kv_seqlen=8192, heads=16, groups=8, dim=128)
    gen.gen_qwen3_ops(layer_id=8, mode=HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED
    
    # print("Test single_micro completed.")