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
                            allclose_iter=5, print_all=False)
    
def test_silu_mul():
    M, N = 32, 9728
    micro = MicroSiluMul(M,N, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
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
    M = 1
    N = 6144
    K = 1024
    # N = 2560
    # K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info = micro.get_kernel(HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED

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
    M = 1
    N = 1024
    K = 3072
    # N = 2560
    # K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM_ADD, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    r = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    
    def target_func():
        return kernel(a, b, r)
    def torch_ref():
        return TorchRef.linear(a, b) + r
    
    profile(target_func, torch_ref)

def test_gqa_decode():
    batch = 1
    heads = 16
    groups = 8
    kv_seqlen = 8192
    valid_kv_seqlen = 5
    dim = 128
    is_causal = False
    
    # config = [64,64,64,2,128,0,true]
    micro = MicroGqaDecode(batch, kv_seqlen, heads, groups, dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)              # [B, N=q_seqlen=1, H=heads,  D=dim]
    k = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=kv_seqlen,  H=groups, D=dim]
    v = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)
    # mask = torch.randint(0, 2, (batch, kv_seqlen, groups), device="cuda", dtype=torch.uint8) # Only 0/1
    edge = torch.empty(10, device="cuda", dtype=torch.int32)
    edge[0].fill_(valid_kv_seqlen)
    mask = torch.ones(batch, kv_seqlen, groups, device="cuda", dtype=torch.uint8)      # no mask
    
    # 上面的mask(batch, kv_seqlen, groups)，维度其实是(batch, q_seqlen, kv_seqlen, groups),groups维度是广播出来的，mask只跟q_seqlen, kv_seqlen有关
    # q_len = 4
    # kv_seqlen = 16
    # mask = torch.tril(torch.ones((q_len, kv_seqlen), device="cuda", dtype=torch.uint8)).unsqueeze(0).expand(batch, -1, -1) # (q_len, kv_seqlen)
    # mask.unsqueeze(2).expand(-1, -1, groups, -1).transpose(1, 2)
    # print(mask, mask.shape)
    
    split = kernel.config[2]
    glse = torch.empty(batch, heads, split, device="cuda", dtype=torch.bfloat16)
    Output_partial = torch.empty(batch, heads, split, dim, device="cuda", dtype=torch.bfloat16)
    
    def target_func():
        return kernel(q, k, v, edge, mask, glse, Output_partial)
    
    k_slice = k[:, :valid_kv_seqlen, :, :]
    v_slice = v[:, :valid_kv_seqlen, :, :]
    def torch_ref():
        return TorchRef.attention_sdpa(q, k_slice, v_slice, is_causal)
        # return TorchRef.attention(q, k, v, mask, glse, Output_partial)
        # return TorchRef.attention_split(q, k, v, mask, glse, Output_partial)
        
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
    
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen=1, H=heads,  D=dim]
    k = torch.randn(batch, seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)   # [B, N=seqlen=1, H=groups, D=dim]
    cos_half = torch.randn(batch, seqlen, dim//2, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.randn(batch, seqlen, dim//2, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    
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
    test_gqa_decode()
    # test_rope()

    # # # gen = MicroAutoGen(1, 2560, 9728)
    # gen = MicroAutoGen(batch_size=1, hidden_size=1024, intermediate_size=3072, 
    #                    kv_seqlen=8192, heads=16, groups=8, dim=128)
    # gen.gen_qwen3_ops(layer_id=99, mode=HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED
    
    # print("Test single_micro completed.")