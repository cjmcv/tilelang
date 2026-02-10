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

from common.micro_autogen import MicroAutoGen

def profile(target_func, torch_ref_func):
    reporter = PerfReporter() 
    reporter.generate_report(target_func, None, 1, 
                            torch_ref_func, None, 
                            warnup_iter=100, test_iter=200, 
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
    kv_seqlen = 2000
    dim = 128
    is_causal = False
    
    # config = [64,64,64,2,128,0,true]
    micro = MicroGqaDecode(batch, kv_seqlen, heads, groups, dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)              # [B, N=q_seqlen=1, H=heads,  D=dim]
    k = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=kv_seqlen,  H=groups, D=dim]
    v = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)
    # mask = torch.randint(0, 2, (batch, kv_seqlen, groups), device="cuda", dtype=torch.uint8) # Only 0/1
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
        return kernel(q, k, v, mask, glse, Output_partial)
    
    def torch_ref():
        return TorchRef.attention_sdpa(q, k, v, is_causal)
        # return TorchRef.attention(q, k, v, mask, glse, Output_partial)
        # return TorchRef.attention_split(q, k, v, mask, glse, Output_partial)
        
    profile(target_func, torch_ref)
    

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos torch.Size([1, 1, 1, 128]) q torch.Size([1, 1, 16, 128]) k torch.Size([1, 1, 8, 128])
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 更简洁的版本：直接使用if表达式
@tilelang.jit(out_idx=[-1])
def rope_kernel_simple_direct(
    batch, kv_seqlen, num_kv_heads, head_dim, 
    BLOCK_BATCH, BLOCK_SEQLEN, BLOCK_HEADS,
    threads, dtype="bfloat16", accum_dtype="float32"
):
    @T.prim_func
    def rope_direct_impl(
        K: T.Tensor((batch, kv_seqlen, num_kv_heads, head_dim), dtype),
        COS: T.Tensor((batch, kv_seqlen, 1, head_dim), dtype),
        SIN: T.Tensor((batch, kv_seqlen, 1, head_dim), dtype),
        K_out: T.Tensor((batch, kv_seqlen, num_kv_heads, head_dim), dtype),
    ):
        with T.Kernel(batch, T.ceildiv(num_kv_heads, BLOCK_HEADS), threads=threads) as (bx, by):
            valid_block_heads = T.min(BLOCK_HEADS, num_kv_heads - by * BLOCK_HEADS)
            
            # 为当前block分配shared memory
            K_sh = T.alloc_shared((kv_seqlen, BLOCK_HEADS, head_dim), dtype)
            half_hdim = head_dim // 2
            COS_sh = T.alloc_shared((kv_seqlen, half_hdim), dtype)
            SIN_sh = T.alloc_shared((kv_seqlen, half_hdim), dtype)
            K_out_sh = T.alloc_shared((kv_seqlen, BLOCK_HEADS, head_dim), dtype)
            
            # 加载数据
            for s in T.serial(kv_seqlen):
                T.copy(COS[bx, s, 0, 0:half_hdim], COS_sh[s, :])
                T.copy(SIN[bx, s, 0, 0:half_hdim], SIN_sh[s, :])
                
                for h in range(BLOCK_HEADS):
                    head_idx = by * BLOCK_HEADS + h
                    if head_idx < num_kv_heads:
                        T.copy(K[bx, s, head_idx, :], K_sh[s, h, :])
            
            # 计算RoPE
            for s, h, d in T.Parallel(kv_seqlen, BLOCK_HEADS, head_dim):
                # 检查边界
                head_idx = by * BLOCK_HEADS + h
                if head_idx < num_kv_heads:
                    k_val = K_sh[s, h, d].astype(accum_dtype)
                    cos_sin_idx = d % half_hdim
                    
                    # 使用if表达式计算rotate_val
                    rotate_val = T.if_then_else(
                        d < half_hdim,
                        -K_sh[s, h, d + half_hdim].astype(accum_dtype),
                        K_sh[s, h, d - half_hdim].astype(accum_dtype)
                    )
                    
                    cos_val = COS_sh[s, cos_sin_idx].astype(accum_dtype)
                    sin_val = SIN_sh[s, cos_sin_idx].astype(accum_dtype)
                    
                    k_rope = k_val * cos_val + rotate_val * sin_val
                    K_out_sh[s, h, d] = k_rope.astype(dtype)
            
            # 写回结果
            for s in T.serial(kv_seqlen):
                for h in range(BLOCK_HEADS):
                    head_idx = by * BLOCK_HEADS + h
                    if head_idx < num_kv_heads:
                        T.copy(K_out_sh[s, h, :], K_out[bx, s, head_idx, :])
    
    return rope_direct_impl
@tilelang.jit(out_idx=-1)
def rope_kernel_k_optimized(batch, seq_len, num_heads, head_dim, 
                        BLOCK_SEQ=64, BLOCK_HEADS=4, BLOCK_DIM=64, threads=128,
                        dtype="bfloat16", accum_dtype="float32"):
    @T.prim_func
    def rope_k_final(
        K: T.Tensor((batch, seq_len, num_heads, head_dim), dtype),
        cos: T.Tensor((batch, seq_len, head_dim), dtype),
        sin: T.Tensor((batch, seq_len, head_dim), dtype),
        K_embed: T.Tensor((batch, seq_len, num_heads, head_dim), dtype),
    ):
        with T.Kernel(batch, 
                      T.ceildiv(num_heads, BLOCK_HEADS),
                      T.ceildiv(seq_len, BLOCK_SEQ),
                      threads=threads) as (bx, by, bz):
            
            batch_id = bx
            head_start = by * BLOCK_HEADS
            seq_start = bz * BLOCK_SEQ
            half_dim = head_dim // 2
            
            # Allocate shared memory
            K_first = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS, half_dim), dtype)   # [0:half]
            K_second = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS, half_dim), dtype)  # [half:end]
            cos_sh = T.alloc_shared((BLOCK_SEQ, head_dim), dtype)  # 需要完整的cos/sin
            sin_sh = T.alloc_shared((BLOCK_SEQ, head_dim), dtype)
            
            # Load cos/sin（完整的head_dim）
            for s, d in T.Parallel(BLOCK_SEQ, head_dim):
                global_seq = seq_start + s
                if global_seq < seq_len:
                    cos_sh[s, d] = cos[batch_id, global_seq, d]
                    sin_sh[s, d] = sin[batch_id, global_seq, d]
            
            # Load K split into two halves
            for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS, half_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                if global_seq < seq_len and global_head < num_heads:
                    K_first[s, h, d] = K[batch_id, global_seq, global_head, d]
                    K_second[s, h, d] = K[batch_id, global_seq, global_head, d + half_dim]
            
            # Compute RoPE - 正确的公式
            for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS, half_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                if global_seq < seq_len and global_head < num_heads:
                    
                    # 获取k的前后部分
                    a = K_first[s, h, d].astype(accum_dtype)      # k_first
                    b = K_second[s, h, d].astype(accum_dtype)     # k_second
                    
                    # 获取对应位置的cos/sin
                    c1 = cos_sh[s, d].astype(accum_dtype)         # cos前一半
                    s1 = sin_sh[s, d].astype(accum_dtype)         # sin前一半
                    c2 = cos_sh[s, d + half_dim].astype(accum_dtype)  # cos后一半
                    s2 = sin_sh[s, d + half_dim].astype(accum_dtype)  # sin后一半
                    
                    # 前一半：a * c1 - b * s1
                    out_first = a * c1 - b * s1
                    # 后一半：b * c2 + a * s2
                    out_second = b * c2 + a * s2
                    
                    # Store back
                    K_embed[batch_id, global_seq, global_head, d] = out_first.astype(dtype)
                    K_embed[batch_id, global_seq, global_head, d + half_dim] = out_second.astype(dtype)
    
    return rope_k_final

def test_rope():
    batch = 1
    heads = 16
    groups = 8
    q_seqlen = 1
    kv_seqlen = 200
    dim = 128
    is_causal = False
    
    # torch.Size([1, 1, 16, 128]) torch.Size([1, 1, 8, 128]) torch.Size([1, 1, 128]) torch.Size([1, 1, 128])
    # config = [64,64,64,2,128,0,true]
    # micro = MicroGqaDecode(batch, kv_seqlen, heads, groups, dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32)
    # kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    q = torch.randn(batch,  q_seqlen,  heads, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=q_seqlen=1, H=heads,  D=dim]
    k = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=kv_seqlen,  H=groups, D=dim]
    cos = torch.randn(batch, kv_seqlen, dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(batch, kv_seqlen, dim, device="cuda", dtype=torch.bfloat16)

    # def triton_ref():
    #     from models.rope import apply_rotary_pos_emb_triton
    #     query_states, key_states = apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=2)
    #     return query_states, key_states
    
    def torch_ref():
        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2)
        return query_states, key_states
    
    # cos = cos.unsqueeze(-2)
    # sin = sin.unsqueeze(-2)
    # kernel = rope_main(batch, q_seqlen, heads, kv_seqlen, groups, dim)
    # q_embed, k_embed = kernel(q, k, cos, sin)
    # kernel = kernel_rope_qk(batch, q_seqlen, heads, groups, dim, 1, 16, 16, 128)
    # kernel.export_sources(kernel_path="demo/gen/single_micro.cu")
    # q_embed, k_embed = kernel(q, k, cos, sin)
    
    print(cos.size(), sin.size(), batch, kv_seqlen, groups, dim)
    kernel = rope_kernel_k_optimized(batch, kv_seqlen, groups, dim, 1, 1, 8)
    # kernel = rope_kernel_simple_direct(batch, kv_seqlen, groups, dim, 1, 1, 8, 128)
    k_embed = kernel(k, cos, sin)
    
    # q_embed_triton, k_embed_triton = triton_ref()
    # print("triton_ref", q_embed_triton, "\n", k_embed_triton)
    q_embed_torch, k_embed_torch = torch_ref()
    print("torch_ref", q_embed_torch, "\n", k_embed_torch)
    
    print("tilelang", k_embed)
    rep = PerfReporter()
    rep.assert_similar(k_embed, k_embed_torch)    
    
    print("shape", k_embed.size(), q_embed_torch.size(), k_embed_torch.size())
    
if __name__ == "__main__":
    # test_silu_mul()
    # test_rms_norm()
    # test_gemm()
    ## test_silu_mul_gemm() # 逻辑有误，silu_mul被重复计算
    # test_gemm_add()
    # test_gqa_decode()
    test_rope()

    # # # gen = MicroAutoGen(1, 2560, 9728)
    # gen = MicroAutoGen(batch_size=1, hidden_size=1024, intermediate_size=3072, 
    #                    kv_seqlen=8192, heads=16, groups=8, dim=128)
    # gen.gen_qwen3_mlp(layer_id=99, mode=HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED
    
    print("Test single_micro completed.")