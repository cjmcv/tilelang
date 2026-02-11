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

@tilelang.jit(out_idx=-1)
def rope_split_n_v2(batch, seq_len, num_heads, head_dim, 
                    BLOCK_SEQ, BLOCK_HEADS, threads=128,
                    dtype="bfloat16", accum_dtype="float32"):
    """
    更现实的RoPE kernel版本，考虑实际分块策略
    """
    @T.prim_func
    def rope_k_realistic(
        K: T.Tensor((batch, seq_len, num_heads, head_dim), dtype),
        cos: T.Tensor((batch, seq_len, head_dim), dtype),
        sin: T.Tensor((batch, seq_len, head_dim), dtype),
        K_embed: T.Tensor((batch, seq_len, num_heads, head_dim), dtype),
    ):
        half_dim = head_dim // 2
        
        with T.Kernel(batch, 
                      T.ceildiv(num_heads, BLOCK_HEADS),
                      T.ceildiv(seq_len, BLOCK_SEQ),
                      threads=threads) as (bx, by, bz):
            
            batch_id = bx
            head_start = by * BLOCK_HEADS
            seq_start = bz * BLOCK_SEQ

            # 完整存储K
            K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS, head_dim), dtype)
            cos_sh = T.alloc_shared((BLOCK_SEQ, head_dim), dtype)
            sin_sh = T.alloc_shared((BLOCK_SEQ, head_dim), dtype)
            
            # 1. 加载cos/sin（每个seq位置只加载一次，被所有heads共享）
            for s, d in T.Parallel(BLOCK_SEQ, head_dim):
                global_seq = seq_start + s
                if global_seq < seq_len:
                    cos_sh[s, d] = cos[batch_id, global_seq, d]
                    sin_sh[s, d] = sin[batch_id, global_seq, d]
            
            # 2. 加载K（完整head_dim）
            for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS, head_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                if global_seq < seq_len and global_head < num_heads:
                    K_sh[s, h, d] = K[batch_id, global_seq, global_head, d]
            
            # 3. 计算RoPE
            for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS, half_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                
                if global_seq < seq_len and global_head < num_heads:
                    # 从shared memory读取（都在同一个block内）
                    a = K_sh[s, h, d].astype(accum_dtype)                    # 前一半
                    b = K_sh[s, h, d + half_dim].astype(accum_dtype)         # 后一半
                    
                    # cos/sin也被所有heads共享
                    c1 = cos_sh[s, d].astype(accum_dtype)                    # cos前一半
                    s1 = sin_sh[s, d].astype(accum_dtype)                    # sin前一半
                    c2 = cos_sh[s, d + half_dim].astype(accum_dtype)         # cos后一半
                    s2 = sin_sh[s, d + half_dim].astype(accum_dtype)         # sin后一半
                    
                    # RoPE公式
                    out_first = a * c1 - b * s1
                    out_second = b * c2 + a * s2
                    
                    K_embed[batch_id, global_seq, global_head, d] = out_first.astype(dtype)
                    K_embed[batch_id, global_seq, global_head, d + half_dim] = out_second.astype(dtype)
    
    return rope_k_realistic

@tilelang.jit(out_idx=[-2, -1])  # 输出最后两个张量：Q_embed和K_embed
def rope_fuse_qk_v2(batch, seq_len, num_heads_q, num_heads_k, head_dim, 
                    BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads=128,
                    dtype="bfloat16", accum_dtype="float32"):
    """
    融合Q和K的RoPE计算kernel
    假设Q和K的seq_len相同，head_dim相同，但head数可能不同
    """
    @T.prim_func
    def rope_qk_realistic(
        Q: T.Tensor((batch, seq_len, num_heads_q, head_dim), dtype),
        K: T.Tensor((batch, seq_len, num_heads_k, head_dim), dtype),
        cos: T.Tensor((batch, seq_len, head_dim), dtype),
        sin: T.Tensor((batch, seq_len, head_dim), dtype),
        Q_embed: T.Tensor((batch, seq_len, num_heads_q, head_dim), dtype),
        K_embed: T.Tensor((batch, seq_len, num_heads_k, head_dim), dtype),
    ):
        half_dim = head_dim // 2
        
        # 使用Q和K的最大head块数
        max_head_blocks = T.max(
            T.ceildiv(num_heads_q, BLOCK_HEADS_Q),
            T.ceildiv(num_heads_k, BLOCK_HEADS_K)
        )
        
        with T.Kernel(batch, 
                      max_head_blocks,
                      T.ceildiv(seq_len, BLOCK_SEQ),
                      threads=threads) as (bx, by, bz):
            
            batch_id = bx
            seq_start = bz * BLOCK_SEQ

            # 分配shared memory
            # cos/sin可以被Q和K共享
            cos_sh = T.alloc_shared((BLOCK_SEQ, head_dim), dtype)
            sin_sh = T.alloc_shared((BLOCK_SEQ, head_dim), dtype)
            
            # 1. 加载cos/sin（每个seq位置只加载一次，被所有heads和Q/K共享）
            for s, d in T.Parallel(BLOCK_SEQ, head_dim):
                global_seq = seq_start + s
                if global_seq < seq_len:
                    cos_sh[s, d] = cos[batch_id, global_seq, d]
                    sin_sh[s, d] = sin[batch_id, global_seq, d]
            
            # 2. 处理Q（如果当前head块对应Q）
            head_q_start = by * BLOCK_HEADS_Q
            if head_q_start < num_heads_q:
                Q_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_Q, head_dim), dtype)
                
                # 加载Q数据
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, head_dim):
                    global_seq = seq_start + s
                    global_head = head_q_start + h
                    if global_seq < seq_len and global_head < num_heads_q:
                        Q_sh[s, h, d] = Q[batch_id, global_seq, global_head, d]
                
                # 计算Q的RoPE
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, half_dim):
                    global_seq = seq_start + s
                    global_head = head_q_start + h
                    
                    if global_seq < seq_len and global_head < num_heads_q:
                        # 读取Q数据
                        a_q = Q_sh[s, h, d].astype(accum_dtype)
                        b_q = Q_sh[s, h, d + half_dim].astype(accum_dtype)
                        
                        # cos/sin被所有heads共享
                        c1 = cos_sh[s, d].astype(accum_dtype)
                        s1 = sin_sh[s, d].astype(accum_dtype)
                        c2 = cos_sh[s, d + half_dim].astype(accum_dtype)
                        s2 = sin_sh[s, d + half_dim].astype(accum_dtype)
                        
                        # Q的RoPE公式
                        out_first_q = a_q * c1 - b_q * s1
                        out_second_q = b_q * c2 + a_q * s2
                        
                        Q_embed[batch_id, global_seq, global_head, d] = out_first_q.astype(dtype)
                        Q_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_q.astype(dtype)
            
            # 3. 处理K（如果当前head块对应K）
            head_k_start = by * BLOCK_HEADS_K
            if head_k_start < num_heads_k:
                K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_K, head_dim), dtype)
                
                # 加载K数据
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, head_dim):
                    global_seq = seq_start + s
                    global_head = head_k_start + h
                    if global_seq < seq_len and global_head < num_heads_k:
                        K_sh[s, h, d] = K[batch_id, global_seq, global_head, d]
                
                # 计算K的RoPE
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, half_dim):
                    global_seq = seq_start + s
                    global_head = head_k_start + h
                    
                    if global_seq < seq_len and global_head < num_heads_k:
                        # 读取K数据
                        a_k = K_sh[s, h, d].astype(accum_dtype)
                        b_k = K_sh[s, h, d + half_dim].astype(accum_dtype)
                        
                        # 使用相同的cos/sin
                        c1 = cos_sh[s, d].astype(accum_dtype)
                        s1 = sin_sh[s, d].astype(accum_dtype)
                        c2 = cos_sh[s, d + half_dim].astype(accum_dtype)
                        s2 = sin_sh[s, d + half_dim].astype(accum_dtype)
                        
                        # K的RoPE公式
                        out_first_k = a_k * c1 - b_k * s1
                        out_second_k = b_k * c2 + a_k * s2
                        
                        K_embed[batch_id, global_seq, global_head, d] = out_first_k.astype(dtype)
                        K_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_k.astype(dtype)
    
    return rope_qk_realistic

@tilelang.jit(out_idx=[-2, -1])
def rope_fuse_qk_v2_optimized(batch, seq_len, num_heads_q, num_heads_k, head_dim, 
                              BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads=128,
                              dtype="bfloat16", accum_dtype="float32"):
    """
    优化版本：利用cos/sin的前后一半相同的特点
    1. 只加载一半的cos/sin到shared memory（节省50% shared memory）
    2. 减少一半的cos/sin加载操作
    3. 使用相同的cos/sin值计算前后half_dim
    """
    @T.prim_func
    def rope_qk_optimized(
        Q: T.Tensor((batch, seq_len, num_heads_q, head_dim), dtype),
        K: T.Tensor((batch, seq_len, num_heads_k, head_dim), dtype),
        cos: T.Tensor((batch, seq_len, head_dim), dtype),
        sin: T.Tensor((batch, seq_len, head_dim), dtype),
        Q_embed: T.Tensor((batch, seq_len, num_heads_q, head_dim), dtype),
        K_embed: T.Tensor((batch, seq_len, num_heads_k, head_dim), dtype),
    ):
        half_dim = head_dim // 2
        
        # 验证cos/sin的前后一半相同（可选，调试用）
        # 注意：在实际产品代码中可能不需要这个断言
        # T.Assert(T.all(cos[..., :half_dim] == cos[..., half_dim:]), "cos前后一半应该相同")
        # T.Assert(T.all(sin[..., :half_dim] == sin[..., half_dim:]), "sin前后一半应该相同")
        
        max_head_blocks = T.max(
            T.ceildiv(num_heads_q, BLOCK_HEADS_Q),
            T.ceildiv(num_heads_k, BLOCK_HEADS_K)
        )
        
        with T.Kernel(batch, 
                      max_head_blocks,
                      T.ceildiv(seq_len, BLOCK_SEQ),
                      threads=threads) as (bx, by, bz):
            
            batch_id = bx
            seq_start = bz * BLOCK_SEQ

            # 优化1：只存储一半的cos/sin到shared memory
            cos_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)  # 节省50%
            sin_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)  # 节省50%
            
            # 1. 加载cos/sin（只加载前一半）
            for s, d in T.Parallel(BLOCK_SEQ, half_dim):
                global_seq = seq_start + s
                if global_seq < seq_len:
                    cos_sh[s, d] = cos[batch_id, global_seq, d]  # 只取前half_dim
                    sin_sh[s, d] = sin[batch_id, global_seq, d]  # 只取前half_dim
            
            # 2. 处理Q（如果当前head块对应Q）
            head_q_start = by * BLOCK_HEADS_Q
            if head_q_start < num_heads_q:
                # 优化2：可以只存储Q的一半，计算时重组
                # 但为了简单，我们先保持完整存储
                Q_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_Q, head_dim), dtype)
                
                # 加载Q数据
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, head_dim):
                    global_seq = seq_start + s
                    global_head = head_q_start + h
                    if global_seq < seq_len and global_head < num_heads_q:
                        Q_sh[s, h, d] = Q[batch_id, global_seq, global_head, d]
                
                # 计算Q的RoPE（使用优化后的cos/sin）
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, half_dim):
                    global_seq = seq_start + s
                    global_head = head_q_start + h
                    
                    if global_seq < seq_len and global_head < num_heads_q:
                        # 读取Q数据
                        a_q = Q_sh[s, h, d].astype(accum_dtype)          # 前一半
                        b_q = Q_sh[s, h, d + half_dim].astype(accum_dtype)  # 后一半
                        
                        # 关键优化：使用相同的cos/sin值（只从cos_sh取前half_dim）
                        cos_val = cos_sh[s, d].astype(accum_dtype)
                        sin_val = sin_sh[s, d].astype(accum_dtype)
                        
                        # 前后half_dim使用相同的cos/sin值
                        # 前half_dim: a*cos - b*sin
                        # 后half_dim: b*cos + a*sin
                        out_first_q = a_q * cos_val - b_q * sin_val
                        out_second_q = b_q * cos_val + a_q * sin_val
                        
                        Q_embed[batch_id, global_seq, global_head, d] = out_first_q.astype(dtype)
                        Q_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_q.astype(dtype)
            
            # 3. 处理K（如果当前head块对应K）
            head_k_start = by * BLOCK_HEADS_K
            if head_k_start < num_heads_k:
                K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_K, head_dim), dtype)
                
                # 加载K数据
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, head_dim):
                    global_seq = seq_start + s
                    global_head = head_k_start + h
                    if global_seq < seq_len and global_head < num_heads_k:
                        K_sh[s, h, d] = K[batch_id, global_seq, global_head, d]
                
                # 计算K的RoPE（使用相同的cos/sin）
                for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, half_dim):
                    global_seq = seq_start + s
                    global_head = head_k_start + h
                    
                    if global_seq < seq_len and global_head < num_heads_k:
                        # 读取K数据
                        a_k = K_sh[s, h, d].astype(accum_dtype)
                        b_k = K_sh[s, h, d + half_dim].astype(accum_dtype)
                        
                        # 使用相同的cos/sin值
                        cos_val = cos_sh[s, d].astype(accum_dtype)
                        sin_val = sin_sh[s, d].astype(accum_dtype)
                        
                        # K的RoPE公式
                        out_first_k = a_k * cos_val - b_k * sin_val
                        out_second_k = b_k * cos_val + a_k * sin_val
                        
                        K_embed[batch_id, global_seq, global_head, d] = out_first_k.astype(dtype)
                        K_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_k.astype(dtype)
    
    return rope_qk_optimized

@tilelang.jit(out_idx=[-2, -1])
def rope_fuse_qk_parallel(batch, seq_len, num_heads_q, num_heads_k, head_dim, 
                      BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads=128,
                      dtype="bfloat16", accum_dtype="float32"):
    """
    使用T.macro提取公共计算逻辑的版本
    """
    half_dim = head_dim // 2
    q_shape = [batch, seq_len, num_heads_q, head_dim]
    k_shape = [batch, seq_len, num_heads_k, head_dim]
    
    @T.macro
    def ComputeRoPE(
        data_ptr: T.Tensor,          # Q或K的输入张量
        cos_ptr: T.Tensor,           # cos张量
        sin_ptr: T.Tensor,           # sin张量  
        out_ptr: T.Tensor,           # 输出张量
        batch_id: T.int32,           # batch索引
        head_start: T.int32,         # head起始索引
        seq_start: T.int32,          # seq起始索引
        num_heads: T.int32,          # head总数
        block_heads: T.int32,        # 每个block的head数
        data_sh: T.SharedBuffer,     # 数据shared memory
        cos_sh: T.SharedBuffer,      # cos shared memory
        sin_sh: T.SharedBuffer,      # sin shared memory
    ):
        """RoPE计算的宏"""
        # 加载cos/sin到shared memory
        for s, d in T.Parallel(BLOCK_SEQ, half_dim):
            global_seq = seq_start + s
            if global_seq < seq_len:
                cos_sh[s, d] = cos_ptr[batch_id, global_seq, d]
                sin_sh[s, d] = sin_ptr[batch_id, global_seq, d]
        
        # 加载数据到shared memory
        for s, h, d in T.Parallel(BLOCK_SEQ, block_heads, head_dim):
            global_seq = seq_start + s
            global_head = head_start + h
            if global_seq < seq_len and global_head < num_heads:
                data_sh[s, h, d] = data_ptr[batch_id, global_seq, global_head, d]
        
        # 计算RoPE并写回
        for s, h, d in T.Parallel(BLOCK_SEQ, block_heads, half_dim):
            global_seq = seq_start + s
            global_head = head_start + h
            
            if global_seq < seq_len and global_head < num_heads:
                a = data_sh[s, h, d].astype(accum_dtype)
                b = data_sh[s, h, d + half_dim].astype(accum_dtype)
                
                cos_val = cos_sh[s, d].astype(accum_dtype)
                sin_val = sin_sh[s, d].astype(accum_dtype)
                
                out_first = a * cos_val - b * sin_val
                out_second = b * cos_val + a * sin_val
                
                out_ptr[batch_id, global_seq, global_head, d] = out_first.astype(dtype)
                out_ptr[batch_id, global_seq, global_head, d + half_dim] = out_second.astype(dtype)
    
    @T.prim_func
    def rope_qk_macro(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        cos: T.Tensor((batch, seq_len, head_dim), dtype),
        sin: T.Tensor((batch, seq_len, head_dim), dtype),
        Q_embed: T.Tensor(q_shape, dtype),
        K_embed: T.Tensor(k_shape, dtype),
    ):
        # 计算总block数
        total_q_blocks = batch * T.ceildiv(num_heads_q, BLOCK_HEADS_Q) * T.ceildiv(seq_len, BLOCK_SEQ)
        total_k_blocks = batch * T.ceildiv(num_heads_k, BLOCK_HEADS_K) * T.ceildiv(seq_len, BLOCK_SEQ)
        total_blocks = total_q_blocks + total_k_blocks
        
        with T.Kernel(total_blocks, threads=threads) as (bidx,):
            
            cos_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)
            sin_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)
            
            # Q block
            if bidx < total_q_blocks:
                blocks_per_batch = T.ceildiv(num_heads_q, BLOCK_HEADS_Q) * T.ceildiv(seq_len, BLOCK_SEQ)
                batch_id = bidx // blocks_per_batch
                remaining = bidx % blocks_per_batch
                head_block = remaining // T.ceildiv(seq_len, BLOCK_SEQ)
                seq_block = remaining % T.ceildiv(seq_len, BLOCK_SEQ)
                
                head_start = head_block * BLOCK_HEADS_Q
                seq_start = seq_block * BLOCK_SEQ
                
                if head_start < num_heads_q:
                    Q_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_Q, head_dim), dtype)
                    ComputeRoPE(
                        Q, cos, sin, Q_embed,
                        batch_id, head_start, seq_start, num_heads_q, BLOCK_HEADS_Q,
                        Q_sh, cos_sh, sin_sh
                    )
            
            # K block  
            else:
                k_bidx = bidx - total_q_blocks
                k_blocks_per_batch = T.ceildiv(num_heads_k, BLOCK_HEADS_K) * T.ceildiv(seq_len, BLOCK_SEQ)
                k_batch_id = k_bidx // k_blocks_per_batch
                k_remaining = k_bidx % k_blocks_per_batch
                k_head_block = k_remaining // T.ceildiv(seq_len, BLOCK_SEQ)
                seq_block = k_remaining % T.ceildiv(seq_len, BLOCK_SEQ)
                
                head_start = k_head_block * BLOCK_HEADS_K
                seq_start = seq_block * BLOCK_SEQ
                
                if head_start < num_heads_k:
                    K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_K, head_dim), dtype)
                    ComputeRoPE(
                        K, cos, sin, K_embed,
                        k_batch_id, head_start, seq_start, num_heads_k, BLOCK_HEADS_K,
                        K_sh, cos_sh, sin_sh
                    )
    return rope_qk_macro

def test_rope():
    batch = 1
    heads = 16
    groups = 8
    seqlen = 1
    dim = 128
    is_causal = False
    
    # torch.Size([1, 1, 16, 128]) torch.Size([1, 1, 8, 128]) torch.Size([1, 1, 128]) torch.Size([1, 1, 128])
    # config = [64,64,64,2,128,0,true]
    # micro = MicroGqaDecode(batch, kv_seqlen, heads, groups, dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32)
    # kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    q = torch.randn(batch,  seqlen,  heads, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen=1, H=heads,  D=dim]
    k = torch.randn(batch, seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)   # [B, N=seqlen=1, H=groups, D=dim]
    cos_half = torch.randn(batch, seqlen, dim//2, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.randn(batch, seqlen, dim//2, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    
    def triton_ref():
        from models.rope import apply_rotary_pos_emb_triton
        q_emb, k_emb = apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)
    # q_embed_triton, k_embed_triton = triton_ref()
    # print("triton_ref", q_embed_triton, "\n", k_embed_triton) 
       
    def torch_ref():
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)
    
    # kernel1 = rope_split_n1_v2(batch, groups, dim, BLOCK_HEADS=8)
    # kernel_q = rope_split_n_v2(batch, seqlen, heads, dim, BLOCK_SEQ=1, BLOCK_HEADS=1)
    # kernel_k = rope_split_n_v2(batch, seqlen, groups, dim, BLOCK_SEQ=8, BLOCK_HEADS=8)
    # kernel = rope_fuse_qk_v2_optimized(batch, seqlen, heads, groups, dim, BLOCK_SEQ=1, BLOCK_HEADS_Q=1, BLOCK_HEADS_K=1)
    kernel = rope_fuse_qk_parallel(batch, seqlen, heads, groups, dim, BLOCK_SEQ=1, BLOCK_HEADS_Q=1, BLOCK_HEADS_K=1)
    # kernel = rope_split_n1_v2(batch, groups, dim, 1, 8)
    # kernel.export_sources(kernel_path="demo/gen/single_micro.cu")
    
    def target_func():
        q_emb, k_emb = kernel(q, k, cos, sin)
        return torch.cat((q_emb, k_emb), dim=-2)
    
    # def target_func2():
    #     return kernel2(k, cos, sin)    
    # q_embed_torch, k_embed_torch = torch_ref()
    # print("torch_ref", q_embed_torch, "\n", k_embed_torch)
    
    # print("tilelang", k_embed)
    # rep = PerfReporter()
    # rep.assert_similar(k_embed, k_embed_torch)    
    profile(target_func, triton_ref)
    
    # print("shape", k_embed.size(), q_embed_torch.size(), k_embed_torch.size())
    
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