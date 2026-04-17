import os
import math
import itertools
import json
import torch
# 
os.environ["TL_DISABLE_WARP_SPECIALIZED"] = "1"

import tilelang
import tilelang.language as T
import tvm
from tvm.tir.stmt_functor import ir_transform

from common.pkt_util import TorchRef, PerfReporter, Qwen3Info
from common.micro_base import HparamSelectMode
from common.micro_linear import MicroLinearStrategy, MicroLinear
from common.micro_rmsnorm import MicroRmsNorm
from common.micro_silu_mul import MicroSiluMul
from common.micro_gqa_decode import MicroGqaDecode
from common.micro_rope import MicroRope

from common.micro_autogen import MicroAutoGen

def test_code_gen():
    @tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
    def kernel_load_B(M, N, BLOCK_M, BLOCK_N, threads, dtype="bfloat16"):
        @T.prim_func
        def rms_norm_load_B(B: T.Tensor((1, N), dtype)):
            with T.Kernel(1, threads=threads) as bx:
                B_shared = T.alloc_shared((1, N), dtype)
                T.copy(B[0:1, :], B_shared)
        return rms_norm_load_B

    @tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
    def kernel_main(M, N, BLOCK_M, BLOCK_N, threads, eps=1e-12, dtype="bfloat16", accum_dtype="float32"):
        @T.prim_func
        def rms_norm(A: T.Tensor((M, N), dtype), B_smem: T.Tensor((N,), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(M, BLOCK_M), threads=threads) as bx:
                A_shared = T.alloc_shared((BLOCK_M, N), dtype)
                A_pow_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
                B_local = T.alloc_fragment((1, N), accum_dtype)

                T.copy(A[bx * BLOCK_M : (bx + 1) * BLOCK_M, :], A_shared)

                for i in T.Parallel(BLOCK_M, N):
                    B_local[i % BLOCK_M, i // BLOCK_M] = B_smem[i]

                T.copy(A_shared, A_local)

                for i, j in T.Parallel(BLOCK_M, N):
                    A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
                T.reduce_sum(A_pow_local, A_powsum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)
                for i, j in T.Parallel(BLOCK_M, N):
                    A_local[i, j] *= A_powsum[i] * B_local[i, j]
                T.copy(A_local, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, :])

        return rms_norm
    
    @tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
    def kernel_main_org(M, N, BLOCK_M, BLOCK_N, threads, eps=1e-12, dtype="bfloat16", accum_dtype="float32"):
        @T.prim_func
        def rms_norm(A: T.Tensor((M, N), dtype), B: T.Tensor((1, N), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(M, BLOCK_M), threads=threads) as bx:
                A_shared = T.alloc_shared((BLOCK_M, N), dtype)
                A_pow_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
                B_shared = T.alloc_shared((1, N), dtype)
                B_local = T.alloc_fragment((1, N), accum_dtype)
                
                T.copy(B[0:1, :], B_shared)
                T.copy(A[bx * BLOCK_M : (bx + 1) * BLOCK_M, :], A_shared)
                
                T.copy(A_shared, A_local)
                T.copy(B_shared, B_local)
                
                for i, j in T.Parallel(BLOCK_M, N):
                    A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
                T.reduce_sum(A_pow_local, A_powsum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)
                for i, j in T.Parallel(BLOCK_M, N):
                    A_local[i, j] *= A_powsum[i] * B_local[0, j]
                T.copy(A_local, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, :])

        return rms_norm
    
    M = 32
    N = 2560
    a = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, N, dtype=torch.bfloat16, device="cuda")
    kernel = kernel_main_org(M, N, 1, 1, 128)
    print(kernel.get_kernel_source())
    print(kernel(a,b)) 
    
    kernel2 = kernel_load_B(M, N, 1, 1, 128)
    print(kernel2.get_kernel_source())
    
def profile(target_func, torch_ref_func):
    reporter = PerfReporter() 
    reporter.generate_report(target_func, torch_ref_func,
                            warnup_iter=100, test_iter=500, 
                            allclose_iter=5, print_mode=1)

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
    
def test_merge_rms_norm():
    M = 16
    M2 = 8
    N = 2560
    micro = MicroRmsNorm(M,N, dtype=T.bfloat16, accum_dtype=T.float32, M2=M2)
    kernel, fn, info = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    
    test_data = micro.gen_test_data(kernel.config)
    def target_func():
        return kernel(*test_data)
    
    a,b = test_data
    a1 = a[ : M, :]
    a2 = a[M : M+M2, :]
    b1 = b[0 : 1, :]
    b2 = b[1 : 2, :]
    def torch_ref():
        c1 = TorchRef.rms_norm(a1, b1) 
        c2 = TorchRef.rms_norm(a2, b2) 
        return torch.cat([c1, c2], dim=0)
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

def test_gqa_decode(num_heads, num_kv_heads, head_dim):
    batch = 1
    max_kv_seqlen = 8192
    target_kv_seqlen = 64
    valid_kv_seqlen = 64
    is_causal = False
    
    # config = [64,64,64,2,128,0,true]
    micro = MicroGqaDecode(batch, max_kv_seqlen, target_kv_seqlen, num_heads, num_kv_heads, head_dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED
    
    test_data = micro.gen_test_data(kernel.config)
    q, k, v, edge, mask, glse, Output_partial = test_data
    
    edge[0].fill_(valid_kv_seqlen)
    def target_func():
        return kernel(q, k, v, edge, mask, glse, Output_partial)
    
    k_slice = k[:, :valid_kv_seqlen, :, :]
    v_slice = v[:, :valid_kv_seqlen, :, :]
        
    # if q.ndim == 3:
    #     q_for_sdpa = q.unsqueeze(2)         # [batch, num_heads, seqlen_q=1, head_dim]
    # else:
    #     q_for_sdpa = q.permute(0, 2, 1, 3)
    # k_for_sdpa = k_slice.permute(0, 2, 1, 3)    # [batch, num_kv_heads, seqlen_kv, head_dim]  groups即是num_kv_heads
    # v_for_sdpa = v_slice.permute(0, 2, 1, 3)  # [batch, num_kv_heads, seqlen_kv, head_dim]
    
    # def torch_ref():
    #     return torch.nn.functional.scaled_dot_product_attention(
    #         q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=is_causal, enable_gqa=True
    #     )
        
    def torch_ref():
        return TorchRef.attention_sdpa(q, k_slice, v_slice, is_causal)
        # return TorchRef.attention(q, k, v, mask, glse, Output_partial)
        # return TorchRef.attention_split(q, k, v, mask, glse, Output_partial)
    profile(target_func, torch_ref)

def test_rope(num_heads, num_kv_heads, head_dim):
    batch = 1
    seqlen = 1
    
    micro = MicroRope(batch, seqlen, num_heads, num_kv_heads, head_dim, dtype=T.bfloat16, accum_dtype=T.float32)
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
    model_tag = "qwen3_06b" # "qwen3_4b"
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)  
    
    # test_silu_mul()
    # test_rms_norm()
    # test_merge_rms_norm()
    # test_gemm()
    ## test_silu_mul_gemm() # 逻辑有误，silu_mul被重复计算
    # test_gemm_add()
     
    # test_gqa_decode(num_heads, num_kv_heads, head_dim)
    # test_rope(num_heads, num_kv_heads, head_dim)

    # test_code_gen()
    
    gen = MicroAutoGen(model_tag, batch_size=1, hidden_size=hidden_size, intermediate_size=intermediate_size, 
                       max_kv_seqlen=8192, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    gen.gen_qwen3_ops(layer_id=99, mode=HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED
    # print(">> Finish gen_qwen3_ops.")
    # print("Test single_micro completed.")
    
    # PerfReporter.draw()