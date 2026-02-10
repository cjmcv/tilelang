import os
import torch
import argparse
import megakernel as mi

from common.pkt_util import TorchRef, PerfReporter
from common.mpk_layers import MpkLayers
from common.autogen.qwen3_mlp_config import Qwen3MlpConfig

if __name__ == "__main__":
    max_batch_size = 1
    batch = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.getenv("MEGAKERNEL_HOME", default=None)+"/demo/gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")

    args = parser.parse_args()
    world_size = 1
    rank = 0

    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    # model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    reporter = PerfReporter() 
    # reporter.memory_footprint_simulation(rank)
    # q_proj: torch.Size([2048, 1024])
    # k_proj: torch.Size([1024, 1024])
    # v_proj: torch.Size([1024, 1024])
    # o_proj: torch.Size([1024, 2048])
    
    hidden_size = 1024
    intermediate_size = 3072
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    seqlen_q = 1
    seqlen_kv = 64
    x_torch = torch.randn((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_layer_norm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    w_qkv_proj_torch = torch.randn(((num_heads+2*num_kv_heads)*head_dim, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_q_norm_torch = torch.randn((seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    w_k_norm_torch = torch.randn((seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    w_cos_torch = torch.randn((batch, seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    w_sin_torch = torch.randn((batch, seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    
    key_cache_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    w_o_proj_torch = torch.randn((hidden_size, num_heads*head_dim), dtype=torch.bfloat16, device="cuda")
    
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")

    ###
    def ref_run():
        o1 = TorchRef.rms_norm(x_torch, w_layer_norm_torch)
        qkv_out = TorchRef.linear(o1, w_qkv_proj_torch)
        
        q_dim = num_heads*head_dim
        kv_dim = num_kv_heads*head_dim
        query_states = qkv_out[:, :q_dim].view(batch, seqlen_q, num_heads , head_dim) 
        key_states = qkv_out[:, q_dim:q_dim+kv_dim].view(batch, seqlen_q, num_kv_heads, head_dim) 
        value_states = qkv_out[:, q_dim+kv_dim:].view(batch, seqlen_q, num_kv_heads, head_dim) 
        
        query_states = TorchRef.rms_norm(query_states, w_q_norm_torch)
        key_states = TorchRef.rms_norm(key_states, w_k_norm_torch)
        
        from models.rope import apply_rotary_pos_emb_triton
        query_states, key_states = apply_rotary_pos_emb_triton(
            query_states, key_states, w_cos_torch, w_sin_torch, unsqueeze_dim=2
        )
        
        step = 63
        key_cache_torch[0, step, :, :] = key_states
        value_cache_torch[0, step, :, :] = value_states
        # print("q", query_states.size())
        # print("cache k", key_cache_torch.size())
        # print("cache v", value_cache_torch.size())
        attn_output = TorchRef.attention_sdpa(query_states, key_cache_torch, value_cache_torch, False)
        
        attn_output = attn_output.reshape(batch, seqlen_q, q_dim)
        attn_output = TorchRef.linear(attn_output, w_o_proj_torch)
        # print(key_states.size())
        # print(value_states.size())
        # print(out.size())
        return attn_output
    
    graph, ref_output = TorchRef.compile_capture(ref_run, is_compile=False)
    
    # def mpk_run():
    #     mpk(batch_size)
        
    ref_output = ref_run()
    print(ref_output)
    # mpk_output = out_torch[:batch_size]

    for _ in range(100):
        graph.replay()
        # ref_run()
        
    # mpk_run()
    ##
    
    # if not args.profiling:
    #     reporter.generate_report(mpk_run, mpk_output, splitk, 
    #                             graph.replay, ref_output, 
    #                             warnup_iter=100, test_iter=200, 
    #                             allclose_iter=5, print_all=False)