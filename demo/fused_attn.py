import os
import torch
import argparse
import megakernel as mi

from common.pkt_util import TorchRef, PerfReporter
from common.mpk_layers import MpkLayers
from common.autogen.qwen3_mega_config import Qwen3MegaConfig

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
    w_layernorm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    w_qkv_proj_torch = torch.randn(((num_heads+2*num_kv_heads)*head_dim, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_q_norm_torch = torch.randn((seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    w_k_norm_torch = torch.randn((seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    
    cos_half = torch.randn((batch, seqlen_q, head_dim//2), dtype=torch.bfloat16, device="cuda")
    sin_half = torch.randn((batch, seqlen_q, head_dim//2), dtype=torch.bfloat16, device="cuda")
    w_cos_torch = torch.cat((cos_half, cos_half), dim=-1)
    w_sin_torch = torch.cat((sin_half, sin_half), dim=-1)
    
    key_cache_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    w_o_proj_torch = torch.randn((hidden_size, num_heads*head_dim), dtype=torch.bfloat16, device="cuda")
    
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")

    ###
    def ref_run():
        o1 = TorchRef.rms_norm(x_torch, w_layernorm_torch)
        qkv_out = TorchRef.linear(o1, w_qkv_proj_torch)
        print(x_torch.size(), w_qkv_proj_torch.size())
        
        q_dim = num_heads*head_dim
        kv_dim = num_kv_heads*head_dim
        query_states = qkv_out[:, :q_dim].view(batch, seqlen_q, num_heads , head_dim) 
        key_states = qkv_out[:, q_dim:q_dim+kv_dim].view(batch, seqlen_q, num_kv_heads, head_dim) 
        value_states = qkv_out[:, q_dim+kv_dim:].view(batch, seqlen_q, num_kv_heads, head_dim) 
        
        query_states = TorchRef.rms_norm(query_states, w_q_norm_torch)
        key_states = TorchRef.rms_norm(key_states, w_k_norm_torch)
        query_states, key_states = TorchRef.apply_rotary_pos_emb_triton(query_states, key_states, w_cos_torch, w_sin_torch, unsqueeze_dim=2)
        
        step = 63
        key_cache_torch[0, step, :, :] = key_states
        value_cache_torch[0, step, :, :] = value_states
        attn_output = TorchRef.attention_sdpa(query_states, key_cache_torch, value_cache_torch, False)
        attn_output = attn_output.reshape(batch*seqlen_q, q_dim)
        attn_output = TorchRef.linear(attn_output, w_o_proj_torch)
        return attn_output
    
    # layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    # mpk = layers.get_mpk()

    # x = mpk.attach_input(torch_tensor=x_torch, name="in")
    # w_layernorm = mpk.attach_input(torch_tensor=w_layernorm_torch, name="w_layernorm")
    # w_qkv_proj = mpk.attach_input(torch_tensor=w_qkv_proj_torch, name="w_qkv_proj")
    # # w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    # # mlp_out = mpk.attach_input(torch_tensor=out_torch, name="mlp_out")
    
    # layernorm_out = mpk.new_tensor(dims=(max_batch_size, hidden_size), dtype=mi.bfloat16, name="layernorm_out", io_category="cuda_tensor")
    # mpk.rmsnorm_layer(
    #     input=x,
    #     weight=w_layernorm,
    #     output=layernorm_out,
    #     sync_mode=(0, 0, 0),
    #     layout=Qwen3MegaConfig.rmsnorm_layout,
    # )
    # # qkv_proj_out = mpk.new_tensor(dims=(max_batch_size, (num_heads+2*num_kv_heads)*head_dim), dtype=mi.bfloat16, name="qkv_proj_out", io_category="cuda_tensor")
    # qkv_proj_out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    # qkv_proj_out = mpk.attach_input(torch_tensor=qkv_proj_out_torch, name="qkv_proj_out")
    # mpk.linear_layer(
    #     input=layernorm_out,
    #     weight=w_qkv_proj,
    #     output=qkv_proj_out,
    #     sync_mode=(0, 0, 0),
    #     layout=Qwen3MegaConfig.linear1_layout,
    # )
    
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
    #     reporter.generate_report(mpk_run, graph.replay, 
    #                             warnup_iter=100, test_iter=200, 
    #                             allclose_iter=5, print_all=False)