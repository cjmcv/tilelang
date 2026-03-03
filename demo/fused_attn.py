import os
import torch
import argparse
import megakernel as mi

from common.pkt_util import TorchRef, PerfReporter, Qwen3Info
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
    
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim = Qwen3Info.get_basic_params(0.6)

    seqlen_q = 1
    max_seqlen_kv = 8192
    x_torch = torch.randn((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_layernorm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    w_qkv_proj_torch = torch.randn(((num_heads+2*num_kv_heads)*head_dim, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_q_norm_torch = torch.randn((seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    w_k_norm_torch = torch.randn((seqlen_q, head_dim), dtype=torch.bfloat16, device="cuda")
    
    cos_half = torch.randn((batch, seqlen_q, head_dim//2), dtype=torch.bfloat16, device="cuda")
    sin_half = torch.randn((batch, seqlen_q, head_dim//2), dtype=torch.bfloat16, device="cuda")
    w_cos_torch = torch.cat((cos_half, cos_half), dim=-1)
    w_sin_torch = torch.cat((sin_half, sin_half), dim=-1)
    
    edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
    key_cache_torch = torch.zeros(batch, max_seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_torch = torch.zeros(batch, max_seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    w_o_proj_torch = torch.randn((hidden_size, num_heads*head_dim), dtype=torch.bfloat16, device="cuda")
    
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")

    ###
    def ref_run(step):
        o1 = TorchRef.rms_norm(x_torch, w_layernorm_torch)
        qkv_out = TorchRef.linear(o1, w_qkv_proj_torch)
        # print("qkv_out", qkv_out)
        # print("qkv_proj", x_torch.size(), w_qkv_proj_torch.size()) # qkv_proj torch.Size([1, 1024]) torch.Size([4096, 1024])
        
        q_dim = num_heads*head_dim
        kv_dim = num_kv_heads*head_dim
        query_states = qkv_out[:, :q_dim].view(batch, seqlen_q, num_heads, head_dim) 
        key_states = qkv_out[:, q_dim:q_dim+kv_dim].view(batch, seqlen_q, num_kv_heads, head_dim) 
        value_states = qkv_out[:, q_dim+kv_dim:].view(batch, seqlen_q, num_kv_heads, head_dim) 
        
        query_states = TorchRef.rms_norm(query_states, w_q_norm_torch)
        key_states = TorchRef.rms_norm(key_states, w_k_norm_torch)
        query_states, key_states = TorchRef.apply_rotary_pos_emb_triton(query_states, key_states, w_cos_torch, w_sin_torch, unsqueeze_dim=2)
        
        key_cache_torch[0, step, :, :] = key_states
        value_cache_torch[0, step, :, :] = value_states
        k_slice = key_cache_torch[:, :step+1, :, :]
        v_slice = value_cache_torch[:, :step+1, :, :]
        # k_slice.zero_()
        # v_slice.zero_()
        attn_output = TorchRef.attention_sdpa(query_states, k_slice, v_slice, False)
        attn_output = attn_output.reshape(batch*seqlen_q, q_dim)
        final_output = TorchRef.linear(attn_output, w_o_proj_torch) + x_torch # res
        print("torch", key_states, value_states, final_output)
        # print("torch:", query_states, "\n", key_states, "\n", value_states, "\n", attn_output, "\n", final_output)
        # print("o_proj", attn_output.size(), w_o_proj_torch.size()) # o_proj torch.Size([1, 1024]) torch.Size([1024, 2048])
        return final_output
    
    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()

    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_layernorm = mpk.attach_input(torch_tensor=w_layernorm_torch, name="w_layernorm")
    w_qkv_proj = mpk.attach_input(torch_tensor=w_qkv_proj_torch, name="w_qkv_proj")
    w_q_norm = mpk.attach_input(torch_tensor=w_q_norm_torch, name="w_q_norm")
    w_k_norm = mpk.attach_input(torch_tensor=w_k_norm_torch, name="w_k_norm")
    w_o_proj = mpk.attach_input(torch_tensor=w_o_proj_torch, name="w_o_proj")
    final_attn_out = mpk.attach_input(torch_tensor=out_torch, name="final_attn_out")
    
    layernorm_out = mpk.new_tensor(dims=(max_batch_size, hidden_size), dtype=mi.bfloat16, name="layernorm_out", io_category="cuda_tensor")
    mpk.rmsnorm_layer(
        input=x,
        weight=w_layernorm,
        output=layernorm_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.rmsnorm_layout,
    )
    # qkv_proj_out = mpk.new_tensor(dims=(max_batch_size, (num_heads+2*num_kv_heads)*head_dim), dtype=mi.bfloat16, name="qkv_proj_out", io_category="cuda_tensor")
    qkv_proj_out_torch = torch.zeros((max_batch_size, (num_heads+2*num_kv_heads)*head_dim), dtype=torch.bfloat16, device="cuda")
    qkv_proj_out = mpk.attach_input(torch_tensor=qkv_proj_out_torch, name="qkv_proj_out")
    mpk.linear_layer(
        input=layernorm_out,
        weight=w_qkv_proj,
        output=qkv_proj_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.qkv_proj_layout,
    )
    
    q_dim = num_heads*head_dim
    kv_dim = num_kv_heads*head_dim
    query_states_torch = qkv_proj_out_torch[:, :q_dim].view(batch*seqlen_q*num_heads, head_dim) 
    key_states_torch = qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch*seqlen_q*num_kv_heads, head_dim) 
    query_states = mpk.attach_input(torch_tensor=query_states_torch, name="query_states")
    key_states = mpk.attach_input(torch_tensor=key_states_torch, name="key_states")
    
    # print(query_states_torch.dim, key_states_torch.dim)
    # todo 合并两个norm
    mpk.rmsnorm_layer(
        input=query_states,
        weight=w_q_norm,
        output=query_states,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.q_norm_layout,
    )
    mpk.rmsnorm_layer(
        input=key_states,
        weight=w_k_norm,
        output=key_states,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.k_norm_layout,
    )
    
    # rope
    q_4dim_torch = query_states_torch.view(batch, seqlen_q, num_heads, head_dim)
    k_4dim_torch = key_states_torch.view(batch, seqlen_q, num_kv_heads, head_dim)
    q_4dim = mpk.attach_input(torch_tensor=q_4dim_torch, name="q_4dim")
    k_4dim = mpk.attach_input(torch_tensor=k_4dim_torch, name="k_4dim")
    cos = mpk.attach_input(torch_tensor=w_cos_torch, name="cos")
    sin = mpk.attach_input(torch_tensor=w_sin_torch, name="sin")
    mpk.rope_layer(
        q=q_4dim,
        k=k_4dim,
        cos=cos,
        sin=sin,
        q_embed=q_4dim,
        k_embed=k_4dim,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.rope_layout,
    )
    
    step = 0
    # attn
    q_3dim_torch = query_states_torch.view(batch, num_heads, head_dim)
    q_3dim = mpk.attach_input(torch_tensor=q_3dim_torch, name="q_3dim")
    value_states_torch = qkv_proj_out_torch[:, q_dim+kv_dim:].view(batch, seqlen_q, num_kv_heads, head_dim)
    
    key_cache_torch[0, step, :, :] = k_4dim_torch
    value_cache_torch[0, step, :, :] = value_states_torch
    
    max_attn_split = 8
    glse_torch = torch.empty(batch, num_heads, max_attn_split, device="cuda", dtype=torch.bfloat16)
    out_partial_torch = torch.empty(batch, num_heads, max_attn_split, head_dim, device="cuda", dtype=torch.bfloat16)
    mask_torch = torch.ones(batch, max_seqlen_kv, num_kv_heads, device="cuda", dtype=torch.uint8)
    
    k_cache = mpk.attach_input(torch_tensor=key_cache_torch, name="k_cache")
    v_cache = mpk.attach_input(torch_tensor=value_cache_torch, name="v_cache")
    edge = mpk.attach_input(torch_tensor=edge_torch, name="edge")
    mask = mpk.attach_input(torch_tensor=mask_torch, name="mask")
    glse = mpk.attach_input(torch_tensor=glse_torch, name="glse")
    out_partial = mpk.attach_input(torch_tensor=out_partial_torch, name="out_partial")
    
    attn_out_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    attn_out = mpk.attach_input(torch_tensor=attn_out_torch, name="attn_out") # 
    mpk.gqa_decode_layer(
        q=q_3dim,
        k_cache=k_cache,
        v_cache=v_cache,
        edge=edge,
        mask=mask,
        glse=glse,
        out_partial=out_partial,
        output=attn_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.gqa_decode_layout_16,
    )
    attn_out_2d_torch = attn_out_torch.view(batch*seqlen_q, q_dim)
    attn_out_2d = mpk.attach_input(torch_tensor=attn_out_2d_torch, name="attn_out_2d")
    
    mpk.linear_with_residual_layer(
        input=attn_out_2d,
        weight=w_o_proj,
        residual=x,
        output=final_attn_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.linear2_layout,
    )
    
    layers.compile_load(meta_tensors=[edge_torch], is_no_compile=args.nc, output_dir=args.output_dir)
    
    # step = 16
    # ref_run(step)
    
    edge_torch[0].fill_(step+1) # kv_seqlen
    mpk(batch)
    print("mpk", k_4dim_torch, value_states_torch, out_torch)
    # print("mpk:", q_4dim_torch, "\n", k_4dim_torch, "\n", value_states_torch, "\n", attn_out_torch, "\n", out_torch)

    ref_run(step)
    # def torch_fix_param():
    #     return ref_run(step)
    # graph, ref_output = TorchRef.compile_capture(torch_fix_param, is_compile=True)
    
    # edge_torch[0].fill_(step)
    # def mpk_run():
    #     mpk(batch)
    #     return out_torch
    # def torch_ref():
    #     graph.replay()
    #     return ref_output
    
    # if not args.profiling:
    #     reporter.generate_report(mpk_run, torch_ref, 
    #                             warnup_iter=100, test_iter=200, 
    #                             allclose_iter=5, print_mode=1)