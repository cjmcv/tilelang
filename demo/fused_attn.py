import os
import torch
import argparse
import megakernel as mi

from common.pkt_util import TorchRef, PerfReporter, Qwen3Info
from common.mpk_layers import MpkLayers
from common.autogen.qwen3_mega_config import Qwen3MegaConfig

def ref_run(x_torch, step, w_layernorm_torch, w_qkv_proj_torch, w_q_norm_torch, w_k_norm_torch, w_cos_torch, w_sin_torch, w_o_proj_torch):
    key_cache_5dim_torch = torch.zeros(batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_5dim_torch = torch.zeros(batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

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
    
    key_cache_5dim_torch[0, step, :, :] = key_states
    value_cache_5dim_torch[0, step, :, :] = value_states
    k_slice = key_cache_5dim_torch[:, :step+1, :, :]
    v_slice = value_cache_5dim_torch[:, :step+1, :, :]
    # k_slice.zero_()
    # v_slice.zero_()
    attn_output = TorchRef.attention_sdpa(query_states, k_slice, v_slice, False)
    attn_output = attn_output.reshape(batch*seqlen_q, q_dim)
    final_output = TorchRef.linear(attn_output, w_o_proj_torch) + x_torch # res
    # print("torch", key_states, value_states, final_output)
    # print("torch:", query_states, "\n", key_states, "\n", value_states, "\n", attn_output, "\n", final_output)
    # print("o_proj", attn_output.size(), w_o_proj_torch.size()) # o_proj torch.Size([1, 1024]) torch.Size([1024, 2048])
    return final_output
    
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
    max_kv_seqlen = 8192
    x_torch = torch.randn((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_layernorm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    w_qkv_proj_torch = torch.randn(((num_heads+2*num_kv_heads)*head_dim, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_q_norm_torch = torch.randn((1, head_dim), dtype=torch.bfloat16, device="cuda")
    w_k_norm_torch = torch.randn((1, head_dim), dtype=torch.bfloat16, device="cuda")
    w_qk_norm_torch = torch.cat([w_q_norm_torch, w_k_norm_torch], dim=0).contiguous()
    
    cos_half = torch.randn((batch, seqlen_q, head_dim//2), dtype=torch.bfloat16, device="cuda")
    sin_half = torch.randn((batch, seqlen_q, head_dim//2), dtype=torch.bfloat16, device="cuda")
    w_cos_torch = torch.cat((cos_half, cos_half), dim=-1)
    w_sin_torch = torch.cat((sin_half, sin_half), dim=-1)
    w_o_proj_torch = torch.randn((hidden_size, num_heads*head_dim), dtype=torch.bfloat16, device="cuda")
    
    ###
    layer_num = 10
    key_cache_5dim_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_5dim_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    q_dim = num_heads*head_dim
    kv_dim = num_kv_heads*head_dim    
    qkv_proj_out_torch = torch.zeros(max_batch_size, q_dim+2*kv_dim, dtype=torch.bfloat16, device="cuda")
    qk_states_torch = qkv_proj_out_torch[:, :q_dim+kv_dim].view(batch*seqlen_q*(num_heads+num_kv_heads), head_dim) 
    
    q_2dim_torch = qkv_proj_out_torch[:, :q_dim].view(batch*seqlen_q*num_heads, head_dim)
    q_3dim_torch = q_2dim_torch.view(batch, num_heads, head_dim)
    q_4dim_torch = q_2dim_torch.view(batch, seqlen_q, num_heads, head_dim)
    
    k_2dim_torch = qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch*seqlen_q*num_kv_heads, head_dim)
    k_4dim_torch = k_2dim_torch.view(batch, seqlen_q, num_kv_heads, head_dim)
    
    v_4dim_torch = qkv_proj_out_torch[:, q_dim+kv_dim:].view(batch, seqlen_q, num_kv_heads, head_dim)
    
    max_attn_split = 8
    edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
    glse_torch = torch.empty(batch, num_heads, max_attn_split, device="cuda", dtype=torch.bfloat16)
    out_partial_torch = torch.empty(batch, num_heads, max_attn_split, head_dim, device="cuda", dtype=torch.bfloat16)
    mask_torch = torch.ones(batch, max_kv_seqlen, num_kv_heads, device="cuda", dtype=torch.uint8)
    
    attn_out_3dim_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    attn_out_2dim_torch = attn_out_3dim_torch.view(batch*seqlen_q, q_dim)
    
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")    
    
    #########################################
    
    layer_id = 0
    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()

    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_layernorm = mpk.attach_input(torch_tensor=w_layernorm_torch, name="w_layernorm")
    w_qkv_proj = mpk.attach_input(torch_tensor=w_qkv_proj_torch, name="w_qkv_proj")
    w_o_proj = mpk.attach_input(torch_tensor=w_o_proj_torch, name="w_o_proj")
    
    # 
    layernorm_out = mpk.new_tensor(dims=(max_batch_size, hidden_size), dtype=mi.bfloat16, name="layernorm_out", io_category="cuda_tensor")
    qkv_proj_out = mpk.attach_input(torch_tensor=qkv_proj_out_torch, name="qkv_proj_out")
    if 1:
        w_qk_norm = mpk.attach_input(torch_tensor=w_qk_norm_torch, name="w_qk_norm")
        qk_states = mpk.attach_input(torch_tensor=qk_states_torch, name="qk_states")
    else:
        w_q_norm = mpk.attach_input(torch_tensor=w_q_norm_torch, name="w_q_norm")
        w_k_norm = mpk.attach_input(torch_tensor=w_k_norm_torch, name="w_k_norm")
        query_states = mpk.attach_input(torch_tensor=q_2dim_torch, name="query_states")
        key_states = mpk.attach_input(torch_tensor=k_2dim_torch, name="key_states")
        
    # rope
    q_4dim = mpk.attach_input(torch_tensor=q_4dim_torch, name="q_4dim")
    k_4dim = mpk.attach_input(torch_tensor=k_4dim_torch, name="k_4dim")
    w_cos = mpk.attach_input(torch_tensor=w_cos_torch, name="cos")
    w_sin = mpk.attach_input(torch_tensor=w_sin_torch, name="sin")
    
    # attn
    q_3dim = mpk.attach_input(torch_tensor=q_3dim_torch, name="q_3dim")
    k_4dim_cache = mpk.attach_input(torch_tensor=key_cache_5dim_torch[layer_id, :, :, :, :], name="k_4dim_cache")
    v_4dim_cache = mpk.attach_input(torch_tensor=value_cache_5dim_torch[layer_id, :, :, :, :], name="v_4dim_cache")
    edge = mpk.attach_input(torch_tensor=edge_torch, name="edge")
    mask = mpk.attach_input(torch_tensor=mask_torch, name="mask")
    glse = mpk.attach_input(torch_tensor=glse_torch, name="glse")
    out_partial = mpk.attach_input(torch_tensor=out_partial_torch, name="out_partial")
    
    attn_out_3dim = mpk.attach_input(torch_tensor=attn_out_3dim_torch, name="attn_out_3dim") # 
    attn_out_2dim = mpk.attach_input(torch_tensor=attn_out_2dim_torch, name="attn_out_2dim")
    
    # linear_res
    linear_res_out = mpk.attach_input(torch_tensor=out_torch, name="linear_res_out")
    #########################################
    mpk.rmsnorm_layer(
        input=x,
        weight=w_layernorm,
        output=layernorm_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.rmsnorm_layout,
    )
    mpk.linear_layer(
        input=layernorm_out,
        weight=w_qkv_proj,
        output=qkv_proj_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.qkv_proj_layout,
    )
    
    if 1:
        mpk.rmsnorm_layer(
            input=qk_states,
            weight=w_qk_norm,
            output=qk_states,
            sync_mode=(0, 0, 0),
            layout=Qwen3MegaConfig.merge_q_k_norm_layout,
        )
    else:
        # print(q_2dim_torch.dim, k_2dim_torch.dim)
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
    extra_layout = (2, 0, 0)
    fused_layout = tuple(a + b for a, b in zip(Qwen3MegaConfig.rope_layout[0], extra_layout)), Qwen3MegaConfig.rope_layout[1]
    mpk.rope_layer(
        q=q_4dim,
        k=k_4dim,
        cos=w_cos,
        sin=w_sin,
        q_embed=q_4dim,
        k_embed=k_4dim,
        sync_mode=(0, 0, 0),
        layout=fused_layout,
        fused_params=[99, 0, *extra_layout, layer_id],
    )
    
    # # attn    
    mpk.gqa_decode_layer(
        q=q_3dim,
        k_cache=k_4dim_cache,
        v_cache=v_4dim_cache,
        edge=edge,
        mask=mask,
        glse=glse,
        out_partial=out_partial,
        output=attn_out_3dim,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.gqa_decode_layout_16,
        
    )
    mpk.linear_with_residual_layer(
        input=attn_out_2dim,
        weight=w_o_proj,
        residual=x,
        output=linear_res_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.linear2_layout,
    )
    
    step = 15
    edge_torch[0].fill_(step) # kv_seqlen
    edge_torch[1].fill_(num_kv_heads*head_dim) # kvcache onestep_size
    edge_torch[2].fill_(batch*max_kv_seqlen*num_kv_heads*head_dim) # kvcache onelayer_size
    meta = [edge_torch, key_cache_5dim_torch, value_cache_5dim_torch, k_4dim_torch, v_4dim_torch]
    layers.compile_load(meta_tensors=meta, is_no_compile=args.nc, output_dir=args.output_dir)

    print(key_cache_5dim_torch[layer_id, 0, step, :, :].data_ptr(), value_cache_5dim_torch[layer_id, 0, step, :, :].data_ptr())
    mpk(batch)

    edge_torch[0].fill_(step)
    def mpk_run():
        mpk(batch)
        return out_torch
    def torch_ref():
        return ref_run(x_torch, step, w_layernorm_torch, w_qkv_proj_torch, w_q_norm_torch, w_k_norm_torch, w_cos_torch, w_sin_torch, w_o_proj_torch)

    reporter.generate_report(mpk_run, torch_ref, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_mode=1)
