
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
from common.pkt_util import TorchRef, PerfReporter, Qwen3Info
from common.mpk_layers import MpkLayers
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.getenv("MEGAKERNEL_HOME", default=None)+"/demo/gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.bfloat16)

    model_size = 0.6
    batch = 1
    q_seqlen = 1
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim = Qwen3Info.get_basic_params(0.6)
    model, tokenizer = Qwen3Info.load_model(0, model_size)
    
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    all_position_embeddings = model.model.rotary_emb(positions)
 
    hidden_states = torch.randn((batch, q_seqlen, hidden_size), dtype=torch.bfloat16, device="cuda")
    attention_mask = None    
    step0 = 0 # todo
    step = torch.full((1, ), 0, dtype=torch.int32, device="cuda")
    stream = None
    print("before forward", hidden_states)
    
    position_embeddings=(all_position_embeddings[0][:, step], all_position_embeddings[1][:, step])
    
    layer_id = 0
    layers = MpkLayers(instance_id=0, kernel_num=1, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
    mpk = layers.get_mpk()
    
    layer_num = 10
    max_kv_seqlen = 8192
    layers.qwen3_alloc_io_buffer(model_size, layer_num, batch, 1, max_kv_seqlen)
    layers.qwen3_create_attn_layer(model, layer_id)
    meta, mpk_out = layers.fill_meta(step0)
    layers.compile_load(meta_tensors=meta, is_no_compile=args.nc, output_dir=args.output_dir)
    
    print(position_embeddings[0].size(), position_embeddings[0][:, step].size())
    layers.w_cos_torch.copy_(position_embeddings[0][:, step])
    layers.w_sin_torch.copy_(position_embeddings[1][:, step])
    
    def torch_ref():
        with torch.no_grad():
            out = model.model.layers[layer_id].forward(hidden_states, attention_mask, position_embeddings, step, stream)
            return out[0]
    
    def mpk_run():
        layers.attn_layer_io.layer_in.pt.copy_(hidden_states.view(batch*q_seqlen, hidden_size))
        mpk(batch)
        return mpk_out
    

    print("mpk_out: ", mpk_run())    
    # print("layer_in", layers.attn_layer_io.layer_in.pt)
    # print("qkv_proj_out", layers.attn_layer_io.qkv_proj_out.pt)
    # print("attn_layer_io.attn_in.q: ", layers.attn_layer_io.attn_in.q.pt)
    # print("kcache:", layers.attn_layer_io.attn_in.kcache.pt[0,0,0]) # layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim
    # print("vcache:", layers.attn_layer_io.attn_in.vcache.pt[0,0,0])
    # print("attn_out:", layers.attn_layer_io.attn_out.three_dim.pt)
    # print("attn_out: ", layers.attn_layer_io.attn_out.two_dim.pt)
    
    print("torch_ref: ", torch_ref())
    
    # reporter = PerfReporter() 
    # reporter.generate_report(mpk_run, torch_ref, 
    #                         warnup_iter=100, test_iter=200, 
    #                         allclose_iter=5, print_mode=1)