
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
from common.pkt_util import TorchRef, PerfReporter, Qwen3Info
from common.mk_layers import MkLayers
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.getenv("MEGAKERNEL_HOME", default=None)+"/demo/gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.bfloat16)

    model_tag = "qwen3_06b"
    batch = 1
    q_seqlen = 1
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim = Qwen3Info.get_basic_params(model_tag)
    model, tokenizer = Qwen3Info.load_model(0, model_tag)
    
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    all_position_embeddings = model.model.rotary_emb(positions)
 
    hidden_states = torch.randn((batch, q_seqlen, hidden_size), dtype=torch.bfloat16, device="cuda")
    attention_mask = None    
    step0 = 0 # todo
    step = torch.full((1, ), 0, dtype=torch.int32, device="cuda")
    stream = None
    print("before forward", hidden_states)
    
    position_embeddings=(all_position_embeddings[0][:, step], all_position_embeddings[1][:, step])
    
    layers = MkLayers(model_tag, instance_id=0, kernel_num=10, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
    mk = layers.get_mk()
    
    layer_num = 10
    max_kv_seqlen = 8192
    layers.qwen3_alloc_io_buffer(model_tag, layer_num, batch, 1, max_kv_seqlen)
    for layer_id in range(layer_num):
        if (layer_id == 0):
            reuse_instance = False
        else:
            reuse_instance = True
        layers.qwen3_create_attn_layer(model, layer_id, reuse_instance)
        layers.qwen3_create_mlp_layer(model, layer_id, reuse_instance)
    meta, mk_attn_out, mk_mlp_out = layers.fill_meta()
    layers.compile_load(meta_tensors=meta, is_no_compile=args.nc, output_dir=args.output_dir)  
    
    
    layers.update_step(step0, cos=position_embeddings[0][:, step], sin=position_embeddings[1][:, step])
    def mk_run():
        mk_mlp_out.copy_(hidden_states.view(batch*q_seqlen, hidden_size))
        for layer_id in range(layer_num):
            layers.attn_layer_io.layer_in.pt.copy_(mk_mlp_out)
            mk(batch, layer_id)
        return mk_mlp_out  


    def torch_ref_tmp():
        torch_io = hidden_states.clone()
        with torch.no_grad():
            for layer_id in range(layer_num):
                out = model.model.layers[layer_id].forward(torch_io, attention_mask, position_embeddings, step, stream)
                torch_io = out[0]
            return torch_io
        
    if 0:
        graph, ref_output = TorchRef.compile_capture(torch_ref_tmp, is_compile=False) # 搜 “kv_seq_len = step + 1” 改成=> 1
        def torch_ref():
            graph.replay()
            return ref_output
    else:
        def torch_ref():
            return torch_ref_tmp()
    
    for i in range(10):
        torch_ref()
    
    torch_ref()    
    mk_run()
    
    # print("mk_out: ", mk_run())    
    # print("torch_ref: ", torch_ref())
    
    reporter = PerfReporter() 
    reporter.generate_report(mk_run, torch_ref, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_mode=1)