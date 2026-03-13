
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
from common.pkt_util import TorchRef, PerfReporter, Qwen3Info
from common.mpk_layers import MpkLayers
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.bfloat16)

    batch = 1
    q_seqlen = 1
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim = Qwen3Info.get_basic_params(0.6)
    model, tokenizer = Qwen3Info.load_model(0)
    
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    all_position_embeddings = model.model.rotary_emb(positions)
 
    hidden_states = torch.randn((batch, q_seqlen, hidden_size), dtype=torch.bfloat16, device="cuda")
    attention_mask = None    
    step = torch.full((1, ), 0, dtype=torch.int32, device="cuda")
    stream = None
    print("before forward", hidden_states)
    
    position_embeddings=(all_position_embeddings[0][step], all_position_embeddings[1][step])
    
    layer_id = 0
    layers = MpkLayers(instance_id=0, kernel_num=1, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
    mpk = layers.get_mpk()
    # def qwen3_alloc_io_buffer(self, model_size, batch, q_seqlen, max_kv_seqlen, key_cache_5dim_torch, value_cache_5dim_torch):
        
    def torch_ref():
        out = model.model.layers[layer_id].forward(hidden_states, attention_mask, position_embeddings, step, stream)
        return out[0]
    
    reporter = PerfReporter() 
    reporter.generate_report(torch_ref, torch_ref, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_mode=1)