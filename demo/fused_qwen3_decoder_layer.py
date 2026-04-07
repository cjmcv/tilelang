
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
from common.pkt_util import TorchRef, PerfReporter, Qwen3Info
from common.mk_layers import MkLayers, MkLayersHybridLayout
    

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
    max_kv_seqlen = 1024
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)
    model, tokenizer = Qwen3Info.load_model(0, model_tag, page_size=max_kv_seqlen)
    
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    all_position_embeddings = model.model.rotary_emb(positions)
 
 
    hidden_states = torch.randn((batch, q_seqlen, hidden_size), dtype=torch.bfloat16, device="cuda")
    attention_mask = None    
    step = torch.full((1, ), 0, dtype=torch.int32, device="cuda")
    stream = None
    print("before forward", hidden_states)
    
    layer_num = num_hidden_layers
    layers = MkLayers(model_tag, instance_id=0, kernel_num=layer_num, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
    params, io_pt, public_pt = MkLayers.qwen3_alloc_torch_buffer(model_tag, layer_num, batch, q_seqlen=1, max_kv_seqlen=max_kv_seqlen)   
    layers.qwen3_create_decoder_layer(model, layer_num, params, io_pt, public_pt, is_long_kv=True, is_no_compile=args.nc, output_dir=args.output_dir)
    
    # layers = MkLayersHybridLayout(model_tag, instance_num=2, kernel_num=layer_num, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
    # layers.qwen3_create_decoder_layer(model_tag, model, layer_num, batch, is_no_compile=args.nc, output_dir=args.output_dir)

    #####################################
    cur_pos = 65
    position_embeddings=(all_position_embeddings[0][:, cur_pos-1:cur_pos], all_position_embeddings[1][:, cur_pos-1:cur_pos])
    def mk_run_one_step():
        mk_out = layers(cur_pos, position_embeddings, hidden_states.view(batch*q_seqlen, hidden_size))
        return mk_out, layers.public_pt.key_cache_5d, layers.public_pt.value_cache_5d 
    
    def torch_ref_tmp_one_step():
        step.fill_(cur_pos - 1)
        torch_io = hidden_states.clone()
        with torch.no_grad():
            for layer_id in range(layer_num):
                out = model.model.layers[layer_id].forward(torch_io, attention_mask, position_embeddings, step, stream)
                torch_io = out[0]
            return torch_io, model.model.kv_cache[0], model.model.kv_cache[1]
    ######################################
    start_pos = 65
    decode_limit = 66
    def mk_run_multi_step():
        mk_io = hidden_states.view(batch*q_seqlen, hidden_size).clone()
        for cur_pos in range(start_pos, decode_limit):
            position_embeddings=(all_position_embeddings[0][:, cur_pos-1:cur_pos], all_position_embeddings[1][:, cur_pos-1:cur_pos])
            mk_out = layers(cur_pos, position_embeddings, mk_io)
            mk_io.copy_(mk_out)
        return mk_out, layers.public_pt.key_cache_5d, layers.public_pt.value_cache_5d # [layer_num, batch, kv_seqlen, kv_heads, head_dim]

    def torch_ref_tmp_multi_step():
        torch_io = hidden_states.clone()    
        with torch.no_grad():
            for cur_pos in range(start_pos, decode_limit):
                step.fill_(cur_pos - 1)
                position_embeddings=(all_position_embeddings[0][:, cur_pos-1:cur_pos], all_position_embeddings[1][:, cur_pos-1:cur_pos])
                for layer_id in range(layer_num):
                    out = model.model.layers[layer_id].forward(torch_io, attention_mask, position_embeddings, step, stream)
                    torch_io = out[0]
            return torch_io, model.model.kv_cache[0], model.model.kv_cache[1]
    ######################################
    
    mk_run = mk_run_one_step
    torch_ref_tmp = torch_ref_tmp_one_step
    
    # mk_run = mk_run_multi_step
    # torch_ref_tmp = torch_ref_tmp_multi_step
    
    if 0:
        graph, ref_output = TorchRef.compile_capture(torch_ref_tmp, is_compile=False) # 搜 “kv_seq_len = step + 1” 改成=> 1
        def torch_ref():
            graph.replay()
            return ref_output
    else:
        def torch_ref():
            return torch_ref_tmp()
    
    # for i in range(10):
    #     torch_ref()
    
    # torch_ref()    
    # mk_run()
    
    # print("mk_out: ", mk_run())    
    # print("torch_ref: ", torch_ref())
    
    
    q1,k1,v1 = mk_run() # [:, 0, :, :, :]
    q2,k2,v2 = torch_ref()
    # # print("q", PerfReporter.assert_similar(q1, q2))
    # # print("k", PerfReporter.assert_similar(k1, k2))
    # # print("v", PerfReporter.assert_similar(v1, v2))
    
    # 第1层没问题，第二层的split方案的结果不对
    print("mpk query0:\n", q1)
    print("torch query0:\n", q2)
    
    # print("mpk key0:\n", k1[0, 0, cur_pos-1:cur_pos, :, :])
    # print("torch key0:\n", k2[0, 0, cur_pos-1:cur_pos, :, :])
    # print("mpk key1:\n", k1[1, 0, cur_pos-1:cur_pos, :, :])
    # print("torch key1:\n", k2[1, 0, cur_pos-1:cur_pos, :, :])
    
    # print("mpk value0:\n", v1[0, 0, cur_pos-1:cur_pos, :, :])
    # print("torch value0:\n", v2[0, 0, cur_pos-1:cur_pos, :, :])
    # print("mpk value1:\n", v1[1, 0, cur_pos-1:cur_pos, :, :])
    # print("torch value1:\n", v2[1, 0, cur_pos-1:cur_pos, :, :])
    
    # print("mpk key0:\n", k1[0, 0, start_pos-1:decode_limit-1, :, :])
    # print("torch key0:\n", k2[0, 0, start_pos-1:decode_limit-1, :, :])
    # print("mpk key1:\n", k1[1, 0, start_pos-1:decode_limit-1, :, :])
    # print("torch key1:\n", k2[1, 0, start_pos-1:decode_limit-1, :, :])
 
    # reporter = PerfReporter() 
    # reporter.generate_report(mk_run, torch_ref, 
    #                         warnup_iter=1, test_iter=1, 
    #                         allclose_iter=2, print_mode=1)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(10):
        torch.cuda.synchronize()
        starter.record()
        mk_run()
        ender.record()
        torch.cuda.synchronize()
        print("mpk time: ", starter.elapsed_time(ender))
        
    for i in range(10):
        torch.cuda.synchronize()
        starter.record()
        torch_ref()
        ender.record()
        torch.cuda.synchronize()
        print("torch time: ", starter.elapsed_time(ender))