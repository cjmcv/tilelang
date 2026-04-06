
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
from common.pkt_util import TorchRef, Qwen3Info
from common.mk_layers import MkLayers, MkLayersHybridLayout

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--output-dir", default=os.getenv("MEGAKERNEL_HOME", default=None)+"/demo/gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")
    
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos token during generation")

    # -------- Args for CI tests ----------
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Decode cap for CI determinism")
    parser.add_argument("--prompt",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Custom prompt text to generate from.",
    )

    args = parser.parse_args()
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    torch.set_default_dtype(torch.bfloat16)

    model_tag = "qwen3_06b"
    model, tokenizer = Qwen3Info.load_model(rank, model_tag)
    total_num_requests = 1# if not args.use_mirage else args.max_num_batched_requests
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    prompt = args.prompt
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)
    # print("aa position_embeddings: ", position_embeddings[0].size(), position_embeddings[1].size()) # torch.Size([1, 32768, 128]) torch.Size([1, 32768, 128])

    # get all model weight tensors
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")
    # print("step: ", step.size())
    # g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    warmup = 0
    # Decode up to user cap or buffer size
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (tokens.size(1) - prompt_lengths[0].item())
    output_len = max(0, min(output_len, tokens.size(1) - prompt_lengths[0].item()))
    
    prompt_len = prompt_lengths[0].item()
    decode_limit = prompt_len + output_len
    
    #############################
    if args.use_mirage == True:
        batch = 1
        q_seqlen = 1
        hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
            = Qwen3Info.get_basic_params(model_tag)
            
        print("num_hidden_layers: ", num_hidden_layers)
        layers = MkLayers(model_tag, instance_id=0, kernel_num=num_hidden_layers, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
        params, io_pt, public_pt = MkLayers.qwen3_alloc_torch_buffer(model_tag, num_hidden_layers, batch, q_seqlen=1)   
        layers.qwen3_create_decoder_layer(model, num_hidden_layers, params, io_pt, public_pt, is_long_kv=True, is_no_compile=args.nc, output_dir=args.output_dir)
        
        # layers = MkLayersHybridLayout(model_tag, instance_num=2, kernel_num=num_hidden_layers, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
        # layers.qwen3_create_decoder_layer(model_tag=model_tag, model=model, layer_num=num_hidden_layers, batch=batch, is_no_compile=args.nc, output_dir=args.output_dir)

        # cur_pos = 64
        # prev_pos = 63
        # hidden_states = torch.randn((batch, q_seqlen, hidden_size), dtype=torch.bfloat16, device="cuda")
        # cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
        # sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
        # layers(cur_pos, (cos_embeddings, sin_embeddings), hidden_states.view(batch*q_seqlen, hidden_size))
        
        #############################
        for cur_pos in range(prompt_len, decode_limit):
            # print(cur_pos - 1)
            step.fill_(cur_pos - 1)
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            # print("cos_embeddings: ", cos_embeddings.size(), "sin_embeddings: ", sin_embeddings.size()) # torch.Size([1, cur_pos, 128]) torch.Size([1, cur_pos, 128])
            logits = model.forward(
                mk_layers=layers,
                cur_pos=cur_pos,
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
            )
            next_token = logits.argmax(dim=-1)
            next_token = next_token[0, -1]
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            if next_token == model.config.eos_token_id:
                break
            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()
    else:
        for cur_pos in range(prompt_len, decode_limit):
            # print(cur_pos - 1)
            step.fill_(cur_pos - 1)
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            # print("cos_embeddings: ", cos_embeddings.size(), "sin_embeddings: ", sin_embeddings.size()) # torch.Size([1, cur_pos, 128]) torch.Size([1, cur_pos, 128])
            logits = model.forward(
                mk_layers=None,
                cur_pos=cur_pos,
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
            )
            next_token = logits.argmax(dim=-1)
            next_token = next_token[0, -1]
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            if next_token == model.config.eos_token_id:
                break
            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()
                
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    #############
    cur_pos = 65
    hidden_states = torch.randn((batch, q_seqlen, hidden_size), dtype=torch.bfloat16, device="cuda")
    def mk_run_one_step():
        mk_out = layers(cur_pos, (cos_embeddings, sin_embeddings), hidden_states.view(batch*q_seqlen, hidden_size))
        return mk_out
    
    for i in range(20):
        torch.cuda.synchronize()
        starter.record()
        mk_run_one_step()
        ender.record()
        torch.cuda.synchronize()
        print("outside time: ", starter.elapsed_time(ender))
    print("dims:", batch, q_seqlen, hidden_size)
    ################
    
    # ########################
    # cur_pos = 100
    # step.fill_(cur_pos - 1)
    # torch.cuda.synchronize()
    # starter.record()
    # logits = model.forward(
    #     mk_layers=layers,
    #     cur_pos=cur_pos,
    #     input_ids=input_ids,
    #     position_embeddings=(cos_embeddings, sin_embeddings),
    #     step=step,
    #     stream=stream,
    # )
    # ender.record()
    # torch.cuda.synchronize()
    # print("time: ", starter.elapsed_time(ender))
    # ###########################
    
    end_idx = prev_pos + 1
    generated_ids = tokens[:, :end_idx]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    print(
        "Prompt length {}, generate length {}, per-token latency {} ms".format(
            prompt_len, cur_pos - prompt_len, run_time / (cur_pos - prompt_len)
        )
    )

    if world_size > 1:
        dist.destroy_process_group()
