import os
import torch
import argparse
from torch import nn
import megakernel as mi

from common.pkt_util import TorchRef, MpkReporter, TestUtil
from common.mpk_layers import MpkLayers
from common.autogen.qwen3_mlp_config import Qwen3MlpConfig

if __name__ == "__main__":
    max_batch_size = 1
    batch_size = 1
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
    
    reporter = MpkReporter() 
    # model, tokenizer = reporter.memory_footprint_simulation(rank)
    # w_rms_torch, w_gatedup_torch, w_down_proj_torch = reporter.get_weight_qwen3_mlp(layer_id=0)
    
    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    
    splitk = 1 # 8
    # hidden_size = 2560        # K
    # intermediate_size = 9728 # torch.randn / ones / TestUtil.create_matrix_arange_col /
    hidden_size = 1024
    intermediate_size = 3072
    
    # w_torch = w_gatedup_torch 
    w_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    # a = torch.ones((100000, 30000), dtype=torch.bfloat16, device="cuda")
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    print("x: ", x_torch.data_ptr(), "w: ", w_torch.data_ptr(), "o: ", out_torch.data_ptr())
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w = mpk.attach_input(torch_tensor=w_torch, name="w")
    linear_out = mpk.attach_input(torch_tensor=out_torch, name="linear_out")

    # grid_dim, block_dim(实际是tile_dim), thread_num(实际是block_dim, 固定为threadIdx.x==128或256, 其他维度为1)
    mpk.linear_layer(
        input=x,
        weight=w,
        output=linear_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MlpConfig.linear1_layout,
    )
    layers.compile_load(args.nc, args.output_dir)
    
    
    # ###
    def ref_run():
        return TorchRef.linear(x_torch[:batch_size], w_torch)
    
    def mpk_run():
        mpk(batch_size)
    
    ref_output = ref_run()
    mpk(batch_size)
    mpk_output = out_torch[:batch_size]
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    ###
    
    reporter.generate_report(mpk_run, mpk_output, splitk, 
                            ref_run, ref_output, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)