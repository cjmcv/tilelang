import os
import torch
import argparse
import megakernel as mi

from common.pkt_util import TorchRef, MpkReporter
from common.mpk_layers import MpkLayers
from common.autogen.qwen3_mlp_config import Qwen3MlpConfig

WITH_RMS_NORM = 1
WITH_RESIDUAL = 1


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

    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    reporter = MpkReporter() 
    # reporter.memory_footprint_simulation(rank)
    
    splitk = 1 # 8
    hidden_size = 2560
    intermediate_size = 9728
    # hidden_size = 1024
    # intermediate_size = 3072
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_rms_norm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_rms_norm = mpk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm")
    w_gatedup = mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
    w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mpk.attach_input(torch_tensor=out_torch, name="mlp_out")

    # m1    
    # rmsnorm_layout = (1, 1, 1), (1, 1, 1)
    linear1_layout = (304, 1, 1), (64, 16, 128)
    silu_mul_layout = (152, 1, 1), (64, 16, 1)
    linear2_layout = (40, 1, 1), (64, 16, 64)
    # # m32
    # rmsnorm_gird, rmsnorm_tile = (32, 1, 1), (1, 1, 1)
    # linear1_gird, linear1_tile = (304, 1, 1), (64, 64, 64)
    # silu_mul_gird, silu_mul_tile = (152, 1, 1), (64, 32, 1)
    # linear2_gird, linear2_tile = (40, 1, 1), (64, 64, 64)
    # # m128
    # rmsnorm_gird, rmsnorm_tile = (128, 1, 1), (1, 1, 1)
    # linear1_gird, linear1_tile = (304, 2, 1), (64, 64, 64)
    # silu_mul_gird, silu_mul_tile = (152, 4, 1), (64, 32, 1)
    # linear2_gird, linear2_tile = (40, 2, 1), (64, 64, 64)
    
    x_residual = x
    if WITH_RMS_NORM:
        rms_out = mpk.new_tensor(dims=(max_batch_size, hidden_size), dtype=mi.bfloat16, name="rms_out", io_category="cuda_tensor")
        mpk.rmsnorm_layer(
            input=x,
            weight=w_rms_norm,
            output=rms_out,
            sync_mode=(0, 0, 0),
            layout=Qwen3MlpConfig.rmsnorm_layout,
        )
        x = rms_out
        
    # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")
    mlp_mid = mpk.new_tensor(dims=(max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    mpk.linear_layer(
        input=x,
        weight=w_gatedup,
        output=mlp_mid,
        sync_mode=(0, 0, 0),
        layout=Qwen3MlpConfig.linear1_layout,
    )
    
    if 1:
        # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
        # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
        # mlp_out_torch = silu_mul_out_torch
        silu_mul_out = mpk.new_tensor(dims=(max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
        mpk.silu_mul_layer(
            input=mlp_mid,
            output=silu_mul_out,
            sync_mode=(2, 0, 0),
            layout=Qwen3MlpConfig.silu_mul_layout,
            # grid_dim=(2, 4, 1), tile_dim=(128, 1, 1),
            # sync_mode=(2, 0, 0),
        )
        if WITH_RESIDUAL:
            mpk.linear_with_residual_layer(
                input=silu_mul_out,
                weight=w_down_proj,
                residual=x_residual,
                output=mlp_out,
                sync_mode=(0, 0, 0),
                layout=Qwen3MlpConfig.linear2_layout,
            )
        else:
            mpk.linear_layer(
                input=silu_mul_out,
                weight=w_down_proj,
                output=mlp_out,
                sync_mode=(0, 0, 0),
                layout=Qwen3MlpConfig.inear2_layout,
            )
    else:
        mpk.silu_mul_linear_layer(
            input=mlp_mid,
            weight=w_down_proj,
            output=mlp_out,
            grid_dim=(20, 1, 1), tile_dim=(128, 64, 64),
            sync_mode=(2, 0, 0),
        )
    layers.compile_load(args.nc, args.output_dir)
    
    ###
    def ref_run():
        if WITH_RMS_NORM:
            return TorchRef.norm_mlp(x_torch[:batch_size], w_rms_norm_torch, w_gatedup_torch, w_down_proj_torch)
        elif WITH_RMS_NORM and WITH_RESIDUAL:
            return TorchRef.norm_mlp(x_torch[:batch_size], w_rms_norm_torch, w_gatedup_torch, w_down_proj_torch) + x_torch[:batch_size]
        else:
            return TorchRef.mlp(x_torch[:batch_size], w_gatedup_torch, w_down_proj_torch)
    graph, ref_output = TorchRef.compile_capture(ref_run, is_compile=False)
    
    def mpk_run():
        mpk(batch_size)
        
    ref_output = ref_run()
    mpk_output = out_torch[:batch_size]

    for _ in range(100):
        graph.replay()
    mpk_run()
    ##
    
    reporter.generate_report(mpk_run, mpk_output, splitk, 
                            graph.replay, ref_output, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)