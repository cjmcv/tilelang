import os
import torch
import argparse
from torch import nn
import megakernel as mi

from common.pkt_util import TorchRef, PerfReporter, TestUtil
from common.mpk_layers import MpkLayers

from common.micro_base import HparamSelectMode
from common.micro_autogen import MicroAutoGen
from common.autogen.qwen3_mega_config import Qwen3MegaConfig

def test_rms_norm(mpk, max_batch_size, batch_size, hidden_size):
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_rms_norm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_rms_norm = mpk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm")
    rms_out = mpk.attach_input(torch_tensor=out_torch, name="rms_out")
    mpk.rmsnorm_layer(
        input=x,
        weight=w_rms_norm,
        output=rms_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.rmsnorm_layout,
    )
    layers.compile_load(args.nc, args.output_dir)

    def torch_ref():
        return TorchRef.rms_norm(x_torch[:batch_size], w_rms_norm_torch)

    def target_func():
        mpk(batch_size)
        return out_torch[:batch_size]
    
    reporter.generate_report(target_func, torch_ref,
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)


def test_linear(mpk, max_batch_size, batch_size, N, K, layout):
    x_torch = torch.randn((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    # w_torch = w_gatedup_torch 
    w_torch = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, N), dtype=torch.bfloat16, device="cuda")
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
        layout=layout,
    )
    layers.compile_load(args.nc, args.output_dir)
    
    def torch_ref():
        return TorchRef.linear(x_torch[:batch_size], w_torch)
    
    def target_func():
        mpk(batch_size)
        return out_torch[:batch_size]
        
    # target_output = target_func()
    # ref_output = torch_ref()
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)

def test_silu_mul(mpk, max_batch_size, batch_size, intermediate_size):
    x_torch = torch.randn((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    silu_mul_out = mpk.attach_input(torch_tensor=out_torch, name="silu_mul_out")
    mpk.silu_mul_layer(
        input=x,
        output=silu_mul_out,
        sync_mode=(0, 0, 0), # (2, 0, 0)
        layout=Qwen3MegaConfig.silu_mul_layout,
    )
    layers.compile_load(args.nc, args.output_dir)

    def torch_ref():
        return TorchRef.silu_and_mul(x_torch[:batch_size])
    
    def target_func():
        mpk(batch_size)
        return out_torch[:batch_size]
        
    # target_output = target_func()    
    # ref_output = torch_ref()
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
    
def test_linear_residual(mpk, max_batch_size, batch_size, N, K, layout):
    x_residual_torch = torch.randn((max_batch_size, N), dtype=torch.bfloat16, device="cuda")
    x_torch = torch.randn((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, N), dtype=torch.bfloat16, device="cuda")

    x_residual = mpk.attach_input(torch_tensor=x_residual_torch, name="res")
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mpk.attach_input(torch_tensor=out_torch, name="mlp_out")

    mpk.linear_with_residual_layer(
        input=x,
        weight=w_down_proj,
        residual=x_residual,
        output=mlp_out,
        sync_mode=(0, 0, 0),
        layout=layout,
    )
    layers.compile_load(args.nc, args.output_dir)
    
    def torch_ref():
        return TorchRef.linear(x_torch[:batch_size], w_down_proj_torch) + x_residual_torch

    def target_func():
        mpk(batch_size)
        return out_torch[:batch_size]
        
    target_output = target_func()    
    ref_output = torch_ref()
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
   
def test_rope(mpk, max_batch_size, batch, heads, groups, dim):
    seqlen = 1
    q_torch = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen=1, H=heads,  D=dim]
    k_torch = torch.randn(batch, seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)   # [B, N=seqlen=1, H=groups, D=dim]
    cos_half = torch.randn(batch, seqlen, dim//2, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.randn(batch, seqlen, dim//2, device="cuda", dtype=torch.bfloat16)
    cos_torch = torch.cat((cos_half, cos_half), dim=-1)
    sin_torch = torch.cat((sin_half, sin_half), dim=-1)
    q_out_torch = torch.empty(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    k_out_torch = torch.empty(batch, seqlen, groups, dim, device="cuda", dtype=torch.bfloat16)
    
    # print("torch: ", q_torch.data_ptr(), k_torch.data_ptr(), v_torch.data_ptr(), mask_torch.data_ptr(), out_torch.data_ptr())
    q = mpk.attach_input(torch_tensor=q_torch, name="q")
    k = mpk.attach_input(torch_tensor=k_torch, name="k")
    cos = mpk.attach_input(torch_tensor=cos_torch, name="cos")
    sin = mpk.attach_input(torch_tensor=sin_torch, name="sin")
    q_out = mpk.attach_input(torch_tensor=q_out_torch, name="q_out")
    k_out = mpk.attach_input(torch_tensor=k_out_torch, name="k_out")

    mpk.rope_layer(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        q_embed=q_out,
        k_embed=k_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.rope_layout,
    )
    layers.compile_load(args.nc, args.output_dir)
    
    def target_func():
        mpk(batch_size)
        return torch.cat((q_out_torch, k_out_torch), dim=-2)
    
    def torch_ref():
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb_triton(q_torch, k_torch, cos_torch, sin_torch, position_ids=None, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)
    
    target_output = target_func()    
    ref_output = torch_ref()
    print("target_output", target_output)
    print("ref_output", ref_output)
    
    print("target_output", target_func())
    print("ref_output", torch_ref())
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
     
# def mpk_tensor(mpk):
#     return mpk

def test_gqa_decode(mpk, max_batch_size, batch, heads, groups, seqlen_kv, dim):
    # 1）两个图切换，以适配两个gqa配置？
    # 2）两个gqa布局，按多的配置，部分block可空跑。！！优先尝试
    #    生成代码时，添加优先主动退出条件(如有生成的时split的kernel则进入，不split的blockz直接退出。)
    # 封装attach_input.
    # 3) (暂停，一轮推理内，step不用改，可以两次推理间在外面修改) 
    #    添加 step, 搜 RuntimeConfig的int *step， 将该指针指向 edge 的内存，搜 kernel::gqa_decode_kernel，将input转为 写死的RuntimeConfig的step。
    #    对应 global_runtime_config[kernel_id].infer_cnt 和 global_runtime_config[kernel_id].batch_size
    #    可以在调用mpk推理后，外面打印edge的内容来检查step的递增情况。step的递增在static_persistent_kernel里进行，在等待前置信号时确定
    split = 8 # TODO 自动配置
    glse_torch = torch.empty(batch, heads, split, device="cuda", dtype=torch.bfloat16)
    out_partial_torch = torch.empty(batch, heads, split, dim, device="cuda", dtype=torch.bfloat16)
    
    valid_kv_seqlen = 64
    q_torch = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)              # [B, N=seqlen_q=1, H=heads,  D=dim]
    k_torch = torch.randn(batch, seqlen_kv, groups, dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    v_torch = torch.randn(batch, seqlen_kv, groups, dim, device="cuda", dtype=torch.bfloat16)
    edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
    edge_torch[0].fill_(valid_kv_seqlen)
    mask_torch = torch.ones(batch, seqlen_kv, groups, device="cuda", dtype=torch.uint8)
    out_torch = torch.empty(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    
    # print("torch: ", q_torch.data_ptr(), k_torch.data_ptr(), v_torch.data_ptr(), mask_torch.data_ptr(), out_torch.data_ptr())
    q = mpk.attach_input(torch_tensor=q_torch, name="q")
    k = mpk.attach_input(torch_tensor=k_torch, name="k")
    v = mpk.attach_input(torch_tensor=v_torch, name="v")
    edge = mpk.attach_input(torch_tensor=edge_torch, name="edge")
    mask = mpk.attach_input(torch_tensor=mask_torch, name="mask")
    glse = mpk.attach_input(torch_tensor=glse_torch, name="glse")
    out_partial = mpk.attach_input(torch_tensor=out_partial_torch, name="out_partial")
    attn_out = mpk.attach_input(torch_tensor=out_torch, name="attn_out")

    mpk.gqa_decode_layer(
        q=q,
        k_cache=k,
        v_cache=v,
        edge=edge,
        mask=mask,
        glse=glse,
        out_partial=out_partial,
        output=attn_out,
        sync_mode=(0, 0, 0),
        layout=Qwen3MegaConfig.gqa_decode_layout_test,
        # layout=((1, 8, 8), (64, 64, 8), (16, 1, 1), (64, 64, 8))
    )
    layers.compile_load(meta_tensors=[edge_torch], is_no_compile=args.nc, output_dir=args.output_dir)
    
    def target_func():
        mpk(batch_size)
        return out_torch
    
    k_slice = k_torch[:, :valid_kv_seqlen, :, :]
    v_slice = v_torch[:, :valid_kv_seqlen, :, :]
    def torch_ref():
        return TorchRef.attention_sdpa(q_torch, k_slice, v_slice, False)
    
    # target_output = target_func()
    # ref_output = torch_ref()
    # print("target_output", target_output)
    # print("ref_output", ref_output)
    
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    print("step", edge_torch[0])
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=0)


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

    # # 13/16/20/21/22/30/45: 0.118, 24-29
    # micro = MicroAutoGen(1, 1024, 3072)
    # micro.gen_qwen3_mlp(1, HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED
       
    reporter = PerfReporter() 
    # model, tokenizer = reporter.memory_footprint_simulation(rank)
    # w_rms_torch, w_gatedup_torch, w_down_proj_torch = reporter.get_weight_qwen3_mlp(layer_id=0)
    
    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    
    splitk = 1 # 8
    # hidden_size = 2560        # K
    # intermediate_size = 9728 # torch.randn / ones / TestUtil.create_matrix_arange_col /
    hidden_size = 1024
    intermediate_size = 3072
    
    # test_rms_norm(mpk, max_batch_size, batch_size, hidden_size)
    # test_linear(mpk, max_batch_size, batch_size, intermediate_size*2, hidden_size, Qwen3MegaConfig.linear1_layout)
    # test_silu_mul(mpk, max_batch_size, batch_size, intermediate_size) # 5us vs 2us，需要加速
    # test_linear_residual(mpk, max_batch_size, batch_size, hidden_size, intermediate_size, Qwen3MegaConfig.linear2_layout)
    
    heads=16
    groups=8
    dim=128
    seqlen_kv=8192
    # test_linear(mpk, max_batch_size, batch_size, (heads+2*groups)*dim, hidden_size, Qwen3MegaConfig.qkv_proj_layout)
    # test_rope(mpk, max_batch_size=1, batch=1, heads=heads, groups=groups, dim=dim)
    test_gqa_decode(mpk, max_batch_size=1, batch=1, heads=heads, groups=groups, seqlen_kv=seqlen_kv, dim=dim)
    # test_linear_residual(mpk, max_batch_size, batch_size, hidden_size, heads*dim, Qwen3MegaConfig.o_proj_layout)
        
    print("Test single_mega completed.")
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "kernel" -o my_profile python demo/single_linear.py --nc
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "persistent_kernel" -o my_profile python demo/single_linear.py