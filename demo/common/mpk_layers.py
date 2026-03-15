
import torch
from types import SimpleNamespace
import megakernel as mi
from common.pkt_util import Qwen3Info
from common.autogen.qwen3_mega_config import Qwen3MegaConfig

class MpkLayers:
    def __init__(self, instance_id, kernel_num, world_size, rank, max_batch_size, trace_name, profiling):
        self.profiler_tensor = None
        if profiling:
            self.profiler_tensor = torch.zeros(3000 * 128, dtype=torch.uint64, device="cuda").contiguous()
            
        # int num_sms_to_use = global_runtime_config[kernel_id].num_workers + num_schedulers / 4;
        num_workers, num_schedulers = mi.get_static_configurations_from_gpu(rank) # n, (sm-n)*4
        print("num_workers: ", num_workers)
        print("num_schedulers: ", num_schedulers)
        
        self.mpk = mi.PersistentKernel(
            instance_id=instance_id,
            kernel_num=kernel_num,
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            meta_tensors={}, #  meta_tensors={"qo_indptr_buffer": self.qo_indptr_buffer,},
            profiler_tensor=self.profiler_tensor,
            trace_name=trace_name,
            use_cutlass_kernel=True,
        )
        self.max_batch_size = max_batch_size
    
    def get_mpk(self):
        return self.mpk

    def compile_load(self, meta_tensors=list(), is_no_compile=False, output_dir="./gen"):
        if is_no_compile is True:
            module_path = output_dir + "/test.cpython-38-x86_64-linux-gnu.so"
            self.mpk.load_module(module_path, meta_tensors)
        else:
            module_path = self.mpk.compile(output_dir=output_dir)
            print("module_path: ", module_path)
            self.mpk.load_module(module_path, meta_tensors)

    def qwen3_alloc_io_buffer(self, model_size, layer_num, batch, q_seqlen, max_kv_seqlen):
        hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim = Qwen3Info.get_basic_params(model_size)
        
        self.batch = batch
        self.q_seqlen = q_seqlen
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.key_cache_5dim_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
        self.value_cache_5dim_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        # 在推理前，根据step拷贝进对应的值, 所有层共享
        self.w_cos_torch = torch.empty((self.batch, self.q_seqlen, self.head_dim), dtype=torch.bfloat16, device="cuda")
        self.w_sin_torch = torch.empty((self.batch, self.q_seqlen, self.head_dim), dtype=torch.bfloat16, device="cuda")
        
        x_torch = torch.randn((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
        
        q_dim = num_heads*head_dim
        kv_dim = num_kv_heads*head_dim    
        qkv_proj_out_torch = torch.zeros(batch, q_dim+2*kv_dim, dtype=torch.bfloat16, device="cuda")
        qk_torch = {
            "2d": qkv_proj_out_torch[:, :q_dim+kv_dim].view(batch*q_seqlen*(num_heads+num_kv_heads), head_dim) 
        }
        q_torch = {
            "2d": qkv_proj_out_torch[:, :q_dim].view(batch*q_seqlen*num_heads, head_dim),
            "3d": qkv_proj_out_torch[:, :q_dim].view(batch*q_seqlen, num_heads, head_dim),
            "4d": qkv_proj_out_torch[:, :q_dim].view(batch, q_seqlen, num_heads, head_dim),
        }
        k_torch = {
            "2d": qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch*q_seqlen*num_kv_heads, head_dim),
            "4d": qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch, q_seqlen, num_kv_heads, head_dim),
        }
        v_torch = {
            "4d": qkv_proj_out_torch[:, q_dim+kv_dim:].view(batch, q_seqlen, num_kv_heads, head_dim),
        }
        
        max_attn_split = 8
        self.edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
        self.glse_torch = torch.empty(batch, num_heads, max_attn_split, device="cuda", dtype=torch.bfloat16)
        self.out_partial_torch = torch.empty(batch, num_heads, max_attn_split, head_dim, device="cuda", dtype=torch.bfloat16)
        self.mask_torch = torch.ones(batch, max_kv_seqlen, num_kv_heads, device="cuda", dtype=torch.uint8)
        
        self.attn_out_3dim_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        attn_out_2dim_torch = self.attn_out_3dim_torch.view(batch*q_seqlen, q_dim)
        self.attn_out_torch = {
            "3d": self.attn_out_3dim_torch,
            "2d": attn_out_2dim_torch
        }
        self.o_proj_res_out_torch = torch.zeros((batch, hidden_size), dtype=torch.bfloat16, device="cuda") # x_torch

        #############################################################################
        # #    rmsnorm (x) -> linear (qkv proj) -> rmsnorm (q/k) -> rope (q/k) -> attn -> linear_res
        # # -> update cos/sin (step)                             -> update kvcache (step)    
        self.attn_layer_io = SimpleNamespace(
            layer_in = SimpleNamespace(pt=x_torch, mpk=self.mpk.attach_input(torch_tensor=x_torch, name="attn_layer_in")),
            layernorm_out = SimpleNamespace(pt=None, mpk=self.mpk.new_tensor(dims=(batch, hidden_size), dtype=mi.bfloat16, name="layernorm_out", io_category="cuda_tensor")),
            qkv_proj_out = SimpleNamespace(pt=qkv_proj_out_torch, mpk=self.mpk.attach_input(torch_tensor=qkv_proj_out_torch, name="qkv_proj_out")),
            qk_norm_states = SimpleNamespace(pt=None, mpk=self.mpk.attach_input(torch_tensor=qk_torch["2d"], name="qk_norm_states")),
            rope_io = SimpleNamespace(
                q = SimpleNamespace(pt=q_torch["4d"], mpk=self.mpk.attach_input(torch_tensor=q_torch["4d"], name="rope_io_q")),
                k = SimpleNamespace(pt=k_torch["4d"], mpk=self.mpk.attach_input(torch_tensor=k_torch["4d"], name="rope_io_k"))
            ),
            kv_curstep = SimpleNamespace(
                k = SimpleNamespace(pt=k_torch["4d"], mpk=self.mpk.attach_input(torch_tensor=k_torch["4d"], name="k_curstep")),
                v = SimpleNamespace(pt=v_torch["4d"], mpk=self.mpk.attach_input(torch_tensor=v_torch["4d"], name="v_curstep")),
            ),
            attn_in = SimpleNamespace(
                q = SimpleNamespace(pt=q_torch["3d"], mpk=self.mpk.attach_input(torch_tensor=q_torch["3d"], name="attn_in_q")),
                kcache = SimpleNamespace(pt=self.key_cache_5dim_torch, mpk=self.mpk.attach_input(torch_tensor=self.key_cache_5dim_torch, name="attn_in_kcache")),
                vcache = SimpleNamespace(pt=self.value_cache_5dim_torch, mpk=self.mpk.attach_input(torch_tensor=self.value_cache_5dim_torch, name="attn_in_vcache")),
                edge = SimpleNamespace(pt=self.edge_torch, mpk=self.mpk.attach_input(torch_tensor=self.edge_torch, name="edge")),
                mask = SimpleNamespace(pt=self.mask_torch, mpk=self.mpk.attach_input(torch_tensor=self.mask_torch, name="attn_in_mask")),
                glse = SimpleNamespace(pt=self.glse_torch, mpk=self.mpk.attach_input(torch_tensor=self.glse_torch, name="attn_in_glse")),
                out_partial = SimpleNamespace(pt=self.out_partial_torch, mpk=self.mpk.attach_input(torch_tensor=self.out_partial_torch, name="attn_out_partial"))
            ),
            attn_out = SimpleNamespace(
                three_dim = SimpleNamespace(pt=self.attn_out_torch["3d"], mpk=self.mpk.attach_input(torch_tensor=self.attn_out_torch["3d"], name="attn_out_3dim")),
                two_dim = SimpleNamespace(pt=self.attn_out_torch["2d"], mpk=self.mpk.attach_input(torch_tensor=self.attn_out_torch["2d"], name="attn_out_2dim")),
            ),
            o_proj_res_out = SimpleNamespace(pt=self.o_proj_res_out_torch, mpk=self.mpk.attach_input(torch_tensor=self.o_proj_res_out_torch, name="o_proj_res_out"))
        )
        
    def qwen3_create_attn_layer(self, model, layer_id):

        w_input_layernorm_torch, w_q_norm_torch, w_k_norm_torch, \
        w_q_torch, w_k_torch, w_v_torch, w_out_proj_torch, \
        k_cache_torch, v_cache_torch = Qwen3Info.get_weight_qwen3_attention(model, layer_id)
        
        self.w_qkv_proj_torch = torch.cat([w_q_torch, w_k_torch, w_v_torch], dim=0).contiguous()
        self.w_qk_norm_torch = torch.cat([w_q_norm_torch, w_k_norm_torch], dim=0).contiguous()
        
        layer_id_str = "_" + str(layer_id)
        self.mpk.rmsnorm_layer(
            input  = self.attn_layer_io.layer_in.mpk,
            weight = self.mpk.attach_input(torch_tensor=w_input_layernorm_torch, name="w_layernorm"+layer_id_str),
            output = self.attn_layer_io.layernorm_out.mpk,
            sync_mode = (0, 0, 0),
            layout = Qwen3MegaConfig.rmsnorm_layout,
        )
        self.mpk.linear_layer(
            input  = self.attn_layer_io.layernorm_out.mpk,
            weight = self.mpk.attach_input(torch_tensor=self.w_qkv_proj_torch, name="w_qkv_proj"+layer_id_str),
            output = self.attn_layer_io.qkv_proj_out.mpk,
            sync_mode = (0, 0, 0),
            layout = Qwen3MegaConfig.qkv_proj_layout,
        )
        self.mpk.rmsnorm_layer(
            input=self.attn_layer_io.qk_norm_states.mpk,
            weight=self.mpk.attach_input(torch_tensor=self.w_qk_norm_torch, name="w_qk_norm"+layer_id_str),
            output=self.attn_layer_io.qk_norm_states.mpk,
            sync_mode=(0, 0, 0),
            layout=Qwen3MegaConfig.merge_q_k_norm_layout,
        )
        # rope
        extra_layout = (2, 0, 0)
        fused_layout = tuple(a + b for a, b in zip(Qwen3MegaConfig.rope_layout[0], extra_layout)), Qwen3MegaConfig.rope_layout[1]
        self.mpk.rope_layer(
            q=self.attn_layer_io.rope_io.q.mpk,
            k=self.attn_layer_io.rope_io.k.mpk,
            cos=self.mpk.attach_input(torch_tensor=self.w_cos_torch, name="cos"+layer_id_str),
            sin=self.mpk.attach_input(torch_tensor=self.w_sin_torch, name="sin"+layer_id_str),
            q_embed=self.attn_layer_io.rope_io.q.mpk,
            k_embed=self.attn_layer_io.rope_io.k.mpk,
            sync_mode=(0, 0, 0),
            layout=fused_layout,
            fused_params=[99, 1, *extra_layout, layer_id],
        )
        
        # attn    
        self.mpk.gqa_decode_layer(
            q=self.attn_layer_io.attn_in.q.mpk,
            k_cache=self.attn_layer_io.attn_in.kcache.mpk,
            v_cache=self.attn_layer_io.attn_in.vcache.mpk,
            edge=self.attn_layer_io.attn_in.edge.mpk,
            mask=self.attn_layer_io.attn_in.mask.mpk,
            glse=self.attn_layer_io.attn_in.glse.mpk,
            out_partial=self.attn_layer_io.attn_in.out_partial.mpk,
            output=self.attn_layer_io.attn_out.three_dim.mpk,
            sync_mode=(0, 0, 0),
            layout=Qwen3MegaConfig.gqa_decode_layout_16,    # todo
            fused_params=[99, 0, layer_id],
        )
        self.mpk.linear_with_residual_layer(
            input=self.attn_layer_io.attn_out.two_dim.mpk,
            weight=self.mpk.attach_input(torch_tensor=w_out_proj_torch, name="w_o_proj"),
            residual=self.attn_layer_io.layer_in.mpk,
            output=self.attn_layer_io.o_proj_res_out.mpk,
            sync_mode=(0, 0, 0),
            layout=Qwen3MegaConfig.linear2_layout,
        )
    
    def fill_meta(self, step):
        edge_torch = self.attn_layer_io.attn_in.edge.pt
        edge_torch[0].fill_(step) # kv_seqlen
        edge_torch[1].fill_(self.num_kv_heads * self.head_dim) # kvcache onestep_size
        edge_torch[2].fill_(self.batch * self.q_seqlen * self.num_kv_heads * self.head_dim) # kvcache onelayer_size
        
        key_cache_5dim_torch = self.attn_layer_io.attn_in.kcache.pt
        value_cache_5dim_torch = self.attn_layer_io.attn_in.vcache.pt
        k_torch_curstep = self.attn_layer_io.kv_curstep.k.pt
        v_torch_curstep = self.attn_layer_io.kv_curstep.v.pt
        meta = [edge_torch, key_cache_5dim_torch, value_cache_5dim_torch, k_torch_curstep, v_torch_curstep]
        return meta, self.attn_layer_io.o_proj_res_out.pt
    
        # #######################################################
        # w_rms_torch, w_gatedup_torch, w_down_proj_torch = Qwen3Info.get_weight_qwen3_mlp(model, layer_id)
            
        # ###################################################################################
    
          
    # def create_qwen3_oproj_norm_mlp(self, gridsize, total_head_dims, hidden_size, intermediate_size, 
    #                                 w_o_proj_torch, w_rms_torch, w_gatedup_torch, w_down_proj_torch):
    #     self.x_torch = torch.randn((self.max_batch_size, total_head_dims), dtype=torch.bfloat16, device="cuda")
    #     self.x_residual_torch = torch.randn((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    #     self.out_torch = torch.zeros((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    #     # gridsize = [max_batch_size, 76, 38, 40]
    #     self.x = self.mpk.attach_input(torch_tensor=self.x_torch, name="in")
    #     self.x_residual = self.mpk.attach_input(torch_tensor=self.x_residual_torch, name="x_residual")
    #     self.w_o_proj = self.mpk.attach_input(torch_tensor=w_o_proj_torch, name="w_o_proj")
    #     self.w_rms = self.mpk.attach_input(torch_tensor=w_rms_torch, name="w_rms")
    #     self.w_gatedup = self.mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
    #     self.w_down_proj = self.mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    #     self.mlp_out = self.mpk.attach_input(torch_tensor=self.out_torch, name="mlp_out")
        
    #     self.o_proj_out = self.mpk.new_tensor(dims=(self.max_batch_size, hidden_size), dtype=mi.bfloat16, name="o_proj_out", io_category="cuda_tensor")
    #     self.mpk.linear_with_residual_layer(
    #         input=self.x,
    #         weight=self.w_o_proj,
    #         residual=self.x_residual,
    #         output=self.o_proj_out,
    #         grid_dim=(gridsize[0], 1, 1),
    #         block_dim=(128, 1, 1),
    #     )
        
    #     ## self.o_proj_out
         
    #     self.rmsnorm_out = self.mpk.new_tensor(dims=(self.max_batch_size, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
    #     self.mpk.rmsnorm_layer(
    #         input=self.o_proj_out,
    #         weight=self.w_rms,
    #         output=self.rmsnorm_out,
    #         grid_dim=(gridsize[1], 1, 1),
    #         block_dim=(128, 1, 1),
    #     )
        
    #     # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    #     # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
    #     self.mlp_mid = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    #     self.mpk.linear_layer(
    #         input=self.rmsnorm_out,
    #         weight=self.w_gatedup,
    #         output=self.mlp_mid,
    #         grid_dim=(gridsize[2], 1, 1),
    #         block_dim=(128, 1, 1),
    #     )
        
    #     # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    #     # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    #     # out_torch = silu_mul_out_torch
    #     self.silu_mul_out = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    #     self.mpk.silu_mul_layer(
    #         input  = self.mlp_mid,
    #         output = self.silu_mul_out,
    #         grid_dim  = (gridsize[3], 1, 1),
    #         block_dim = (128, 1, 1),
    #     )
    #     self.mpk.linear_with_residual_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
    #         input = self.silu_mul_out,
    #         weight = self.w_down_proj,
    #         residual = self.o_proj_out,
    #         output = self.mlp_out,
    #         grid_dim = (gridsize[4], 1, 1), # (64, 1, 1)
    #         block_dim = (128, 1, 1),
    #     )
            
    #     return self.x_torch, self.x_residual_torch, self.out_torch
    
    # def create_qwen3_norm_mlp(self, gridsize, hidden_size, intermediate_size, w_rms_torch, w_gatedup_torch, w_down_proj_torch):
    #     self.x_torch = torch.randn((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    #     self.out_torch = torch.zeros((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    #     # gridsize = [max_batch_size, 76, 38, 40]
    #     self.x = self.mpk.attach_input(torch_tensor=self.x_torch, name="in")
    #     self.w_rms = self.mpk.attach_input(torch_tensor=w_rms_torch, name="w_rms")
    #     self.w_gatedup = self.mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
    #     self.w_down_proj = self.mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    #     self.mlp_out = self.mpk.attach_input(torch_tensor=self.out_torch, name="mlp_out")
        
    #     self.rmsnorm_out = self.mpk.new_tensor(dims=(self.max_batch_size, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
    #     self.mpk.rmsnorm_layer(
    #         input=self.x,
    #         weight=self.w_rms,
    #         output=self.rmsnorm_out,
    #         grid_dim=(gridsize[0], 1, 1),
    #         block_dim=(128, 1, 1),
    #     )
        
    #     # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    #     # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
    #     self.mlp_mid = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    #     self.mpk.linear_layer(
    #         input=self.rmsnorm_out,
    #         weight=self.w_gatedup,
    #         output=self.mlp_mid,
    #         grid_dim=(gridsize[1], 1, 1),
    #         block_dim=(128, 1, 1),
    #     )
        
    #     # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    #     # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    #     # out_torch = silu_mul_out_torch
    #     self.silu_mul_out = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    #     self.mpk.silu_mul_layer(
    #         input  = self.mlp_mid,
    #         output = self.silu_mul_out,
    #         grid_dim  = (gridsize[2], 1, 1),
    #         block_dim = (128, 1, 1),
    #     )
    #     self.mpk.linear_with_residual_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
    #         input = self.silu_mul_out,
    #         weight = self.w_down_proj,
    #         residual = self.x,
    #         output = self.mlp_out,
    #         grid_dim = (gridsize[3], 1, 1), # (64, 1, 1)
    #         block_dim = (128, 1, 1),
    #     )
            
    #     return self.x_torch, self.out_torch