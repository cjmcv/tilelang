
import torch
from types import SimpleNamespace
import megakernel as mi
from common.pkt_util import Qwen3Info

# model_tag:  "qwen3_06b" / "qwen3_4b"
class MpkLayers:
    def __init__(self, model_tag, instance_id, kernel_num, world_size, rank, max_batch_size, trace_name, profiling):
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
            model_tag=model_tag,
        )
        self.max_batch_size = max_batch_size
        self.w_qkv_proj_torch = []
        self.w_qk_norm_torch = []
        self.w_mlp_gateup_proj = []
        
        self.Qwen3MegaConfig = None
        if (model_tag == "qwen3_4b"):
            from common.autogen.qwen3_4b_mega_config import Qwen3MegaConfig4b
            self.Qwen3MegaConfig = Qwen3MegaConfig4b
        elif (model_tag == "qwen3_06b"):
            from common.autogen.qwen3_06b_mega_config import Qwen3MegaConfig06b
            self.Qwen3MegaConfig = Qwen3MegaConfig06b
            
        self.replaceable_weight_mapping = {}
        
    def get_mpk(self):
        return self.mpk
    def get_layout(self):
        return self.Qwen3MegaConfig
    
    def add_weight_pair(self, kernel_id, base_weight, target_weight):
        if kernel_id not in self.replaceable_weight_mapping:
            self.replaceable_weight_mapping[kernel_id] = []
        self.replaceable_weight_mapping[kernel_id].append((base_weight, target_weight))
        
    def compile_load(self, meta_tensors=list(), is_no_compile=False, output_dir="./gen"):
        self.mpk.append_replaceable_weights(self.replaceable_weight_mapping)
        
        if is_no_compile is True:
            module_path = output_dir + "/test.cpython-38-x86_64-linux-gnu.so"
            self.mpk.load_module(module_path, meta_tensors)
        else:
            module_path = self.mpk.compile(output_dir=output_dir)
            print("module_path: ", module_path)
            self.mpk.load_module(module_path, meta_tensors)

    def qwen3_alloc_io_buffer(self, model_tag, layer_num, batch, q_seqlen, max_kv_seqlen):
        hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim = Qwen3Info.get_basic_params(model_tag)
        
        self.batch = batch
        self.q_seqlen = q_seqlen
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # 所有层共享
        self.public_pt = SimpleNamespace(
            # 在rope中完成更新
            key_cache_5d = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16),  # [B, N=seqlen_kv,  H=groups, D=dim]
            value_cache_5d = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16),
            # 在推理前，根据step拷贝进对应的值
            cos = torch.empty((self.batch, self.q_seqlen, self.head_dim), dtype=torch.bfloat16, device="cuda"),
            sin = torch.empty((self.batch, self.q_seqlen, self.head_dim), dtype=torch.bfloat16, device="cuda"),
        )
        
        #############################################################################        
        # attn layer
        # #    rmsnorm (x) -> linear (qkv proj) -> rmsnorm (q/k) -> rope (q/k) -> attn -> linear_res
        # # -> update cos/sin (step)                             -> update kvcache (step)   
        
        ##################
        # torch tensor
        x_torch = torch.empty((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
        
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
        edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
        glse_torch = torch.empty(batch, num_heads, max_attn_split, device="cuda", dtype=torch.bfloat16)
        out_partial_torch = torch.empty(batch, num_heads, max_attn_split, head_dim, device="cuda", dtype=torch.bfloat16)
        mask_torch = torch.ones(batch, max_kv_seqlen, num_kv_heads, device="cuda", dtype=torch.uint8)
        
        attn_out_3dim_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        attn_out_2dim_torch = attn_out_3dim_torch.view(batch*q_seqlen, q_dim)
        attn_out_torch = {
            "3d": attn_out_3dim_torch,
            "2d": attn_out_2dim_torch
        }
        o_proj_res_out_torch = torch.zeros((batch, hidden_size), dtype=torch.bfloat16, device="cuda") # x_torch

        # mpk tensor
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
                kcache = SimpleNamespace(pt=self.public_pt.key_cache_5d, mpk=self.mpk.attach_input(torch_tensor=self.public_pt.key_cache_5d, name="attn_in_kcache")),
                vcache = SimpleNamespace(pt=self.public_pt.value_cache_5d, mpk=self.mpk.attach_input(torch_tensor=self.public_pt.value_cache_5d, name="attn_in_vcache")),
                edge = SimpleNamespace(pt=edge_torch, mpk=self.mpk.attach_input(torch_tensor=edge_torch, name="edge")),
                mask = SimpleNamespace(pt=mask_torch, mpk=self.mpk.attach_input(torch_tensor=mask_torch, name="attn_in_mask")),
                glse = SimpleNamespace(pt=glse_torch, mpk=self.mpk.attach_input(torch_tensor=glse_torch, name="attn_in_glse")),
                out_partial = SimpleNamespace(pt=out_partial_torch, mpk=self.mpk.attach_input(torch_tensor=out_partial_torch, name="attn_out_partial"))
            ),
            attn_out = SimpleNamespace(
                three_dim = SimpleNamespace(pt=attn_out_torch["3d"], mpk=self.mpk.attach_input(torch_tensor=attn_out_torch["3d"], name="attn_out_3dim")),
                two_dim = SimpleNamespace(pt=attn_out_torch["2d"], mpk=self.mpk.attach_input(torch_tensor=attn_out_torch["2d"], name="attn_out_2dim")),
            ),
            layer_out = SimpleNamespace(pt=o_proj_res_out_torch, mpk=self.mpk.attach_input(torch_tensor=o_proj_res_out_torch, name="attn_layer_out"))
        )
        
        #############################################################################        
        # mlp layer
        # #    rmsnorm (x) -> linear (gateup_proj) -> act (silu_mul) -> linear_res (down_proj)
        # torch tensor
        # mlp_layer_out_torch = torch.zeros((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
        # mpk tensor
        self.mlp_layer_io = SimpleNamespace(
            layer_in = self.attn_layer_io.layer_out,
            layernorm_out = SimpleNamespace(pt=None, mpk=self.mpk.new_tensor(dims=(batch, hidden_size), dtype=mi.bfloat16, name="mlp_rms_out", io_category="cuda_tensor")),
            mlp_mid = SimpleNamespace(pt=None, mpk=self.mpk.new_tensor(dims=(batch, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")),
            silu_mul_out = SimpleNamespace(pt=None, mpk=self.mpk.new_tensor(dims=(batch, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")),
            layer_out = self.attn_layer_io.layer_in,
        )
    
    def qwen3_create_attn_layer(self, model, layer_id, reuse_instance = False):
        w_input_layernorm_torch, w_q_norm_torch, w_k_norm_torch, \
        w_q_torch, w_k_torch, w_v_torch, w_out_proj_torch, \
        k_cache_torch, v_cache_torch = Qwen3Info.get_weight_qwen3_attention(model, layer_id)
        
        self.w_qkv_proj_torch.append(torch.cat([w_q_torch, w_k_torch, w_v_torch], dim=0).contiguous())
        self.w_qk_norm_torch.append(torch.cat([w_q_norm_torch, w_k_norm_torch], dim=0).contiguous())
        
        if (reuse_instance == True):
            layer_id_str = "_" + str(layer_id)
            self.mpk.attach_input(torch_tensor=w_input_layernorm_torch, name="w_layernorm"+layer_id_str)
            self.mpk.attach_input(torch_tensor=self.w_qkv_proj_torch[layer_id], name="w_qkv_proj"+layer_id_str)
            self.mpk.attach_input(torch_tensor=self.w_qk_norm_torch[layer_id], name="w_qk_norm"+layer_id_str)
            self.mpk.attach_input(torch_tensor=w_out_proj_torch, name="w_o_proj"+layer_id_str)
            
            self.add_weight_pair(layer_id, "w_layernorm", "w_layernorm"+layer_id_str)
            self.add_weight_pair(layer_id, "w_qkv_proj", "w_qkv_proj"+layer_id_str)
            self.add_weight_pair(layer_id, "w_qk_norm", "w_qk_norm"+layer_id_str)
            self.add_weight_pair(layer_id, "w_o_proj", "w_o_proj"+layer_id_str)
            return 
        
        self.mpk.rmsnorm_layer(
            input  = self.attn_layer_io.layer_in.mpk,
            weight = self.mpk.attach_input(torch_tensor=w_input_layernorm_torch, name="w_layernorm"),
            output = self.attn_layer_io.layernorm_out.mpk,
            sync_mode = (0, 0, 0),
            layout = self.Qwen3MegaConfig.rmsnorm_layout,
        )
        self.mpk.linear_layer(
            input  = self.attn_layer_io.layernorm_out.mpk,
            weight = self.mpk.attach_input(torch_tensor=self.w_qkv_proj_torch[layer_id], name="w_qkv_proj"),
            output = self.attn_layer_io.qkv_proj_out.mpk,
            sync_mode = (0, 0, 0),
            layout = self.Qwen3MegaConfig.qkv_proj_layout,
        )
        self.mpk.rmsnorm_layer(
            input  = self.attn_layer_io.qk_norm_states.mpk,
            weight = self.mpk.attach_input(torch_tensor=self.w_qk_norm_torch[layer_id], name="w_qk_norm"),
            output = self.attn_layer_io.qk_norm_states.mpk,
            sync_mode=(0, 0, 0),
            layout = self.Qwen3MegaConfig.merge_q_k_norm_layout,
        )
        # rope
        extra_layout = (2, 0, 0)
        fused_layout = tuple(a + b for a, b in zip(self.Qwen3MegaConfig.rope_layout[0], extra_layout)), self.Qwen3MegaConfig.rope_layout[1]
        self.mpk.rope_layer(
            q=self.attn_layer_io.rope_io.q.mpk,
            k=self.attn_layer_io.rope_io.k.mpk,
            cos=self.mpk.attach_input(torch_tensor=self.public_pt.cos, name="cos"), # 所有层共享，不需要替换
            sin=self.mpk.attach_input(torch_tensor=self.public_pt.sin, name="sin"),
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
            layout=self.Qwen3MegaConfig.gqa_decode_layout_16,    # todo
            fused_params=[99, 0, layer_id],
        )
        self.mpk.linear_with_residual_layer(
            input=self.attn_layer_io.attn_out.two_dim.mpk,
            weight=self.mpk.attach_input(torch_tensor=w_out_proj_torch, name="w_o_proj"),
            residual=self.attn_layer_io.layer_in.mpk,
            output=self.attn_layer_io.layer_out.mpk,
            sync_mode=(0, 0, 0),
            layout=self.Qwen3MegaConfig.linear2_layout,
        )
    
    def qwen3_create_mlp_layer(self, model, layer_id, reuse_instance = False):
        w_rms_norm_torch, w_gate_proj, w_up_proj, w_down_proj_torch = Qwen3Info.get_weight_qwen3_mlp(model, layer_id)
        self.w_mlp_gateup_proj.append(torch.cat((w_gate_proj, w_up_proj), 0).contiguous())
        
        if (reuse_instance == True):
            layer_id_str = "_" + str(layer_id)
            self.mpk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm"+layer_id_str)
            self.mpk.attach_input(torch_tensor=self.w_mlp_gateup_proj[layer_id], name="w_gatedup"+layer_id_str)
            self.mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj"+layer_id_str)
            
            self.add_weight_pair(layer_id, "w_norm", "w_norm"+layer_id_str)
            self.add_weight_pair(layer_id, "w_gatedup", "w_gatedup"+layer_id_str)
            self.add_weight_pair(layer_id, "w_down_proj", "w_down_proj"+layer_id_str)
            return 
        
        self.mpk.rmsnorm_layer(
            input = self.mlp_layer_io.layer_in.mpk,
            weight = self.mpk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm"),
            output = self.mlp_layer_io.layernorm_out.mpk,
            sync_mode=(0, 0, 0),
            layout=self.Qwen3MegaConfig.rmsnorm_layout,
        )
        self.mpk.linear_layer(
            input  = self.mlp_layer_io.layernorm_out.mpk,
            weight = self.mpk.attach_input(torch_tensor=self.w_mlp_gateup_proj[layer_id], name="w_gatedup"),
            output = self.mlp_layer_io.mlp_mid.mpk,
            sync_mode=(0, 0, 0),
            layout = self.Qwen3MegaConfig.linear1_layout,
        )
        self.mpk.silu_mul_layer(
            input  = self.mlp_layer_io.mlp_mid.mpk,
            output = self.mlp_layer_io.silu_mul_out.mpk,
            sync_mode=(2, 0, 0),
            layout = self.Qwen3MegaConfig.silu_mul_layout,
        )
        self.mpk.linear_with_residual_layer(
            input    = self.mlp_layer_io.silu_mul_out.mpk,
            weight   = self.mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj"),
            residual = self.mlp_layer_io.layer_in.mpk,
            output   = self.mlp_layer_io.layer_out.mpk,
            sync_mode=(0, 0, 0),
            layout   = self.Qwen3MegaConfig.linear2_layout,
        )
        
    def fill_meta(self):
        edge_torch = self.attn_layer_io.attn_in.edge.pt
        # edge_torch[0].fill_(step) # kv_seqlen
        edge_torch[1].fill_(self.num_kv_heads * self.head_dim) # kvcache onestep_size
        edge_torch[2].fill_(self.batch * self.q_seqlen * self.num_kv_heads * self.head_dim) # kvcache onelayer_size
        
        k_torch_curstep = self.attn_layer_io.kv_curstep.k.pt
        v_torch_curstep = self.attn_layer_io.kv_curstep.v.pt
        meta = [edge_torch, self.public_pt.key_cache_5d, self.public_pt.value_cache_5d, k_torch_curstep, v_torch_curstep]
        return meta, self.attn_layer_io.layer_out.pt, self.mlp_layer_io.layer_out.pt
    
    def update_step(self, step, cos, sin):
        edge_torch = self.attn_layer_io.attn_in.edge.pt
        edge_torch[0].fill_(step) # kv_seqlen
        self.public_pt.cos.copy_(cos)
        self.public_pt.sin.copy_(sin)
        