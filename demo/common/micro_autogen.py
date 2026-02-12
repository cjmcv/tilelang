import os
import shutil
from pathlib import Path

import torch
import tilelang
import tilelang.language as T

from .pkt_util import TestUtil, TorchRef
from .micro_base import HparamSelectMode
from .micro_linear import MicroLinearStrategy, MicroLinear
from .micro_rmsnorm import MicroRmsNorm
from .micro_silu_mul import MicroSiluMul
from .micro_gqa_decode import MicroGqaDecode
from common.micro_rope import MicroRope

class MicroAutoGen:
    def __init__(self, batch_size, hidden_size, intermediate_size, kv_seqlen, heads, groups, dim):
        print(tilelang.__version__) # 预先加载完成FFI的静态初始化，以免初始化发生在tuning的多线程场景导致崩溃
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.kv_seqlen = kv_seqlen
        self.heads = heads
        self.groups = groups
        self.dim = dim
        
        self.dtype = T.bfloat16
        self.accum_dtype = T.float32
        
    def _save_target(self, micro, mode: HparamSelectMode, code_dir, config_file, name):
        kernel, file_name, info = micro.get_kernel(mode)
        print(file_name, info)
        shutil.copy2(file_name, code_dir)
        config_file.write(f"    {name} = {info}\n")
        
    def gen_qwen3_ops(self, layer_id: int, mode: HparamSelectMode):
        
        megakernel_home = os.getenv("MEGAKERNEL_HOME", default=None)
        if megakernel_home is None:
            raise EnvironmentError("The environment variable MEGAKERNEL_HOME is not set.")
        code_path = megakernel_home + f"/src/megakernel/persistent_kernel/tasks/autogen/m{self.batch_size}/"
        code_dir = Path(code_path)
        code_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = megakernel_home + f"/demo/common/autogen/"
        config_dir = Path(config_path)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        ## L40
        # rmsnorm_layout = (1, 1, 1), (1, 1, 1)
        # linear1_layout = (192, 1, 1), (32, 16, 256)
        # silu_mul_layout = (96, 1, 1), (32, 16, 1)
        # linear2_layout = (32, 1, 1), (32, 16, 256)        
        ## rtx4050
        # rmsnorm_layout = (1, 1, 1), (1, 1, 1)
        # linear1_layout = (96, 1, 1), (64, 16, 64)
        # silu_mul_layout = (48, 1, 1), (64, 16, 1)
        # linear2_layout = (32, 1, 1), (32, 16, 128)
    
        with open(config_path+"qwen3_mega_config.py", "w", encoding="utf-8") as config_file:
            config_file.write(f"class Qwen3MegaConfig:\n")
            if (layer_id == 0 or layer_id == 99):
                kernel = MicroRmsNorm(self.batch_size, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "rmsnorm_layout")
            if (layer_id == 1 or layer_id == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM, self.batch_size, self.intermediate_size*2, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "linear1_layout")
            if (layer_id == 2 or layer_id == 99):   
                kernel = MicroSiluMul(self.batch_size, self.intermediate_size, dtype=T.bfloat16, accum_dtype=T.float32)
                self._save_target(kernel, mode, code_dir, config_file, "silu_mul_layout")
            if (layer_id == 3 or layer_id == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM_ADD, self.batch_size, self.hidden_size, self.intermediate_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "linear2_layout")
            if (layer_id == 4 or layer_id == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM, self.batch_size, (self.heads+2*self.groups)*self.dim, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "qkv_proj_layout")
            if (layer_id == 5 or layer_id == 99):
                kernel = MicroRope(self.batch_size, 1, self.heads, self.groups, self.dim, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "rope_layout")
            if (layer_id == 6 or layer_id == 99):
                kernel = MicroGqaDecode(self.batch_size, self.kv_seqlen, self.heads, self.groups, self.dim, False, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "gqa_decode_layout")
            
            
        