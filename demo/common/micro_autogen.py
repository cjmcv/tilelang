import os
import shutil
from pathlib import Path

import torch
import tilelang.language as T

from .pkt_util import TestUtil, TorchRef
from .micro_base import HparamSelectMode
from .micro_linear import MicroLinearStrategy, MicroLinear
from .micro_rmsnorm import MicroRmsNorm
from .micro_silu_mul import MicroSiluMul

class MicroAutoGen:
    def __init__(self, batch_size, hidden_size, intermediate_size):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = T.bfloat16
        self.accum_dtype = T.float32
        
    def _save_target(self, micro, mode: HparamSelectMode, code_dir, config_file, name):
        kernel, file_name, info = micro.get_kernel(mode)
        print(file_name, info)
        shutil.copy2(file_name, code_dir)
        config_file.write(f"    {name} = {info}\n")
        
    def gen_qwen3_mlp(self, mode: HparamSelectMode, idx: int):
        
        megakernel_home = os.getenv("MEGAKERNEL_HOME", default=None)
        if megakernel_home is None:
            raise EnvironmentError("The environment variable MEGAKERNEL_HOME is not set.")
        code_path = megakernel_home + f"/src/megakernel/persistent_kernel/tasks/autogen/m{self.batch_size}/"
        code_dir = Path(code_path)
        code_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = megakernel_home + f"/demo/common/autogen/"
        config_dir = Path(config_path)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path+"qwen3_mlp_config.py", "w", encoding="utf-8") as config_file:
            config_file.write(f"class Qwen3MlpConfig:\n")
            # rmsnorm_layout = (1, 1, 1), (1, 1, 1)
            # linear1_layout = (304, 1, 1), (64, 16, 128)
            # silu_mul_layout = (152, 1, 1), (64, 16, 1)
            # linear2_layout = (40, 1, 1), (64, 16, 64)
            if (idx == 0 or idx == 99):
                kernel = MicroRmsNorm(self.batch_size, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "rmsnorm_layout")
            if (idx == 1 or idx == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM, self.batch_size, self.intermediate_size*2, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "linear1_layout")
            if (idx == 2 or idx == 99):   
                kernel = MicroSiluMul(self.batch_size, self.intermediate_size, dtype=T.bfloat16, accum_dtype=T.float32)
                self._save_target(kernel, mode, code_dir, config_file, "silu_mul_layout")
            if (idx == 3 or idx == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM_ADD, self.batch_size, self.hidden_size, self.intermediate_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target(kernel, mode, code_dir, config_file, "linear2_layout")
            
        