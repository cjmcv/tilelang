import os
import shutil
from pathlib import Path

import torch
import tilelang.language as T

from .pkt_util import TestUtil, TorchRef
from .micro_base import HparamSelectMode
from .micro_linear import MicroLinearStrategy, MicroLinear
from .micro_rms_norm import MicroRmsNorm
from .micro_silu_mul import MicroSiluMul

class MicroAutoGen:
    def __init__(self, batch_size, hidden_size, intermediate_size):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = T.bfloat16
        self.accum_dtype = T.float32
        
    def gen_qwen_mlp(self, mode: HparamSelectMode, idx: int):
        
        megakernel_home = os.getenv("MEGAKERNEL_HOME", default=None)
        if megakernel_home is None:
            raise EnvironmentError("The environment variable MEGAKERNEL_HOME is not set.")
        target_path = megakernel_home + f"/src/megakernel/persistent_kernel/tasks/autogen/m{self.batch_size}/"
        target_dir = Path(target_path)
        target_dir.mkdir(parents=True, exist_ok=True)        
        
        if (idx == 0 or idx == 99):
            rmsnorm = MicroRmsNorm(self.batch_size, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
            rmsnorm_kernel, rmsnorm_file_name, rmsnorm_info = rmsnorm.get_kernel(mode)
            print(rmsnorm_file_name, rmsnorm_info)
            shutil.copy2(rmsnorm_file_name, target_dir)
            
        if (idx == 1 or idx == 99):
            linear1 = MicroLinear(MicroLinearStrategy.GEMM, self.batch_size, self.intermediate_size*2, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
            linear1_kernel, linear1_file_name, linear1_info = linear1.get_kernel(mode)
            print(linear1_file_name, linear1_info)
            shutil.copy2(linear1_file_name, target_dir)
         
        if (idx == 2 or idx == 99):   
            silu_mul = MicroSiluMul(self.batch_size, self.intermediate_size, dtype=T.bfloat16, accum_dtype=T.float32)
            silu_mul_kernel, silu_mul_file_name, silu_mul_info = silu_mul.get_kernel(mode)
            print(silu_mul_file_name, silu_mul_info)
            shutil.copy2(silu_mul_file_name, target_dir)
            
        if (idx == 3 or idx == 99):
            linear2 = MicroLinear(MicroLinearStrategy.GEMM_ADD, self.batch_size, self.hidden_size, self.intermediate_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
            linear2_kernel, linear2_file_name, linear2_info = linear2.get_kernel(mode)
            print(linear2_file_name, linear2_info)
            shutil.copy2(linear2_file_name, target_dir)