import os
import math
from enum import Enum
import itertools
import json
from pathlib import Path

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor
import torch
from tvm.tir import stmt_functor, Block, For, PrimFunc
import tilelang
import tilelang.language as T

from common.pkt_util import TestUtil, TorchRef
  
class HparamSelectMode(Enum):
    HEURISTIC = 0
    TUNING = 1
    TUNED = 2

class BaseMicroKernel:
    def __init__(self):
        self.megakernel_home = os.getenv("MEGAKERNEL_HOME", default=None)
        if self.megakernel_home is None:
            raise EnvironmentError("The environment variable MEGAKERNEL_HOME is not set.")
        prop = torch.cuda.get_device_properties(0)
        self.save_path = self.megakernel_home + "/demo/gen/sm" + str(prop.major) + str(prop.minor) + "/"
        target_dir = Path(self.save_path)
        target_dir.mkdir(parents=True, exist_ok=True)

    def get_smem_bytes(self, prim_func: PrimFunc):
        smem_bytes = 0
        num_stages = 1
        
        def collect(node):
            nonlocal smem_bytes
            nonlocal num_stages
            if isinstance(node, Block):
                for buf in node.alloc_buffers:
                    scope = buf.scope()
                    if str(scope).startswith("shared"):
                        numel = 1
                        for s in buf.shape:
                            numel *= int(s)
                        smem_bytes += numel * (buf.dtype.bits // 8)
            if isinstance(node, For):
                num_stages = node.annotations.get("num_stages", 1)
                    
        stmt_functor.post_order_visit(prim_func.body, collect)
        return smem_bytes*num_stages

    def replace_line(self, text: str, src_target: str, skip_count: int, dst_target: str) -> str:
        lines = text.splitlines(True)
        processed_lines = []
        target_count = 0
        
        for line in lines:
            if src_target in line:
                target_count += 1
                if target_count == skip_count:
                    continue
                processed_lines.append(dst_target)
            else:
                processed_lines.append(line)
        return "".join(processed_lines)

    def write_tuned_hparams_to_json(self, latency_hparams_list, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for latency, hparams in latency_hparams_list:
                single_config = {
                    "latency": latency,
                    "hparams": hparams
                }
                json_line = json.dumps(single_config, ensure_ascii=False, separators=(",", ":"))
                f.write(json_line + "\n")
        
        print(f"Save: {file_path}")

    def read_tuned_hparams_from_json(self, file_path):
        latency_hparams_list = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line:
                        continue
                    try:
                        json_data = json.loads(line)
                        latency_hparams_list.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Line {line_num}: Failed to parse JSON: {e}, content: {line}")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except Exception as e:
            print(f"Unknown error during file reading: {e}")
        
        return latency_hparams_list
