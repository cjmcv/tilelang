import os
import math
from enum import Enum
import itertools
from collections.abc import Iterable
from typing import ParamSpec, TypeVar, Literal, Any
import concurrent.futures
from tqdm.auto import tqdm

from tilelang.jit.kernel import JITKernel
from tilelang.language.v2 import PrimFunc
from tvm.target import Target

import json
from pathlib import Path

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor
import torch
import tvm
from tvm.tir import stmt_functor, Block, For, PrimFunc
from tvm.tir.stmt_functor import ir_transform
import tilelang
import tilelang.language as T

from common.pkt_util import TestUtil, TorchRef
  
class HparamSelectMode(Enum):
    HEURISTIC = 0
    TUNING = 1
    TUNED = 2

# print("artifact: ", artifact)
# T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 49152, "target": T.target({"arch": "sm_89", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 304, "blockIdx.y": 1, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"], "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1]})

# analyzer = LaunchInfoAnalyzer(kernel.prim_func)
# analyzer.get_threads_layout()
# print(analyzer.grid_dim)
class LaunchInfoAnalyzer:
    def __init__(self, fn: PrimFunc):
        self.prim_func = fn
        self.ir_module = tvm.IRModule({"main": fn})
        self.grid_dim = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        self.block_dim = {"threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}
        self.dyn_shared_memory_buf = 0
        self.loop_stack = []
        
    def get_threads_layout(self):
        """
        Traverse and transform the IR module to extract performance-related information.
        Returns:
            self: The LaunchInfoAnalyzer instance.
        """

        def _ftransform(f, mod, ctx):
            # Initialize the set of global buffers
            self.global_buffers = set(f.buffer_map.values())

            def _pre_visit(stmt):
                """
                Pre-visit callback for IR nodes.
                Args:
                    stmt: The current IR node being visited.
                """
                # print(type(stmt), stmt, "\n\n")
                if isinstance(stmt, tvm.tir.AttrStmt):
                    # Handle thread extent attributes
                    # print(stmt.attr_key)
                    if stmt.attr_key == "thread_extent":
                        iter_var = stmt.node
                        thread_tag = iter_var.thread_tag
                        if thread_tag in self.grid_dim:
                            extent = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
                            self.grid_dim[thread_tag] = extent
                        elif thread_tag in self.block_dim:
                            extent = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
                            self.block_dim[thread_tag] = extent
                elif isinstance(stmt, tvm.tir.For):
                    # Push loop extent onto the stack
                    self.loop_stack.append(stmt.extent)
                # elif isinstance(stmt, tvm.tir.Evaluate):
                #     # Handle Evaluate nodes containing calls
                #     value = stmt.value
                #     if isinstance(value, tvm.tir.Call):
                #         if value.op.name == "tl.copy":
                #             self._analyze_copy(value)
                #         elif value.op.name == "tl.gemm":
                #             self._analyze_gemm(value)
                return None

            def _post_visit(stmt):
                """
                Post-visit callback for IR nodes.
                Args:
                    stmt: The current IR node being visited.
                """
                if isinstance(stmt, tvm.tir.For) and self.loop_stack:
                    self.loop_stack.pop()
                return None

            # Use IR transformation to traverse and modify the function body
            new_body = ir_transform(f.body, _pre_visit, _post_visit)
            return f.with_body(new_body)

        # Apply the custom PrimFunc pass
        tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)(self.ir_module)
        return self
    
    def get_smem_bytes(self):
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
                    
        stmt_functor.post_order_visit(self.prim_func.body, collect)
        return smem_bytes*num_stages
class BaseMicroKernel:
    def __init__(self):
        self.megakernel_home = os.getenv("MEGAKERNEL_HOME", default=None)
        if self.megakernel_home is None:
            raise EnvironmentError("The environment variable MEGAKERNEL_HOME is not set.")
        prop = torch.cuda.get_device_properties(0)
        self.save_path = self.megakernel_home + "/demo/gen/sm" + str(prop.major) + str(prop.minor) + "/"
        target_dir = Path(self.save_path)
        target_dir.mkdir(parents=True, exist_ok=True)

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
            for latency, hparams, idx in latency_hparams_list:
                single_config = {
                    "latency": latency,
                    "hparams": hparams,
                    "idx": idx,
                }
                json_line = json.dumps(single_config, ensure_ascii=False, separators=(",", ":"))
                f.write(json_line + "\n")
        
        print(f"Save: {file_path}")

    def read_tuned_hparams_from_json(self, kernel_name):
        latency_hparams_list = []
        read_file_path = self.save_path+kernel_name+"_tuned.json"
            
        try:
            with open(read_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line:
                        continue
                    try:
                        json_data = json.loads(line)
                        latency_hparams_list.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Line {line_num}: Failed to parse JSON: {e}, content: {line}")
        except FileNotFoundError:
            print(f"Error: File {read_file_path} not found.")
        except Exception as e:
            print(f"Unknown error during file reading: {e}")
            
        return latency_hparams_list

    def run_tuning(self, kernel_name, hparam_space, get_kernel_func):
        tuned_file_path = self.save_path + kernel_name + "_tuned.json"
        print(f"Start tuning with a total of {len(hparam_space)} schemes.")

        latency_hparams_list = []
        
        is_compile_parallel = True
        # # compile
        if (is_compile_parallel):
            num_workers = 8
            with concurrent.futures.ThreadPoolExecutor(num_workers, "tl-par-comp") as executor:
                futures = []
                future_map = {}
                for idx, hparams in enumerate(hparam_space):
                    future = executor.submit(get_kernel_func, selected_hparams=hparams)
                    future_map[future] = idx
                    futures.append(future)
                kernels = [... for _ in futures]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Parallel Compiling",
                ):
                    idx = future_map[future]
                    kernels[idx] = future.result()
    
        # profile
        for idx, hparams in enumerate(hparam_space):
            try:
                if (is_compile_parallel):
                    kernel = kernels[idx]
                else:
                    kernel = get_kernel_func(hparams)
                profiler = kernel.get_profiler()
                latency = round(profiler.do_bench(backend="cupti"), 5)
                status = "success"
            except Exception as e:
                latency = None, 
                status = f"{e}"   
        
            if status == "success":
                latency_hparams_list.append((latency, hparams, idx))
            print(f">>>>> tuning({idx}-{status}): {latency} -> {hparams}")
            
        latency_hparams_list.sort(key=lambda x: x[0])
        self.write_tuned_hparams_to_json(latency_hparams_list, tuned_file_path)
        best_latency, selected_hparams, idx = latency_hparams_list[0]
        print(f"[Tuning], the best result: {best_latency} ms -> {selected_hparams}")
        
        return latency_hparams_list