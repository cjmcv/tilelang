import os
import math
import itertools
import json

# 
import tilelang
import tilelang.language as T
import tvm
from tvm.tir.stmt_functor import ir_transform

from common.pkt_util import TestUtil, TorchRef
from common.micro_kernel_base import HparamSelectMode
from common.micro_linear import MicroLinear

# print("artifact: ", artifact)
# T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 49152, "target": T.target({"arch": "sm_89", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 304, "blockIdx.y": 1, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"], "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1]})
class Analyzer:
    def __init__(self, fn):
        self.fn = tvm.IRModule({"main": fn})
        self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        self.thread_counts = {"threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}
        self.dyn_shared_memory_buf = 0
        self.loop_stack = []
        
    def ir_pass(self):
        """
        Traverse and transform the IR module to extract performance-related information.
        Returns:
            self: The Analyzer instance.
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
                print(type(stmt), stmt, "\n\n")
                if isinstance(stmt, tvm.tir.AttrStmt):
                    # Handle thread extent attributes
                    print(stmt.attr_key)
                    if stmt.attr_key == "thread_extent":
                        iter_var = stmt.node
                        thread_tag = iter_var.thread_tag
                        if thread_tag in self.block_counts:
                            extent = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
                            self.block_counts[thread_tag] = extent
                        elif thread_tag in self.thread_counts:
                            extent = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
                            self.thread_counts[thread_tag] = extent
                elif isinstance(stmt, tvm.tir.For):
                    # Push loop extent onto the stack
                    self.loop_stack.append(stmt.extent)
                elif isinstance(stmt, tvm.tir.Evaluate):
                    # Handle Evaluate nodes containing calls
                    value = stmt.value
                    if isinstance(value, tvm.tir.Call):
                        if value.op.name == "tl.copy":
                            self._analyze_copy(value)
                        elif value.op.name == "tl.gemm":
                            self._analyze_gemm(value)
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
        tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)(self.fn)
        return self
    
def main():
    M = 1
    N = 19456
    K = 2560
    # config = [64,64,64,2,128,0,true]
    linear = MicroLinear(M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel = linear.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING

    import torch

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    c = kernel(a, b)
    ref_c = TorchRef.linear(a, b)

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)
    
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    
    # analyzer = Analyzer(kernel.prim_func)
    # analyzer.ir_pass()
    # print("analyzer: ", analyzer.block_counts, analyzer.thread_counts, analyzer.dyn_shared_memory_buf, analyzer.loop_stack)
    # print("a:", kernel.artifact)
    print("0:", kernel.prim_func.attrs)
    print("1:", kernel.adapter.params)
    print("2:", kernel.adapter.func)
    print("3:", kernel.config)
    print("4:", kernel.prim_func)
    # print("5:", kernel.prim_func.body)
    # print("5:", kernel.prim_func.metadata)
    
    latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


if __name__ == "__main__":
    main()