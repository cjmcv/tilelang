"""The profiler and convert to torch utils"""


from __future__ import annotations

import sys
import os
from typing import Callable, Any, Literal
from functools import partial
import torch
from contextlib import suppress
from dataclasses import dataclass
import tvm
from tilelang.utils.tensor import (
    get_tensor_supply,
    TensorSupplyType,
    torch_assert_close,
    is_float8_dtype,
)
from tilelang.engine.param import KernelParam
from tilelang.jit.tvm_ffi import BaseKernelAdapter


class suppress_stdout_stderr:
    """Context manager to suppress stdout and stderr output.

    Source: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/bench.py
    """

    def __enter__(self):
        # Open null device files
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        # Save original file descriptors
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        # Save original stdout/stderr objects
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        # Redirect file descriptors and streams to null device
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file

        return self

    def __exit__(self, *_):
        # Restore original stdout/stderr objects
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        # Restore original file descriptors
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        # Close duplicated file descriptors
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        # Close null device files
        self.outnull_file.close()
        self.errnull_file.close()


IS_CUDA = torch.cuda.is_available()
device = "cuda:0"
Event = torch.cuda.Event


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    backend: Literal["event", "cupti"] = "event",
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
) -> float | list[float]:
    """Benchmark the runtime of a PyTorch function with L2 cache management.

    This function provides accurate GPU kernel timing by:
    - Clearing L2 cache between runs for consistent measurements
    - Auto-calculating warmup and repeat counts based on kernel runtime
    - Supporting multiple profiling backends (CUDA events or CUPTI)
    - Offering flexible result aggregation (mean/median/min/max/quantiles)

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds (default: 25)
        rep: Target total benchmark time in milliseconds (default: 100)
        _n_warmup: Manual override for warmup iterations (default: 0 = auto)
        _n_repeat: Manual override for benchmark iterations (default: 0 = auto)
        quantiles: Performance percentiles to compute (e.g., [0.5, 0.95])
        fast_flush: Use faster L2 cache flush with int32 vs int8 (default: True)
        backend: Profiler backend - "event" (CUDA events) or "cupti" (default: "event")
        return_mode: Result aggregation method - "mean", "median", "min", or "max"

    Returns:
        Runtime in milliseconds (float) or list of quantile values if quantiles specified
    """
    assert return_mode in ["min", "max", "mean", "median"], f"Invalid return_mode: {return_mode}"

    # Initial function call and synchronization
    fn()
    torch.cuda.synchronize()

    # Create L2 cache flush buffer (256 MB)
    # Fast flush uses int32 (4 bytes), regular uses int8 (1 byte)
    cache_size = int(256e6 // 4) if fast_flush else int(256e6)
    cache_dtype = torch.int if fast_flush else torch.int8
    cache = torch.empty(cache_size, dtype=cache_dtype, device="cuda")

    # Estimate kernel runtime with 5 iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    start_event.synchronize()
    end_event.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Calculate warmup and repeat counts (minimum 1 iteration each)
    n_warmup = _n_warmup if _n_warmup > 0 else max(1, int(warmup / estimate_ms))
    n_repeat = _n_repeat if _n_repeat > 0 else max(1, int(rep / estimate_ms))

    # Warmup phase
    for _ in range(n_warmup):
        fn()

    # Benchmarking phase
    if backend == "event":
        return _bench_with_cuda_events(fn, cache, n_repeat, quantiles, return_mode)
    elif backend == "cupti":
        return _bench_with_cupti(fn, cache, n_repeat)
    else:
        raise ValueError(f"Unknown profiler backend: {backend}")


def _bench_with_cuda_events(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
) -> float | list[float]:
    """Benchmark using CUDA events for timing."""
    # Create timing events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    # Run benchmark iterations
    for i in range(n_repeat):
        cache.zero_()  # Clear L2 cache
        start_events[i].record()
        fn()
        end_events[i].record()

    # Synchronize and collect timings
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_events, end_events)],
        dtype=torch.float,
    )

    # Return quantiles if requested
    if quantiles is not None:
        quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return quantile_values[0] if len(quantile_values) == 1 else quantile_values

    # Return aggregated result
    return getattr(torch, return_mode)(times).item()


def _bench_with_cupti(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
) -> float:
    """Benchmark using CUPTI profiler for detailed kernel timing."""
    with suppress_stdout_stderr():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
        )

        with profiler:
            for _ in range(2):
                for _ in range(n_repeat):
                    cache.zero_()
                    fn()
                profiler.step()

    # Calculate average kernel time, excluding cache-clearing overhead
    total_cuda_time = 0.0
    excluded_time = 0.0
    excluded_kernels = "at::native::vectorized_elementwise"

    for event in profiler.key_averages():
        total_cuda_time += event.self_device_time_total
        if excluded_kernels in event.key:
            excluded_time += event.self_device_time_total

    kernel_time_us = (total_cuda_time - excluded_time) / n_repeat
    return kernel_time_us * 1e-3  # Convert microseconds to milliseconds

@dataclass
class Profiler:
    """A profiler class for benchmarking and validating kernel implementations.

    Attributes:
        params: List of kernel parameters defining the input/output specifications
        result_idx: Indices indicating which parameters are output tensors
        supply_type: Type of tensor supply to use (e.g., random, zeros, etc.)
        adapter: Optional kernel adapter for interfacing with different backends
    """

    params: list[KernelParam]
    result_idx: list[int]
    supply_type: TensorSupplyType
    adapter: BaseKernelAdapter | None = None

    def __post_init__(self):
        """Initialize tensor supply after dataclass initialization"""
        self.result_idx = self._legalize_result_idx(self.result_idx)
        self.supply = get_tensor_supply(self.supply_type)

    def _legalize_result_idx(self, result_idx: list[int] | None = None) -> list[int]:
        params = self.params
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(f"result_idx should be an integer between {-len(params)} and {len(params) - 1}")
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif not isinstance(result_idx, list):
            raise ValueError("result_idx should be a list of integers")

        return result_idx

    def with_default_adapter(self, adapter: BaseKernelAdapter) -> Profiler:
        self.adapter = adapter
        return self

    def _get_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins

    def _get_params(self, with_output=False):
        params = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                params.append(self.params[i])
        return params

    def assert_allclose(
        self,
        reference_program: Callable,
        input_tensors: list[torch.Tensor] | None = None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        max_mismatched_ratio=0.01,
    ):
        """Validates kernel output against a reference implementation.

        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        ins = self._get_inputs() if input_tensors is None else input_tensors
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        elif isinstance(lib_outs, tuple):
            lib_outs = list(lib_outs)
        elif lib_outs is None:
            lib_outs = []

        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif isinstance(ref_outs, tuple):
            ref_outs = list(ref_outs)
        elif ref_outs is None:
            ref_outs = []

        ref_tensors = ins + ref_outs
        lib_tensors = ins + lib_outs

        assert len(lib_tensors) == len(ref_tensors), "len(lib_tensors) not equals to len(ref_tensors) !"
        # torch.set_printoptions(edgeitems=torch.inf)
        for lhs, rhs in zip(lib_tensors, ref_tensors):
            # close_mask = torch.isclose(lhs, rhs, rtol=rtol, atol=atol)
            # total_elements = lhs.numel()
            # num_not_close = (~close_mask).sum().item()
            # percentage_not_close = (num_not_close / total_elements) * 100
            # print(f"{percentage_not_close:.2f}% of the elements are not close.")
            # print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
            if lhs is not None and rhs is not None:
                # in case of numsplit template, the ref output may be None
                # which means the value is invalid, so we skip the comparison
                torch_assert_close(
                    lhs if not is_float8_dtype(lhs.dtype) else lhs.to(torch.float32),
                    rhs if not is_float8_dtype(rhs.dtype) else rhs.to(torch.float32),
                    rtol=rtol,
                    atol=atol,
                    max_mismatched_ratio=max_mismatched_ratio,
                    base_name="tilelang",
                    ref_name="ref",
                )

    def manual_assert_close(
        self,
        reference_program: Callable,
        input_tensors: list[torch.Tensor] | None = None,
        manual_check_prog: Callable = None,
    ):
        """Validates kernel output against a reference implementation.

        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        ins = self._get_inputs() if input_tensors is None else input_tensors
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif ref_outs is None:
            ref_outs = []
        assert len(lib_outs) == len(ref_outs), f"{len(lib_outs)=} not equals to {len(ref_outs)=} !"
        torch.set_printoptions(edgeitems=torch.inf)
        manual_check_prog(lib_outs, ref_outs)

    def assert_consistent(self, repeat=10):
        """Checks for kernel consistency across multiple runs.

        Args:
            repeat: Number of times to repeat the consistency check
        """
        # Used to check no race condition inside the kernel
        ins = self._get_inputs()
        ref_outs = self.func(*ins)

        for _ in range(repeat):
            lib_outs = self.func(*ins)
            for lhs, rhs in zip(lib_outs, ref_outs):
                assert torch.allclose(lhs, rhs), [
                    "result is not consistent",
                    lhs,
                    rhs,
                ]

    def run_once(self, func: Callable | None = None):
        ins = self._get_inputs()
        if not func:
            func = self.__call__
        return func(*ins)

    def determine_profiler(self, func: Callable | None = None):
        """Determines which profiler backend to use based on function type.

        Args:
            func: Function to be profiled
            profiler: Explicitly specified profiler type or "auto" for automatic detection

        Returns:
            str: The determined profiler type ("torch" or "tvm")
        """
        if isinstance(func, tvm.runtime.Module):
            return "tvm"
        else:
            return "torch"

    def do_bench(
        self,
        func: Callable | None = None,
        warmup: int = 25,
        rep: int = 100,
        n_warmup: int = 50,
        n_repeat: int = 50,
        input_tensors: list[torch.Tensor] = None,
        backend: Literal["event", "cupti"] = "event",
        quantiles: list[float] | None = None,
        return_mode: Literal["min", "max", "mean", "median"] = "mean",
    ) -> float:
        """Benchmarks the execution time of a given function.

        Args:
            func: Function to benchmark (uses adapter if None)
            warmup: Warmup time in milliseconds
            rep: Number of repetitions for timing
            n_warmup: Number of warmup iterations
            n_repeat: Number of timing iterations
            profiler: Which profiling backend to use
            input_tensors: Optional pre-generated input tensors

        Returns:
            float: Average execution time in milliseconds
        """
        profiler = self.determine_profiler(func)
        if profiler == "torch":
            if func is None:
                assert self.adapter is not None, "benchmarking function should be provided"
                func = self.adapter
            ins = self._get_inputs() if input_tensors is None else input_tensors
            bench_func = partial(func, *ins)
            return do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
                quantiles=quantiles,
                backend=backend,
                return_mode=return_mode,
            )
        elif profiler == "tvm":
            assert func is not None, "func should not be None"
            assert isinstance(func, tvm.runtime.Module), f"func should be a TVM module, but got {type(func)}"

            ins = self._get_inputs(with_output=True) if input_tensors is None else input_tensors
            target = "cuda"

            with suppress(Exception):
                target = self.mod.imported_modules[0].type_key

            assert target in ["cuda", "hip"], f"Unknown target: {target}"

            device = tvm.cuda(0) # if target == "cuda" else tvm.rocm(0)
            time_evaluator = self.mod.time_evaluator(self.mod.entry_name, device, number=rep, repeat=n_repeat)
            # Transform Latency to ms
            return time_evaluator(*ins).mean * 1e3
        else:
            raise ValueError(f"Unknown profiler: {profiler}")

    @property
    def func(self):
        assert self.adapter is not None, "adapter should be provided"
        return self.adapter

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
