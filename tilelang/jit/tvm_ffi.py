"""Utilities to adapt TVM FFI kernels to Torch tensors.

This adapter intentionally captures PyTorch's current CUDA stream and device
via light-weight callables so that, when the wrapped function is invoked,
the execution observes the same stream context as the active Torch code.
On non-CUDA builds, the stream/device fall back to 0/CPU semantics.
"""

from __future__ import annotations

from typing import Callable, Any
from abc import ABC, abstractmethod

import torch
from tilelang import tvm
from tvm import runtime, tir
from tvm.target import Target
from tvm.relax import TensorType
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module
from tilelang.engine.param import KernelParam


class BaseKernelAdapter(ABC):
    func: Callable | None = None

    def __init__(self, mod, params: list[KernelParam], result_idx: list[int]) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self._post_init()

    def _legalize_result_idx(self, result_idx: list[int] | None) -> list[int]:
        params = self.params
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(f"result_idx should be an integer between {-len(params) - 1} and {len(params) - 1}")
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif isinstance(result_idx, list):
            for i, idx in enumerate(result_idx):
                if idx >= len(params) or idx < -len(params):
                    raise ValueError(f"result_idx should be an integer between {-len(params) - 1} and {len(params) - 1}")
                if idx < 0:
                    result_idx[i] = len(params) + idx
        else:
            raise ValueError("result_idx should be a list of integers")

        return result_idx

    @abstractmethod
    def _convert_torch_func(self) -> callable:
        pass

    # --- Common helpers to align with PyTorch stream/device semantics ---
    @staticmethod
    def get_current_stream_functor() -> Callable[[], int]:
        """Return a callable that reads Torch's current CUDA stream pointer.

        The returned lambda yields the raw CUDA stream handle of the current
        PyTorch stream on the active device. It's a thunk (evaluated at call
        time) so that any upstream stream guards are respected. If CUDA is
        unavailable, it returns a lambda that yields 0.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda._lazy_init()
                current_device = torch._C._cuda_getDevice
                get_stream = torch._C._cuda_getCurrentRawStream
                return lambda: get_stream(current_device())
            except Exception:
                # Fallback to Python API if internal handles are unavailable
                return lambda: int(torch.cuda.current_stream().cuda_stream)
        # CPU or CUDA unavailable: no stream semantics
        return lambda: 0

    @staticmethod
    def get_current_device_functor() -> Callable[[], torch.device]:
        """Return a callable that yields Torch's current device.

        Similar to the stream functor, we capture a callable that, when called,
        fetches the current device according to PyTorch. On CPU or when CUDA is
        unavailable, returns ``torch.device('cpu')``.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda._lazy_init()
                current_device = torch._C._cuda_getDevice
                return lambda: torch.device("cuda", current_device())
            except Exception:
                return lambda: torch.device("cuda", torch.cuda.current_device())
        # CPU fallback
        return lambda: torch.device("cpu")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if kernel_only:
            return self.mod.imports[0].inspect_source()
        else:
            return self.mod.inspect_source() + "\n\n" + self.mod.imports[0].inspect_source()

    def _post_init(self):
        self.func = self._convert_torch_func()


class TVMFFIKernelAdapter(BaseKernelAdapter):
    """Adapter that runs a TVM runtime.Executable with Torch tensors.

    Notes
    - We capture the "current" PyTorch CUDA stream/device as thunks (callables)
      rather than materializing them at construction time. This ensures the
      actual stream/device is read just-in-time when the function runs, matching
      the user's current Torch context (e.g., after a stream guard/switch).
    - The stream pointer returned is a raw CUDA stream handle compatible with
      TVM's device API; on CPU or when CUDA is unavailable, we return 0.
    """

    # Class attributes to store compiled kernel information
    target: str | Target = "cuda"
    ir_module: tvm.IRModule | None = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None
    executable: tvm.runtime.Executable | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None
    # host_mod
    host_mod: tvm.IRModule | None = None
    # device_mod
    device_mod: tvm.IRModule | None = None
    # rt_mod
    rt_mod: tvm.runtime.Module | None = None
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: dict[tir.Var, tuple[int, int, int]] | None = None

    # Stream/device functors are inherited from BaseKernelAdapter
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        rt_mod: tvm.runtime.Module | None = None,
        host_kernel_source: str | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        """Initialize the adapter with the given TIR function or module.

        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.host_kernel_source = host_kernel_source
        self.device_kernel_source = device_kernel_source

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target.canon_target(determine_target(target))

        self.host_mod = host_mod
        self.device_mod = device_mod
        self.rt_mod = rt_mod
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags
        self.dynamic_symbolic_map = self._process_dynamic_symbolic()

        # self.wrapper = TLWrapper(self.target)
        # self.wrapper.assign_optimized_module(self.ir_module)
        # self.wrapper.assign_pass_configs(pass_configs)
        # self.wrapper.assign_host_module(host_mod)
        # self.wrapper.assign_device_module(device_mod)
        # self.host_kernel_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))
        # print("get_kernel_source", host_mod)
        # print("tvm_ffi", self.host_kernel_source)
        
        self._post_init()

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
        """Extract information about dynamic shapes from the TIR function.

        Maps symbolic variables to their corresponding (id, buffer_index, dimension)
        for runtime shape resolution.
        id represents shape or stride, 0 represents shape, 1 represents stride
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if isinstance(param, tir.Var) and (param not in dynamic_symbolic_map):
                dynamic_symbolic_map[param] = (2, i, -1)
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map) and (shape not in params):
                        dynamic_symbolic_map[shape] = (0, i, j)
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, stride in enumerate(buffer.strides):
                    if isinstance(stride, tir.Var) and (stride not in dynamic_symbolic_map) and (stride not in params):
                        dynamic_symbolic_map[stride] = (1, i, j)
        return dynamic_symbolic_map

    def _convert_torch_func(self) -> Callable[..., Any]:
        # Capture thunks that reflect Torch's current stream and device.
        # These are evaluated at call time to align TVM execution with the
        # caller's active PyTorch stream/device.
        # current_stream_functor = self.get_current_stream_functor()
        current_device_functor = self.get_current_device_functor()

        # Convert TVM types to native Python types during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        param_dtypes = [param.torch_dtype() for param in self.params]
        # Convert TVM shape arrays to native Python lists
        param_shapes = []

        for param in self.params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            param_shapes.append(native_shape)

        if self.executable is None:
            self.executable = runtime.Executable(self.rt_mod)

        dynamic_symbolic_map = self._process_dynamic_symbolic()
        executable = self.executable

        # Prepare helpers for friendly dtype error messages
        prim_func = self.prim_func
        buffer_map = prim_func.buffer_map
        params = prim_func.params
        # Expected dtype string per parameter index (for buffers only)
        expected_dtype_strs: list[str | None] = []
        # Track whether each param is a buffer (has dtype) vs scalar
        is_buffer_param: list[bool] = []
        for p in params:
            if p in buffer_map:
                expected_dtype_strs.append(str(buffer_map[p].dtype))
                is_buffer_param.append(True)
            else:
                expected_dtype_strs.append(None)
                is_buffer_param.append(False)

        # Map torch dtype to TVM-style dtype string
        def torch_dtype_to_tvm_str(dtype: torch.dtype) -> str:
            try:
                import torch as _torch
            except Exception:  # pragma: no cover
                # Fallback, though torch should always be available here
                return str(dtype)
            fp8_e4m3fn = getattr(_torch, "float8_e4m3fn", None)
            fp8_e4m3fnuz = getattr(_torch, "float8_e4m3fnuz", None)
            fp8_e5m2 = getattr(_torch, "float8_e5m2", None)
            fp8_e5m2fnuz = getattr(_torch, "float8_e5m2fnuz", None)
            if fp8_e4m3fn is not None and dtype == fp8_e4m3fn:
                return "float8_e4m3"
            if fp8_e4m3fnuz is not None and dtype == fp8_e4m3fnuz:
                return "float8_e4m3fnuz"
            if fp8_e5m2 is not None and dtype == fp8_e5m2:
                return "float8_e5m2"
            if fp8_e5m2fnuz is not None and dtype == fp8_e5m2fnuz:
                return "float8_e5m2"
            # Strip torch. prefix for readability
            s = str(dtype)
            return s[6:] if s.startswith("torch.") else s

        def func(*inputs: torch.Tensor | Any):
            # Validate input count strictly
            expected_inputs = len(self.params) - len(self.result_idx)
            if len(inputs) != expected_inputs:
                raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")

            # Resolve the device used for outputs. Prefer the first tensor input's device
            # if available, otherwise use PyTorch's current device.
            out_device: torch.device | None = None

            # Stitch the full positional argument list expected by the TVM executable
            ins_idx: int = 0
            tensor_list: list[torch.Tensor] = []

            # Prepare input and output tensors
            for i in range(len(self.params)):
                if i in self.result_idx:
                    dtype = param_dtypes[i]
                    shape = []
                    # Now working with native Python list, no FFI calls needed
                    for s in param_shapes[i]:
                        if isinstance(s, tir.Var):
                            for key in dynamic_symbolic_map:
                                if str(s) == str(key):
                                    ref_id, ref_tensor_idx, ref_shape_idx = dynamic_symbolic_map[key]
                                    if ref_id == 2:
                                        shape.append(inputs[ref_tensor_idx])
                                    elif ref_id == 0:
                                        shape.append(tensor_list[ref_tensor_idx].shape[ref_shape_idx])
                                    elif ref_id == 1:
                                        shape.append(tensor_list[ref_tensor_idx].stride()[ref_shape_idx])
                        else:  # Already converted to Python int during initialization
                            shape.append(s)

                    if out_device is None:
                        out_device = current_device_functor()

                    if len(shape) == 0:
                        param_name = self.params[i].name if hasattr(self.params[i], "name") else f"parameter_{i}"
                        raise ValueError(
                            f"Cannot create output tensor (name={param_name}) - 0-dimensional tensors are not supported. "
                            f"Expected shape: {shape}"
                        )
                    tensor = torch.empty(*shape, dtype=dtype, device=out_device)
                else:
                    tensor = inputs[ins_idx]
                    ins_idx += 1
                tensor_list.append(tensor)

            executable(*tensor_list)

            # Return outputs in the requested form
            if len(self.result_idx) == 1:
                return tensor_list[self.result_idx[0]]
            return [tensor_list[i] for i in self.result_idx]

        return func

    def get_host_source(self):
        """Returns the source code of the host module."""
        if self.host_kernel_source is not None:
            return self.host_kernel_source
        return self.rt_mod.inspect_source()

    def get_device_source(self):
        """Returns the source code of the device module."""
        if self.device_kernel_source is not None:
            return self.device_kernel_source
        return self.rt_mod.imports[0].inspect_source()

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.get_device_source()
        else:
            return self.get_device_source() + "\n\n" + self.get_host_source()

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)



####################################################################################

import ctypes
from tilelang.jit.wrapper import TLWrapper
# from tilelang.jit.libgen import LibraryGenerator
from tilelang.utils.tensor import map_torch_type

try:
    from tilelang_cython_wrapper import CythonKernelWrapper
except ImportError:
    raise


def is_symbolic_expr(expr) -> bool:
    """Check if the expression is a symbolic expression.
    A symbolic expression can be a simple tvm.Var, or an tvm.PrimExpr containing tvm.Var.
    """
    return not isinstance(expr, tir.IntImm) and isinstance(expr, tir.PrimExpr)


class CythonKernelAdapter(BaseKernelAdapter):
    """Adapter class that converts TVM/TIR functions to callable CUDA kernels using cython.

    This adapter handles:
    1. Converting TIR functions to compiled CUDA libraries
    2. Managing dynamic shapes in tensor operations
    3. Wrapping C++ kernels for Python/PyTorch usage
    """

    # Class attributes to store compiled kernel information
    target: str | Target = "cuda"
    ir_module: tvm.IRModule | None = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None
    kernel_global_source: str | None = None  # Alias for device_kernel_source for compatibility
    lib: ctypes.CDLL | None = None  # Compiled library handle
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: dict[tir.Var, tuple[int, int]] | None = None
    # Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
    ptr_map: dict[int, str] | None = None
    # Maps buffer variables to their corresponding dtypes
    buffer_dtype_map: dict[tir.Var, tuple[int, torch.dtype]] | None = None
    # Maps buffer variables to their corresponding static shapes and strides,
    # e.g., {
    #     "A": [(0, 16), (1, 16)] -> represents A.shape/strides = (16, 16)
    # }
    static_shape_map: dict[tir.Var, tuple[int, list[tuple[int, int]]]] | None = None
    static_strides_map: dict[tir.Var, tuple[int, list[tuple[int, int]]]] | None = None
    # Contains contiguous buffers
    static_contiguous_list: list[tir.Var] | None = None
    # Maps buffer variables to their corresponding devices
    buffer_device_map: dict[tir.Var, tuple[int, torch.device]] | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        """Initialize the adapter with the given TIR function or module.

        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.device_kernel_source = device_kernel_source
        self.kernel_global_source = device_kernel_source  # Set alias for compatibility

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target.canon_target(determine_target(target))

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.ptr_map = self._process_ptr_map()
        self.buffer_device_map = self._process_buffer_device()

        static_buffer_infos = self._process_static_buffer_infos()
        self.static_shape_map = static_buffer_infos[0]
        self.static_strides_map = static_buffer_infos[1]
        self.static_contiguous_list = static_buffer_infos[2]

        self.verbose = verbose
        # self.wrapper = TLWrapper(self.target)
        # # self.lib_generator = LibraryGenerator(self.target, verbose=verbose)
        # # self.lib_generator.assign_pass_configs(pass_configs)
        # # self.lib_generator.assign_compile_flags(compile_flags)

        # self.wrapper.assign_optimized_module(self.ir_module)
        # self.wrapper.assign_pass_configs(pass_configs)
        # self.wrapper.assign_host_module(host_mod)
        # self.wrapper.assign_device_module(device_mod)
        # self.host_kernel_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))
        # print(self.host_kernel_source)
        
        # self.lib_generator.update_lib_code(self.host_kernel_source)
        # self.lib_generator.compile_lib()
        # self.lib = self.lib_generator.load_lib()

        # self.lib.get_last_error.restype = ctypes.c_char_p
        # result = self.lib.init()
        # if result != 0:
        #     error_msg = self.lib.get_last_error().decode("utf-8")
        #     error_msg += f"\n{self.lib_code}"
        #     raise RuntimeError(f"Initialization failed: {error_msg}")

        self.cython_wrapper = CythonKernelWrapper(self.result_idx, self.params, self.lib)
        self.cython_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
        self.cython_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
        self.cython_wrapper.set_static_shape_map(self.static_shape_map)
        self.cython_wrapper.set_static_strides_map(self.static_strides_map)
        self.cython_wrapper.set_static_contiguous_list(self.static_contiguous_list)
        self.cython_wrapper.set_buffer_device_map(self.buffer_device_map)
        self.cython_wrapper.set_ptr_map(self.ptr_map)
        self._post_init()
        print("hello cython: ", self.cython_wrapper.forward)


    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int, int, int]]:
        """Extract information about dynamic shapes from the TIR function.

        Maps symbolic variables to their corresponding (id, buffer_index, dimension, stride_scale)
        for runtime shape resolution.
        id represents shape or stride, 0 represents shape, 1 represents stride.
        stride_scale compensates for sub-byte dtypes (e.g. float4_e2m1fn) where torch strides
        are in storage units but the kernel expects logical element strides.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map) and (shape not in params):
                        dynamic_symbolic_map[shape] = (0, i, j, 1)
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                element_bits = buffer.dtype.bits * buffer.dtype.lanes
                stride_scale = 8 // element_bits if element_bits < 8 else 1
                for j, stride in enumerate(buffer.strides):
                    if isinstance(stride, tir.Var) and (stride not in dynamic_symbolic_map) and (stride not in params):
                        dynamic_symbolic_map[stride] = (1, i, j, stride_scale)
        return dynamic_symbolic_map

    def _process_buffer_dtype(self) -> dict[tir.Var, tuple[int, torch.dtype]]:
        """Extract information about buffer dtypes from the TIR function.

        Maps buffer variables to their corresponding dtypes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_dtype_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name, dtype = buffer.name, buffer.dtype
                buffer_dtype_map[name] = (i, map_torch_type(dtype))
        return buffer_dtype_map

    def _process_ptr_map(self) -> dict[int, str]:
        """Extract information about pointer arguments from the TIR function.

        Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
        """
        func = self.prim_func
        params = func.params
        ptr_map = {}
        for i, param in enumerate(params):
            if param.dtype == "handle":
                ptr_map[i] = param.name
        return ptr_map

    def _process_static_buffer_infos(
        self,
    ) -> tuple[dict[tir.Var, tuple[int, list[tuple[int, int]]]], dict[tir.Var, tuple[int, list[tuple[int, int]]]], list[tuple[tir.Var]]]:
        """Extract information about static shapes from the TIR function.

        Maps buffer variables to their corresponding static shapes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        static_shape_map = {}
        static_strides_map = {}
        static_contiguous_list = list()
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                static_shape, static_strides = [], []
                for j, s in enumerate(buffer.shape):
                    if isinstance(s, tir.IntImm):
                        static_shape.append((j, s.value))
                    elif is_symbolic_expr(s):
                        static_shape.append((j, -1))  # -1 for symbolic
                    else:
                        raise ValueError(f"Unsupported shape type: {type(s)}")
                for j, s in enumerate(buffer.strides):
                    if isinstance(s, tir.IntImm):
                        static_strides.append((j, s.value))
                is_contiguous, prod = True, 1
                for dim, stride in reversed(list(zip(buffer.shape, buffer.strides))):
                    is_contiguous &= bool(stride == prod)
                    prod *= dim
                static_shape_map[buffer.name] = (i, static_shape)
                static_strides_map[buffer.name] = (i, static_strides)
                if is_contiguous:
                    static_contiguous_list.append((i, buffer.name))
        return static_shape_map, static_strides_map, static_contiguous_list

    def _process_buffer_device(self) -> dict[tir.Var, tuple[int, torch.device]]:
        """Extract information about buffer devices from the TIR function.

        Maps buffer variables to their corresponding devices.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_device_map = {}
        device = None
        device = torch.device("cuda")
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name = buffer.name
                buffer_device_map[name] = (i, device)
        return buffer_device_map

    def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
        """Low-level function to call the compiled CUDA kernel.

        Converts PyTorch tensor pointers to C void pointers for ctypes interface.
        """
        ctypes_args = [ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def _convert_torch_func(self) -> Callable:
        """Returns a PyTorch-compatible function wrapper for the kernel."""

        def lambda_forward(*args, stream: int = -1, skip_tensor_validation: bool = False):
            """
            Args:
                args: List of input tensors
                stream: CUDA stream ID, default to -1, will use the current stream if not specified
                skip_tensor_validation: Whether to skip tensor attributes validation which
                includes shape, dtype, device, etc.
            """
            return self.cython_wrapper.forward([*args], stream=stream, skip_tensor_validation=skip_tensor_validation)

        return lambda_forward

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)

    @property
    def srcpath(self):
        """Returns the source path of the compiled library."""
        return self.lib_generator.srcpath

    @property
    def libpath(self):
        """Returns the path to the compiled library."""
        return self.lib_generator.libpath

    @property
    def lib_code(self):
        """Returns the code of the compiled library."""
        return self.lib_generator.lib_code

    @property
    def is_dynamic(self):
        """Indicates whether the kernel handles dynamic shapes."""
        return self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.device_kernel_source
        else:
            # Wrapper only has host kernel source
            assert self.host_kernel_source is not None, "Wrapped source is not available"
            return self.host_kernel_source

    def get_host_source(self):
        """Returns the source code of the host function."""
        return self.host_kernel_source