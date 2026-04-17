import os
from megakernel.core import PyAllReduce
from megakernel.cuda_wrapper import CudaRTLibrary

if __name__ == "__main__":
    size_in_bytes = 1024
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    print(pointer, handle)
    
    ar = PyAllReduce(123)
    print(ar)