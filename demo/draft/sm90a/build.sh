#!/bin/bash
# naive_gemm/gemm_multistage.cu
# streamk_gemm/gemm_streamk.cu
# gemm_simple.cu
# -L../build/lib -ltilelang_module -ltilelang \
# -L../build/tvm -ltvm_ffi \

rm a.out
nvcc -arch=sm_90 \
     --no-jit-fallback \
     -std=c++20 \
     -I../../../3rdparty/cutlass/include \
     -I../../../src/ \
     main.cu \
     -lcuda

echo "Compile Done!"

./a.out