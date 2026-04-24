#!/bin/bash

cd "$(dirname "$0")"

rm -f a.out

nvcc -gencode arch=compute_90,code=sm_90 \
     -std=c++20 \
     -I../../../3rdparty/cutlass/include \
     -I../../../src/ \
     main.cu \
     -lcuda

echo "Compile Done!"
./a.out