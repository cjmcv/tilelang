#!/bin/bash

rm -f test_tma a.out
nvcc -gencode arch=compute_90a,code=sm_90a \
     -I../../../../3rdparty/cutlass/include \
     -I../../../../src/ \
     test_tma.cu \
     -lcuda

echo "Compile Done!"

./test_tma