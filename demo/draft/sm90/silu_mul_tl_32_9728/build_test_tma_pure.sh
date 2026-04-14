#!/bin/bash

rm -f test_tma_pure a.out
nvcc -gencode arch=compute_90a,code=sm_90a \
     test_tma_pure.cu \
     -lcuda

echo "Compile Done!"

./test_tma_pure