#!/bin/bash
# build.sh - 编译 TMA 1D Copy 示例

set -e

# 编译参数
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
NVCC="${CUDA_PATH}/bin/nvcc"

# 源文件和输出
SRC_FILE="tma_copy_1d.cu"
OUTPUT="tma_copy_1d"

# 编译选项
# -arch=sm_90a: Hopper 架构
# -O3: 优化
# -Xptxas: 额外 PTXAS 选项
# -cudart: CUDA runtime
NVCC_FLAGS="-arch=sm_90a -O3 --expt-relaxed-constexpr"
PTXAS_FLAGS="-Xptxas=-dlcm=ca"
EXTRA_FLAGS="-cudart shared"

echo "=== Building TMA 1D Copy Example ==="
echo "Source: $SRC_FILE"
echo "Output: $OUTPUT"
echo "CUDA Path: $CUDA_PATH"
echo ""

# 编译
echo "Compiling..."
$NVCC $NVCC_FLAGS $PTXAS_FLAGS $EXTRA_FLAGS $SRC_FILE -o $OUTPUT

echo ""
echo "Build successful: $OUTPUT"
echo ""
echo "Run with: ./$OUTPUT"