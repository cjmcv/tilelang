# Makefile - 直接编译 silu_mul_tl_32_9728

# CUDA 路径
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# 项目根目录
PROJECT_ROOT = ../../../../
TL_TEMPLATES = $(PROJECT_ROOT)/src/tl_templates
TL_TEMPLATE_CUDA = $(TL_TEMPLATES)/cuda
TL_TEMPLATE_CUDA_INST = $(TL_TEMPLATES)/cuda/instruction
CUTLASS = $(PROJECT_ROOT)/3rdparty/cutlass/include
CUTE = $(PROJECT_ROOT)/3rdparty/cute
TVM_CUTLASS = $(PROJECT_ROOT)/3rdparty/tvm/3rdparty/libflash_attn/cutlass/include
TVM_CUTE = $(PROJECT_ROOT)/3rdparty/tvm/3rdparty/libflash_attn/cute

# 编译选项
ARCH = -arch=sm_90a
NVCC_FLAGS = $(ARCH) -Xcompiler=-fPIC -lineinfo --expt-relaxed-constexpr --use_fast_math -std=c++17

# Include 路径
INCLUDES = -I$(PROJECT_ROOT)/src \
           -I$(TL_TEMPLATE_CUDA) \
           -I$(TL_TEMPLATE_CUDA_INST) \
           -I$(CUTLASS) \
           -I$(CUTE) \
           -I$(TVM_CUTLASS) \
           -I$(TVM_CUTE)

# 目标
TARGET = silu_mul_host
SRCS = host.cu kernel.cu

# 库
LIBS = -lcuda -lcudart

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRCS) -o $@ $(LIBS)

clean:
	rm -f $(TARGET)