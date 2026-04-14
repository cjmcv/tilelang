# CMakeLists.txt - 编译 silu_mul_tl_32_9728 kernel

cmake_minimum_required(VERSION 3.18)
project(silu_mul_tl_32_9728 LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# 包含 TileLang 模板路径
include_directories(
    ${CMAKE_SOURCE_DIR}/../../../src
    ${CMAKE_SOURCE_DIR}/../../../src/tl_templates/cuda
    ${CMAKE_SOURCE_DIR}/../../../src/tl_templates/cuda/instruction
    ${CMAKE_SOURCE_DIR}/../../../3rdparty/cutlass/include
    ${CMAKE_SOURCE_DIR}/../../../3rdparty/cute
    ${CMAKE_SOURCE_DIR}/../../../3rdparty/tvm/3rdparty/libflash_attn/cutlass/include
    ${CMAKE_SOURCE_DIR}/../../../3rdparty/tvm/3rdparty/libflash_attn/cute
)

# 启用 CUDA 着色器/PTX 编译
enable_language(CUDA)

# 编译选项
set(CMAKE_CUDA_ARCHITECTURES 90)  # Hopper (sm_90a)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 编译选项
set(CUDA_FLAGS
    -Xcompiler=-fPIC
    -lineinfo
    --expt-relaxed-constexpr
    --use_fast_math
)

# 如果使用 NVRTC，需要单独处理
# 这里假设 kernel 已经通过 TileLang/TVM 编译好了
if(NOT DEFINED KERNEL_CUBIN)
    message(WARNING "KERNEL_CUBIN not defined, assuming kernel is compiled at runtime")
endif()

# Host 端可执行文件
add_executable(silu_mul_host
    host.cu
)

# 链接 CUDA 库
target_link_libraries(silu_mul_host
    CUDA::cuda_driver
    CUDA::cudart
)

# 设置 include 目录
target_include_directories(silu_mul_host PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# 设置输出目录
set_target_properties(silu_mul_host PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 打印配置信息
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "TileLang templates: ${CMAKE_SOURCE_DIR}/src/tl_templates")