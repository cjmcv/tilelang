# 限制编译cpu数量
export CMAKE_BUILD_PARALLEL_LEVEL=4 

# 一键安装
pip install . -v -e

# 装完依赖库后的代替方案
mkdir build && cmake -B build -G Ninja && cmake --build build --parallel 4
export PYTHONPATH=/home/cjmcv/project/tilelang:$PYTHONPATH