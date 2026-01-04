# 限制编译cpu数量 并 一键安装
export CMAKE_BUILD_PARALLEL_LEVEL=4 
pip install . -v -e

# 装完依赖库后的代替方案
mkdir build && cmake -B build -G Ninja && cmake --build build --parallel 4
export PYTHONPATH=/home/cjmcv/project/tilelang:$PYTHONPATH

# 1. 先编译tvm
cd 3rdparty/tvm && rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .

echo "set(CMAKE_BUILD_TYPE Release)" >> config.cmake
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(BUILD_SHARED_LIBS ON)" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
echo "set(CMAKE_CUDA_ARCHITECTURES 89)" >> config.cmake
echo "set(USE_CUDA   ON)" >> config.cmake

cmake .. && make -j6

# 2. 添加软连接并编译tilelang
mkdir -p build/tvm

ln -s $PWD/3rdparty/tvm/build/libtvm_runtime.so  build/tvm/
ln -s $PWD/3rdparty/tvm/build/libtvm.so          build/tvm/
ln -s $PWD/3rdparty/tvm/build/lib/libtvm_ffi.so  build/tvm/

cmake -B build -G Ninja && cmake --build build --parallel 6