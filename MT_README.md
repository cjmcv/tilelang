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
rm -rf build && mkdir -p build/tvm

ln -s $PWD/3rdparty/tvm/build/libtvm_runtime.so  build/tvm/
ln -s $PWD/3rdparty/tvm/build/libtvm.so          build/tvm/
ln -s $PWD/3rdparty/tvm/build/lib/libtvm_ffi.so  build/tvm/

cmake -B build -G Ninja && cmake --build build --parallel 8

# 3. 使用
export PYTHONPATH=/home/cjmcv/project/megakernel/:$PYTHONPATH
pushd demo && python microkernels/example_gemm.py && popd

# 4. megakernel 编译
python megakernel_setup.py build_ext --inplace
export MEGAKERNEL_HOME=/home/cjmcv/project/megakernel/
pushd demo && python single_silu_mul.py && popd

# 清submodule
git submodule deinit -f 3rdparty/tvm/
git rm -f 3rdparty/tvm/
rm -rf .git/modules/3rdparty/tvm/

git submodule add https://github.com/apache/tvm.git 3rdparty/tvm

# 备注
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
-> 改回使用tvm的，tvm的只是没有_ge, _lt等扩展，功能一致
@tvm.testing.requires_cuda
@tvm.testing.requires_cuda_compute_version(9, 0)

# TODO
1. 将项目转到mpk里进行编译；
2. 修改cuda code gen，转化到所需格式；


# 使用官方tvm，未通过
/home/cjmcv/project/tilelang/src/transform/inject_assumes.cc:86:48: error: ‘tilelang_assume’ is not a member of ‘tvm::tir::attr’
   86 |         body = AttrStmt(simplified, tir::attr::tilelang_assume,

/home/cjmcv/project/tilelang/src/transform/layout_inference.cc:504:41: error: ‘class tvm::arith::Analyzer’ has no member named ‘Clone’
  504 |       analyzer_vec_.push_back(analyzer_.Clone());


/home/cjmcv/project/tilelang/src/target/rt_mod_cuda.cc:24:54: error: ‘kDLGridConstant’ is not a member of ‘tvm::runtime’
   24 |           info.arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
-> grid_constant

OSError: /home/cjmcv/project/tilelang/build/lib/libtilelang_module.so: undefined symbol: _ZN3tvm3tir24DetectBufferVarAccessLCAERKNS0_8PrimFuncE
class BufferAllocationLocator : public StmtExprMutator { ：在tvm也有一份，需要知道更改的目的

ValueError: Invalid object type: <class 'tvm.tir.expr.Var'>