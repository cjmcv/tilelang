# 限制编译cpu数量 并 一键安装
export CMAKE_BUILD_PARALLEL_LEVEL=4 
pip install . -v -e

# 装完依赖库后的代替方案
mkdir build && cmake -B build -G Ninja && cmake --build build --parallel 4
export PYTHONPATH=/home/cjmcv/project/tilelang:$PYTHONPATH

################################
# 1. 先编译tvm: https://tvm.apache.org/docs/install/from_source.html#step-2-get-source-from-github
conda install -c conda-forge llvmdev=15.0.7  # llvm-config --version
apt install -y libzstd-dev libxml2-dev

cd 3rdparty/tvm && rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .

echo "set(CMAKE_BUILD_TYPE Release)" >> config.cmake
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(BUILD_SHARED_LIBS ON)" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
echo "set(CMAKE_CUDA_ARCHITECTURES 89)" >> config.cmake  # 90
echo "set(USE_CUDA   ON)" >> config.cmake

cmake .. && make -j6

cd ../3rdparty/tvm-ffi &&  pip install .

# 2. 回到主目录，添加软连接并编译tilelang
cd ../../../../
rm -rf build && mkdir -p build/tvm

ln -s $PWD/3rdparty/tvm/build/libtvm_runtime.so  build/tvm/
ln -s $PWD/3rdparty/tvm/build/libtvm.so          build/tvm/
ln -s $PWD/3rdparty/tvm/build/lib/libtvm_ffi.so  build/tvm/

cmake -B build -G Ninja && cmake --build build --parallel 8

# 3. megakernel 编译
python megakernel_setup.py build_ext --inplace

# 4. 使用tilelang
export MEGAKERNEL_HOME=/home/cjmcv/project/megakernel && export PYTHONPATH=$MEGAKERNEL_HOME:$PYTHONPATH
export MEGAKERNEL_HOME=/data/team/cjm/mg && export PYTHONPATH=$MEGAKERNEL_HOME:$PYTHONPATH

pushd demo && python micro_test.py && popd
pushd demo && python fused_mlp.py && popd

# 清submodule
git submodule deinit -f 3rdparty/tvm/
git rm -f 3rdparty/tvm/
rm -rf .git/modules/3rdparty/tvm/
git submodule add https://github.com/apache/tvm.git 3rdparty/tvm

# TODO
1. 基于single_linear尝试检索所有micro kernel配置，尝试找到超越torch的方案，并分析tilelang的自己launch和集成后的耗时是否有一定规律？
   从 single_linear.py 修改，调用 MicroAutoGen(1, 1024, 3072)，MicroAutoGen扩展通过id号选定配置
2. 考虑tilelang端只生成代码而不编译，看能否减少耗时；
3. 考虑新增megakernel的并行编译；
4. L40并行编译崩溃问题；
5. gemv对比性能
6. 分析：block派发逻辑是否固定，还是属于抢占式派发，每次都不同。
7. 分析：gemm1的4block -> silu_mul的2block，02->0, 13->1，能否只写回gemm1的后两个block 23，前两个block 01保留在smem，延递silu_mul上。
   尝试: 依托block的固定smem，通过多传入偏移量，实现跨task共享。
8. 排查block数量不能超过sm数量的本质原因。

# 备注
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
-> 改回使用tvm的，tvm的只是没有_ge, _lt等扩展，功能一致
@tvm.testing.requires_cuda
@tvm.testing.requires_cuda_compute_version(9, 0)




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