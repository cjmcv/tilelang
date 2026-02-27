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
export MEGAKERNEL_HOME=/data/team/cjm/mg89 && export PYTHONPATH=$MEGAKERNEL_HOME:$PYTHONPATH

pushd demo && python micro_test.py && popd
pushd demo && python fused_mlp.py && popd

# 指令
nsys profile --trace=cuda,nvtx --output=my_nsys
ncu --set full --section "SpeedOfLight_RooflineChart" -k "persistent_kernel" -o my_profile python...
"kernel"
compute-sanitizer --tool memcheck python demo/single_mega.py --nc
compute-sanitizer --tool memcheck --shared-memory-check yes ./your_cuda_program

# 清submodule
git submodule deinit -f 3rdparty/tvm/
git rm -f 3rdparty/tvm/
rm -rf .git/modules/3rdparty/tvm/
git submodule add https://github.com/apache/tvm.git 3rdparty/tvm

# TODO
0. 实现fused_attn.py: mpk(batch, step), 添加step参数用于控制推理步数。gqa kernel动态选择时的布局兼容问题，如何处理？
1. 可以先完成固定step的推理集成。
1. block_dim: PersistentKernel中“TBGraph(CyTBGraph(grid_dim, block_dim”中的block_dim疑似没用，仅仅用于kernel模板参数的确认与校验。是否真的需要校验？
2. 考虑tilelang端只生成代码而不编译，看能否减少耗时；
3. 考虑新增megakernel的并行编译；
5. gemv对比性能
7. 分析：gemm1的4block -> silu_mul的2block，02->0, 13->1，能否只写回gemm1的后两个block 23，前两个block 01保留在smem，延递silu_mul上。
   尝试: 依托block的固定smem，通过多传入偏移量，实现跨task共享。


8. 排查block数量不能超过sm数量的本质原因。（因为kernel限制一个sm仅持有一个block，当worker超过sm数量时，worker将会占据所有gpu资源，scheduler因缺少资源难以被启动，导致worker也接不到任务卡住）
2. 跨步同步，不必每个task都读和写一次gmem。task添加标记，event的首task启动，后面连续多个不需要等待，计算完一次写。（大显卡不需要处理这个问题？）
4. L40并行编译崩溃问题；（已解决， tilelang的ffi注册只能在单线程下进行，进入多线程前，应掉调用一下tilelang，提前触发其注册）
6. 分析：block派发逻辑是否固定，还是属于抢占式派发，每次都不同。（主推静态）

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