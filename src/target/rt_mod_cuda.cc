#include "../transform/common/attr.h"
#include "codegen_cuda.h"
#include "runtime/cuda/cuda_module.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

// <NT> TVM中的 PrimFunc 代表一个底层、命令式的张量函数, 所有高级 DSL 最终都降级为 PrimFunc，所有 TIR Passes 都以 PrimFunc 为操作对象
//    * 基本定义: class PrimFuncNode : public BaseFuncNode {
//                 public:
//                  Array<tir::Var> params;           // 函数参数列表
//                  tir::Stmt body;                    // 函数体（语句）
//                  Type ret_type;                     // 返回类型
//                  Map<tir::Var, tir::Buffer> buffer_map;  // 参数到缓冲区的映射
//                  Map<String, ObjectRef> attrs;       // 函数属性
//                  Array<PrimExpr> precondition;       // 执行前提条件
//                  static constexpr const char* _type_key = "tir.PrimFunc";
//                };
//    * 参数: @T.prim_func
//            def example(
//              A: T.Buffer((128, 128), "float32"),  # ← 参数
//              B: T.Buffer((128, 128), "float32"),  # ← 参数
//            ):
//              ...
//          在 PrimFunc 内部: params = [Var("A"), Var("B")]  # 参数列表
//   * 缓冲区映射 (buffer_map):
//         buffer_map = {
//           Var("A"): Buffer(name="A", shape=[128, 128], dtype="float32"),
//           Var("B"): Buffer(name="B", shape=[128, 128], dtype="float32"),
//         }
//   * 函数体 (body): 是一个 Stmt 节点，包含所有语句: 
//             ├── body: Stmt                    # 函数体
//             │   ├── ForNode                   # 循环
//             │   ├── IfThenElseNode            # 条件
//             │   ├── LetStmtNode               # 变量绑定
//             │   ├── AllocateNode              # 内存分配
//             │   ├── AttrStmtNode              # 属性
//             │   ├── BufferStoreNode           # 存储
//             │   └── EvaluateNode              # 求值
//   * 属性: @T.prim_func
//           def example(A, B, C):
//             T.func_attr({
//             "global_symbol": "my_kernel",     # 函数符号名
//             "tir.noalias": True,               # 无别名指针
//             "target": T.target("cuda"),        # 目标平台
//             "cuda.is_kernel": True,            # 是否是 CUDA 内核
//           })
//             ...

// <NT> TVM中的 IRModule 是贯穿整个编译流程的核心数据结构，本质上就是一个函数字典。
//   在 TileLang 中，IRModule 主要包含 PrimFunc，是连接 DSL、优化 Passes 和代码生成的桥梁，贯穿全程。
//   可包含多个函数（PrimFunc/RelayFunc等），是 Pass 操作的基本单位，所有前端最终都生成 IRModule。
//   * 基本定义: class IRModuleNode : public Object {
//               public:
//                 Map<GlobalVar, BaseFunc> functions;            // 函数映射表（关键）
//                 Map<GlobalTypeVar, TypeData> type_definitions; // 类型定义 (用于 Relay)
//                 Map<String, ObjectRef> attrs;                  // 模块属性
//                 Map<GlobalVar, String> source_map;             // 源映射 (调试信息)
//                 static constexpr const char* _type_key = "IRModule";
//               };

// <NT> TVM与tilelang的关系
//  TVM:      Python DSL (Relax DSL) → Relax IR → [Relax passes] → TIR → [TIR passes] → CodeGen 
//            Python DSL (Relax DSL) → Relax IR → [Relax passes] → TE（Tensor Expression，声明式算子，传统方式) -> TIR（自动生成） → [TIR passes] → CodeGen 
//                                                               → 如复用 TOPI 算子库、手动定制算子调度，需要用TE.
//            Python DSL (Relax DSL) → Relax IR → [Relax passes] → tir.script（命令式手写) -> TIR（直接产出） → [TIR passes] → CodeGen 
//  TileLang: Python DSL (TileLang DSL) → TIR (直接) → [TIR passes] → CodeGen
//
//  * 在TIR和CodeGen之间，有tuning阶段，用到 AutoTVM(基于模板调优)/Ansor（AutoScheduler，无模板调优）
//  * TVM也可以绕过Relax DSL，直接使用 TE 或 tir.script 来实现算子。
// 
//  TileLang也不需要经过Relax IR / Relay IR, Relay/Relax用于计算图与算子融合, 属于“模型编译器”范畴。
// 而tilelang定位是“内核编译器”，如需融合，直接在DSL里手动进行。而tvm里的tir.script也是命令式手写，跟 TileLang DSL是完全对应的。
//  所以简化链路得到： TVM: tir.script -> Parser -> TIR → [TIR passes] → CodeGen 
//                      TileLang DSL -> Parser -> TIR → [TIR passes] → CodeGen 
//  TileLang在三个层面上对tvm进行了扩展:
//  1. DSL层 TIR Script 的语法增强。（见下面的TileLang DSL 的抽象层级）
//  2. Pass层的专用优化策略，见transform层。
//  3. CodeGen层 深度集成tl_templates，提供高度优化的基础组件，大部分内容来自cutlass/cute，使更容易获得极致性能。
//     这也是推动国产化的关键设计环节，芯片厂家围绕这个线路可以将自己的核心优化代码单元封装在这里，由tilelang进行拼凑。
//
//  <NT> TileLang DSL 的抽象层级与TVM的完全不同，主要有以下特点:
// -- 硬件意识：直接暴露硬件特性（TMA/WGMMA/TMEM）如 T.wgmma_ss / T.tcgen05_mma_ss
// -- 显式控制：内存、线程、同步全部显式表达，如 T.alloc_shared
// -- 内核导向：不是描述计算，而是编写内核
// -- 可预测性：代码和生成的指令有一一对应关系
// -- 熟悉感：对 CUDA 开发者友好
//  好处是比TVM DSL更能贴近底层，就更容易拿到高性能。比如TVM的TE，主要是描述“做什么”，由编译器决定“怎么做”。
// 而tilelang DSL可以直接描述“怎么做”，如T.tma_load直接指定用tma，T.wgmma_ss指定用wgmma，T.wgmma_ss精准控制同步等。
// 可以实现 新特性集成快，资源控制精准，容易调试无黑盒。再配合cutlass/cute的代码快，容易拿到峰值性能。
// 但也需要使用者有相关技术基础，如无cuda基础，使用tilelang也难以写出好的cuda kernel。
//
// <NT> TileLang的DSL具体是怎么工作的
// 1. 用户编写DSL，并使用@T.prim_func修饰 + T.原语
// 2. 前端解析生成 PrimFunc / IRModule
// 3. 基于tvm的基础设施优化流水线。
//    阶段1: LowerAndLegalize - 内存作用域推断 - 高级原语降级 (如 T.gemm) - 布局推断 (Layout Inference)
//    阶段2: OptimizeForTarget- 软件流水线 (T.Pipelined) - 并行循环 (T.Parallel) 线程映射 - 硬件特定优化 (如注入TMA指令)
// 4. CodeGenTileLangCUDA 生成cuda代码
// 所以 PrimFunc / IRModule 就是贯穿整体的核心单元。
// 

namespace tvm {
namespace codegen {

static std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          info.arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      info.arg_types.push_back(dtype);
    }
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      info.launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      info.launch_param_tags.push_back(
          runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<ffi::Array<Integer>>("cluster_dims").defined()) {
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tir::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          info.launch_param_tags.push_back(tag);
        }
      }
    }
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    fmap[static_cast<std::string>(global_symbol.value())] = info;
  }
  return fmap;
}

// <NT> BuildTileLangCUDA 的基本时间线
// 1. 创建代码生成器
// CodeGenTileLangCUDA cg;
// 2. 添加多个函数
// cg.AddFunction(gvar1, func1);  // ← 这里遍历 func1 的 body
// cg.AddFunction(gvar2, func2);  // ← 这里遍历 func2 的 body
// ...
// 3. 最后调用 Finish
// std::string final_code = cg.Finish();  // ← 收集所有生成的代码

ffi::Module BuildTileLangCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_compile")) {
    // Fetch current pass context config and pass into the compile callback
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    ptx = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "cubin";
  } else {
    ICHECK(0);
  }
  return runtime::CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

ffi::Module BuildTileLangCUDAWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return runtime::CUDAModuleCreate("ptx", "ptx", ExtractFuncInfo(mod), code);
}

// <NT> 用于注册FFI(Foreign Function Interface)函数的静态初始化块
// TVM_FFI_STATIC_INIT_BLOCK是一个宏，用于定义一个静态初始化块，在程序启动时会自动执行其中的代码。
// 在这个块中注册了两个FFI函数，分别是"target.build.tilelang_cuda"和"target.build.tilelang_cuda_without_compile"，
// 它们分别对应BuildTileLangCUDA和BuildTileLangCUDAWithoutCompile两个C++函数。
// 如：
//   # 在Python中可以通过FFI调用这些注册的函数
//   fcompile = tvm.get_global_func("target.build.tilelang_cuda")
//   cuda_kernel = fcompile(some_tir_module)  # 返回编译好的CUDA内核
//   # 或者只生成代码不编译
//   fcgen = tvm.get_global_func("target.build.tilelang_cuda_without_compile")
//   cuda_code = fcgen(some_tir_module)
// tilelang里这两个ffi函数会在tilelang/engine/lower.py里调用:
//    global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda") + "_without_compile"
// 
// <NT> TVM FFI函数的统一签名: using FType = void(TVMArgs args, TVMRetValue* rv);
// 即所有通过tvm.get_global_func()获取的函数，在调用时都会：接收一个TVMArgs对象(包含所有传入的参数);返回一个TVMRetValue对象（包含返回值）
// 如 fcompile = tvm.get_global_func("target.build.tilelang_cuda")
//    result = fcompile(mod, target) # mod和target会被封装成TVMArgs传入C++函数BuildTileLangCUDA中，返回的ffi::Module对象会被封装成TVMRetValue返回给Python端。
// 所以对应BuildTileLangCUDA的c++函数签名是 ffi::Module BuildTileLangCUDA(IRModule mod, Target target)

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile",
           BuildTileLangCUDAWithoutCompile);
}

} // namespace codegen
} // namespace tvm
