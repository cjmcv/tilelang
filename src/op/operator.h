/*!
 * \file tl/op/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_OP_H_
#define TVM_TL_OP_OP_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

#include "../layout/layout.h"

// <NT> operator 定义了如何将TileLang DSL中的高级操作（如 T.gemm、T.copy）转换为底层 TIR 的基础设施。
// 基本流程: T.copy(A,B)   -> ParseOperator -> CopyOpNode -> CopyOp.Lower -> tir.Call / cp.async
//          T.gemm(A,B,C) -> ParseOperator -> GemmOpNode -> GemmOp.Lower -> tir.Call / wgmma

namespace tvm {
namespace tl {

using namespace tir;

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
using LayoutMap = Map<Buffer, Layout>;
using BufferMap = Map<Var, Buffer>;

// <NT> InferLevel 布局推断的严格程度
//  kFree = 0,    // 自由推断，允许各种可能，可以尝试各种布局组合，如 row/col-major, swizzled 等
//                   对应 探索阶段，给优化器最大自由度
//  kCommon = 1,  // 常规推断，平衡灵活性和确定性，通常选择最常用的布局，如 倾向行主序，但也可接受其他
//                   对应 权衡阶段，平衡性能和确定性
//  kStrict = 2,  // 严格推断，要求完全确定，必须返回唯一确定的布局，不能有歧义，如 只能是行主序
//                   对应 最终阶段，必须确定唯一答案
// 不同的严格程度，是编译器在不同阶段做出不同决策的关键。
enum class InferLevel : uint8_t {
  kFree = 0,
  kCommon = 1,
  kStrict = 2,
};

/// Convert InferLevel enum to string for debugging
inline const char *InferLevelToString(InferLevel level) {
  switch (level) {
  case InferLevel::kFree:
    return "Free";
  case InferLevel::kCommon:
    return "Common";
  case InferLevel::kStrict:
    return "Strict";
  default:
    return "Unknown";
  }
}

// <NT> LowerArgs 降级上下文，从 TileOp 到 TIR 传递上下文
// target: 目标硬件（cuda/ascend等）
// Range thread_bounds: 线程范围
// Var thread_var: 线程变量
// AddWorkspace: 添加workspace的回调
// LayoutMap layout_map: 布局映射
// Map<Buffer, Buffer> buffer_remap: 一个buffer映射到另一个buffer，常用于 内存重用优化(如复用smem) / 双缓冲实现(流水线) / 视图变换
// Map<Var, PrimExpr> let_var_to_expr: 一个从 Let 语句变量到其绑定表达式的映射表。
// bool in_pipeline: 标记是否在流水线内
//
// * Range 的基本定义
//     class RangeNode : public Object {
//     public:
//       PrimExpr min;      // 起始值（通常是 0）
//       PrimExpr extent;   // 范围大小（线程数）
//       // 实际表示的区间是 [min, min + extent)
//     };
//     Range r = Range::FromMinExtent(0, 128);  // 使用示例，区间 [0, 128)
// * Var 是 TVM 中的变量节点
//     class VarNode : public Object {
//       String name_hint;  // 变量名提示（如 "threadIdx.x"）
//       DataType dtype;    // 数据类型（通常是 int32）
//     };
//     Var tx = Var("threadIdx.x", DataType::Int(32)); // 使用示例
//
// * Let 语句，在TVM中用于引入局部变量
//     let_var_to_expr 主要是为了通过 Let 间接访问 buffer。
//     @T.prim_func
//     def kernel(A: T.Buffer((128, 128))):
//       # 通过 Let 引入中间变量
//       let ptr = A.data  # A 的指针
//       let offset = compute_offset()  # 计算偏移
//       value = ptr[offset]  # 通过 ptr 和 offset 访问 buffer，实际上是在访问 A
//    此时，需要知道 ptr 和 offset 最终对应的是哪个 buffer。
//    所以会有: Map<Var, PrimExpr> let_var_to_expr = {
//               ptr: A.data,           // ptr 是 A 的指针
//               offset: 42,            // offset 是常量 42
//               temp: ptr + offset,    // temp 是表达式
//             };
// * in_pipeline 里true和false的区别:
//    1) 同步策略: 流水线内异步拷贝，由流水线机制处理数据同步问题；非流水线则使用同步拷贝，立即完成确保数据可用。
//    2) 内存分配策略: 流水线内多缓冲区(multi-stage)；非流水线单缓冲区; 
//    3) 寄存器分配: 流水线内更多寄存器保存中间状态

struct LowerArgs {
  Target target;
  Range thread_bounds;
  Var thread_var;
  AddWorkspaceCallback AddWorkspace;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
  // Map from LetStmt variable to its bound expression, for resolving
  // fragment buffer accesses through let bindings
  Map<Var, PrimExpr> let_var_to_expr;
  // Whether the current TileOp is nested inside a pipelined loop
  // (i.e. a surrounding loop annotated with num_stages > 0).
  bool in_pipeline = false;
};

// <NT>  LayoutInferArgs - 布局推断上下文
// arith::Analyzer analyzer: 算术分析器
// bool buffer_oob: （Buffer Out-Of-Bounds）标记buffer是否越界，一旦某个操作被发现有越界访问，这个信息会一直传递下去
//
// * Analyzer 的基本功能
//     class Analyzer {
//     public:
//       PrimExpr Simplify(const PrimExpr& expr);       // 1. 表达式简化: 如 简化索引，使更容易分析布局关系
//       void Bind(const Var& var, const Range& range); // 2. 约束求解: 如 绑定线程范围约束
//       bool CanProve(const PrimExpr& cond);           // 3. 检查条件: 如 检查内存是否连续， bool is_contiguous = CanProve(a_col == a_row + 1);  或 检查边界 甚至是 bank conflict 等
//       ModularSet mod(const PrimExpr& expr);       // 4. 模数分析
//       ConstIntBound bound(const PrimExpr& expr);  // 5. 范围分析
//     };
struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  arith::Analyzer *analyzer;
  bool buffer_oob = false;
  Map<Buffer, Buffer> buffer_remap;
  // Map from LetStmt variable to its bound expression, for resolving
  // fragment buffer accesses through let bindings
  Map<Var, PrimExpr> let_var_to_expr;
  // Whether the current TileOp is nested inside a pipelined loop
  // (i.e. a surrounding loop annotated with num_stages > 0).
  bool in_pipeline = false;
};

class TileOperator;

// <NT> TileOperator: 每个具体的算子（如 CopyOp、GemmOp）都会继承这个类，实现这三个核心方法。
// * Lower: 将算子降级为 TIR 语句
// * InferLayout: 推断算子的内存布局
// * Clone: 克隆算子（用于复制）
//
// TileOperatorNode (Object) 本质是数据节点，存储算子的实际数据和虚函数表
// TileOperator (ObjectRef) 本质是智能句柄，为用户可见的智能指针，通过引用计数句柄自动管理节点的生命周期
//              类名 TileOperator, 基类 ObjectRef, 指向的数据节点类型 TileOperatorNode
// 是 智能指针 + 抽象基类 的组合，用户只和 TileOperator 句柄交互，永远不会直接操作 TileOperatorNode。
// 同样，tvm的其他核心类也采用了这种方式，如: PrimFunc（句柄） vs PrimFuncNode（数据）
//                                         IRModule vs IRModuleNode
//                                         Pass vs PassNode
//                                         Target vs TargetNode
// 用户只需要: 通过工厂函数获取句柄，通过句柄调用方法，传递句柄给其他函数，必要时用 as<T>() 安全转换.
// 内部实现者: 继承 Node 类实现具体功能, 在工厂函数中创建 Node 并返回句柄, 通过虚函数提供多态行为.
class TileOperatorNode : public Object {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const = 0;

  virtual LayoutMap InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const = 0;

  virtual TileOperator Clone() const = 0;

  TVM_FFI_DECLARE_OBJECT_INFO("tl.TileOperator", TileOperatorNode, Object);
};

class TileOperator : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TileOperator, ObjectRef,
                                             TileOperatorNode);
};

// <NT> GetVarFromAccessPtr 从内存访问指针access_ptr表达式中提取出底层的buffer变量。
// 在 TVM/TileLang 的 IR 中，内存访问通常被包装在特定的 access_ptr 调用中，而实际操作的 buffer 变量被藏在参数里。这个函数就是解包这个调用的。
// 里面包装了 tvm_access_ptr：TVM 早期的标准方式，直接暴露 buffer_var
//           tl.access_ptr：TileLang 优化的方式，通过 BufferLoad 间接引用
// 并对外提供了统一接口, 在代码生成/内存分析/别名检测/优化 Passes中都会经常用到。
Var GetVarFromAccessPtr(const PrimExpr &expr);

// <NT> ParseOperator 从 TIR 解析回 TileOperator 的反向映射。
// 正向过程: TileLang DSL -> ParseOperator -> TileOperator -> lower -> TIR (-> <<ParseOperator>> -> TileOperator)
// 这两个函数处于上面正向过程的后面括号部分，一般过程中不需要用到，通常充当 编译器开发者 的内部工具使用。
// 用于: 编译器内部优化 / 调试和分析 / 跨阶段信息传递。
// 比如：__global__ void kernel(float* a, float* b) {
//        *a = *b;  // ← 算子开发者 只关心这个
//      }
//      // 编译器内部：
//      // - 解析 AST
//      // - 生成 LLVM IR
//      // - 优化 IR
//      // - 生成 PTX
//      IR* ir = ParseAST(ast);  // ← 如果在开发 NVCC 编译器，则需要这种内部函数
//      OptimizeIR(ir);
//
// Call: 是算子的唯一载体，所有算子最终都表现为 Call 节点，所以从 Call 解析是最直接的路径。
// Expr（Expression）- 表达式：是计算值，会返回一个结果。Call 是表达式的一种。
// Stmt(Statement) 语句：语句是执行动作，不产生值。算子的独立执行单元，是 Call 的容器, Call 作为表达式，必须被包裹在stmt中才能独立存在
// 如: T.copy(A, B, 16) # DSL
//     # TIR
//     Evaluate(                # ← Stmt
//       Call(                  # ← Call
//        op=Op("tl.tileop.copy"),
//         args=[A, B, 16]
//       )
//     )
//     op = ParseOperator(stmt)  # 通过 Stmt 解析
//     op = ParseOperator(call)   # 或直接通过 Call 解析

TileOperator ParseOperator(Call call);
TileOperator ParseOperator(Stmt stmt);

// <NT> 算子注册，将 C++ 算子类与 TVM 的算子注册机制绑定在一起。
// OpBuilderFunc: 作为工厂函数，从参数构建具体的算子实例
//   * 输入：Array<PrimExpr>（参数列表）和 Map<String, ObjectRef>（注解）
//   * 输出：TileOperator（算子对象）
//   ps: "注解" annotations 是一种编译期的元数据传递机制，能用于指导编译器 / 指定硬件特性 / 优化内存访问 / 控制流水线等。
//      使用 Map<String, ObjectRef> 表示，其中ObjectRef 是 TVM 中所有对象的基类，可以表示任何类型。String使人类可读，且序列化友好。
//      比如  # 算子注解
//         @T.prim_func
//         def gemm_kernel(A, B, C):
//             # 函数属性 attr 
//             T.func_attr({
//                "global_symbol": "gemm",
//                 "target": T.target("cuda"),
//             })
//             # 算子注解 annotations
//             T.gemm(
//               A_shared, B_shared, C,
//               annotations={
//                 "layout_a": "row_major", # 布局信息
//                 "layout_b": "col_major",
//                 "accumulator_dtype": "float32", # 精度控制
//                 "input_dtype": "float16",
//                 "arch": "hopper", # 硬件选择
//                 "use_wgmma": True,
//                 "prefetch": True, # 性能提示
//                 "num_stages": 3,
//               }
//             )
//      注解 annotations 需要和 函数属性 func_attr 区分开，annotations用于单个算子或表达式，func_attr 用于整个函数或模块。
//      下面还有个Op注册的属性 (set_attr)，适用于所有该类型算子，在程序初始化时设置。
//
//       层次	                  示例	                      作用域	          设置时机	     访问方式
//  ------------------------------------------------------------------------------------------------------
//  -- 函数属性	  T.func_attr({"global_symbol": "gemm"})	 整个PrimFunc	     函数定义时	  func->GetAttr()
//  -- 算子注解	  T.gemm(..., annotations={...})	         单个算子实例	      算子调用时	  op->annotations
//  -- Op 属性	  .set_attr<...>("TLOpBuilder", ...)	    所有该类型算子	    程序初始化	Op::Get()->GetAttr()

//      
using OpBuilderFunc =
    ffi::TypedFunction<TileOperator(Array<PrimExpr>, Map<String, ObjectRef>)>;

#define TIR_REGISTER_TL_TILE_OP(Entry, OpName)                                 \
  const Op &Entry::Get() {                                                     \
    static const Op &op = Op::Get("tl.tileop." #OpName);                       \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl.tileop." #OpName)                                        \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)             \
      .set_attr<OpBuilderFunc>(                                                \
          "TLOpBuilder",                                                       \
          [](Array<PrimExpr> args, Map<String, ObjectRef> annotations) {       \
            return Entry(args, annotations);                                   \
          })

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_OP_H_
