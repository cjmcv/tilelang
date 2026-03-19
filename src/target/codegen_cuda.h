/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
#define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "target/source/codegen_c.h"

// <NT> CodeGenTileLangCUDA，继承自tvm的CodeGenC，负责将高级的TIR操作转换为具体的CUDA内核代码。
// 分 behavior 和 visitor 两部分:
//    behavior: 一些特定的代码生成行为，如如何打印函数前缀、如何处理存储同步等；
//    visitor: 对TIR中的不同节点类型进行访问和处理，如ForNode、CallNode、BufferLoadNode等。
// 生成代码最终形态会是一个由 TVM提供的基础代码 搭配 tl基于cutlass/cute封装的tl_templates 组成的一个完整cuda kernel。
// 
// 问：为什么不采用继承CodeGenCUDA？
// -- 因为CodeGenCUDA是为较低级别的TIR设计的，使用TVM内置的运行时和内存管理，直接生成标准的、通用的CUDA代码。继承可能会带来不必要的复杂性和限制。
//   CodeGenTileLangCUDA需要处理更高级别的TIR操作，并且需要与tl_templates紧密集成，这些都是CodeGenCUDA没有考虑到的。
//   此外，CodeGenTileLangCUDA还需要支持一些特定于TileLang的功能，如访问ptr元数据、处理特殊的内存同步等，这些都可能与CodeGenCUDA的设计不兼容。
// * CodeGenCUDA生成的标准模式
// __global__ void kernel(float* __restrict__ A, float* __restrict__ B) {
//   __shared__ float smem[1024];
//   // 标准的CUDA代码
// }
// * CodeGenTileLangCUDA生成的优化模式，直接站在TVM和cutlass/cute两个巨人的肩膀上，完美补充了tvm生成cuda代码的短板。
// #include <tl_templates/cuda/gemm.h>
// #include <tl_templates/cuda/copy.h>
// __global__ __launch_bounds__(128, 1) void kernel(float* A, float* B) {
//   tl::warpgroup_commit_batch();
//   tl::tma_load(desc, ...);
//   // 使用TileLang专用模板
// }
//
// <NT> TVM TIR 中的节点类型
// 一 表达式节点 (Expr Nodes)
//   1. RampNode - 向量步进表达式,生成一个等差数列，用于向量化内存访问和计算。
//    TIR表示：ramp(base, stride, lanes) => ramp(0, 1, 4)       // 生成 [0, 1, 2, 3]
//    TileLang中的使用场景: A[ramp(0, 1, 4)] = B[ramp(0, 1, 4)]  // 向量化内存访问
//   2. BroadcastNode - 标量广播, 将标量值复制到向量的所有通道。
//    TIR表示：broadcast(value, lanes) => broadcast(3.14, 4)  // 生成 [3.14, 3.14, 3.14, 3.14]
//    TileLang中的使用: float4 val = broadcast(0.0f, 4)       // 初始化向量寄存器
//   3. FloatImmNode - 浮点立即数, 表示浮点类型的常量值, 如 3.14159f, 1.0e-10
//   4. CallNode - 函数调用, 表示各种内置函数和外部调用, 如TileLang中的各种intrinsic
//    Call(DataType::Handle(), tl::tma_load(), {desc, smem_ptr, ...})      // TMA加载
//    Call(DataType::Handle(), tl::wgmma_ss(), {desc_a, desc_b, acc, ...}) // WGMMA计算
//    Call(DataType::Int(32), tl::get_lane_idx(), {})                      // 获取线程束内索引
//    Call(DataType::Handle(), builtin::ptx_cp_async(), {dst, src, size})  // 异步拷贝
//   5. CastNode - 类型转换
//    TIR表示: 类型转换 cast(DataType::Float(16), y)  // 转换为float16
//    TileLang中的使用: fp8_e4m3 c = cast<fp8_e4m3>(float_val)  // FP32 -> FP8
//   6. MinNode - 最小值
//    bound = min(idx, max_bound)  // 边界检查
//    warp_reduce_min(val)         // 线程束归约求最小值
//   7. MaxNode - 最大值, 同上
//
// 二 语句节点 (Stmt Nodes)
//   1. EvaluateNode - 求值语句
//    TIR表示: Evaluate(call)  // 执行函数调用但不使用返回值
//    TileLang使用: Evaluate(tl::tma_store_arrive())  // 触发TMA存储但不关心返回值
//                  Evaluate(tl::cluster_sync())      // 执行集群同步
//   2. AllocateNode - 内存分配
//    TIR表示：分配内存 Allocate(ptr, dtype, [size], condition, body)
//    TileLang中的各种作用域: Allocate(smem_ptr, float16, [1024], const_true(), body)                // 共享内存
//                           Allocate(mbarrier_ptr, uint64_t, [barrier_count], const_true(), body)  // 屏障
//                           Allocate(wmma_frag, float32, [8], const_true(), body)                  // WMMA寄存器分片
//   3. AttrStmtNode - 属性语句
//    TIR表示：添加属性注解 AttrStmt(node, attr_key, value, body)
//    TileLang中的属性: AttrStmt(iv, "thread_extent", 32, body)               // 线程绑定
//                     AttrStmt(buffer, "fragment_shape", "16x16x16", body)  // WMMA分片形状
//                     AttrStmt(loop, "pragma_unroll", 4, body)              // 循环展开
//                     AttrStmt(_, "async_commit_queue_scope", 0, body)      // 异步队列作用域
//   4. BufferLoadNode - Buffer加载
//    TIR表示：从buffer加载数据 BufferLoad(buffer, [index])
//    TileLang中的使用: float4 val = BufferLoad(A, [i*4])  // 向量化加载
//                     half2 h2 = BufferLoad(smem, [tid])  // 从共享内存加载
//   5. BufferStoreNode - Buffer存储
//    TIR表示：存储数据到buffer BufferStore(buffer, value, [index])
//    TileLang中的使用: BufferStore(C, acc, [i])           // 存储结果
//                     BufferStore(smem, reg_val, [tid])  // 存储到共享内存

// <NT> CodeGenTileLangCUDA的调用链路
//   (engine/lower.py) lower -> device_codegen -> target.build.tilelang_cuda -> (src/target/rt_mode_cuda.cc) BuildTileLangCUDA 
// -> CodeGenTileLangCUDA::AddFunction() -> CodeGenTileLangCUDA::Finish() -> CodeGenC::Finish()

namespace tvm {
namespace codegen {

class CodeGenTileLangCUDA final : public CodeGenC {
public:
  CodeGenTileLangCUDA();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  std::string GetVecLoad(DataType t, const BufferNode *buffer,
                         PrimExpr base) final;
  void PrintVecStore(const BufferNode *buffer, DataType t, PrimExpr base,
                     const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const MinNode *op, std::ostream &os) final;
  void VisitExpr_(const MaxNode *op, std::ostream &os) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void PrintFunctionSignature(const ffi::String &function_name,
                              const PrimFunc &func, std::ostream &os);

protected:
  void ReserveKeywordsAsUnique_();
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // Global curand state
  std::string curand_random_generator_state;
  std::string curand_random_generator_state_type;

  // <NT> enable_* 跟踪需要的特性，用于在Finish()时自动包含相应的头文件
  // tcgen05: 
  //   特性维度	     WMMA (Volta/Ampere)	 WGMMA (Hopper)	            UMMA / tcgen05 (Blackwell)
  //   核心设计	     以寄存器为中心	        以共享内存为中心	           以张量内存 (TMEM) 为中心
  //   指令示例	     mma.sync	w            gmma.mma_async	            tcgen05.mma / tcgen05.ld
  //   指令发射粒度	 Warp 级同步(32线程)	  Warp Group 级同步(128线程)  单线程编排 (标量指令)
  //   操作数来源	   寄存器文件 (RF)	      A: RF/SMEM, B: SMEM	       A/B: SMEM, 累加器 C/D: TMEM
  //   主要瓶颈	     寄存器压力大	          SMEM带宽，累加器仍占RF	     极大缓解RF压力，TMEM容量成为新考量
  //  * UMMA (Universal MMA): UMMA 原生支持包括 FP8、FP6 和 FP4 在内的更低精度数据类型，并允许在同一个指令中混合使用不同精度的操作数（例如，A 矩阵使用 FP8，B 矩阵使用 FP4）。
  //  它还内置了对块缩放 (block-scaling) 等去量化技术的硬件支持。
  //  * TMEM: 张量内存, 与UMMA搭配时，累加器可以放在TMA中，而不是传统的寄存器里，极大缓解RF压力
  // warp shuffle: 一般用于warp_reduce，采用内置函数 __shfl_sync() / __shfl_up_sync() / __shfl_down_sync() / __shfl_xor_sync()
  
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable fp6
  bool enable_fp6_{false};
  // whether enable fp4
  bool enable_fp4_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable sparse gemm
  bool enable_sparse_gemm_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need tl mma instruction header
  bool need_mma_instruction_h_{false};
  // whether need tl wgmma instruction header
  bool need_wgmma_instruction_h_{false};
  // whether need tl tcgen05mma instruction header
  bool need_tcgen05mma_instruction_h_{false};
  // whether need tl mma_sm70 instruction header
  bool need_mma_sm70_instruction_h_{false};
  // whether need tcgen_05 common header
  bool need_tcgen05_common_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // whether need cooperative_groups.h
  bool need_cooperative_groups_{false};
  // whether need curand_kernel.h
  bool need_curand_kernel_h_{false};
  // whether need cluster.h
  bool need_cluster_h_{false};
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ =
      Op::GetAttrMap<bool>("cuda.need_warp_shuffle");

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The name of the mbarrier array in shared memory
  // The same as injected_mbarrier_name_ in transform/common/mbarrier.h
  const std::string mbarrier_name_ = "mbarrier";
  // The type name of the mbarrier array
  const std::string mbarrier_dtype_ = "Barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
  std::unordered_map<const VarNode *, IntImm> unroll_factor;
  // Map from VarNode to packed buffer variable name for fp4 packed storage
  std::unordered_map<const VarNode *, std::string> fp4_packed_buffers_;
  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);
  void PrintWmmaScope(const std::string &scope, DataType t,
                      const VarNode *variable, std::ostream &os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const VarNode *variable,
                              int32_t size);

  std::vector<std::string> eviction_policy_names_ = {
      "EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST"};
  std::unordered_set<std::string> bf16_supported_ops_ = {
      "bf1622float2", "bf1622int16", "float22bf162", "bf162bf162"};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUDA_H_
