/*!
 *  \file lower_shared_barrier.cc
 *  \brief Convert shared.barrier buffers to plain shared + ptx init.
 */
#include "../op/builtin.h"
#include "tvm/ir/type.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>
// <NT> TVM的基础概念
// -- attr: 用于存储属性的命名空间，属性是一种键值对，可以附加在TIR节点上，用于传递额外的信息。
//          在这个pass中，我们定义了一个名为kBarrierInit的属性键，用于记录每个屏障分配的到达计数（arrive counts）。
// -- StmtExprMutator: 用于遍历和修改TIR 语句 和 表达式 的基类。通过继承这个类，重写其中的VisitStmt_和VisitExpr_方法来实现对特定类型的语句或表达式的处理逻辑。
// -- StmtMutator: 用于遍历和修改TIR 语句 的基类。与StmtExprMutator不同的是，StmtMutator只能处理语句节点，而不能处理表达式节点。
//                 在这个pass中，我们使用了StmtExprMutator，因为我们需要同时处理语句和表达式节点。
//                 既然StmtExprMutator能处理语句和表达式，为什么还要有StmtMutator呢？这是因为在某些情况下，我们可能只需要处理语句节点，而不需要处理表达式节点，这时使用StmtMutator会更简单和高效。
//                 例如，在一些只需要修改控制流结构的优化pass中，我们可能只需要处理ForNode、IfNode等语句节点，而不需要处理BufferLoadNode、CallNode等表达式节点，这时使用StmtMutator会更合适。
// -- VisitStmt_: 这是StmtExprMutator中的一个虚函数，用于访问TIR中的 语句 节点。当我们重写这个函数时，可以根据不同类型的语句节点来实现特定的处理逻辑。
//                在这个pass中，我们重写了VisitStmt_来处理BlockNode，以实现对包含屏障buffer的代码块进行访问重写，插入PTX指令来初始化屏障。
// -- VisitExpr_: 这是StmtExprMutator中的一个虚函数，用于访问TIR中的 表达式 节点。当我们重写这个函数时，可以根据不同类型的表达式节点（如BufferLoadNode）来实现特定的处理逻辑。
//                在这个pass中，我们重写了VisitExpr_来处理BufferLoadNode，以实现对屏障buffer的访问重写，将其转换为普通的共享内存buffer访问。
//
// 语句（Stmt）节点有: BlockNode、BufferStoreNode、AttrStmtNode 等。通常表示程序的控制流结构，如循环、条件分支、代码块等
// -- BlockNode: 表示一个 代码块 的节点，包含了该块内的语句、分配的缓冲区、匹配的缓冲区等信息。
// -- BufferStoreNode: 表示 向缓冲区存储数据 的语句节点，包含了要存储的缓冲区、值和索引信息。
// -- AttrStmtNode: 表示一个 属性语句 的节点，包含了属性的键、值和作用范围等信息。
// 表达式（PrimExpr）节点有: BufferLoadNode 等,表示计算或数据访问，如变量引用、算术运算、函数调用等
// -- BufferLoadNode: 表示 从缓冲区加载数据 的表达式节点，包含了要加载的缓冲区和索引信息。

// -- Buffer: 这是TIR中表示内存缓冲区的对象，包含了缓冲区的名称、数据类型、形状、存储范围等信息。

// <NT> 对共享内存屏障的实现是通过一个特殊的 buffer (shared.barrier) 来表示的，
// 这个 pass 的作用是把这个特殊的 buffer 转换为普通的共享内存 buffer，并在合适的位置插入 PTX 指令来初始化屏障。
// 这样做的好处是可以复用现有的共享内存管理机制，同时也更贴近底层硬件的实现细节，方便后续针对不同架构进行优化和调整。

namespace tvm {
namespace tl {

namespace attr {
// BlockAttr, Recording the arrive counts for each barrier allocation
constexpr const char *kBarrierInit = "barrier_init";
} // namespace attr

using namespace tir;

class SharedBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, bool disable_shuffle_elect = false) {
    SharedBarrierRewriter rewriter(disable_shuffle_elect);
    return rewriter(std::move(body));
  }

private:
  SharedBarrierRewriter(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = tvm::ffi::GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;

    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    // <NT> 存储域要区分开来，只有shared.barrier和shared.cluster_barrier才是屏障buffer
    // cluster_barrier是sm90引入的新的屏障类型，支持跨block的线程同步，区别于shared.barrier只能在block内同步。
    // Only check buffers allocated in THIS block, not accumulated from parent
    // blocks
    Array<Buffer> barrier_buffers;
    for (auto buffer : alloc_buffers) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      if (!ptr_type)
        continue;
      auto storage_scope = ptr_type->storage_scope;
      if (storage_scope == "shared.barrier" ||
          storage_scope == "shared.cluster_barrier") {
        barrier_buffers.push_back(buffer);
        if (storage_scope == "shared.cluster_barrier") {
          has_cluster_barrier_ = true;
        }
      }
    }

    if (barrier_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(thread_var_.defined()) << "thread_var_ is not defined";

    for (auto buffer : barrier_buffers) {
      ICHECK(buffer->name != "mbarrier")
          << "Shared barrier's name 'mbarrier' is reserved";
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    /*
    Transform:
        mbarrier_list = T.alloc_barrier(arrive_counts: list[int], "handle",
    scope="shared.barrier")

    into:
        # This is emitted by the definition of T.alloc_barrier
        mbarrier_list = T.alloc_buffer(len(arrive_counts), "handle",
    scope="shared.barrier")

        # This is emitted by this pass
        if tx == 0:
          for i in range(len(arrive_counts)):
            T.ptx_init_barrier_thread_count(mbarrier_list[i], arrive_counts[i])
    */

    // Extract the arrive counts from the block attr "barrier_init"
    // The attr is a Map<Var, Array<PrimExpr>> where key is buffer.data and
    // value is arrive counts
    ICHECK(op->annotations.count(attr::kBarrierInit))
        << "barrier_init is not defined";
    auto barrier_init_map = op->annotations.Get(attr::kBarrierInit)
                                ->as<Map<Var, Array<PrimExpr>>>()
                                .value();

    // Create init calls for each barrier buffer
    // Initialize each barrier element with its respective arrive count
    Array<Stmt> init_mbarrier_calls_;
    for (auto buffer : barrier_buffers) {
      auto data = buffer->data;
      ICHECK(barrier_init_map.count(data))
          << "Barrier buffer " << buffer->name
          << " not found in barrier_init annotation";
      auto arrive_counts = barrier_init_map.at(data);
      ICHECK(arrive_counts.size() ==
             static_cast<size_t>(buffer->shape[0].as<IntImmNode>()->value))
          << "The number of arrive counts (" << arrive_counts.size()
          << ") must match the barrier buffer size (" << buffer->shape[0]
          << ") for buffer " << buffer->name;

      for (size_t i = 0; i < arrive_counts.size(); i++) {
        auto call =
            Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                 {BufferLoad(buffer,
                             {IntImm(DataType::Int(32), static_cast<int>(i))}),
                  arrive_counts[i]});
        init_mbarrier_calls_.push_back(Evaluate(call));
      }
    }
    if (init_mbarrier_calls_.empty())
      return block;

    Array<Stmt> new_body;
    PrimExpr condition;
    if (!disable_shuffle_elect_) {
      condition = Call(DataType::Bool(), tl_shuffle_elect(), {0});
    } else {
      condition = EQ(thread_var_->var, 0);
    }
    new_body.push_back(IfThenElse(condition,
                                  init_mbarrier_calls_.size() == 1
                                      ? init_mbarrier_calls_.back()
                                      : SeqStmt(init_mbarrier_calls_),
                                  Stmt()));

    new_body.push_back(
        Evaluate(Call(DataType::Handle(), ptx_fence_barrier_init(), {})));
    new_body.push_back(Evaluate(
        Call(DataType::Handle(), builtin::tvm_storage_sync(),
             {StringImm(has_cluster_barrier_ ? "cluster" : "shared")})));
    new_body.push_back(block->body);

    block.CopyOnWrite()->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, load->indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, store->indices);
    }
    return store;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  // Disable shuffle elect for the warp specialized kernel
  bool disable_shuffle_elect_;
  // Whether the block has a cluster barrier
  bool has_cluster_barrier_ = false;
};

PrimFunc LowerSharedBarrier(PrimFunc f, bool disable_shuffle_elect) {
  f.CopyOnWrite()->body =
      SharedBarrierRewriter::Rewrite(f->body, disable_shuffle_elect);
  return f;
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass LowerSharedBarrier() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    return tl::LowerSharedBarrier(std::move(f), disable_shuffle_elect);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerSharedBarrier", LowerSharedBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
