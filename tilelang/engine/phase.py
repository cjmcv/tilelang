from __future__ import annotations
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma, is_hopper


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if (not is_cuda_target(target)) or (not have_tma(target)):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def allow_tma_and_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not have_tma(target):
        return False
    disable_tma_lower = pass_ctx.config.get("tl.disable_tma_lower", False)
    return not disable_tma_lower and allow_warp_specialized(pass_ctx=pass_ctx, target=target)


def allow_fence_proxy(target: Target | None = None) -> bool:
    return have_tma(target)


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_aggressive_merge = bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        # This is a workaround to avoid the bug in the MergeSharedMemoryAllocations pass
        # when warp specialization is enabled, as different warp threads may access different
        # buffers, but the liveness analysis is hard because we need to do pipeline.
        enable_aggressive_merge = False
    return enable_aggressive_merge


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)
    return enabled


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """
    Check whether the module is valid before lowering. If not, raise a user-friendly error
    in Python side instead of letting the error dive into the complicated TVM/C++ stack.
    Note: This is a validation-only pipeline of passes and does not modify or return the module.
    """

    # Debug
    # tilelang.analysis.ASTPrinter()(mod)
    # Check if there are any invalid nested loops.
    tilelang.analysis.NestedLoopChecker()(mod)
    # Check if there are any invalid symbolic T.Parallel + fragment access.
    tilelang.analysis.FragmentLoopChecker()(mod)

# <NT> TVM 中的transform主要包含: 
#   1) ModulePass，应用于整个IRModule，处理包含多个函数 / 算子的完整模块（比如跨函数内联、模块级常量合并、依赖分析）
#   2）FunctionPass，应用于单个relax.Function，针对 Relax IR（TVM 高层图 IR）的函数优化（比如算子融合、图化简、类型推导）。
#   3）PrimFuncPass，应用于单个 tir.PrimFunc，针对 TensorIR（底层计算 IR）的函数优化（比如循环展开、内存布局优化、向量化）
#   4）ExprMutator/ExprVisitor，单个表达式 / 算子，非严格意义的 pass，但常作为 pass 内部逻辑（遍历 / 修改单个表达式，比如常量折叠）
#   即主要有1/2/3个平级的transform pass，其中 ModulePass 会同时涉及到 FunctionPass 和 PrimFuncPass 的作用域，而 tilelang 的主要优化点在 PrimFuncPass。
#
# <NT> LowerAndLegalize: 语义合法化（lower过程中，包含 语义合法化 + 性能合法化）。
#                        将前端 Tile IR 逐步合法化、降阶，转换为适用于下游优化与代码生成的形式。
# 调用链路： JITKernel.__init__ -> _compile_and_create_adapter -> lower -> LowerAndLegalize
#                                                                      -> OptimizeForTarget
# 1. LetInline: 默认策略是“后续用到时再 inline”，但如果 pass config 里开了 force_let_inline，就一次性全部展开成普通 SSA(Static Single Assignment，静态单赋值) 语句。
#               前端为了可读性会生成很多 Let(var, value, body) 节点，该 pass 将单 use 的 Let 全部 inline，消除中间变量，减少 TIR 节点数，让后续模式匹配更容易命中。
#    * Let(var, value, body) 的核心作用就是：“引入一个局部、不可变的命名绑定，明确计算顺序与作用域，为后续优化 Pass 提供结构清晰的 ANF 形式的 IR。”
#    * SSA 的核心规则只有一条：每个变量只能被赋值一次，且在使用前必须先定义。有着 无歧义/优化友好/分析高效 的作用。
#      -> 非SSA代码，编译器需要跟踪，否则有歧义         |    SSA代码，无歧义（不同框架（TVM/LLVM/MLIR）的 SSA 语句语法不同，但核心规则一致）
#      -> x = 1          # 第一次赋值                 |    x1 = 1         # x 的第1个版本（唯一赋值）
#         x = x + 2      # 第二次赋值（覆盖）          |    x2 = x1 + 2    # x 的第2个版本（唯一赋值）
#         y = x * 3      # 使用的是第二次赋值的 x      |    y1 = x2 * 3    # 明确使用 x2，无歧义
# 2. AddWrapperForSingleBufStore: 把 “一条语句里既读又写同一个 buffer” 的 TileLang 写法，拆成 “先读→修改→再写” 的三地址形式，
#                                 让后续 TIR 优化和硬件 codegen 不必处理“读-改-写”原地更新语义，降低合法化复杂度。
# 3. LegalizeNegativeIndex: 支持python风格的负下标，如A[-1, -2]。而该pass则把所有下标规范化成非负形式, 如用户写的A[-1]转为A[extent - 1]，否则编译负下标会出错。
#                           同时把新的下标用 assume 标注上界，方便 TVM 的算术证明器消去冗余判断。
# 4. InjectAssumes: 在“可能越界”的算子（如 resize、strided_slice、pad、pooling …）周围，自动插入 tir.assume 断言，如判断除法是否整除、下标是否对齐等。
#                   看到这些 assume 后，可以激进地做强度折减、边界消除。例如 assume(n % 128 == 0) 后，vectorize 宽度可直接取 128，而无需生成 fallback 分支。
# 5. Simplify: 标准 TIR 算术/逻辑/循环折叠简化。
# 6. LayoutReducer: TileLang 的 reduction 变量可以声明在任意 scope（thread、warp、ctaid、grid）。
#                   该 pass 根据 target 的 warp size、simd width，把 reduction 的“逻辑线程 ID” 映射到“物理线程 ID”，并插入 shuffle、shared memory、atomic 等实现策略。
#                    输出：每个 reduction 节点都带上 layout=Shuffle|Shared|Atomic 标记，方便后端直接 codegen。
# 7. LayoutInference: 对 shared memory、fragment（matrix core 寄存器数组）做 layout 推断：
#                    计算每个 tile 的 shape、stride、alignment，根据 swizzle、padding、bank_size 决定偏移量，避免 bank conflict。
#                    把 tl.shared, tl.fragment 的抽象声明改写成 TIR 的 Allocate + Buffer + 具体 stride 表达式。
#                    如没有这一步，后端无法知道 shared memory 到底该放哪、怎么 vectorize。
# 8. LowerTileOp: 把 TileLang 的高阶指令如tl.dot(A, B, C, tile_k=16) 降维成 TIR 的 for (k=0; k<16; k++) … 三重循环 + load_vector_4 + mma_sync 等底层内联调用。
#                 从此开始，IR 里不再出现 tile_op 节点，只剩普通 loop + call_intrin。
# 9. LowerL2Persistent: TileLang 支持把某些共享内存缓冲区标记为 “L2 persistent”，即跨多个 CTA 复用。该 pass 把逻辑上“一次分配、多次复用”的语句改写成：
#                       在 kernel 开头做一次 Allocate，把指针通过 kernel 参数或 global memory 传进来，在 consumer CTA 里直接 Attach 该 buffer。
#                       生成的 TIR 里出现 tir.allocate([persistent_scope="L2"], …)，后端可据此生成 CUDA 的 cudaGraph / global memory slab。
# 10.LegalizeVectorizedLoop: 检查所有 for 循环的 vectorize 标注：* 迭代次数是否静态且为 vector_width 的倍数
#                                                              * 循环体内部是否有条件分支、函数调用导致无法向量化
#                                                              * 内存访问是否对齐
#                                                              * 不合法就回退到 serial 或 unroll，并在 attr 里写原因，方便开发者 debug。
# 11.LegalizeSafeMemoryAccess: 给每个 load/store 插入边界检查: * 如果下标是动态表达式，生成 if (index >= 0 && index < bound) … else trap()
#                                                            * 如果 shared memory 访问可能越界，生成 __builtin_trap() 或 tir.tvm_throw_last_error()
#                              目的：在 debug 模式下立即挂掉，而不是 silently corrupt；release 模式下可通过 tir.Likely 提示后端优化器把分支判为冷路径。
# 12.Simplify: 再次调用，上一步插了很多 if (likely(true)) … 的保底分支，Simplify 可以把“显然为真/假”的条件剪掉，避免后端生成多余代码。
#                       同时把新出现的 select(x, load, 0) 等模式化简成 load（当 x 为真时）。
# 13.HoistNonRestrictParams: 是 TileLang 在“彻底降维成 TIR”之前做的最后一次“信息抬升”。
#             把散落在 block 作用域里的 tl.assume(...)、tl.align(...)、tl.restrict(...) 等“程序员口头承诺”，
#             收集起来并贴到 PrimFunc 的 attrs 上，让 TVM/LLVM 的后端优化器一眼就能看到，从而生成更激进的机器码。
#   TVM 的 TIR 世界里，信息分三级：
#    1）PrimFunc 级 attr，有attr::kAssume, attr::kNoAlias, attr::kAlign, tir::restrict等，后端（LLVM、CUDA、ROCm）在生成函数序言 / 读取 parameter 属性时，只看这一级。
#    2）Block 级 stmt，tir.assume(x > 0) 出现在 tir::BlockNode 的 init 或 body 里。
#       对 TVM 的算术推导器（arith::Analyzer）可见，但对 LLVM 不可见，因为 LLVM 拿到的是已经把 Block 降维成 for/if 的 CodeGen。
#    3）Loop 级 annotation，例如 for (i, 0, n) { for_attr({vectorize: 4}) } 只影响该循环，跨循环就丢了。
#
# 至此，IRModule 里的 TileLang 高阶构造已全部降维成“标准 TIR + 少量 intrinsics”：
#  * 所有内存访问都有合法下标、对齐、边界保证;  * shared/fragment layout 已推断并 swizzle
#  * reduction 已映射到 warp/shuffle/shared; * vectorize 宽度已校验
#  * 动态形状已插 assume，方便后续 bound inference.

def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    """
    Bind target information and progressively legalize and lower frontend Tile IR into a form suitable for downstream optimization and codegen.

    This pass pipeline:
    - Binds the provided target to the module.
    - Legalizes frontend Tile IR into TVM-compatible constructs.
    - Simplifies expressions.
    - Configures reducer layouts and performs layout inference for fragments and shared memory.
    - Lowers high-level tile operations and L2 persistent maps.
    - Legalizes vectorized loops and inserts safety checks for memory accesses.
    - Re-simplifies to remove redundancies introduced by safety checks.
    - Attempts loop vectorization for dynamic-shaped loops.

    Parameters:
        mod (IRModule): The input IR module containing frontend Tile IR.
        target (Target): Target device information to bind into the module.

    Returns:
        IRModule: The transformed module, ready for target-specific optimization passes.
    """
    mod = tir.transform.BindTarget(target)(mod)

    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Lower l2 persistent map
    mod = tilelang.transform.LowerL2Persistent()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


# <NT> OptimizeForTarget: 性能合法化（lower过程中，包含 语义合法化 + 性能合法化）。
#                        将前面LowerAndLegalize得到的tir，进一步做特定后端的性能优化。
# 调用链路： JITKernel.__init__ -> _compile_and_create_adapter -> lower -> LowerAndLegalize
#                                                                      -> OptimizeForTarget
# 0. pass_ctx = tilelang.transform.get_pass_context()
#    拿到当前 build 的 PassContext，后面所有 allow_* 函数都靠它读 pass_ctx.config（例如 "tl_enable_warp_specialized": True），从而在不同架构/调试模式下走不同分支。
# 1. Shared 资源降维
#  1.1) LowerSharedBarrier：TileLang 前端允许写 tl.shared_barrier.arrive() / wait()，表示 thread-block 内 对某块 shared memory 的异步生产-消费。
#               该 pass 把它换成 具体的 barrier object（CUDA 11+ 的 __barrier 或软件模拟的 int barrier[threads]），并给每个 barrier 分配唯一 slot id，为后面 InjectTmaBarrier/ThreadSync 提供“句柄”。
#  1.2）LowerSharedTmem: Hopper 架构有 164 KB “shared memory” 可配置成 shared 或 tmem (tensor memory)。
#               该 pass 把 tl.shared.tmem_allocate 节点换成真正的 Allocate + 新 scope=tmem，并记录基址偏移，供 RewriteWgmmaSync 计算 wgmma.mma_async 的地址立即数。
# 2. 是否走 Warp Specialized + TMA 的路线
#  走 ws+tma 路线：
#   2.1) IfStmtBinding: 把 if (warp_id < constant) 这类“编译期已知真值”的条件绑定到 warp-specialized 版本号，生成 ws_version = (pred) ? 0 : 1 的常量，
#               供后续 MultiVersionBuffer 为不同版本分配独立 shared 块
#   2.2) MultiVersionBuffer: 为 pipelined 的每个 stage、每个 warp-specialized 版本 独立分配 shared buffer（名字后缀 _ws0_stage0、_ws1_stage1…），避免同名 buffer 被复用导致同步复杂度爆炸。
#   2.3) WarpSpecialized: 把 if (ws_version == 0) { compute A } else { compute B } 的两条路径 拆成两个独立的 PrimFunc（__device__ void kernel_ws0, kernel_ws1），并在 host 端
#               生成 cudaLaunchKernel 时根据 warp_id / warps_per_block 选码。在 shared memory 里插 producer-consumer 信号量（barrier slot），实现 warp 间异步流水线。并给 wgmma.mma_async 计算 tmem 地址偏移。
#   2.4) InjectTmaBarrier: TMA (Tensor Memory Accelerator) 的 cp.async.bulk.tensor 需要 global → tmem 的异步 barrier。
#               该 pass 在 tma.load 之后插入 tma.barrier.arrive 指令，并在 consumer 端插 tma.barrier.wait，与 shared barrier 共用 slot 管理。
#   2.5) PipelinePlanning: 在 warp-specialized 场景下，plan 会考虑 双缓冲 vs 三缓冲 的 shared 大小、TMA pipeline depth，目标是把 global memory latency 完全藏在 compute 后面。
#   2.6) InjectSoftwarePipeline: 根据 plan 结果，把 for (k, 0, K, stage) 循环 展开成 prologue + steady + epilogue，并在正确的 stage 点插入 barrier.arrive/wait 
#               以及 cp.async.commit_group/wait_group。还要处理 TMA bulk 与 shared pipeline 的并行。
#   2.7) LowerOpaqueBlock: TileLang 为了保留 block 语义，会生成 tir::OpaqueBlock 节点。该 pass 把 OpaqueBlock 降成普通 tir::SeqStmt，方便后面 MergeIfStmt 做跨 warp 的 if 合并。
#              * 什么是OpaqueBlock：在tilelang中，常会有
#                @tl.program
#                def gemm():
#                    with tl.block():          # ← 这里
#                        for i in tl.range(16):
#                            C[...] = A[...] * B[...]
#                with tl.block() 在 AST 里被翻译成 单入口单出口、带局部分配、带 barrier 的复合语句。降到 TIR 时，为了保留“这是同一个 block” 这一信息
#                （方便后续 PipelinePlanning / WarpSpecialized / ThreadSync 知道哪些语句必须在同一 scope 内），TileLang 先 不急着展开，而是包一层 tir::OpaqueBlock。
#              * OpaqueBlock 在 TIR 里就是一个 Stmt 节点，与 tir::Block（TVM Schedule 用的可重构 Block）不同，OpaqueBlock 不允许被 Schedule 再变换，只是临时容器。
#   2.8) MergeIfStmt: 把相邻且条件相同的 if (ws_version == x) 合并，减少分支碎片。
#   2.9) RewriteWgmmaSync: 把 wgmma.mma_async intrinsic 换成真正的 PTX wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 指令，并计算 tmem 地址立即数（必须 256-byte 对齐）。
#   2.10) InjectFenceProxy: Hopper 的 wgmma 与 shared 之间需要 fence.proxy.async 保证内存序。该 pass 在正确的程序点插 fence。
#  走 传统路线：
#   2.1) IfStmtBinding / MergeIfStmt: 同样走，但 不做 warp-specialized 拆分，只给 async-copy pipeline 用。
#   2.2) PlanAndUpdateBufferAllocationLocation: 计算 shared buffer 生命周期，把 跨 stage 复用 的 buffer 合并，减少 shared 内存峰值。
#                         与上面ws+tma路线的差异：ws+tma因为 warp-specialized 已拆函数，buffer 不能跨 warp 复用；传统版可以激进合并。
#   2.3) PipelinePlanning / InjectSoftwarePipeline: 类似，但只针对 cp.async.ca (Ampere) 或 ldmatrix 做双缓冲，不涉 TMA。
#   2.4) InjectFenceProxy (option): Ampere 没有 wgmma，但若开了 cp.async.proxy 仍需要 fence.proxy.async 保证顺序.
# 3. 公共优化阶段
#   3.1) LowerOpaqueBlock: ws+tma路线已经走过一次，为空；传统线路是第一次做。
#   3.2) Simplify: 把 pipeline 展开后的冗余 if (stage >= 0) 等剪掉。
#   3.3) NarrowDataType: 所有 int64 下标 缩到 32-bit（只要静态证明不会溢出 2^31-1）。减少寄存器压力，NVVM 对 32-bit 整数乘加可省指令。
#   3.4) FlattenBuffer: 把多维 Buffer 压成一维 flat_buffer + base+offset 形式，方便后面StorageRewrite / MergeSharedMemoryAllocations 做 跨 buffer 的偏移量合并。
#   3.5) ConfigIndexBitwidth: 根据 flatten 后的偏移量范围，给 tir::Var 重新设置 dtype=int32/uint16，进一步节省地址计算指令。
#   3.6) 再次 Simplify / VectorizeLoop / StorageRewrite: VectorizeLoop 只在 allow_vectorize==True 时生效，把常量宽度 for (i, 0, 4) 换成 tir.vectorize 标注，最终生成 ld.v4 / fma.v4。
#                                           StorageRewrite 把 同一个 scope 里生命周期不重叠 的 Allocate 节点合并到同一块 slab，减少 CUDA 的 dynamic_shared__ 申请次数。
#   3.7) UnrollLoop: 把小的常数循环完全展开，为后面 RemoveNoOp 创造 DCE 机会。
#   3.8) RenormalizeSplitPattern: TVM 的 Split 可能生成 floordiv(x, 256) * 256 + floormod(x, 256)，该 pass 把它重写成 x 让后续 RemoveNoOp 删掉冗余表达式。
#   3.9) RemoveNoOp / RewriteUnsafeSelect / HoistIfThenElse: 清理死代码、把 select(cond, load, 0) 换成 if (cond) load else 0 方便 LLVM 向量化、把 invariant 的 if 提到循环外。
# 4. 内存正确性与设备划分
#   4.1) VerifyMemory: 检查 shared/global memory 访问是否越界、对齐是否满足硬件要求；失败即报错，避免 kernel 跑飞。
#   4.2) AnnotateEntryFunc: 把 primfn 标成 __global__ 入口，供 SplitHostDevice 识别。
#   4.3) InferFragment: 为 Hopper 的 wgmma fragment 补全 mma.sync 所需的 layout=row/col 属性，防止 LLVM  codegen 找不到 intrinsic。
#   4.4) LowerThreadAllreduce: TileLang 只用一个 thread-block 维度，但 warp-level reduce 需要 __shfl_xor_sync。该 pass 把 tir.reduce 换成 warp::shuffle + shared 两段式实现，并插 __syncthreads。
# 5. Host / Device 分离 & 同步收尾
#   5.1) LowerHopperIntrin: 把所有 tl.hopper.* 内联函数（如 wgmma, tma.load) 换成最终 PTX 字符串，嵌入 tir::Call("llvm.inline_asm", ...)
#   5.2) 可选 ThreadSync("global"): 若 kernel 用了 global memory 的 producer-consumer（persistent threadblock），则插 __threadfence_system() 级别全局屏障。
#   5.3) AnnotateDeviceRegions / SplitHostDevice: 把 __global__ kernel 拆到 mod->functions["kernel_name_device"]；
#                                                 host 端生成 kernel_name(args) → cudaLaunchKernel 的 wrapper。
#   5.4) MergeSharedMemoryAllocations: 在 每个 device 函数内部，把多个 Allocate 合并成一条 __shared__ uint8 slab[N]，然后重写所有 buffer->data 为 slab + offset。
#                                      开关 enable_aggressive_merge：Hopper  warp-specialized 场景下关闭（不同 warp 不能复用），传统模式开启（可跨 stage 复用）。
#   5.5) ThreadSync("shared") / ThreadSync("shared.dyn"): 在 shared memory 生命周期边界插入 __syncthreads()；对 dynamic_shared__ 额外插 __barrier_sync(0)，保证初始化完成。
#   5.6) InjectPTXAsyncCopy: 把 cp.async.ca/shared/global 内联换成 cuda::barrier_arrive / wait 的 PTX 序列；若之前已走 TMA 路线，则这里只处理 cp.async.ca 的 shared 双缓冲部分。
#   5.7) 可选 AnnotateWarpGroupRegAlloc: 为 Hopper 的 wgmma 计算所需 128 register/thread 需求，在 LLVM 里生成 .maxnreg 注解，防止寄存器溢出。
#   5.8) MakePackedAPI: 生成 TVM 标准 int kernel_name_wrapper(void** args)，供 Python / GraphExecutor 调用。
#   5.9) LowerDeviceKernelLaunch: 若 kernel 里又 launch 了子 kernel（dynamic kernel launch），则把 tir::Call("tir.tl_device_launch", ...) 换成 CUDA 的 cudaGetParameterBuffer + cudaLaunchDevice 序列。
# 6. Persistent threadblock 转换
#   6.1) PersistThreadblock: 把原本 gridDim = (M/128, N/128) 的静态 grid 换成 persistent 模式：
#                            * 只 launch 少量 CTA（等于 SM 数）；    * 每个 CTA 内部 while (work_queue.next(tile)) { compute }；
#                            * 通过 global memory queue 动态领取 tile，避免 GPU 尾部 wave 空闲。
#                            * 该 pass 会重写 gridDim、插 atomicCAS 取任务、在 kernel 末尾插 continue 循环。
# 至此，mod 里只剩下：host 端 wrapper（负责 cudaLaunchKernel / memcpy）
#                   device 端 __global__ kernel（已经满是 PTX 内联、barrier、fence、wgmma、寄存器对齐、shared 合并、persistent 循环）
#                   可以交给 tir->CodegenCUDA → NVVM → PTX → cubin，直接跑在卡上。
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    # Lower the barrier.arrive into specific initialization slot
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    # Lower the shared.tmem into specific initialization slot
    mod = tilelang.transform.LowerSharedTmem()(mod)
    # which may be introduced by the LegalizeSafeMemoryAccess
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.MultiVersionBuffer()(mod)
        mod = tilelang.transform.WarpSpecialized()(mod)
        mod = tilelang.transform.InjectTmaBarrier()(mod)
        # if tma is not enabled, we can also do pipeline planning
        # to get better performance with async copy
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        # warp_specialized pass will pack the if stmt into the block
        # so we need to lower the opaque block first
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)
        if is_hopper(target):
            mod = tilelang.transform.RewriteWgmmaSync()(mod)
        mod = tilelang.transform.InjectFenceProxy()(mod)
    else:
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)
        if allow_fence_proxy(target=target):
            # in hopper device, wgmma is an async proxy
            # so we need to inject a fence proxy before it
            mod = tilelang.transform.InjectFenceProxy()(mod)

    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    mod = tilelang.transform.LowerHopperIntrin()(mod)
    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    # Inject PTX async copy must behind the thread sync pass
    # as ptx async copy won't be recognized as a valid buffer load
    mod = tilelang.transform.InjectPTXAsyncCopy()(mod)
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # Transform threadblock to persistent threadblock
    mod = tilelang.transform.PersistThreadblock()(mod)

    return mod
