// ============================================================================
// 头文件包含
// ============================================================================
// TensorLLM 模板库: 包含 MMA (矩阵乘法累加)、GEMM、Copy、Reduce、LDSM 和线程块 Swizzle 等 CUDA 模板
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
// BF16 数据类型支持 (如启用)
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

// ============================================================================
// CUDA Kernel 函数声明
// ============================================================================
// 外部声明: 供 CUDA 运行时调用的 GEMM Linear Kernel
// A_desc, B_desc, C_desc: TMA (Tensor Memory Access) 描述符, 分别对应输入矩阵 A、B 和输出矩阵 C
extern "C" __global__ void linear_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);

// ============================================================================
// CUDA Kernel 函数定义
// ============================================================================
// __launch_bounds__(256, 1): 最大线程块大小 256, 最小活跃 SM 数量 1
// __grid_constant__: 网格级别的常量内存优化
extern "C" __global__ void __launch_bounds__(256, 1) linear_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  // -------------------------------------------------------------------------
  // 共享内存声明
  // -------------------------------------------------------------------------
  // 动态共享内存, 按 1024 字节对齐, 用于 TMA 加载/存储的缓冲区
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];

  // -------------------------------------------------------------------------
  // 寄存器变量声明
  // -------------------------------------------------------------------------
  // C_local: 累加器寄存器, 存储 MMA 计算的中间结果 (8 个 float, 对应 8 个输出元素)
  float C_local[8];
  // A_local, B_local: 每个线程的局部寄存器, 用于存储从共享内存加载的矩阵分块
  bfloat16_t A_local[8];
  bfloat16_t B_local[8];

  // -------------------------------------------------------------------------
  // 屏障 (Barrier) 内存声明
  // -------------------------------------------------------------------------
  // mbarrier_mem: 用于 TMA 操作的硬件屏障, 共 6 个 (3 对, 用于双缓冲)
  __shared__ uint64_t mbarrier_mem[6];
  // 将原始内存重新解释为 Barrier 类型, 便于调用屏障操作
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);

  // -------------------------------------------------------------------------
  // 线程 0 初始化阶段
  // -------------------------------------------------------------------------
  // tl::tl_shuffle_elect<0>(): 仅线程 0 执行初始化 (warp 内选举)
  if (tl::tl_shuffle_elect<0>()) {
    // 预取 TMA 描述符到 GPU, 加速后续 TMA 操作
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
    // 初始化 6 个屏障, 每个屏障参与计数为 128 (warpgroup 大小)
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  // 栅栏 + 同步: 确保所有线程都完成初始化
  tl::fence_barrier_init();
  __syncthreads();

  // -------------------------------------------------------------------------
  // 分支处理: TMA 加载 vs MMA 计算
  // -------------------------------------------------------------------------
  // 线程 128 及以上: 负责 TMA 加载 (生产者线程)
  if (128 <= ((int)threadIdx.x)) {
    // 释放 warpgroup 寄存器 (因为加载线程不需要那么多寄存器)
    tl::warpgroup_reg_dealloc<24>();
    // 获取 2D 光栅化后的 block 索引 (用于确定当前线程块处理的输出块位置)
    const dim3 blockIdx = tl::rasterization2DRow<10>();

    // -------------------------------------------------------------------------
    // 主循环: 迭代 16 次, 完成整个矩阵乘法
    // -------------------------------------------------------------------------
    for (int k = 0; k < 16; ++k) {
      // 等待对应的双缓冲阶段完成加载
      // (k % 3): 双缓冲索引
      // (((k % 6) / 3) ^ 1): 等待另一方的完成信号
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));

      // 仅 warpgroup 0 的线程执行 TMA 操作
      if (tl::tl_shuffle_elect<128>()) {
        // -------------------------------------------------------------------------
        // 加载矩阵 A 的一个分块 (使用 TMA)
        // -------------------------------------------------------------------------
        // expect_transaction(2048): 预期加载 2048 字节
        mbarrier[(k % 3)].expect_transaction(2048);
        tl::fence_proxy_async();
        // TMA 加载: 从全局内存加载到共享内存
        // 偏移 = (k % 3) * 1024 + 12288 (A 矩阵在共享内存中的偏移)
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 1024) + 12288)])), (k * 64), 0);

        // -------------------------------------------------------------------------
        // 加载矩阵 B 的一个分块 (使用 TMA)
        // -------------------------------------------------------------------------
        // expect_transaction(8192): 预期加载 8192 字节
        mbarrier[(k % 3)].expect_transaction(8192);
        tl::fence_proxy_async();
        // TMA 加载: 偏移 = (k % 3) * 4096, 列偏移 = blockIdx.x * 64
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 4096)])), (k * 64), (((int)blockIdx.x) * 64));
      }
      // 当前线程块完成任务后到达屏障
      mbarrier[(k % 3)].arrive();
    }
  }
  // -------------------------------------------------------------------------
  // 计算线程分支 (线程 0-127): 负责 MMA 计算 (消费者线程)
  // -------------------------------------------------------------------------
  else {
    // 为计算阶段分配 240 个寄存器
    tl::warpgroup_reg_alloc<240>();
    // 获取 2D 光栅化后的 block 索引
    const dim3 blockIdx = tl::rasterization2DRow<10>();

    // -------------------------------------------------------------------------
    // 初始化累加器寄存器为 0
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      // 使用 float2 将两个 float 初始化为 0.0
      *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }

    // -------------------------------------------------------------------------
    // 主循环: 迭代 16 次, 完成整个矩阵乘法
    // -------------------------------------------------------------------------
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      // 等待对应的双缓冲阶段数据加载完成
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));

      // -------------------------------------------------------------------------
      // 每个迭代内执行 4 次 MMA 操作 (累加 4 个分块的结果)
      // -------------------------------------------------------------------------
      for (int ki = 0; ki < 4; ++ki) {
        // 使用复杂的索引计算从共享内存加载 A_local 和 B_local
        // 这些索引基于 threadIdx.x 的位操作, 用于将数据分配到正确的线程寄存器
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((k_1 % 3) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, A_local + 0);

        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((k_1 % 3) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + 0);

        // -------------------------------------------------------------------------
        // MMA 同步矩阵乘法累加
        // -------------------------------------------------------------------------
        // 第一次 MMA: C_local[0:4] += A_local[0:8] * B_local[0:8]
        // kBFloat16: 输入数据类型, kFloat32: 累加器数据类型
        // 形状: M=16, N=8, K=16, 转置=False, 整数累加=True
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));

        // 第二次 MMA: C_local[4:8] += A_local[0:8] * B_local[4:8]
        // 累加到 C_local 的后半部分
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
      // 计算完成后到达加载屏障, 通知加载线程可以开始下一轮加载
      mbarrier[((k_1 % 3) + 3)].arrive();
    }

    // -------------------------------------------------------------------------
    // 线程同步 (部分同步, 仅同步前 128 个线程)
    // -------------------------------------------------------------------------
    tl::__sync_thread_partial<3, 128>();

    // -------------------------------------------------------------------------
    // 将计算结果存储到共享内存
    // -------------------------------------------------------------------------
    // 使用 stmatrix_x4 指令以矩阵形式存储, 减少寄存器压力
    tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)C_local[0]), ((bfloat16_t)C_local[1])), __pack_half2(((bfloat16_t)C_local[2]), ((bfloat16_t)C_local[3])), __pack_half2(((bfloat16_t)C_local[4]), ((bfloat16_t)C_local[5])), __pack_half2(((bfloat16_t)C_local[6]), ((bfloat16_t)C_local[7])));
    tl::fence_proxy_async();

    // 部分同步, 确保所有线程都完成存储
    tl::__sync_thread_partial<3, 128>();

    // -------------------------------------------------------------------------
    // TMA 存储: 将结果从共享内存写回全局内存
    // -------------------------------------------------------------------------
    if (tl::tl_shuffle_elect<128>()) {
      // TMA 存储: 偏移 = blockIdx.x * 64
      tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 64), 0);
      // TMA 存储的到达-等待同步
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


// ============================================================================
// 策略信息注释 (由代码生成器添加)
// ============================================================================
// 策略名称: linear_gemm_tl_1_6144_1024
// selected_hparams: [16, 64, 64, 1, 3, 128, 0, True] (超参数配置)
// smem: 30720 bytes (共享内存大小)
// use_cooperative_groups: 0 (未使用协作组)
// layout: (96, 1, 1), (64, 16, 64) (矩阵布局配置)
// block_dim: (256, 1, 1) (线程块维度)
// latency: 0 ms vs [ref-0 sim-0], idx: 28 (性能指标)