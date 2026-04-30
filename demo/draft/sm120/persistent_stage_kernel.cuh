#pragma once
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

// Atomic operations for cross-block synchronization
__device__ __forceinline__ unsigned long long int
    atom_add_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int val) {
  unsigned long long int old_val;
  asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
               : "=l"(old_val)
               : "l"(addr), "l"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ unsigned long long int
    ld_acquire_sys_u64(unsigned long long int *addr) {
  unsigned long long int val;
  asm volatile("ld.acquire.sys.u64 %0, [%1];"
               : "=l"(val)
               : "l"(addr)
               : "memory");
  return val;
}

namespace kernel {

enum class PipelineMode {
  NORMAL,     // All 3 stages in one block
  SPLIT       // Block 0: stage0, Blocks 1-20: stage1+stage2 with cross-block sync
};

template <int NUM_STAGES>
struct StageInfo {
  int start_block;
  int end_block;
  bool is_producer;
  bool is_consumer;
};

class PersistentStageKernel {
public:
  static constexpr int MAX_STAGES = 4;
  static constexpr int NUM_SYNC_COUNTERS = 2;

private:
  PipelineMode mode_;

  // For split mode: pointers to global sync counters
  unsigned long long* sync_counter_stage0_done_;  // Stage 0 sets this when done
  unsigned long long* sync_counter_stage1_done_;   // Stage 1 sets this when done (for stage 2)

  // Configuration
  int total_blocks_;
  int stage0_blocks_;      // How many blocks run stage 0
  int stage1_blocks_;      // How many blocks run stage 1
  int stage2_blocks_;      // How many blocks run stage 2

public:
  __device__ __forceinline__ void init(PipelineMode mode,
                                        unsigned long long* sync_counter_stage0,
                                        unsigned long long* sync_counter_stage1,
                                        int total_blocks) {
    mode_ = mode;
    sync_counter_stage0_done_ = sync_counter_stage0;
    sync_counter_stage1_done_ = sync_counter_stage1;
    total_blocks_ = total_blocks;
    stage0_blocks_ = 1;
    stage1_blocks_ = total_blocks - 1;
    stage2_blocks_ = total_blocks - 1;
  }

  // Dynamically adjust mode at runtime
  __device__ __forceinline__ void set_mode(PipelineMode mode) {
    mode_ = mode;
  }

  __device__ __forceinline__ PipelineMode get_mode() const {
    return mode_;
  }

  // Allow reconfiguring sync counters (useful for dynamic adjustment)
  __device__ __forceinline__ void set_sync_counters(unsigned long long* counter_stage0, unsigned long long* counter_stage1) {
    sync_counter_stage0_done_ = counter_stage0;
    sync_counter_stage1_done_ = counter_stage1;
  }

  // Update block configuration dynamically
  __device__ __forceinline__ void set_block_config(int total_blocks) {
    total_blocks_ = total_blocks;
    stage0_blocks_ = 1;
    stage1_blocks_ = total_blocks - 1;
    stage2_blocks_ = total_blocks - 1;
  }

  __device__ __forceinline__ int get_stage0_block_id() const {
    return 0;
  }

  __device__ __forceinline__ int get_stage1_start_block() const {
    return 1;
  }

  __device__ __forceinline__ int get_stage1_end_block() const {
    return stage1_blocks_;
  }

  __device__ __forceinline__ int get_stage2_start_block() const {
    return 1;
  }

  __device__ __forceinline__ int get_stage2_end_block() const {
    return stage2_blocks_;
  }

  __device__ __forceinline__ bool is_stage0_block() const {
    return blockIdx.x == get_stage0_block_id();
  }

  __device__ __forceinline__ bool is_stage1_block() const {
    return blockIdx.x >= get_stage1_start_block() && blockIdx.x < get_stage1_end_block();
  }

  __device__ __forceinline__ bool is_stage2_block() const {
    return blockIdx.x >= get_stage2_start_block() && blockIdx.x < get_stage2_end_block();
  }

  __device__ __forceinline__ bool is_split_mode() const {
    return mode_ == PipelineMode::SPLIT;
  }

  // Wait for stage 0 (rms_norm) to complete (used by stage 2 blocks)
  // Note: stage 1 does NOT wait - it runs concurrently with stage 0
  __device__ __forceinline__ void wait_stage0_complete() {
    if (!is_split_mode()) return;
    // Only stage 2 waits for stage 0
    if (!is_stage2_block()) return;

    // Poll until stage 0 signals completion
    while (true) {
      uint64_t val = ld_acquire_sys_u64(sync_counter_stage0_done_);
      if (val >= stage0_blocks_) break;
      __nanosleep(2);
    }
  }

  // Stage 1 does NOT wait - it runs concurrently with stage 0

  // Wait for stage 1 to complete (used by stage 2 blocks)
  __device__ __forceinline__ void wait_stage1_complete() {
    if (!is_split_mode()) return;
    // Only stage 2 waits for stage 1
    if (!is_stage2_block()) return;

    // Poll until stage 1 signals completion
    while (true) {
      uint64_t val = ld_acquire_sys_u64(sync_counter_stage1_done_);
      if (val >= stage1_blocks_) break;
      __nanosleep(2);
    }
  }

  // Wait for both stage 0 AND stage 1 to complete (used by stage 2)
  __device__ __forceinline__ void wait_all_previous_complete() {
    if (!is_split_mode()) return;
    if (!is_stage2_block()) return;

    // Wait for stage 0 first
    while (true) {
      uint64_t val0 = ld_acquire_sys_u64(sync_counter_stage0_done_);
      if (val0 >= stage0_blocks_) break;
      __nanosleep(2);
    }
    // Then wait for stage 1
    while (true) {
      uint64_t val1 = ld_acquire_sys_u64(sync_counter_stage1_done_);
      if (val1 >= stage1_blocks_) break;
      __nanosleep(2);
    }
  }

  // Signal stage 0 completion (only called by stage 0 blocks)
  __device__ __forceinline__ void signal_stage0_complete() {
    if (!is_split_mode()) return;
    if (!is_stage0_block()) return;

    if (threadIdx.x == 0) {
      atom_add_release_gpu_u64(sync_counter_stage0_done_, 1);
    }
    __syncthreads();
  }

  // Signal stage 1 completion (only called by stage 1 blocks)
  __device__ __forceinline__ void signal_stage1_complete() {
    if (!is_split_mode()) return;
    if (!is_stage1_block()) return;

    if (threadIdx.x == 0) {
      atom_add_release_gpu_u64(sync_counter_stage1_done_, 1);
    }
    __syncthreads();
  }

  // Compute barrier: blocks until all blocks in the same stage have arrived
  __device__ __forceinline__ void compute_barrier(int num_threads = 256) {
    __syncthreads();
  }
};

// Global helper functions for the kernel
__device__ __forceinline__ void init_sync_counters(uint64_t* counter0, uint64_t* counter1, int stage0_count, int stage1_count) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *counter0 = 0;
    *counter1 = 0;
  }
  __syncthreads();
}

} // namespace kernel