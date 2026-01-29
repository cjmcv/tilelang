/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "profiler.h"
#include "tasks/common/copy_sm80.cuh"
#include "tasks/common/bfloat16.h"
#ifdef MPK_ENABLE_TMA
#include "tma.cuh"
#endif
#include "mpk_atoms.cuh"
#include "runtime_header.h"
#ifdef USE_NVSHMEM
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(MEGAKERNEL_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"
#elif defined(MEGAKERNEL_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"
#else
// #include "tasks/ampere/task_header.cuh"
#include "tasks/autogen/task_header.cuh"
#endif

#define LIKELY(x)       __builtin_expect(!!(x), 1)
#define UNLIKELY(x)     __builtin_expect(!!(x), 0)

using bfloat16 = type::bfloat16_t;
using namespace megakernel::runtime;
using namespace kernel;
// Configurations for the MPK runtime
// #define MPK_MAX_NUM_BATCHED_REQUESTS 16
// #define MPK_MAX_NUM_BATCHED_TOKENS 64
// #define MPK_MAX_NUM_PAGES 1024
// #define MPK_PAGE_SIZE 64

#if defined(MEGAKERNEL_GRACE_HOPPER)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(MEGAKERNEL_GRACE_BLACKWELL)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#else
#define WORKER_NUM_THREADS 128
#define SINGLE_KERNEL_NUM_THREADS 128
#endif
#define INIT_NUM_THREADS 128

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

// #define MPK_ENABLE_VERBOSE
__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config);

__device__ __forceinline__ bool is_termination_event(size_t event_loc,
                                                     EventDesc e) {
  return (event_loc == 0);
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return (task_id >> 32);
}

__device__ __forceinline__ size_t get_task_position_index(TaskId task_id) {
  return (task_id & 0xffffffff);
}

__device__ __forceinline__ TaskId compute_task_id(size_t iteration_num,
                                                  size_t position_index) {
  return ((iteration_num << 32) | position_index);
}

__global__ void prepare_kernel(RuntimeConfig config,
                               int end_of_task_graph_event_pos) {
  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < 2 * config.num_workers;
       i += blockDim.x * gridDim.x) {
    config.worker_queue_last_ready_task_id[i] = 0;
  }
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_schedulers + 1;
       i += blockDim.x * gridDim.x) {
    config.sched_queue_last_ready_event_id[i] = 0;
    config.sched_queue_next_free_event_id[i] = 0;
  }
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }
  // Send event to scheduler[0]
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *config.infer_cnt = 0;

    assert(config.all_events[end_of_task_graph_event_pos].event_type == EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
}

__global__ void static_prepare_kernel(RuntimeConfig config) {
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }
}

__device__ __forceinline__ int get_rand_sched_id(size_t event_index,
                                                 int worker_id,
                                                 int num_workers,
                                                 int num_schedulers) {
  // const size_t seed = 0xac4c1b51;
  // size_t x = event_index * seed;
  // x ^= x >> 17;
  // x *= worker_id;
  //  x *= 0xed5ad4bb;
  // x ^= x >> 11;
  size_t x = worker_id;
  return x / ((num_workers + num_schedulers - 1) / num_schedulers);
}

__device__ __forceinline__ void
    get_first_last_ids(unsigned long long int num_elements,
                       unsigned long long int num_workers,
                       unsigned long long int my_id,
                       unsigned long long int *my_first_element,
                       unsigned long long int *my_last_element) {
  unsigned long long int num_elements_per_worker = num_elements / num_workers;
  unsigned long long int reminder = num_elements % num_workers;
  if (my_id < reminder) {
    *my_first_element = (num_elements_per_worker + 1) * my_id;
    *my_last_element = *my_first_element + num_elements_per_worker + 1;
  } else {
    *my_first_element = num_elements_per_worker * my_id + reminder;
    *my_last_element = *my_first_element + num_elements_per_worker;
  }
}

__device__ __forceinline__ void terminate_schedulers(RuntimeConfig config) {
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = 0; i < num_schedulers; i++) {
    // size_t last_event_id =
    //     atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    size_t last_event_id =
        atom_add_release_gpu_u64(&config.sched_queue_next_free_event_id[i], 1);
    st_relaxed_gpu_u64(
        &config.sched_queues[i][last_event_id % config.per_sched_queue_len], 0);
    // Use st.relaxed to make sure sched_queue updates are visible to scheduler
    // CTAs before incrementing its last_ready_event_id
    size_t old;
    do {
      // old = atomicCAS(&config.sched_queue_last_ready_event_id[i],
      //                 last_event_id,
      //                 last_event_id + 1);
      old = atom_cas_release_gpu_u64(&config.sched_queue_last_ready_event_id[i],
                                     last_event_id,
                                     last_event_id + 1);
    } while (old != last_event_id);
  }
}

// __device__ __forceinline__ void init_launch(RuntimeConfig config) {
//   // 只需要1个block负责
//   if (threadIdx.x == 0) {
//     *config.infer_cnt = 0;
//   }
// }

__device__ __forceinline__ void prepare_queue(RuntimeConfig config) {
  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < 2 * config.num_workers;
       i += blockDim.x * gridDim.x) {
    config.worker_queue_last_ready_task_id[i] = 0;
  }
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  int num_schedulers = config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_schedulers + 1;
       i += blockDim.x * gridDim.x) {
    config.sched_queue_last_ready_event_id[i] = 0;
    config.sched_queue_next_free_event_id[i] = 0;
  }
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }
  // Send event to scheduler[0]
  // 第 config.num_workers 个 block 负责
  int end_of_task_graph_event_pos = config.num_events - 1;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *config.infer_cnt = 0;

    assert(config.all_events[end_of_task_graph_event_pos].event_type == EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
}

__device__ __forceinline__ void worker_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(gridDim.x == config.num_workers);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
}

__device__ __forceinline__ void scheduler_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(config.num_workers <= MAX_NUM_WORKERS);
}

__device__ __forceinline__ void persistent_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int const num_schedulers = config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  assert(num_schedulers % num_schedulers_per_sm == 0);
  assert(gridDim.x == config.num_workers + num_schedulers / num_schedulers_per_sm);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
  // assert(blockDim.x >= 128);

  ///////////////////////////////////////////////////////////////////
  // int end_of_task_graph_event_pos = global_runtime_config[kernel_id].num_events - 1;
  // prepare_kernel<<<dim3(global_runtime_config[kernel_id].num_workers, 1, 1),
  //                  dim3(128, 1, 1)>>>(global_runtime_config,
  //                                     end_of_task_graph_event_pos);
  // if (blockIdx.x == config.num_workers) {
  //   init_launch(config);
  // }
  if (blockIdx.x < config.num_workers) {
    prepare_queue(config);
  }
}

__global__ __launch_bounds__(WORKER_NUM_THREADS, 1) 
void static_persistent_kernel(RuntimeConfig config) {
  // persistent_checker(config);
  #ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0, 1, (threadIdx.x % WORKER_NUM_THREADS == 0));
  
  #endif
  const int worker_id = blockIdx.x;

  int task_num = config.static_worker_tasks_index[worker_id][0];
  int *task_ids = &config.static_worker_tasks_index[worker_id][1];
  for (int i = 0; i < task_num ; i++) {
    // if (threadIdx.x == 0) {
    //   // printf("task_ids1[%d]: %d\n", worker_id, i);// 
    //   printf("task_ids2[%d]: %d\n", worker_id, task_ids[i]);// 
    // }
      
    int task_idx = task_ids[i]; // worker_id * 9 + i;
    if (task_idx > config.num_tasks)
      return;

    TaskDesc *task_desc = &config.all_tasks[task_idx];
    // Successfully fetched a new task

    size_t event_index = get_event_position_index(task_desc->dependent_event);
    EventDesc *dep_event_desc = &config.all_events[event_index];

    // if (threadIdx.x == 0) {
    //   if (task_desc->dependent_event != EVENT_INVALID_ID) {
    //     // Wait until the event has been triggered enough times
    //     EventId event_id = task_desc->dependent_event;
    //     assert(get_event_gpu_id(event_id) == config.my_gpu_id);
    //     size_t event_index = get_event_position_index(event_id);
        
    //     EventCounter needed_counts = static_cast<EventCounter>(config.all_event_num_triggers[event_index]);
    //     EventCounter actual_counts = 0;
    //     // 等待前置任务的 Event 计数达到预期值
    //     while (actual_counts < needed_counts) {
    //       actual_counts = ld_acquire_sys_u64(&config.all_event_counters[event_index]);
    //       // printf("dep(%d):(%d vs %d), ", event_index, actual_counts, needed_counts);
    //       __nanosleep(10);
    //     }
    //   }
    // }
    // __syncthreads();

  #ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_idx);
    }
  #endif
    _execute_task(task_desc, config); 
  #ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, task_idx);
    }
  #endif

    // // Trigger event
    // if (threadIdx.x == 0) {
    //   EventId event_id = task_desc->trigger_event;
    //   size_t event_index = get_event_position_index(event_id);
    //   EventCounter count = atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
    //   // printf("tri(%d):(%d), ", event_index, count);
    // }
  }
}

__device__ __forceinline__ void execute_worker(RuntimeConfig config) {
  // Make sure overall smem usage here do not exceed 3KB
  // last_task_pos: 8 B
  // next_task_pos: 8 B 
  //             = 16 B
  // remaining: 3056 B = 3*1024-16
  // printf("worker: %d.\n", blockIdx.x);
  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(32,
      (megakernel::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 16) / (int)(sizeof(TaskDesc) + sizeof(TaskId)));
  // printf("TASK_DESCS_BUFFER_LENGTH: %d.\n", TASK_DESCS_BUFFER_LENGTH);
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
  __shared__ size_t next_task_pos;
  __shared__ size_t last_task_pos;

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0, 1, (threadIdx.x % WORKER_NUM_THREADS == 0));
  size_t task_counter = 0;
#endif

  const int worker_id = blockIdx.x;
  if (threadIdx.x == 0) {
    next_task_pos = 0;
    last_task_pos = 0;
  }

  int queue_pos = 0, queue_len = 0;
  while (true) {
    // fetch next task from a task queue if task_descs is empty
    if (queue_pos == queue_len) {
      if (threadIdx.x == 0) {
        // Leader 线程检查全局队列是否有新任务
        while (next_task_pos == last_task_pos) {
          last_task_pos = ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id[worker_id]);
          if (next_task_pos < last_task_pos) {
            // printf("task_pos: %d, %d\n", next_task_pos,last_task_pos);
            break;
          }
          // printf("*");
          __nanosleep(10); // nanosleep to avoid overwhelming I/O
        }
        assert(next_task_pos + config.per_worker_queue_len > last_task_pos);
      }
      __syncthreads();
      int num_loaded_tasks = min((int)(last_task_pos - next_task_pos), TASK_DESCS_BUFFER_LENGTH);
      // Load task ids
      if (threadIdx.x < num_loaded_tasks) {
        task_ids[threadIdx.x] = ld_relaxed_gpu_u64(&config.worker_queues[worker_id][(next_task_pos + threadIdx.x) % config.per_worker_queue_len]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#ifdef MPK_ENABLE_VERBOSE
        for (int i = 0; i < num_loaded_tasks; i++) {
          printf(
              "[%d][FTCH] worker_id(%d) queue_idx(%d) next_task_pos(%llu"
              ") last_task_pos(%llu) "
              "task_id(%llu) task_type(%d) event_id(%llx) \n",
              config.my_gpu_id,
              worker_id,
              queue_idx,
              next_task_pos,
              last_task_pos,
              get_task_position_index(task_ids[i]),
              config.all_tasks[get_task_position_index(task_ids[i])].task_type,
              config.all_tasks[get_task_position_index(task_ids[i])].trigger_event);
        }
#endif
        next_task_pos += num_loaded_tasks;
      }
      // Load task descs
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128b copy-async
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE; i += blockDim.x) {
        int task_idx = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(config.all_tasks + get_task_position_index(task_ids[task_idx])) + offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
    }
    TaskDesc *task_desc = task_descs + queue_pos;
    // Make sure task is ready before start execution
    if (threadIdx.x == 0) {
      if (task_desc->dependent_event != EVENT_INVALID_ID) {
        // Wait until the event has been triggered enough times
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts = static_cast<EventCounter>(config.all_event_num_triggers[event_index]) * get_task_iteration_num(task_ids[queue_pos]);
        EventCounter actual_counts = 0;
        // 等待前置任务的 Event 计数达到预期值
        while (actual_counts < needed_counts) {
          actual_counts = ld_acquire_sys_u64(&config.all_event_counters[event_index]);
          __nanosleep(10);
        }
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_counter);
    }
#endif

    // Successfully fetched a new task
    if (UNLIKELY(task_desc->task_type == TASK_TERMINATE)) { return; }    // Terminate
    else if (UNLIKELY(task_desc->task_type == TASK_BEGIN_TASK_GRAPH)) {} // Do nothing
    else { _execute_task(task_desc, config); } // CJM
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, task_counter++);
    }
#endif

    // Trigger event
    if (threadIdx.x == 0) {
      EventId event_id = task_desc->trigger_event;
      size_t event_index = get_event_position_index(event_id);

      // Case 1: Trigger a local non-nvshmem event
      EventCounter count = atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
      int num_triggers = config.all_event_num_triggers[event_index];
#ifdef MPK_ENABLE_VERBOSE
      printf("[%d][DONE] worker_id(%d) iter_num(%llu) task_idx(%llu) "
              "event_id(%llu) "
              "event_type(local) count(%llu)\n",
              config.my_gpu_id,
              worker_id,
              get_task_iteration_num(task_ids[queue_pos]),
              get_task_position_index(task_ids[queue_pos]),
              event_id,
              count);
#endif

      // 如果是最后一个触发者，将 Event 加入 Scheduler 队列
      if ((count + 1) == static_cast<EventCounter>(num_triggers) * get_task_iteration_num(task_ids[queue_pos])) {
#ifdef MPK_ENABLE_PROFILING
        PROFILER_EVENT_START(TASK_SCHD_EVENTS, task_counter);
#endif
        EventDesc event_desc = config.all_events[event_index];
        // Add the event to the schedule_queue
        // Note that events launching massive tasks are scheduled to the global sched_queue
        if (event_desc.event_type != EVENT_EMPTY) { // Do nothing for empty event
          // 选择目标 scheduler（随机或全局）
          int sched_id = config.num_local_schedulers;
          if (event_desc.event_type != EVENT_LAUNCH_DEPENDENT_TASKS) {
            sched_id = worker_id / ((config.num_workers + config.num_local_schedulers - 1) / config.num_local_schedulers);
          }
          // 入队并通知
          size_t last_event_pos = atom_add_release_gpu_u64(&config.sched_queue_next_free_event_id[sched_id], 1);
          st_relaxed_gpu_u64(&config.sched_queues[sched_id][last_event_pos % config.per_sched_queue_len], event_index);
          // Use st.relaxed to make sure that the updated event_index is
          // visible to the scheduler CTA before updating its last_ready_event_id
          size_t old;
          do {
            old = atom_cas_release_gpu_u64(&config.sched_queue_last_ready_event_id[sched_id], last_event_pos, last_event_pos + 1);
          } while (old != last_event_pos);
        }
#ifdef MPK_ENABLE_PROFILING
        PROFILER_EVENT_END(TASK_SCHD_EVENTS, task_counter++);
#endif
      }
    }
    queue_pos += 1;
  }
}

// need to alter as there is only one warp per block
__device__ __forceinline__ void execute_scheduler(RuntimeConfig config, int offset) {
  int const num_schedulers = config.num_local_schedulers;
  // if we have more than 4 warps per thread block
  // only the first 4 warps will run schedulers
  // 应使用 warp 级同步，不能用 __syncthreads()
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  // printf("num_schedulers_per_sm: %d.\n", num_schedulers_per_sm);
  int const warp_id = threadIdx.x / 32;
  // CANNOT use syncthreads below
  if (threadIdx.x % 32 == 0 && warp_id < num_schedulers_per_sm) {
    int const sched_id = blockIdx.x * num_schedulers_per_sm + warp_id + offset;
    // if (threadIdx.x == 0) {
    //   int sched_id = (blockIdx.x - config.num_workers);
    size_t iteration_num = 0;
    EventId *sched_queues[2];
    int sched_queue_ids[2];
    sched_queues[0] = config.sched_queues[sched_id];       // 本地队列 分配给该 scheduler 的 events
    sched_queues[1] = config.sched_queues[num_schedulers]; // 所有 scheduler 共同处理的大批量任务 events，EVENT_LAUNCH_DEPENDENT_TASKS，local schedulers also (collectively) process events from the global queue
    sched_queue_ids[0] = sched_id;
    sched_queue_ids[1] = num_schedulers;
   
    unsigned long long int my_first_worker, my_last_worker;
    get_first_last_ids(config.num_workers, config.num_local_schedulers, sched_id,
                        &my_first_worker, &my_last_worker);

    // ONLY can run when comment this chunk
#ifdef MPK_ENABLE_VERBOSE
    printf("[SCHD] sched_id(%d) first_worker(%llu) last_worker(%llu)\n",
           sched_id,
           my_first_worker,
           my_last_worker);
#endif
    size_t cur_event_pos[2] = {0, 0}, last_event_pos[2] = {0, 0};
    size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
    for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; i++) {
      worker_queue_next_free_task_pos[i] = 0;
    }

    int next_worker = my_first_worker;
    int queue_idx = 0;
    while (true) {
      while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
        last_event_pos[queue_idx] = ld_acquire_gpu_u64(&config.sched_queue_last_ready_event_id[sched_queue_ids[queue_idx]]);
        if (cur_event_pos[queue_idx] < last_event_pos[queue_idx]) {
          // printf("event_pos: %d, %d", cur_event_pos[queue_idx], last_event_pos[queue_idx]);
          break;
        } else {
          queue_idx = (queue_idx == 1) ? 0 : 1;
        }
        // printf("-");
        // nanosleep to avoid overwhelming I/O
        __nanosleep(10);
      }
      // Make sure the schedule queue is not overflow
      assert(cur_event_pos[queue_idx] + config.per_sched_queue_len > last_event_pos[queue_idx]);
      // Launch new tasks
      // Use ld.acquire to read latest events
      EventId event_id = ld_relaxed_gpu_u64(&sched_queues[queue_idx][cur_event_pos[queue_idx] % config.per_sched_queue_len]);
      EventDesc e = config.all_events[event_id];
      if (is_termination_event(event_id, e)) {
        // terminate all workers
        if (sched_id < config.num_local_schedulers) {
          for (int i = my_first_worker; i < my_last_worker; i++) {
            size_t last_task_id = worker_queue_next_free_task_pos[i - my_first_worker]++;
            st_relaxed_gpu_u64(&config.worker_queues[i][last_task_id % config.per_worker_queue_len], 0);
            atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[i], 1);
          }
        }
        return;
      }
      // This is the ending task of the current task graph
      if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
#ifdef MPK_ENABLE_VERBOSE
        printf("[SCHD] END_OF_TASK_GRAPH\n");
#endif
        // Check if we want to continue
        if (*config.infer_cnt != 0) {
          // printf("terminate_schedulers EVENT_END_OF_TASK_GRAPH.\n");
          terminate_schedulers(config);
        } else {
          // printf("hello EVENT_END_OF_TASK_GRAPH.\n");
          *config.infer_cnt = 1;
          // Launch task 1 (begin_task_graph) for the next iteration
          size_t last_task_id = worker_queue_next_free_task_pos[next_worker - my_first_worker]++;
          st_relaxed_gpu_u64(&config.worker_queues[next_worker][last_task_id % config.per_worker_queue_len],
                             compute_task_id(iteration_num + 1, 1 /*begin_task_graph*/));
          // Use st.relaxed to make sure writes to worker_queues is visible to
          // worker CTAs before we increase its last_ready_task_id
          atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[next_worker], 1);
#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD]EVENT_END_OF_TASK_GRAPH schd_id(%d) "
                 "iter_num(%llu) task_idx(1) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num + 1,
                 next_worker,
                 last_task_id + 1);
#endif
          next_worker = (next_worker == my_last_worker - 1) ? my_first_worker : next_worker + 1;
        }
      } else if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
        // printf("hello EVENT_LAUNCH_DEPENDENT_TASKS.\n");
        iteration_num = iteration_num + 1;
        // assign event in a round-robin fashion
        // Split event across local schedulers
        assert(sched_id < config.num_local_schedulers);
        for (size_t i = 0; i < (e.last_task_id - e.first_task_id + config.num_workers - 1) / config.num_workers; i++) {
          for (size_t j = my_first_worker; j < my_last_worker; j++) {
            size_t position_index = e.first_task_id + i * config.num_workers + j;
            if (position_index < e.last_task_id) {
              size_t last_task_id = worker_queue_next_free_task_pos[next_worker - my_first_worker]++;
              st_relaxed_gpu_u64(&config.worker_queues[next_worker][last_task_id % config.per_worker_queue_len],
                                 compute_task_id(iteration_num, position_index));
              // Use st.relaxed to make sure writes to worker_queues is visible
              // to worker CTAs before we increase its last_ready_task_id
              atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[next_worker], 1);
#ifdef MPK_ENABLE_VERBOSE
              if (sched_id == 0) {
                printf("[%d][SCHD] EVENT_LAUNCH_DEPENDENT_TASKS schd_id(%d) "
                       "iter_num(%llu) task_idx(%llu) "
                       "worker_id(%d) "
                       "worker_last_ready_pos(%llu)"
                       "event_id(%llu)"
                       "event_range(%llu-%llu)\n",
                       config.my_gpu_id,
                       sched_id,
                       iteration_num,
                       position_index,
                       next_worker,
                       last_task_id + 1,
                       event_id,
                       e.first_task_id,
                       e.last_task_id);
              }
#endif
              next_worker = (next_worker == my_last_worker - 1)? my_first_worker : next_worker + 1;
            }
          }
        }
      } else {
        // printf("hello main: %d.\n", e.event_type);
        for (size_t i = e.first_task_id; i < e.last_task_id; i++) {
          // 选择下一个 worker（Round-Robin）
          size_t last_task_id = worker_queue_next_free_task_pos[next_worker - my_first_worker]++;
          // 写入 worker 的任务队列
          st_relaxed_gpu_u64(&config.worker_queues[next_worker][last_task_id % config.per_worker_queue_len],
                             compute_task_id(iteration_num, i));
          // Use st.relaxed to make sure writes to worker_queues is visible to
          // worker CTAs before we increase its last_ready_task_id
          // 通知 worker 有新任务（release 语义）
          atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[next_worker], 1);

#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD] EXECUTE_TASK schd_id(%d) iter_num(%llu) "
                 "task_idx(%llu) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num,
                 i,
                 next_worker,
                 last_task_id + 1);
#endif

          next_worker = (next_worker == my_last_worker - 1) ? my_first_worker : next_worker + 1;
        }
      }
      cur_event_pos[queue_idx] += 1;
    }
  }
}

__global__ __launch_bounds__(WORKER_NUM_THREADS, 1) 
void persistent_kernel(RuntimeConfig config) {
  persistent_checker(config);
  // printf("<%d,%d>", gridDim.x, blockIdx.x);
  if (blockIdx.x < config.num_workers) {
    // printf("execute_worker: (%d.%d)", blockIdx.x, config.num_workers);
    execute_worker(config);
  } else {
    // printf("execute_scheduler: (%d.%d).", blockIdx.x, config.num_workers);
    execute_scheduler(config, -(4 * config.num_workers));
  }
}

__global__ __launch_bounds__(WORKER_NUM_THREADS, 1) void worker_kernel(RuntimeConfig config) {
  worker_checker(config);
  execute_worker(config);
}

__global__ void scheduler_kernel(RuntimeConfig config) {
  scheduler_checker(config);
  execute_scheduler(config, 0);
}

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#ifdef USE_NVSHMEM
  dst_ptr = nvshmem_malloc(size);
#else
  cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
#ifdef USE_NVSHMEM
  nvshmem_free(ptr);
#else
  cudaFree(ptr);
#endif
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(int kernel_id,
                                    std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id);

static int used_kernel_num = 0;
static RuntimeConfig global_runtime_config[20];

extern "C" void init_persistent_kernel(int kernel_id,
                                       std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers) {

  global_runtime_config[kernel_id].num_workers = num_workers;
  global_runtime_config[kernel_id].num_local_schedulers = num_local_schedulers;
  global_runtime_config[kernel_id].num_remote_schedulers = num_remote_schedulers;
  global_runtime_config[kernel_id].profiler_buffer = profiler_buffer;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  cudaSetDevice(my_rank);

#ifdef USE_NVSHMEM
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);
#else
  int mype = 0;
  int npes = 1;
#endif

  global_runtime_config[kernel_id].infer_cnt = gpu_malloc<int>(sizeof(int));
  global_runtime_config[kernel_id].per_worker_queue_len = 1024;
  global_runtime_config[kernel_id].per_sched_queue_len = 1024;
  global_runtime_config[kernel_id].num_gpus = npes;
  global_runtime_config[kernel_id].my_gpu_id = mype;
  global_runtime_config[kernel_id].num_graphs = 1;
  global_runtime_config[kernel_id].split_worker_scheduler = false;
  global_runtime_config[kernel_id].is_static_schedule = true;

  std::vector<FullTaskDesc> all_fulltasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(kernel_id, all_fulltasks, all_events, first_tasks, npes, mype);
  
  std::vector<TaskDesc> all_tasks;
  for (auto const &ft : all_fulltasks) {
    TaskDesc task_desc(ft);
    // if (ft.task_type == TASK_PAGED_ATTENTION_SPLIT_KV_SM100 || ft.task_type
    // == TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100) {
    //   printf("ft.kv_idx %d\n", ft.kv_idx);
    //   printf("ft.merge_task_offset %d\n", ft.merge_task_offset);
    // }
    // Reinterpret part of TaskDesc to save xfer_size information
    if (ft.task_type == TASK_NVSHMEM_COPY) {
      int size_in_bytes = 2;
      for (int i = 0; i < ft.inputs[0].num_dims; i++) {
        size_in_bytes *= ft.inputs[0].dim[i];
      }
      task_desc.task_metadata.xfer_size_in_bytes = size_in_bytes;
    }
    all_tasks.push_back(task_desc);
  }

  if (global_runtime_config[kernel_id].is_static_schedule == true) {
    global_runtime_config[kernel_id].num_tasks = all_tasks.size();
    int num_workers = global_runtime_config[kernel_id].num_workers;
    int tasks_each_worker = (all_tasks.size() + num_workers - 1) / num_workers;
    int capacity_each_worker = tasks_each_worker * 1.5;
    // printf("tasks_each_worker: %d.\n", tasks_each_worker);

    // 按dep event对task分组
    std::vector<std::vector<int>> event_task_ids;
    event_task_ids.resize(all_events.size());
    for (int i=0; i<all_tasks.size(); i++) {
      TaskDesc task_desc = all_tasks[i];
      if (task_desc.task_type != TASK_TERMINATE && task_desc.task_type != TASK_BEGIN_TASK_GRAPH) {
        if (task_desc.dependent_event != EVENT_INVALID_ID) {
          event_task_ids[task_desc.dependent_event].push_back(i);
        }
        else {
          event_task_ids[0].push_back(i);
        }
      }
    }
    ///////////////////////////////////////////////
    // Static Scheduling Scheme
    std::vector<std::vector<int>> host_tasks_index;
    host_tasks_index.resize(num_workers);
    for (int i=0; i<num_workers; i++) {
      host_tasks_index[i].resize(capacity_each_worker);
      host_tasks_index[i][0] = 0; // 0 for cnt
    }

    // // 1. 按顺序直接赋值
    // for (int i=0; i<num_workers; i++) {
    //   int cnt = 0;
    //   for (int j=0; j<tasks_each_worker; j++) {
    //     int task_idx = i*tasks_each_worker+j;
    //     if (task_idx < all_tasks.size()) {
    //       cnt++;
    //       host_tasks_index[i][j+1] = task_idx;
    //     }
    //   }
    //   host_tasks_index[i][0] = cnt;
    // }

    // 2. 按event分组填充task到worker
    int wid = 0;
    for (int ei=0; ei<event_task_ids.size(); ei++) {
      int task_id = 0;
      int task_num = event_task_ids[ei].size();
      if (task_num == 0) continue;
      int each_worker_num = task_num / num_workers; // (task_num + num_workers - 1) / num_workers;
      int first_round_num = each_worker_num * num_workers;
      printf("ei: %d, %d, %d.\n", ei, task_num, each_worker_num);

      // 均等分
      while (task_id < first_round_num) {
        for (int j=0; j < each_worker_num; j++) {
          // printf("worker %d, %d, %d, %d.\n", wid, j, host_tasks_index[wid][0], event_task_ids[ei][task_id]);
          host_tasks_index[wid][host_tasks_index[wid][0]+1] = event_task_ids[ei][task_id];
          host_tasks_index[wid][0]++; task_id++;
          if (task_id == first_round_num) {
            // printf("break.");
            break;
          }
        }
        wid = (wid+1) % num_workers;
      }
      // 剩余的逐个分
      while (task_id < task_num) {
        for (int j=0; j<1; j++) {
          host_tasks_index[wid][host_tasks_index[wid][0]+1] = event_task_ids[ei][task_id];
          host_tasks_index[wid][0]++; task_id++;
          if (task_id == task_num) {
            // printf("break.");
            break;
          }
        }
        wid = (wid+1) % num_workers;
      }
    }

    // 前置依赖免检标记
      

    for (int i=0; i<all_tasks.size(); i++) {
      TaskDesc task_desc = all_tasks[i];
      printf("task_desc[%d]: type %d, block(%d,%d,%d), dep %d, tri %d, varid %d.\n", i, task_desc.task_type, task_desc.bx, task_desc.by, task_desc.bz, task_desc.dependent_event, task_desc.trigger_event, task_desc.variant_id);
    }
    for (int i=0; i<all_events.size(); i++) {
      EventDesc event_desc = all_events[i];
      printf("event_desc[%d]: type %d, tri %d, task (%d, %d).\n", i, event_desc.event_type, event_desc.num_triggers, event_desc.first_task_id, event_desc.last_task_id);
    }
    for (int i=0; i<event_task_ids.size(); i++) {
      printf("event_group[%d]-(%d): ", i, event_task_ids[i].size());
      for (int j=0; j<event_task_ids[i].size(); j++) {
        printf("%d, ", event_task_ids[i][j]);
      }
      printf("\n");
    }
    for (int i=0; i<num_workers; i++) {
      int num = host_tasks_index[i][0];
      printf("worker[%d]-(%d): ", i, num);
      for (int j=0; j<num; j++) {
        printf("%d, ", host_tasks_index[i][j+1]);
      }
      printf("\n");
    }
    ////////////////////////////////////////////////

    std::vector<int*> host_tasks_index_arr;
    for (int i = 0; i < num_workers; i++) {
      int *device_tasks_index = gpu_malloc<int>(capacity_each_worker * sizeof(int));
      cudaMemcpy(device_tasks_index, host_tasks_index[i].data(), capacity_each_worker * sizeof(int), cudaMemcpyHostToDevice);
      host_tasks_index_arr.push_back(device_tasks_index);
    }

    global_runtime_config[kernel_id].static_worker_tasks_index = gpu_malloc<int*>(num_workers * sizeof(int*));
    cudaMemcpy(global_runtime_config[kernel_id].static_worker_tasks_index,
               host_tasks_index_arr.data(),
               num_workers * sizeof(int*), cudaMemcpyHostToDevice);
  }

  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  global_runtime_config[kernel_id].worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) * sizeof(unsigned long long int));
  //  Initialize scheduler queue last event id
  //  We maintain one extra scheduler queue for the global scheduler
  global_runtime_config[kernel_id].sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) * sizeof(unsigned long long int));
  global_runtime_config[kernel_id].sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) * sizeof(unsigned long long int));
  //  Initialize all event counters
  global_runtime_config[kernel_id].all_event_counters = gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
  global_runtime_config[kernel_id].all_event_num_triggers = gpu_malloc<int>(all_events.size() * sizeof(int));

  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  cudaMemcpy(global_runtime_config[kernel_id].all_event_num_triggers,
             host_all_event_counters.data(),
             all_events.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  // cudaMemset(global_runtime_config[kernel_id].all_event_counters,
  //            0,
  //            all_events.size() * sizeof(EventCounter));
  //  Initialize all tasks
  global_runtime_config[kernel_id].all_tasks = gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  cudaMemcpy(global_runtime_config[kernel_id].all_tasks,
             all_tasks.data(),
             all_tasks.size() * sizeof(TaskDesc),
             cudaMemcpyHostToDevice);
  // Initialize all events
  global_runtime_config[kernel_id].num_events = (int)all_events.size();
  global_runtime_config[kernel_id].all_events = gpu_malloc<EventDesc>(all_events.size() * sizeof(EventDesc));
  cudaMemcpy(global_runtime_config[kernel_id].all_events,
             all_events.data(),
             all_events.size() * sizeof(EventDesc),
             cudaMemcpyHostToDevice);
  // Initialize worker queues
  {
    std::vector<TaskId *> host_worker_queues;
    for (int i = 0; i < (num_workers * 2); i++) {
      TaskId *worker_queue = gpu_malloc<TaskId>(
          global_runtime_config[kernel_id].per_worker_queue_len * sizeof(TaskId));
      host_worker_queues.push_back(worker_queue);
    }
    global_runtime_config[kernel_id].worker_queues = gpu_malloc<TaskId *>((num_workers * 2) * sizeof(TaskId *));
    cudaMemcpy(global_runtime_config[kernel_id].worker_queues,
               host_worker_queues.data(),
               (num_workers * 2) * sizeof(TaskId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize scheduler queues
  {
    std::vector<EventId *> host_sched_queues;
    for (int i = 0; i < (num_schedulers + 1); i++) {
      EventId *sched_queue = gpu_malloc<EventId>(global_runtime_config[kernel_id].per_sched_queue_len * sizeof(EventId));
      host_sched_queues.push_back(sched_queue);
    }
    global_runtime_config[kernel_id].sched_queues = gpu_malloc<EventId *>((num_schedulers + 1) * sizeof(EventId *));
    cudaMemcpy(global_runtime_config[kernel_id].sched_queues,
               host_sched_queues.data(),
               (num_schedulers + 1) * sizeof(EventId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize first tasks
  {
    global_runtime_config[kernel_id].first_tasks = gpu_malloc<TaskId>(first_tasks.size() * sizeof(TaskId));
    cudaMemcpy(global_runtime_config[kernel_id].first_tasks,
               first_tasks.data(),
               first_tasks.size() * sizeof(TaskId),
               cudaMemcpyHostToDevice);
  }

  // Set configuration for kernels
  cudaFuncSetAttribute(worker_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  cudaFuncSetAttribute(scheduler_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  cudaFuncSetAttribute(persistent_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  cudaFuncSetAttribute(static_persistent_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  // Create worker and scheduler streams
  cudaStreamCreate(&global_runtime_config[kernel_id].worker_stream);
  cudaStreamCreate(&global_runtime_config[kernel_id].scheduler_stream);
}

void print_smem_size() {
  int device_id = 0;
  cudaSetDevice(device_id);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  // Print GPU shared memory hardware limits (all in English)
  printf("=== GPU Shared Memory Hardware Limits ===\n");
  printf("GPU Model: %s\n", prop.name);
  // printf("Max Dynamic Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlockDynamic / 1024);
  if (prop.sharedMemPerBlock < MAX_DYNAMIC_SHARED_MEMORY_SIZE) {
    printf("Warning: Dynamic shared memory may be insufficient!\n");
  }
  printf("Max Total Shared Memory per Block (Static + Dynamic): %lu KB, prepare to allocate %d KB (Dynamic).\n", prop.sharedMemPerBlock / 1024, MAX_DYNAMIC_SHARED_MEMORY_SIZE / 1024);
  printf("Total Shared Memory per SM: %lu KB\n", prop.sharedMemPerMultiprocessor / 1024);
}

// Entry point for C/C++
// TODO: change launch config
extern "C" void launch_persistent_kernel(int kernel_id, int batch_size) {
  // int device;
  // cudaGetDevice(&device);
  // int sm_count;
  // cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  global_runtime_config[kernel_id].batch_size = batch_size;

  int num_schedulers = global_runtime_config[kernel_id].num_local_schedulers +
                       global_runtime_config[kernel_id].num_remote_schedulers;
  if (global_runtime_config[kernel_id].split_worker_scheduler) {
    //  Prepare next persistent kernel by resetting queue pointers
    int end_of_task_graph_event_pos = global_runtime_config[kernel_id].num_events - 1;
    prepare_kernel<<<dim3(global_runtime_config[kernel_id].num_workers, 1, 1),
                     dim3(128, 1, 1)>>>(global_runtime_config[kernel_id],
                                        end_of_task_graph_event_pos);
    cudaDeviceSynchronize();

    // The split kernel does not support NVSHMEM because
    // nvshmemx_collective_launch launches kernels sequentially, which blocks
    // the interaction between the worker kernel and the scheduler kernel
    worker_kernel<<<dim3(global_runtime_config[kernel_id].num_workers, 1, 1),
                    dim3(WORKER_NUM_THREADS, 1, 1),
                    MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                    global_runtime_config[kernel_id].worker_stream>>>(
        global_runtime_config[kernel_id]);

    scheduler_kernel<<<dim3(global_runtime_config[kernel_id].num_local_schedulers, 1, 1),
                       dim3(32, 1, 1),
                       0 /*smem*/,
                       global_runtime_config[kernel_id].scheduler_stream>>>(
        global_runtime_config[kernel_id]);
  } else {
    // printf("a single persistent kernel\n");
    if (global_runtime_config[kernel_id].is_static_schedule == true) {
      int end_of_task_graph_event_pos = global_runtime_config[kernel_id].num_events - 1;
      static_prepare_kernel<<<dim3(global_runtime_config[kernel_id].num_workers, 1, 1),
                              dim3(128, 1, 1)>>>(global_runtime_config[kernel_id]);
      cudaDeviceSynchronize();

      static_persistent_kernel<<<dim3(global_runtime_config[kernel_id].num_workers, 1, 1),
          dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
          MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
          global_runtime_config[kernel_id]);      
    }
    else {
      int num_sms_to_use = global_runtime_config[kernel_id].num_workers + num_schedulers / 4;
      #ifdef USE_NVSHMEM
          void *args[] = {&global_runtime_config[kernel_id]};
          nvshmemx_collective_launch((void const *)persistent_kernel,
                                     dim3(num_sms_to_use, 1, 1),
                                     dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                                     args,
                                     MAX_DYNAMIC_SHARED_MEMORY_SIZE /*sharedmem*/,
                                     0 /*stream*/);
      #else
      // print_smem_size();
      persistent_kernel<<<dim3(num_sms_to_use, 1, 1),
                          dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                          MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
          global_runtime_config[kernel_id]);      
    }
#endif
  }
  // cudaError_t err = cudaDeviceSynchronize();
  // if (err != cudaSuccess) {
  //   printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  // }
  // printf("Finished Launch Persistent Kernel\n");
}

extern "C" void finalize_persistent_kernel(int kernel_id) {
  gpu_free(global_runtime_config[kernel_id].worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config[kernel_id].sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config[kernel_id].sched_queue_next_free_event_id);
  gpu_free(global_runtime_config[kernel_id].all_event_counters);
  gpu_free(global_runtime_config[kernel_id].all_event_num_triggers);
  gpu_free(global_runtime_config[kernel_id].all_tasks);
  gpu_free(global_runtime_config[kernel_id].all_events);

  gpu_free(global_runtime_config[kernel_id].infer_cnt);

  int num_workers = global_runtime_config[kernel_id].num_workers;

  if (global_runtime_config[kernel_id].is_static_schedule == true) {
    std::vector<int*> host_tasks_index(num_workers);
    cudaMemcpy(host_tasks_index.data(),
             global_runtime_config[kernel_id].static_worker_tasks_index,
             num_workers * sizeof(int *),
             cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_workers; i++) {
      gpu_free(host_tasks_index[i]);
    }
    gpu_free(global_runtime_config[kernel_id].static_worker_tasks_index);
  }

  std::vector<TaskId *> host_worker_queues(num_workers * 2);
  cudaMemcpy(host_worker_queues.data(),
             global_runtime_config[kernel_id].worker_queues,
             (num_workers * 2) * sizeof(TaskId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 2 * num_workers; i++) {
    gpu_free(host_worker_queues[i]);
  }
  gpu_free(global_runtime_config[kernel_id].worker_queues);
  int num_schedulers = global_runtime_config[kernel_id].num_local_schedulers +
                       global_runtime_config[kernel_id].num_remote_schedulers;
  std::vector<EventId *> host_sched_queues(num_schedulers + 1);
  cudaMemcpy(host_sched_queues.data(),
             global_runtime_config[kernel_id].sched_queues,
             (num_schedulers + 1) * sizeof(EventId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_schedulers + 1; i++) {
    gpu_free(host_sched_queues[i]);
  }
  gpu_free(global_runtime_config[kernel_id].sched_queues);
  gpu_free(global_runtime_config[kernel_id].first_tasks);
#ifdef USE_NVSHMEM
  nvshmem_barrier_all();
  nvshmem_finalize();
#endif
  // Free worker and scheduler streams
  cudaStreamDestroy(global_runtime_config[kernel_id].worker_stream);
  cudaStreamDestroy(global_runtime_config[kernel_id].scheduler_stream);
}
