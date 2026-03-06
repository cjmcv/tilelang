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
// #include "tasks/common/copy_sm80.cuh"
// #include "tasks/common/bfloat16.h"
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

// using bfloat16 = type::bfloat16_t;
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

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

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
  // int step = *config.step;
  const int worker_id = blockIdx.x;
  int task_num = config.static_worker_tasks_index[worker_id][0];
  int *task_ids = &config.static_worker_tasks_index[worker_id][1];

  for (int i = 0; i < task_num ; i++) {
    int task_idx = task_ids[i]; // worker_id * 9 + i;
    TaskDesc *task_desc = &config.all_tasks[task_idx];
    size_t event_index = get_event_position_index(task_desc->dependent_event);
    EventDesc *dep_event_desc = &config.all_events[event_index];
    if (threadIdx.x == 0) {
      // printf("step: %d.\n", *config.step);
      if (task_desc->dependent_event != EVENT_INVALID_ID) {
        // Wait until the event has been triggered enough times
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        
        EventCounter needed_counts = static_cast<EventCounter>(config.all_event_num_triggers[event_index]);
        EventCounter actual_counts = 0;
        // 等待前置任务的 Event 计数达到预期值
        while (actual_counts < needed_counts) {
          actual_counts = ld_acquire_sys_u64(&config.all_event_counters[event_index]);
          // printf("dep(%d):(%d vs %d), ", event_index, actual_counts, needed_counts);
          __nanosleep(10);
        }
      }
    }
    __syncthreads();

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

    // Trigger event
    if (threadIdx.x == 0) {
      EventId event_id = task_desc->trigger_event;
      size_t event_index = get_event_position_index(event_id);
      EventCounter count = atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
      // printf("tri(%d):(%d), ", event_index, count);
    }
  }
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
  // printf("meta_tensors_size: %d.\n", meta_tensors.size());
  global_runtime_config[kernel_id].step = nullptr;
  if (meta_tensors.size() > 0) { global_runtime_config[kernel_id].step = (int*)meta_tensors[0]; }
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

  // is_static_schedule
  {
    global_runtime_config[kernel_id].num_tasks = all_tasks.size();
    int num_workers = global_runtime_config[kernel_id].num_workers;
    int tasks_each_worker = (all_tasks.size() + num_workers - 1) / num_workers;
    int capacity_each_worker = (tasks_each_worker + 1) * 1.5; // each_worker: 0:len, 1:task0, 2:task1... capacity=task_num+1
    // printf("capacity_each_worker: %d.\n", capacity_each_worker);

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
    //       host_tasks_index[i][j + 1] = task_idx;
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

      int base_num = task_num / num_workers; // (task_num + num_workers - 1) / num_workers;
      int remainder = task_num % num_workers;
      
      int tasks_assigned = 0;
      int idx = 0;  // 对应该轮的第几个worker，每个event重置1次，使前remainder个worker多拿一个task
      while (tasks_assigned < task_num) {
        int target_count = base_num + (idx < remainder ? 1 : 0);          // wid 当前该拿多少
        int actual_count = min(target_count, task_num - tasks_assigned);  // 边界处理
        // printf("[%d]actual_count: %d.\n", ei, actual_count);

        // TaskDesc task_desc = all_tasks[event_task_ids[ei][tasks_assigned]];
        // if (task_desc.task_type == TASK_SILU_MUL) {
        //   static int cnt = 0;
        //   if (cnt < (20-4)*2 && wid != 0 && wid != 17 && wid != 18 && wid != 19) {
        //     cnt++; 
        //     wid = (wid + 1) % num_workers;
        //     continue;            
        //   }
        // }

        for (int i = 0; i < actual_count; i++) {
          host_tasks_index[wid][host_tasks_index[wid][0] + 1] = event_task_ids[ei][tasks_assigned + i];
          host_tasks_index[wid][0]++;
        }
        tasks_assigned += actual_count;
        idx++;
        wid = (wid + 1) % num_workers;
      }
    }

    // 前置依赖免检标记
    // TODO  

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

  cudaMemset(global_runtime_config[kernel_id].all_event_counters, 0, 
    sizeof(EventCounter) * global_runtime_config[kernel_id].num_events);
  static_persistent_kernel<<<dim3(global_runtime_config[kernel_id].num_workers, 1, 1),
      dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
      MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
      global_runtime_config[kernel_id]);      

  // cudaError_t err = cudaDeviceSynchronize();
  // if (err != cudaSuccess) {
  //   printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  // }
  // printf("Finished Launch Persistent Kernel\n");
}

extern "C" void finalize_persistent_kernel(int kernel_id) {
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
