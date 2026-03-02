#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace megakernel::runtime;
size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

void construct_task_graph(int num_gpus,
                          int my_gpu_id,
                          std::vector<FullTaskDesc> &all_tasks,
                          std::vector<EventDesc> &all_events,
                          std::vector<TaskId> &first_tasks,
                          std::map<std::string, void*> const &all_tensors) {
  std::filesystem::path file_path(__FILE__);
  std::ifstream json_file(file_path.parent_path().string()+"/task_graph.json");
  nlohmann::json json_task_graph;
  json_file >> json_task_graph;
  for (json const &task : json_task_graph["all_tasks"]) {
    FullTaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    task_desc.task_metadata.request_id = task.at("request_id").get<int>();
    task_desc.task_metadata.expert_offset = task.at("expert_offset").get<int>();
    task_desc.task_metadata.kv_idx = task.at("kv_idx").get<int>();
    task_desc.task_metadata.merge_task_offset = task.at("merge_task_offset").get<int>();
    if (task.at("trigger_event").is_number_integer()) {
      task_desc.trigger_event = task.at("trigger_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    task_desc.num_inputs = 0;
    for (json const &tensor : task["inputs"]) {
      TensorDesc input;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      input.base_ptr = static_cast<char*>(all_tensors.at(name)); // +offset
      input.bx = tensor.at("bx").get<int>();
      input.by = tensor.at("by").get<int>();
      input.bz = tensor.at("bz").get<int>();
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      input.num_dims = tensor.at("dims").size();
      input.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < input.num_dims; i++) {
        input.dim[i] = tensor["dims"][i].get<int>();
        input.stride[i] = tensor["strides"][i].get<int>();
      }
      task_desc.inputs[task_desc.num_inputs++] = input;
    }
    task_desc.num_outputs = 0;
    for (json const &tensor : task["outputs"]) {
      TensorDesc output;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      output.base_ptr = static_cast<char*>(all_tensors.at(name)); // +offset
      output.bx = tensor.at("bx").get<int>();
      output.by = tensor.at("by").get<int>();
      output.bz = tensor.at("bz").get<int>();
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      output.num_dims = tensor.at("dims").size();
      output.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < output.num_dims; i++) {
        output.dim[i] = tensor["dims"][i];
        output.stride[i] = tensor["strides"][i];
      }
      task_desc.outputs[task_desc.num_outputs++] = output;
    }
    #ifdef MPK_ENABLE_TMA
    if (task.at("task_type") > TASK_HOPPER_TASK_BEGIN && task.at("task_type") < TASK_HOPPER_TASK_END) {
      create_tma_desc_by_task(task_desc);
    }
    if (task.at("task_type") > TASK_SM100_TMA_START_TASK && task.at("task_type") < TASK_SM100_TMA_END_TASK) {
      create_tma_desc_by_task(task_desc);
    }
    #endif
    all_tasks.push_back(task_desc);
  }
  for (json const &e : json_task_graph["all_events"]) {
    EventType event_type = static_cast<EventType>(e.at("event_type").get<int>());
    int num_triggers = e.at("num_triggers").get<int>();
    int first_task_id = e.at("first_task_id").get<int>();
    int last_task_id = e.at("last_task_id").get<int>();
    all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));
  }
  for (json const &t : json_task_graph["first_tasks"]) {
    first_tasks.push_back(t.get<int>());
  }
}

void adjust_params_with_kernel_id(int kernel_id, std::map<std::string, void*> &all_tensors);
static void _init_persistent_kernel(int kernel_id,
                                    std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id) {
  assert(num_gpus = 1);
  std::map<std::string, void*> all_tensors;
  char *q = (char*)(0x504a08200);
  all_tensors["q"] = q;
  char *k = (char*)(0x504e00000);
  all_tensors["k"] = k;
  char *v = (char*)(0x505e00000);
  all_tensors["v"] = v;
  char *edge = (char*)(0x504a09200);
  all_tensors["edge"] = edge;
  char *mask = (char*)(0x504a09400);
  all_tensors["mask"] = mask;
  char *glse = (char*)(0x504a00000);
  all_tensors["glse"] = glse;
  char *out_partial = (char*)(0x504a00200);
  all_tensors["out_partial"] = out_partial;
  char *attn_out = (char*)(0x504a19400);
  all_tensors["attn_out"] = attn_out;
  all_tensors["nullptr"] = nullptr;
  adjust_params_with_kernel_id(kernel_id, all_tensors);
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
  cudaDeviceSynchronize();
}

__device__ __forceinline__
void _execute_task(TaskDesc const* task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_GQA_DECODE && task_desc->variant_id == 0) {
      kernel::gqa_decode_kernel<bfloat16, 128, 0, 1, 16, 8, 128>(
      task_desc->bx, task_desc->by, task_desc->bz,
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->input_ptrs[3],
      runtime_config.step, // task_desc->input_ptrs[4],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      task_desc->output_ptrs[2]);

  }
  else if (task_desc->task_type == TASK_GQA_DECODE && task_desc->variant_id == 1) {
      kernel::gqa_decode_kernel<bfloat16, 128, 1, 1, 16, 8, 128>(
      task_desc->bx, task_desc->by, task_desc->bz,
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->input_ptrs[3],
      runtime_config.step, // task_desc->input_ptrs[4],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      task_desc->output_ptrs[2]);

  }
}
// Plugin 
void adjust_params_with_kernel_id(int kernel_id, std::map<std::string, void*> &all_tensors) {
}

#include <Python.h>
#include <cuda_runtime.h>

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer;
  std::vector<void*> meta_tensors;
  int kernel_id, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "iOOiiii", &kernel_id, &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(kernel_id, meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers);
  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  int kernel_id = 0, batch_size = 0, step = 0;
  if (!PyArg_ParseTuple(args, "iii", &kernel_id, &batch_size, &step)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  launch_persistent_kernel(kernel_id, batch_size, step);

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  int kernel_id = 0;
  if (!PyArg_ParseTuple(args, "i", &kernel_id)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  finalize_persistent_kernel(kernel_id);

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__megakernel_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
};

PyMODINIT_FUNC PyInit___megakernel_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
