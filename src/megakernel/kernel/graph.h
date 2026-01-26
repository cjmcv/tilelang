/* Copyright 2023-2025 CMU
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

#pragma once

#include "megakernel/kernel/customized.h"
#include "megakernel/kernel/device_tensor.h"
#include "megakernel/kernel/operator.h"
#include "megakernel/kernel/runtime.h"
#include "megakernel/threadblock/graph.h"
#include "megakernel/kernel/task_register.h"
#include <vector>

#include "megakernel/code_keeper.h"

#include <queue>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace megakernel {
namespace kernel {

using json = nlohmann::json;
using namespace runtime;
using TaskConfigMap = std::unordered_map<
    megakernel::kernel::KNOperator const*,
    std::tuple<int, int, megakernel::runtime::TaskType, int>
>;

using namespace megakernel::runtime;
namespace kn = megakernel::kernel;
namespace tb = megakernel::threadblock;

size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

bool is_nvshmem_event(size_t event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

struct Dim3Comparator {
  bool operator()(dim3 const &a, dim3 const &b) const {
    if (a.x != b.x) {
      return a.x < b.x;
    }
    if (a.y != b.y) {
      return a.y < b.y;
    }
    return a.z < b.z;
  }
};


class Graph {

public:
  Graph(dim3 _gpu_dim = {1, 1, 1})
    : gpu_dim(_gpu_dim) {
   dmem_data_offset = 0;
  }

  ~Graph() {
    while (!operators.empty()) {
      KNOperator *op = operators.back();
      std::vector<DTensor>& output_tensors = op->get_output_dtensors();
      for (size_t i=0; i<output_tensors.size(); i++) {
        this->free(output_tensors[i]);
      }
      delete op;
      operators.pop_back();
    }
  }

  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;

  // input operator
  DTensor new_input(std::vector<int> const &dims,
                    std::vector<size_t> const &strides,
                    megakernel::type::DataType data_type,
                    megakernel::layout::DmemLayout layout) {
    KNInputOp *op = new KNInputOp(dims, strides, data_type, layout);
    assert(op != nullptr);
    std::vector<DTensor>& output_tensors = op->get_output_dtensors();
    for (int i=0; i<output_tensors.size(); i++) {
      this->allocate(output_tensors[i]);
    }
    operators.push_back(op);
    return op->output_tensors[0];
  }
  DTensor *new_input_ptr(std::vector<int> const &dims,
                         std::vector<size_t> const &strides,
                         megakernel::type::DataType data_type,
                         megakernel::layout::DmemLayout layout) {
    KNInputOp *op = new KNInputOp(dims, strides, data_type, layout);
    assert(op != nullptr);
    std::vector<DTensor>& output_tensors = op->get_output_dtensors();
    for (int i=0; i<output_tensors.size(); i++) {
      this->allocate(output_tensors[i]);
    }
    operators.push_back(op);
    return &op->output_tensors[0];
  }
  // KNOperator *create_input_op(std::vector<int> const &dims,
  //                             std::vector<size_t> const &strides,
  //                             megakernel::type::DataType data_type,
  //                             megakernel::layout::DmemLayout layout) {
  //   KNInputOp *op = new KNInputOp(this, dims, strides, data_type, layout);
  //   return op;
  // }
  // customized operator
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  megakernel::threadblock::Graph const &_graph) {
    KNOperator *op = create_customized_op(inputs, _graph);
    assert(op != nullptr);
    operators.push_back(op);
    return op->output_tensors;
  }

  int customized(std::vector<DTensor const *> _inputs,
                 DTensor **outputs,
                 megakernel::threadblock::Graph const *bgraph) {
    std::vector<DTensor> inputs;
    for (auto const &t : _inputs) {
      inputs.push_back(t == nullptr ? DTensor::EMPTY_TENSOR : *t);
    }
    KNOperator *op = create_customized_op(inputs, *bgraph);
    assert(op != nullptr);
    operators.push_back(op);
    for (size_t i = 0; i < op->output_tensors.size(); i++) {
      outputs[i] = &op->output_tensors[i];
    }
    return op->output_tensors.size();
  }
  KNOperator *create_customized_op(std::vector<DTensor> const &inputs,
                                   megakernel::threadblock::Graph const &_graph) {
    // Assert that _graph's dtensor inputs align with inputs
    {
      int num_inputs = 0;
      for (auto const &op : _graph.operators) {
        if (op->op_type == megakernel::type::TB_INPUT_OP) {
          megakernel::threadblock::TBInputOp const *input_op =
              static_cast<megakernel::threadblock::TBInputOp const *>(op);
          assert(inputs[num_inputs] == input_op->dtensor);
          num_inputs++;
        }
      }
      assert(num_inputs == (int)inputs.size());
    }


    KNCustomizedOp *op = new KNCustomizedOp(this, inputs, _graph);
    return op;
  }
  // persistent kernel functions
  void attach_torch_tensor(DTensor const *input,
                           void *torch_ptr,
                           char const *name) {
    io_config.emplace(input->guid, IODesc(IODesc::TorchTensor, std::string(name), *input, torch_ptr));
  }
  void attach_cuda_tensor(DTensor const *input, char const *name) {
    io_config.emplace(input->guid, IODesc(IODesc::CUDAMallocTensor, std::string(name), *input));
  }
  void attach_nvshmem_tensor(DTensor const *input, char const *name) {
    io_config.emplace(input->guid, IODesc(IODesc::NVSHMEMMallocTensor, std::string(name), *input));
  }
  DTensor *fuse_tensors(std::vector<DTensor const *> inputs,
                        int fused_dim,
                        int num_groups,
                        char const *name) {
    // Currently assert that we fuse along the 0-th dim (for weights)
    assert(fused_dim == 0);
    assert(inputs.size() > 0);
    std::vector<int> dims;
    for (int i = 0; i < inputs[0]->num_dims; i++) {
      dims.push_back(inputs[0]->dim[i]);
    }
    for (size_t t = 1; t < inputs.size(); t++) {
      dims[0] += inputs[t]->dim[0];
      assert(inputs[0]->num_dims == inputs[t]->num_dims);
      for (int i = 1; i < inputs[t]->num_dims; i++) {
        assert(dims[i] == inputs[t]->dim[i]);
      }
      assert(inputs[0]->data_type == inputs[t]->data_type);
    }
    std::vector<size_t> strides(dims.size(), 1);
    for (int i = inputs[0]->num_dims - 1; i >= 0; i--) {
      if (i == inputs[0]->num_dims - 1) {
        strides[i] = 1;
      } else {
        strides[i] = strides[i + 1] * dims[i + 1];
      }
    }
    DTensor *fused =
        new_input_ptr(dims, strides, inputs[0]->data_type, layout::DmemRowMajor);
    IODesc desc(IODesc::FusedTorchTensor, std::string(name), *fused);
    desc.num_groups = num_groups;
    for (size_t t = 0; t < inputs.size(); t++) {
      assert(io_config.find(inputs[t]->guid) != io_config.end());
      IODesc sub_desc = io_config.find(inputs[t]->guid)->second;
      desc.sub_descs.push_back(sub_desc);
      io_config.erase(inputs[t]->guid);
    }
    io_config.emplace(fused->guid, desc);
    return fused;
  }

  DTensor *shuffle_tensors(std::vector<DTensor const *> inputs,
                           int shuffled_dim,
                           int num_groups,
                           char const *name){
    // Currently assert that we shuffle along the 0-th dim (for weights)
    assert(shuffled_dim == 0);
    assert(inputs.size() > 0);
    std::vector<int> dims;
    for (int i = 0; i < inputs[0]->num_dims; i++) {
      dims.push_back(inputs[0]->dim[i]);
    }
    for (size_t t = 1; t < inputs.size(); t++) {
      dims[0] += inputs[t]->dim[0];
      assert(inputs[0]->num_dims == inputs[t]->num_dims);
      for (int i = 1; i < inputs[t]->num_dims; i++) {
        assert(dims[i] == inputs[t]->dim[i]);
      }
      assert(inputs[0]->data_type == inputs[t]->data_type);
    }
    std::vector<size_t> strides(dims.size(), 1);
    for (int i = inputs[0]->num_dims - 1; i >= 0; i--) {
      if (i == inputs[0]->num_dims - 1) {
        strides[i] = 1;
      } else {
        strides[i] = strides[i + 1] * dims[i + 1];
      }
    }
    DTensor *shuffled =
        new_input_ptr(dims, strides, inputs[0]->data_type, layout::DmemRowMajor);
    IODesc desc(IODesc::ShuffledTorchTensor, std::string(name), *shuffled);
    desc.num_groups = num_groups;
    for (size_t t = 0; t < inputs.size(); t++) {
      assert(io_config.find(inputs[t]->guid) != io_config.end());
      IODesc sub_desc = io_config.find(inputs[t]->guid)->second;
      desc.sub_descs.push_back(sub_desc);
      io_config.erase(inputs[t]->guid);
    }
    io_config.emplace(shuffled->guid, desc);
    return shuffled;
  }

  // helper functions
  int get_num_input_dtensors() const {
    int num_inputs = 0;
    for (auto const &op : this->operators) {
      if (op->op_type == megakernel::type::KN_INPUT_OP) {
        num_inputs++;
      }
    }
    return num_inputs;
  }
  // int get_num_output_dtensors() const;
  int get_input_dtensors(DTensor **inputs) const {
    int num_inputs = 0;
    for (auto const &op : this->operators) {
      if (op->op_type == megakernel::type::KN_INPUT_OP) {
        assert(op->output_tensors.size() == 1);
        inputs[num_inputs++] = &op->output_tensors[0];
      }
    }
    return num_inputs;
  }

  int get_input_dtensor_shape_and_stride(DTensor const *input,
                                         int *strides,
                                         int *dims) const{
    for (auto const &op : this->operators) {
      if (op == input->owner_op) {
        assert(op->op_type == megakernel::type::KN_INPUT_OP &&
                "input is not an KNInputOp");
        KNInputOp *input_op = static_cast<KNInputOp *>(op);
        int num_dims = (int)input_op->input_strides.size();
        for (int i = 0; i < num_dims; i++) {
          strides[i] = input_op->input_strides[i];
          dims[i] = input->dim[i];
        }
        return num_dims;
      }
    }
    assert(false && "Cannot find input dtensor");
    return 0;
  }

  bool allocate(DTensor &tensor) {
    // assert that the start of the tensor is 16 bytes aligned
    assert(dmem_data_offset % 16 == 0);
    off_t ret = dmem_data_offset;

    size_t aligns_size = ((tensor.data_size() + 15) & ~15);
    dmem_data_offset += aligns_size;

    allocated_data_tensors.push_back(std::make_pair(ret, aligns_size));
    tensor.data_offset = ret;

    return true;
  }
  void free(DTensor &tensor) {
    assert(allocated_data_tensors.size() > 0);
    assert(allocated_data_tensors.back().first == tensor.data_offset);
    assert(allocated_data_tensors.back().second ==
           ((tensor.data_size() + 15) & ~15));
    dmem_data_offset -= allocated_data_tensors.back().second;
    allocated_data_tensors.pop_back();
    tensor.data_offset = -1;
  }

  // runtime::TaskGraphResult generate_task_graph(int num_gpus, int my_gpu_id);
  TaskGraphResult generate_task_graph(int _num_gpus, int _my_gpu_id) {
    printf("Call generate_task_graph.\n");
    std::vector<FullTaskDesc> all_tasks;
    std::vector<EventDesc> all_events;
    std::vector<TaskId> first_tasks;
    int num_gpus, my_gpu_id;
    std::map<kernel::KNOperator *, std::map<dim3, TaskId, Dim3Comparator>>
        all_task_maps;
    num_gpus = _num_gpus;
    my_gpu_id = _my_gpu_id;
    // add the termination event to the event lists
    EventDesc e(EVENT_TERMINATION, 1, 0, 0);
    all_events.push_back(e);
    FullTaskDesc t(TASK_TERMINATE, 0 /*variant_id*/);
    all_tasks.push_back(t);
    register_mugraph(*this,
                     num_gpus,
                     my_gpu_id,
                     all_tasks,
                     all_events,
                     first_tasks,
                     all_task_maps,
                     task_config);
    assert(sanity_check(*this, all_tasks, all_events, first_tasks));
    return print_task_graph(*this,
                            num_gpus,
                            my_gpu_id,
                            all_tasks,
                            all_events,
                            first_tasks,
                            all_task_maps,
                            task_config,
                            io_config,
                            true /*use_json_format*/);
  }
  
  void register_task(char const *task_type, std::vector<int> params) {
    std::string name = std::string(task_type);
    KNOperator const *op = operators.back();
    assert(op->op_type == type::KN_CUSTOMIZED_OP);
    KNCustomizedOp const *customized = static_cast<KNCustomizedOp const *>(op);
    TaskRegister *task_register = TaskRegister::get_instance();
    if (name == "embedding") {
      int variant_id = task_register->register_embedding_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_EMBEDDING, variant_id);
    } else if (name == "rmsnorm") {
      int variant_id = task_register->register_rmsnorm_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM, variant_id);
    } else if (name == "rmsnorm_linear") {
      int variant_id = task_register->register_rmsnorm_linear_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(3, 1, TASK_RMS_NORM_LINEAR, variant_id);
    } else if (name == "attention") {
      int variant_id = task_register->register_attention_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(7, 1, TASK_ATTENTION_1, variant_id);
    } else if (name == "paged_attention") {
      int variant_id = task_register->register_paged_attention_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(7, 1, TASK_PAGED_ATTENTION_1, variant_id);
    } else if (name == "single_batch_extend_attention") {
      int variant_id = task_register->register_single_batch_extend_attention_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(7, 1, TASK_SINGLE_BATCH_EXTEND_ATTENTION, variant_id);
    } else if (name == "linear") {
      int variant_id = task_register->register_linear_task(customized->bgraph, params, false /*with_residual*/, false /*with_silu_mul*/);
      task_config[op] = std::make_tuple(2, 1, TASK_LINEAR, variant_id);
    } else if (name == "silu_mul_linear") {
      int variant_id = task_register->register_linear_task(customized->bgraph, params, false /*with_residual*/, true /*with_silu_mul*/);
      task_config[op] = std::make_tuple(2, 1, TASK_LINEAR, variant_id);
    } else if (name == "linear_with_residual") {
      int variant_id = task_register->register_linear_task(customized->bgraph, params, true /*with_residual*/, false /*with_silu_mul*/);
      task_config[op] = std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL, variant_id);
    } else if (name == "silu_mul") {
      int variant_id = task_register->register_silu_mul_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL, variant_id);
    } else if (name == "identity") {
      int variant_id = task_register->register_identity_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 1, TASK_IDENTITY, variant_id);
    } else if (name == "silu_mul_linear_with_residual") {
      int variant_id = task_register->register_silu_mul_linear_with_residual_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(3, 1, TASK_SILU_MUL_LINEAR_WITH_RESIDUAL, variant_id);
    } else if (name == "argmax") {
      task_config[op] = std::make_tuple(1, 1, TASK_ARGMAX, 0);
    } else if (name == "argmax_partial") {
      int variant_id = task_register->register_argmax_partial_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 2, TASK_ARGMAX_PARTIAL, variant_id);
    } else if (name == "argmax_reduce") {
      int variant_id = task_register->register_argmax_reduce_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_ARGMAX_REDUCE, variant_id);
    } else if (name == "allreduce") {
      // `register_reduce_task` will register two tasks, but we only record one
      int variant_id = task_register->register_reduce_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_ALLREDUCE, variant_id);
    } else if (name == "find_ngram_partial") {
      int variant_id = task_register->register_find_ngram_partial_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 1, TASK_FIND_NGRAM_PARTIAL, variant_id);
    } else if (name == "find_ngram_global") {
      int variant_id = task_register->register_find_ngram_global_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_FIND_NGRAM_GLOBAL, variant_id);
    } else if (name == "target_verify_greedy") {
      int variant_id = task_register->register_target_verify_greedy_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_TARGET_VERIFY_GREEDY, variant_id);
    }
    // Hopper tasks
    else if (name == "linear_hopper") {
      int variant_id = task_register->register_linear_hopper_task(customized->bgraph, params, false /*with_residual*/);
      task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_HOPPER, variant_id);
    } else if (name == "linear_with_residual_hopper") {
      int variant_id = task_register->register_linear_hopper_task(customized->bgraph, params, true /*with_residual*/);
      task_config[op] = std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL_HOPPER, variant_id);
    } else if (name == "paged_attention_hopper") {
      int variant_id = task_register->register_paged_attention_hopper_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(7, 1, TASK_PAGED_ATTENTION_HOPPER, variant_id);
    } else if (name == "rmsnorm_hopper") {
      int variant_id = task_register->register_rmsnorm_hopper_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM_HOPPER, variant_id);
    } else if (name == "linear_swapAB_hopper") {
      int variant_id = task_register->register_linear_swapAB_hopper_task(customized->bgraph, params, false /*with_residual*/);
      task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_SWAPAB_HOPPER, variant_id);
    } else if (name == "linear_swapAB_with_residual_hopper") {
      int variant_id = task_register->register_linear_swapAB_hopper_task(customized->bgraph, params, true /*with_residual*/);
      task_config[op] = std::make_tuple(3, 1, TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER, variant_id);
    } else if (name == "linear_cutlass_hopper") {
      int variant_id = task_register->register_linear_cutlass_hopper_task(customized->bgraph, params, false /*with_residual*/);
      task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_CUTLASS_HOPPER, variant_id);
    } else if (name == "linear_cutlass_with_residual_hopper") {
      int variant_id = task_register->register_linear_cutlass_hopper_task(customized->bgraph, params, true /*with_residual*/);
      task_config[op] = std::make_tuple(3, 1, TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER, variant_id);
    } else if (name == "silu_mul_hopper") {
      int variant_id = task_register->register_silu_mul_hopper_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL_HOPPER, variant_id);
    } else if (name == "embedding_hopper") {
      int variant_id = task_register->register_embedding_hopper_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_EMBEDDING_HOPPER, variant_id);
    } else if (name == "moe_w13_linear_sm90") {
      int variant_id = task_register->register_moe_linear_sm90_task(customized->bgraph, params, true /*w13_linear*/);
      task_config[op] = std::make_tuple(4, 1, TASK_MOE_W13_LINEAR_SM90, variant_id);
    } else if (name == "moe_w2_linear_sm90") {
      int variant_id = task_register->register_moe_linear_sm90_task(customized->bgraph, params, false /*w13_linear*/);
      task_config[op] = std::make_tuple(4, 1, TASK_MOE_W2_LINEAR_SM90, variant_id);
    } else if (name == "splitk_linear_swapAB_hopper") {
      int variant_id = task_register->register_splitk_linear_swapAB_hopper_task(customized->bgraph, params, false /*with_residual*/);
      task_config[op] = std::make_tuple(2, 1, TASK_SPLITK_LINEAR_SWAPAB_HOPPER, variant_id);
    } else if (name == "paged_attention_split_kv_hopper") {
      int variant_id = task_register->register_paged_attention_split_kv_hopper_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(7, 2, TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER, variant_id);
    }
    // SM100 tasks
    else if (name == "linear_sm100") {
      int variant_id = task_register->register_linear_sm100_task(customized->bgraph, params, false /*with_residual*/);
      task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_SM100, variant_id);
    } else if (name == "splitk_linear_sm100") {
      int variant_id = task_register->register_splitk_linear_sm100_task(customized->bgraph, params, false /*with_residual*/);
      task_config[op] = std::make_tuple(2, 1, TASK_SPLITK_LINEAR_SM100, variant_id);
    } else if (name == "linear_with_residual_sm100") {
      int variant_id = task_register->register_linear_sm100_task(customized->bgraph, params, true /*with_residual*/);
      task_config[op] =std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL_SM100, variant_id);
    } else if (name == "paged_attention_sm100") {
      int variant_id = task_register->register_paged_attention_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(7, 1, TASK_ATTN_SM100, variant_id);
    } else if (name == "argmax_partial_sm100") {
      int variant_id = task_register->register_argmax_partial_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 2, TASK_ARGMAX_PARTIAL_SM100, variant_id);
    } else if (name == "argmax_reduce_sm100") {
      int variant_id = task_register->register_argmax_reduce_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_ARGMAX_REDUCE_SM100, variant_id);
    } else if (name == "sampling_sm100") {
      int variant_id = task_register->register_sampling_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 1, TASK_SAMPLING_SM100, variant_id);
    } else if (name == "tensor_init") {
      int variant_id = task_register->register_tensor_init_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_TENSOR_INIT, variant_id);
    } else if (name == "moe_topk_softmax_sm100") {
      int variant_id = task_register->register_moe_topk_softmax_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 3, TASK_MOE_TOPK_SOFTMAX_SM100, variant_id);
    } else if (name == "moe_w13_linear_sm100") {
      int variant_id = task_register->register_moe_linear_sm100_task(customized->bgraph, params, true /*w13_linear*/);
      task_config[op] = std::make_tuple(4, 1, TASK_MOE_W13_LINEAR_SM100, variant_id);
    } else if (name == "moe_silu_mul") {
      int variant_id = task_register->register_moe_silu_mul_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL, variant_id);
    } else if (name == "moe_w2_linear_sm100") {
      int variant_id = task_register->register_moe_linear_sm100_task(customized->bgraph, params, false /*w13_linear*/);
      task_config[op] = std::make_tuple(4, 1, TASK_MOE_W2_LINEAR_SM100, variant_id);
    } else if (name == "moe_mul_sum_add_sm100") {
      int variant_id = task_register->register_moe_mul_sum_add_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(3, 1, TASK_MOE_MUL_SUM_ADD_SM100, variant_id);
    } else if (name == "paged_attention_split_kv_sm100") {
      int variant_id = task_register->register_paged_attention_split_kv_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(7, 2, TASK_PAGED_ATTENTION_SPLIT_KV_SM100, variant_id);
    } else if (name == "paged_attention_split_kv_merge_sm100") {
      int variant_id = task_register->register_paged_attention_split_kv_merge_sm100_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100, variant_id);
    } else {
      printf("Unsupported task name: %s\n", name.c_str());
      assert(false && "Unsupported task type");
    }
  }
  
private:
  bool sanity_check(megakernel::kernel::Graph const &graph,
                    std::vector<FullTaskDesc> const &all_tasks,
                    std::vector<EventDesc> const &all_events,
                    std::vector<TaskId> const &first_tasks) {
    std::unordered_set<EventId> triggered_events;
    std::unordered_set<TaskId> executed_tasks;
    std::vector<int> event_counts(all_events.size(), 0);
    for (size_t i = 0; i < all_events.size(); i++) {
      event_counts[i] = all_events[i].num_triggers;
    }
    std::queue<TaskId> task_queue;
    std::queue<EventId> event_queue;
    printf("First tasks: %d\n", (int)first_tasks.size());
    for (size_t i = 0; i < first_tasks.size(); i++) {
      task_queue.push(first_tasks[i]);
    }
    while (!(task_queue.empty() && event_queue.empty())) {
      // Execute tasks
      while (!task_queue.empty()) {
        TaskId task = task_queue.front();
        task_queue.pop();
        assert(executed_tasks.count(task) == 0);
        executed_tasks.insert(task);
        FullTaskDesc desc = all_tasks[task];
        if (desc.trigger_event != EVENT_INVALID_ID) {
          EventId event_id = desc.trigger_event;
          size_t event_pos = event_id & 0xffffffff;
          // event_pos 0 is the end of task graph event
          if (event_pos == 0) {
            continue;
          }
          // These events counts are manually adjusted. Each task of nvshmem cpy
          // will update BS times of event counter, not just once.
          if (desc.task_type == runtime::TASK_NVSHMEM_COPY) {
            event_counts[event_pos] -= desc.inputs[0].dim[0] - 1;
          }
          assert(event_counts[event_pos] > 0);
          event_counts[event_pos]--;
          if (event_counts[event_pos] == 0) {
            event_queue.push(event_id);
          }
        }
      }
      while (!event_queue.empty()) {
        EventId event_id = event_queue.front();
        event_queue.pop();
        assert(triggered_events.count(event_id) == 0);
        triggered_events.insert(event_id);
        size_t event_pos = event_id & 0xffffffff;
        EventDesc desc = all_events[event_pos];
        for (TaskId tid = desc.first_task_id; tid < desc.last_task_id; tid++) {
          task_queue.push(tid);
        }
      }
    }
    printf("Number of all events: %zu\n", all_events.size());
    printf("Number of all tasks: %zu\n", all_tasks.size());
    printf("Number of triggered events: %zu\n", triggered_events.size());
    printf("Number of executed tasks: %zu\n", executed_tasks.size());
    return true;
  }

  void print_register_info(std::string prefix_str,
                          TaskType task_type,
                          std::vector<EventDesc> &all_events,
                          std::vector<FullTaskDesc> &all_tasks,
                          std::map<dim3, TaskId, Dim3Comparator> const &pre_task_map,
                          std::map<dim3, TaskId, Dim3Comparator> &cur_task_map) {
    printf("%s: task[%d] register_info (all_tasks: %d, all_events: %d, pre_task_map: %d, cur_task_map: %d).\n", 
      prefix_str.c_str(), task_type, all_tasks.size(), all_events.size(), pre_task_map.size(), cur_task_map.size());
    for (size_t i=0; i<all_tasks.size(); i++) {
      printf("task %d: depend: %d, trigger: %d\n", i, all_tasks[i].dependent_event, all_tasks[i].trigger_event);
    }
    for (size_t i=0; i<all_events.size(); i++) {
      printf("event %d: num: %d, first: %d, last: %d\n", i, all_events[i].num_triggers, all_events[i].first_task_id, all_events[i].last_task_id);
    }
    for (const auto& [key, task_id] : cur_task_map) {
      printf("dim3(%d, %d, %d)->TaskId: %d.\n", key.x, key.y, key.z, task_id);
    }
  }
  void print_final_register_info(std::string prefix_str,
                          std::vector<EventDesc> &all_events,
                          std::vector<FullTaskDesc> &all_tasks) {
    printf("%s: final_register_info (all_tasks: %d, all_events: %d).\n", 
      prefix_str.c_str(), all_tasks.size(), all_events.size());
    for (size_t i=0; i<all_tasks.size(); i++)
      printf("task %d: depend: %d, trigger: %d\n", i, all_tasks[i].dependent_event, all_tasks[i].trigger_event);
    for (size_t i=0; i<all_events.size(); i++)
      printf("event %d: num: %d, first: %d, last: %d\n", i, all_events[i].num_triggers, all_events[i].first_task_id, all_events[i].last_task_id);
  }
  void create_events_add_tasks(
    TaskType task_type,
    int depth,
    int const my_gpu_id,
    int3 const input_map,
    int3 const output_map,
    dim3 const consumer_grid_dim,
    dim3 const producer_grid_dim,
    std::vector<EventDesc> &all_events,
    std::vector<FullTaskDesc> &all_tasks,
    std::vector<FullTaskDesc> const &cur_op_tasks,
    std::map<dim3, TaskId, Dim3Comparator> const &pre_task_map,
    std::map<dim3, TaskId, Dim3Comparator> &cur_task_map) {
    
    int fence_mode = input_map.x + input_map.y*10 + input_map.z*100;

    std::vector<std::pair<std::vector<dim3>, std::vector<dim3>>> pv;
    size_t event_num = 1;
    // 获取映射信息
    if (fence_mode == 0) {
      // 同步所有
      event_num = 1;
      for (size_t i=0; i<event_num; i++) {
        std::vector<dim3> producer;
        std::vector<dim3> consumer;

        dim3 bid;
        for (bid.x = 0; bid.x < producer_grid_dim.x; bid.x++) {
          for (bid.y = 0; bid.y < producer_grid_dim.y; bid.y++) {
            for (bid.z = 0; bid.z < producer_grid_dim.z; bid.z++) {
              producer.push_back(bid);
            }
          }
        }
        for (bid.x = 0; bid.x < consumer_grid_dim.x; bid.x++) {
          for (bid.y = 0; bid.y < consumer_grid_dim.y; bid.y++) {
            for (bid.z = 0; bid.z < consumer_grid_dim.z; bid.z++) {
              consumer.push_back(bid);
            }
          }
        } 
        std::pair<std::vector<dim3>, std::vector<dim3>> p{producer, consumer};
        pv.push_back(p);
      }      
    }
    else if (fence_mode == 1) {
      // 沿着x轴均分, y轴和z轴则all对all
      // 如[4,8]-[2,1] => (0-1,0-7)-(0,0), (2-3,0-7)-(1,0)
      //   [4,8]-[4,1] => (0,0-7)-(0,0), (1,0-7)-(1,0), (2,0-7)-(2,0), (3,0-7)-(3,0)
      //   [4,8]-[2,8] => (0-1,0-7)-(0,0-7), (2-3,0-7)-(1,0-7)
      //   [4,8]-[2,4] => (0-1,0-7)-(0,0-3), (2-3,0-7)-(1,0-3)
      int producer_step_x, consumer_step_x;
      if (producer_grid_dim.x >= consumer_grid_dim.x) {
        assert(producer_grid_dim.x % consumer_grid_dim.x == 0);
        producer_step_x = producer_grid_dim.x / consumer_grid_dim.x;
        consumer_step_x = 1;
        event_num = consumer_grid_dim.x;
      }
      else {
        assert(consumer_grid_dim.x % producer_grid_dim.x == 0);
        consumer_step_x = consumer_grid_dim.x / producer_grid_dim.x;
        producer_step_x = 1;
        event_num = producer_grid_dim.x;
      }
      
      for (size_t i=0; i<event_num; i++) {
        dim3 bid;
        std::vector<dim3> producer;
        std::vector<dim3> consumer;
        for (bid.x = i*producer_step_x; bid.x < i*producer_step_x+producer_step_x; bid.x++) {
          for (bid.y = 0; bid.y < producer_grid_dim.y; bid.y++) {
            for (bid.z = 0; bid.z < producer_grid_dim.z; bid.z++) {
              producer.push_back(bid);
            }
          }
        }
        for (bid.x = i*consumer_step_x; bid.x < i*consumer_step_x+consumer_step_x; bid.x++) {
          for (bid.y = 0; bid.y < consumer_grid_dim.y; bid.y++) {
            for (bid.z = 0; bid.z < consumer_grid_dim.z; bid.z++) {
              consumer.push_back(bid);
            }
          }
        }
        std::pair<std::vector<dim3>, std::vector<dim3>> p{producer, consumer};
        pv.push_back(p);
      }
    }
    else if (fence_mode == 2) {
      // 沿着x轴跨一半后均分, y轴和z轴则all对all
      // silu_mul专用
      // 如[4,8]-[2,1] => (0/2,0-7)-(0,0), (1/3,0-7)-(1,0)
      //   [8,8]-[2,1] => (01/45,0-7)-(0,0), (23/67,0-7)-(1,0),
      //   [8,8]-[2,4] => (01/45,0-7)-(0,0-3), (23/67,0-7)-(1,0-3),
      assert(producer_grid_dim.x > consumer_grid_dim.x);
      // assert(producer_grid_dim.x % consumer_grid_dim.x == 0);
      int producer_mid_x = producer_grid_dim.x / 2;
      int gcd = std::gcd(producer_mid_x, consumer_grid_dim.x); // 76, 20 => 4 => 76/4=19，20/4=5 => 19 vs 5
      int producer_step_x = producer_mid_x / gcd;
      int consumer_step_x = consumer_grid_dim.x / gcd;
      // printf("check: %f, %d.\n", producer_step_x, window);
      event_num = gcd;
      for (size_t i=0; i<event_num; i++) {
        dim3 bid;
        std::vector<dim3> producer;
        std::vector<dim3> consumer;

        // printf("(%d, %d).\n", start, end);
        for (bid.x = i*producer_step_x; bid.x < i*producer_step_x+producer_step_x; bid.x++) {
          for (bid.y = 0; bid.y < producer_grid_dim.y; bid.y++) {
            for (bid.z = 0; bid.z < producer_grid_dim.z; bid.z++) {
              producer.push_back(bid);
              producer.push_back({bid.x+producer_mid_x, bid.y, bid.z});
            }
          }
        }
        for (bid.x = i*consumer_step_x; bid.x < i*consumer_step_x+consumer_step_x; bid.x++) {
          for (bid.y = 0; bid.y < consumer_grid_dim.y; bid.y++) {
            for (bid.z = 0; bid.z < consumer_grid_dim.z; bid.z++) {
              consumer.push_back(bid);
            }
          }
        }
        std::pair<std::vector<dim3>, std::vector<dim3>> p{producer, consumer};
        pv.push_back(p);
      }
    }
    else {
      printf("fence_mode (%d) is not supported!\n", fence_mode);
      assert(0);
    }

    // 使用映射信息构建当前节点task，并填充上一节点task的trigger event
    for (size_t i=0; i<event_num; i++) {
      // 一个新的event，之前的最后一个task的下一位id就是当前event的第一个task
      EventDesc event_desc;
      event_desc.num_triggers = 0;
      event_desc.first_task_id = all_tasks.size();

      // 添加当前算子的task
      const std::vector<dim3>& producer = pv[i].first;
      const std::vector<dim3>& consumer = pv[i].second;
      for (size_t i=0; i<consumer.size(); i++) {
        dim3 bid = consumer[i];
        cur_task_map[bid] = all_tasks.size();
        int offset = bid.x * consumer_grid_dim.y * consumer_grid_dim.z +
                    bid.y * consumer_grid_dim.z + bid.z;
        all_tasks.push_back(cur_op_tasks[offset]);
      }
      event_desc.last_task_id = all_tasks.size();
      
      // Set producer tasks
      // 设置前置算子的task
      for (size_t i=0; i<producer.size(); i++) {
        dim3 bid = producer[i];
        assert(pre_task_map.find(bid) != pre_task_map.end());
        int task_id = pre_task_map.find(bid)->second;
        // encode gpu_id
        all_tasks[task_id].trigger_event = get_event_id(
            my_gpu_id, all_events.size(), false /*nvshmem_event*/);
        event_desc.num_triggers++;
      }

      event_desc.event_type =
          event_desc.last_task_id >= event_desc.first_task_id + 8
              ? EVENT_LAUNCH_MASSIVE_TASKS : EVENT_LAUNCH_TASKS;
      all_events.push_back(event_desc);
      printf("create_events_add_tasks event_desc.event_type: %d.\n", event_desc.event_type);
    }
  }

  void register_mugraph(
      megakernel::kernel::Graph const &graph,
      int num_gpus,
      int my_gpu_id,
      std::vector<FullTaskDesc> &all_tasks,
      std::vector<EventDesc> &all_events,
      std::vector<TaskId> &first_tasks,
      std::map<kernel::KNOperator *, std::map<dim3, TaskId, Dim3Comparator>>
          &all_task_maps,
      std::unordered_map<kn::KNOperator const *,
                        std::tuple<int, int, TaskType, int>> const
          &task_configs) {
    // push a begin-graph task and a event to launch dependent asks
    {
      EventDesc e(EVENT_LAUNCH_DEPENDENT_TASKS, 1, 0, 0);
      FullTaskDesc t(TASK_BEGIN_TASK_GRAPH, 0 /*variant_id*/);
      t.trigger_event = get_event_id(my_gpu_id, all_events.size(), false);
      all_tasks.push_back(t);
      all_events.push_back(e);
    }
    std::vector<tb::TBInputOp *> pre_output_ops;
    kn::KNCustomizedOp const *pre_op = nullptr;
    std::map<dim3, TaskId, Dim3Comparator> pre_task_map;
    std::unordered_set<size_t> nvshmem_events_idx;
    for (auto const &op : graph.operators) {
      if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
        continue;
      }
      std::tuple<int, int, TaskType, int> task_config = task_configs.find(op)->second;
      std::map<dim3, TaskId, Dim3Comparator> cur_task_map;
      assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
      // Customized op
      kn::KNCustomizedOp const *cur_op = dynamic_cast<kn::KNCustomizedOp const *>(op);
      tb::Graph const &bgraph = cur_op->bgraph;
      dim3 bid;
      std::vector<FullTaskDesc> tasks;
      std::vector<tb::TBInputOp *> input_ops;
      std::vector<tb::TBInputOp *> output_ops;
      int num_inputs = std::get<0>(task_config);
      int num_outputs = std::get<1>(task_config);
      TaskType task_type = std::get<2>(task_config);
      int variant_id = std::get<3>(task_config);
      assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
      for (auto const &op : bgraph.operators) {
        assert(op->op_type == megakernel::type::TB_INPUT_OP);
        if (input_ops.size() < (size_t)num_inputs) {
          input_ops.push_back(static_cast<tb::TBInputOp *>(op));
        } else {
          output_ops.push_back(static_cast<tb::TBInputOp *>(op));
        }
      }
      
      // Step 1: add all tasks based on their blockIdx
      // (bid.x, bid.y, bid.z) ordering
      for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
        for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
          for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
            FullTaskDesc task(task_type, variant_id);
            // Initialize input tensors to the task
            for (auto const &input : input_ops) {
              TensorDesc desc;
              assert(input->output_tensors.size() == 1);
              tb::STensor stensor = input->output_tensors[0];
              desc.num_dims = stensor.num_dims;
              desc.data_type = stensor.data_type;
              // Assume always partition head group on gridDim.y dimension
              for (int d = stensor.num_dims - 1; d >= 0; d--) {
                desc.dim[d] = stensor.dim[d];
                desc.stride[d] = (d == stensor.num_dims - 1) ? 1 : desc.stride[d + 1] * input->dtensor.dim[d + 1];
              }
              task.inputs[task.num_inputs++] = desc;
            }
            // Initialize output tensors to the task
            for (auto const &output : output_ops) {
              TensorDesc desc;
              assert(output->output_tensors.size() == 1);
              tb::STensor stensor = output->output_tensors[0];
              desc.num_dims = stensor.num_dims;
              desc.data_type = stensor.data_type;
              for (int d = stensor.num_dims - 1; d >= 0; d--) {
                desc.dim[d] = stensor.dim[d];
                desc.stride[d] = (d == stensor.num_dims - 1) ? 1 : desc.stride[d + 1] * output->dtensor.dim[d + 1];
              }
              task.outputs[task.num_outputs++] = desc;
            }
            tasks.push_back(task);
          }
        }
      }
      // Step 2: create events between operators
      if (pre_op == nullptr) {
        // 无前置算子，直接创建task即可
        dim3 bid;
        for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
          for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
            for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
              cur_task_map[bid] = all_tasks.size();

              int offset = bid.x * bgraph.grid_dim.y * bgraph.grid_dim.z +
                          bid.y * bgraph.grid_dim.z + bid.z;

              first_tasks.push_back(all_tasks.size());
              all_tasks.push_back(tasks[offset]); // 每一个block是一个task，下标按将三维grid压扁，形成abcabc的顺序
            }
          }
        }
      } else {
        // Step 2.1: analyze dependencies between thread blocks of the two ops
        int num_shared_tensors = 0;
        int3 input_map, output_map;
        for (auto const &input : input_ops) {
          for (auto const &output : pre_output_ops) {
            if (input->dtensor.guid == output->dtensor.guid) {
              input_map = input->input_map;
              output_map = output->input_map;
              num_shared_tensors++;
            }
            printf("task_type: %d: guid: %ld, %ld.\n", task_type, input->dtensor.guid, output->dtensor.guid);
          }
        }

        // Step 2.2: create events and add tasks - CJM
        // dfs_create_events_add_tasks
        create_events_add_tasks(task_type,
                                    0,                       /*depth*/
                                    my_gpu_id,               /*my_gpu_id*/
                                    input_map,               /*input_map*/
                                    output_map,              /*output_map*/
                                    bgraph.grid_dim,         /*consumer_grid_dim*/
                                    pre_op->bgraph.grid_dim, /*producer_grid_dim*/
                                    all_events,
                                    all_tasks,
                                    tasks,        /*cur_op_tasks*/
                                    pre_task_map, /*pre_task_map*/
                                    cur_task_map /*cur_task_map)*/);
      }
      // print_register_info("a", task_type, all_events, all_tasks, pre_task_map, cur_task_map);
      pre_output_ops = output_ops;
      pre_op = cur_op;
      pre_task_map = cur_task_map;
      all_task_maps.emplace(op, cur_task_map);
    }
    // Update the trigger event for all tasks in pre_task_map
    for (auto const &it : pre_task_map) {
      all_tasks[it.second].trigger_event =
          get_event_id(my_gpu_id, all_events.size(), false /*nvshmem_event*/);
    }
    all_events.push_back(
        EventDesc(EVENT_END_OF_TASK_GRAPH, pre_task_map.size(), 0, 0));

    // Prelaunch all tasks at the begining of an iteration
    all_events[1].first_task_id = 2;
    all_events[1].last_task_id = all_tasks.size();
    for (size_t e = 2; e < all_events.size(); e++) {
      if (all_events[e].event_type == EVENT_LAUNCH_TASKS ||
          all_events[e].event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
        all_events[e].event_type = EVENT_EMPTY;
        bool is_nvshmem_event = false;
        if (nvshmem_events_idx.count(e) > 0) {
          is_nvshmem_event = true;
        }
        for (size_t t = all_events[e].first_task_id; t < all_events[e].last_task_id; t++) {
          all_tasks[t].dependent_event = get_event_id(my_gpu_id, e, is_nvshmem_event /*nvshmem_event*/);
        }
      }
    }
    // print_final_register_info("final", all_events, all_tasks);
  }

  TaskGraphResult print_task_graph(
      megakernel::kernel::Graph const &graph,
      int num_gpus,
      int my_gpu_id,
      std::vector<FullTaskDesc> const &all_tasks,
      std::vector<EventDesc> const &all_events,
      std::vector<TaskId> const &first_tasks,
      std::map<kernel::KNOperator *, std::map<dim3, TaskId, Dim3Comparator>> const
          &all_task_maps,
      std::unordered_map<kn::KNOperator const *,
                        std::tuple<int, int, TaskType, int>> const &task_configs,
      std::map<megakernel::type::GuidType, IODesc> const &io_configs,
      bool use_json_format) {

    using megakernel::runtime::IODesc;
    megakernel::transpiler::CodeKeeper code;
    megakernel::transpiler::CodeKeeper tgbody;
    tgbody.inc_indent();
    code.e("#include \"persistent_kernel.cuh\"");
    if (use_json_format) {
      code.e("#include <nlohmann/json.hpp>");
      code.e("#include <fstream>");
      code.e("#include <filesystem>");
      code.e("using json = nlohmann::json;");
    }
    code.e("using namespace megakernel::runtime;");
    code.e("size_t get_event_id(int my_gpu_id, size_t event_pos, bool "
          "nvshmem_event) {");
    code.e("size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | "
          "event_pos);");
    code.e("if (nvshmem_event) {");
    code.e("event_id = event_id | EVENT_NVSHMEM_TAG;");
    code.e("}");
    code.e("return event_id;");
    code.e("}");
    code.e("");

    // function that loads json file and generates task graph
    if (use_json_format) {
      code.e("void construct_task_graph(int num_gpus,");
      code.e("                          int my_gpu_id,");
      code.e("                          std::vector<FullTaskDesc> &all_tasks,");
      code.e("                          std::vector<EventDesc> &all_events,");
      code.e("                          std::vector<TaskId> &first_tasks,");
      code.e("                          std::map<std::string, void*> const "
            "&all_tensors) {");
      code.e("std::filesystem::path file_path(__FILE__);");
      code.e("std::ifstream "
            "json_file(file_path.parent_path().string()+\"/task_graph.json\");");
      code.e("nlohmann::json json_task_graph;");
      code.e("json_file >> json_task_graph;");
      // load tasks
      code.e("for (json const &task : json_task_graph[\"all_tasks\"]) {");
      code.e("FullTaskDesc "
            "task_desc(static_cast<TaskType>(task.at(\"task_type\")),");
      code.e("            task.at(\"variant_id\"));");
      code.e("task_desc.task_metadata.request_id = "
            "task.at(\"request_id\").get<int>();");
      code.e("task_desc.task_metadata.expert_offset = "
            "task.at(\"expert_offset\").get<int>();");
      code.e("task_desc.task_metadata.kv_idx = task.at(\"kv_idx\").get<int>();");
      code.e("task_desc.task_metadata.merge_task_offset = "
            "task.at(\"merge_task_offset\").get<int>();");
      code.e("if (task.at(\"trigger_event\").is_number_integer()) {");
      code.e("task_desc.trigger_event = task.at(\"trigger_event\").get<unsigned "
            "long long int>();");
      code.e("}");
      code.e("else {");
      code.e("assert(false);");
      code.e("}");
      code.e("if (task.at(\"dependent_event\").is_number_integer()) {");
      code.e("task_desc.dependent_event = "
            "task.at(\"dependent_event\").get<unsigned long long int>();");
      code.e("}");
      code.e("else {");
      code.e("assert(false);");
      code.e("}");

      // load inputs
      code.e("task_desc.num_inputs = 0;");
      code.e("for (json const &tensor : task[\"inputs\"]) {");
      code.e("TensorDesc input;");
      code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
      code.e("assert(all_tensors.find(name) != all_tensors.end());");
      // code.e("off_t offset = tensor.at(\"offset\").get<off_t>();"); //CJM
      code.e("input.base_ptr = static_cast<char*>(all_tensors.at(name)); // +offset");
      code.e("input.bx = tensor.at(\"bx\").get<int>();");
      code.e("input.by = tensor.at(\"by\").get<int>();");
      code.e("input.bz = tensor.at(\"bz\").get<int>();");
      code.e(
          "assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
      code.e("input.num_dims = tensor.at(\"dims\").size();");
      code.e("input.data_type = tensor.at(\"data_type\").get<int>();");
      code.e("for (int i = 0; i < input.num_dims; i++) {");
      code.e("input.dim[i] = tensor[\"dims\"][i].get<int>();");
      code.e("input.stride[i] = tensor[\"strides\"][i].get<int>();");
      code.e("}");

      code.e("task_desc.inputs[task_desc.num_inputs++] = input;");
      code.e("}");
      // load outputs
      code.e("task_desc.num_outputs = 0;");
      code.e("for (json const &tensor : task[\"outputs\"]) {");
      code.e("TensorDesc output;");
      code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
      code.e("assert(all_tensors.find(name) != all_tensors.end());");
      // code.e("off_t offset = tensor.at(\"offset\").get<off_t>();"); //CJM
      code.e("output.base_ptr = static_cast<char*>(all_tensors.at(name)); // +offset");
      code.e("output.bx = tensor.at(\"bx\").get<int>();");
      code.e("output.by = tensor.at(\"by\").get<int>();");
      code.e("output.bz = tensor.at(\"bz\").get<int>();");
      code.e("assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
      code.e("output.num_dims = tensor.at(\"dims\").size();");
      code.e("output.data_type = tensor.at(\"data_type\").get<int>();");
      code.e("for (int i = 0; i < output.num_dims; i++) {");
      code.e("output.dim[i] = tensor[\"dims\"][i];");
      code.e("output.stride[i] = tensor[\"strides\"][i];");
      code.e("}");

      code.e("task_desc.outputs[task_desc.num_outputs++] = output;");
      // code.e("printf(\"task_desc.num_outputs: %d\", task_desc.num_outputs);");
      code.e("}");

      // create TMA desc for each task
      code.e("#ifdef MPK_ENABLE_TMA");
      // Hopper Tasks
      code.e("if (task.at(\"task_type\") > TASK_HOPPER_TASK_BEGIN && "
            "task.at(\"task_type\") < TASK_HOPPER_TASK_END) {");
      code.e("create_tma_desc_by_task(task_desc);");
      code.e("}");
      // SM100 Tasks
      code.e("if (task.at(\"task_type\") > TASK_SM100_TMA_START_TASK && "
            "task.at(\"task_type\") < TASK_SM100_TMA_END_TASK) {");
      code.e("create_tma_desc_by_task(task_desc);");
      code.e("}");
      code.e("#endif");
      code.e("all_tasks.push_back(task_desc);");
      code.e("}");
      // load events
      code.e("for (json const &e : json_task_graph[\"all_events\"]) {");
      code.e("EventType event_type = "
            "static_cast<EventType>(e.at(\"event_type\").get<int>());");
      code.e("int num_triggers = e.at(\"num_triggers\").get<int>();");
      code.e("int first_task_id = e.at(\"first_task_id\").get<int>();");
      code.e("int last_task_id = e.at(\"last_task_id\").get<int>();");
      code.e("all_events.push_back(EventDesc(event_type, num_triggers, "
            "first_task_id, last_task_id));");
      code.e("}");
      // load first tasks
      code.e("for (json const &t : json_task_graph[\"first_tasks\"]) {");
      code.e("first_tasks.push_back(t.get<int>());");
      code.e("}");
      code.e("}");
      code.e("");
    }

    code.e("void adjust_params_with_kernel_id(int kernel_id, std::map<std::string, void*> &all_tensors);");
    code.e("static void _init_persistent_kernel(int kernel_id,");
    code.e("                                    std::vector<FullTaskDesc> &all_tasks,");
    code.e("                                    std::vector<EventDesc> &all_events,");
    code.e("                                    std::vector<TaskId> &first_tasks,");
    code.e("                                    int num_gpus,");
    code.e("                                    int my_gpu_id) {");
    code.e("assert(num_gpus = $);", num_gpus);

    if (use_json_format) {
      code.e("std::map<std::string, void*> all_tensors;");
    }
    for (auto const &iter : io_configs) {
      IODesc desc = iter.second;
      switch (desc.type) {
        case IODesc::TorchTensor: {
          code.e("char *$ = (char*)($);", desc.name, desc.torch_data_ptr);
          if (use_json_format) {
            code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
          }
          break;
        }
        case IODesc::FusedTorchTensor: {
          for (auto const &sdesc : desc.sub_descs) {
            code.e("char *$ = (char*)($);", sdesc.name, sdesc.torch_data_ptr);
            if (use_json_format) {
              code.e("all_tensors[\"$\"] = $;", sdesc.name, sdesc.name);
            }
          }
          break;
        }
        case IODesc::CUDAMallocTensor: {
          code.e("void *$;", desc.name);
          size_t size = megakernel::type::get_datatype_size(
              static_cast<type::DataType>(desc.tensor.data_type));
          for (int i = 0; i < desc.tensor.num_dims; i++) {
            size *= desc.tensor.dim[i];
          }
          code.e("CUDA_CHECK(cudaMalloc(&$, $));", desc.name, size);
          if (use_json_format) {
            code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
          }
          break;
        }
        case IODesc::NVSHMEMMallocTensor: {
          size_t size = megakernel::type::get_datatype_size(
              static_cast<type::DataType>(desc.tensor.data_type));
          for (int i = 0; i < desc.tensor.num_dims; i++) {
            size *= desc.tensor.dim[i];
          }
          code.e("void *$ = nvshmem_malloc($);", desc.name, size);
          code.e("assert($ != nullptr);", desc.name);
          if (use_json_format) {
            code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
          }
          break;
        }
        case IODesc::ShuffledTorchTensor: {
          code.e("char *$;", desc.name);
          size_t size = megakernel::type::get_datatype_size(
              static_cast<type::DataType>(desc.tensor.data_type));
          for (int i = 0; i < desc.tensor.num_dims; i++) {
            size *= desc.tensor.dim[i];
          }
          code.e("CUDA_CHECK(cudaMalloc(&$, $));", desc.name, size);

          size_t bytes_per_row = size / desc.tensor.dim[0];
          size_t bytes_per_group = 0;
          std::vector<size_t> bytes_per_tensor;
          for (size_t i = 0; i < desc.sub_descs.size(); i++) {
            bytes_per_group +=
                bytes_per_row * desc.sub_descs[i].tensor.dim[0] / desc.num_groups;
            bytes_per_tensor.push_back(bytes_per_row *
                                      desc.sub_descs[i].tensor.dim[0] /
                                      desc.num_groups);
          }
          size_t start_addr_offset = 0;
          for (size_t i = 0; i < desc.sub_descs.size(); i++) {
            code.e("CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>($ + "
                  "$), $, "
                  "reinterpret_cast<const void *>($), $, $, $, "
                  "cudaMemcpyDeviceToDevice));",
                  desc.name,         /*dst address*/
                  start_addr_offset, /*dst bytes offset between each copy*/
                  bytes_per_group,   /*dst bytes offset between each copy*/
                  desc.sub_descs[i].torch_data_ptr, /*src address*/
                  bytes_per_tensor[i], /*src bytes offset between each copy*/
                  bytes_per_tensor[i], /*width*/
                  desc.num_groups      /*height*/
            );
            start_addr_offset += bytes_per_tensor[i];
          }
          if (use_json_format) {
            code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
          }
          break;
        }
        default:
          assert(false);
      }
    }

    json json_task_graph = {
        {"all_tasks", {}}, {"all_events", {}}, {"first_tasks", {}}};
    // generate task[0]
    {
      tgbody.e("all_tasks.push_back(FullTaskDesc(TASK_TERMINATE));");
      json_task_graph["all_tasks"].push_back(
          json{{"task_type", TASK_TERMINATE},
              {"variant_id", 0},
              {"inputs", {}},
              {"outputs", {}},
              {"trigger_event", EVENT_INVALID_ID},
              {"dependent_event", EVENT_INVALID_ID},
              {"request_id", -1},
              {"expert_offset", -1},
              {"kv_idx", -1},
              {"merge_task_offset", -1}});
    }
    // generate task[1]
    {
      tgbody.e("all_tasks.push_back(FullTaskDesc(TASK_BEGIN_TASK_GRAPH));");
      json_task_graph["all_tasks"].push_back(
          json{{"task_type", TASK_BEGIN_TASK_GRAPH},
              {"variant_id", 0},
              {"inputs", {}},
              {"outputs", {}},
              {"trigger_event", get_event_id(my_gpu_id, 1 /*event_pos*/, false /*is_nvshmem*/)},
              {"dependent_event", EVENT_INVALID_ID},
              {"request_id", -1},
              {"expert_offset", -1},
              {"kv_idx", -1},
              {"merge_task_offset", -1}});
    }

    // generate all other tasks
    size_t task_pos = 2;
    for (auto const &op : graph.operators) {
      if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
        continue;
      }
      assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
      std::tuple<int, int, TaskType, int> task_config =
          task_configs.find(op)->second;

      assert(all_task_maps.find(op) != all_task_maps.end());
      std::map<dim3, TaskId, Dim3Comparator> const &task_map =
          all_task_maps.find(op)->second;
      // Customized op
      kn::KNCustomizedOp const *cur_op =
          dynamic_cast<kn::KNCustomizedOp const *>(op);
      tb::Graph const &bgraph = cur_op->bgraph;
      dim3 bid;
      std::vector<tb::TBInputOp *> input_ops;
      std::vector<tb::TBInputOp *> output_ops;
      int num_inputs = std::get<0>(task_config);
      // int num_outputs = std::get<1>(task_config);
      TaskType task_type = std::get<2>(task_config);
      for (auto const &op : bgraph.operators) {
        assert(op->op_type == megakernel::type::TB_INPUT_OP);
        if (input_ops.size() < (size_t)num_inputs) {
          input_ops.push_back(static_cast<tb::TBInputOp *>(op));
        } else {
          output_ops.push_back(static_cast<tb::TBInputOp *>(op));
        }
      }

      for (int i = 0;
          i < bgraph.grid_dim.x * bgraph.grid_dim.y * bgraph.grid_dim.z;
          i++) {
        // printf("iii0: %d\n", i);
        FullTaskDesc task_desc = all_tasks[task_pos];
        assert(task_desc.task_type == task_type || task_type == TASK_ALLREDUCE);
        // find current task in grid_dim
        for (int j = 0; j < bgraph.grid_dim.x * bgraph.grid_dim.y * bgraph.grid_dim.z; j++) {
          bid.x = j / (bgraph.grid_dim.y * bgraph.grid_dim.z);
          bid.y = (j % (bgraph.grid_dim.y * bgraph.grid_dim.z)) / bgraph.grid_dim.z;
          bid.z = j % bgraph.grid_dim.z;
          // printf("iiiz: %d, %d, %d\n", bid.x, bid.y, bid.z);
          TaskId task_id = task_map.at(bid);
          // printf("iiiz1: %d, %d, %d\n", bid.x, bid.y, bid.z);
          if (task_pos == (task_id & 0xffffffff)) {
            break;
          }
        }
        // printf("iii1: %d\n", i);
        TaskId task_id = task_map.at(bid);
        assert(task_pos == (task_id & 0xffffffff));
        tgbody.e("// task[$]", task_pos);
        tgbody.e("{");
        tgbody.e("FullTaskDesc task_desc(static_cast<TaskType>($));",
                task_desc.task_type);
        size_t gpu_id = ((task_desc.trigger_event >> 32) & 0xffff);
        // size_t event_pos = (task_desc.trigger_event & 0xffffffff);
        bool is_nvshmem_event =
            ((task_desc.trigger_event & EVENT_NVSHMEM_TAG) > 0);
        assert(gpu_id == my_gpu_id);
        assert(!is_nvshmem_event);
        json json_task;
        json_task = {
            {"task_type", task_desc.task_type},
            {"variant_id", task_desc.variant_id},
            {"inputs", {}},
            {"outputs", {}},
            {"trigger_event", task_desc.trigger_event},
            {"dependent_event", task_desc.dependent_event},
            {"request_id", task_desc.task_metadata.request_id},
            {"expert_offset", task_desc.task_metadata.expert_offset},
            {"kv_idx", task_desc.task_metadata.kv_idx},
            {"merge_task_offset", task_desc.task_metadata.merge_task_offset}};

        /////////////////////////
        // Input 
        /////////////////////////
        for (int i = 0; i < task_desc.num_inputs; i++) {
          if (input_ops[i]->dtensor == kernel::DTensor::EMPTY_TENSOR) {
            json json_dims = json::array();
            json json_strides = json::array();
                                                        //  {"offset", 0}, CJM
            json_task["inputs"].push_back(json{{"base_ptr", "nullptr"},
                                              {"bx", 0},
                                              {"by", 0},
                                              {"bz", 0},
                                              {"data_type", type::DT_UNKNOWN},
                                              {"dims", json_dims},
                                              {"strides", json_strides}});
            continue;
          }
          // off_t offset = 0;
          int num_dims = input_ops[i]->dtensor.num_dims;
          int3 input_map = input_ops[i]->input_map;
          IODesc io_desc = io_configs.find(input_ops[i]->dtensor.guid)->second;
          assert(input_ops[i]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
          if (io_desc.type == IODesc::FusedTorchTensor) {
            // Currently assert that we fuse the 0-th dim (i.e., 0)
            int fused_group_size = 0;
            std::vector<int> group_sizes;
            for (auto const &sub_desc : io_desc.sub_descs) {
              assert(sub_desc.tensor.num_dims == num_dims);
              assert(sub_desc.tensor.dim[0] % io_desc.num_groups == 0);
              int my_group_size = sub_desc.tensor.dim[0] / io_desc.num_groups;
              fused_group_size += my_group_size;
              group_sizes.push_back(my_group_size);
            }
            assert(io_desc.tensor.dim[0] ==
                  fused_group_size * io_desc.num_groups);
            assert(io_desc.tensor.num_dims == num_dims);
            int fused_dim_off = 0;
            if (input_map.x == 0) {
              fused_dim_off = io_desc.tensor.dim[0] / bgraph.grid_dim.x * bid.x;
            }
            if (input_map.y == 0) {
              fused_dim_off = io_desc.tensor.dim[0] / bgraph.grid_dim.y * bid.y;
            }
            if (input_map.z == 0) {
              fused_dim_off = io_desc.tensor.dim[0] / bgraph.grid_dim.z * bid.z;
            }
            int fused_dim_off_in_group = fused_dim_off % fused_group_size;
            size_t index = 0;
            while (index < group_sizes.size()) {
              if (fused_dim_off_in_group >= group_sizes[index]) {
                fused_dim_off_in_group -= group_sizes[index];
                index++;
              } else {
                break;
              }
            }
            IODesc sub_desc = io_desc.sub_descs[index];
            int fused_dim_off_subtensor =
                fused_dim_off / fused_group_size * group_sizes[index] +
                fused_dim_off_in_group;
            // Assert that it is within range
            assert(fused_dim_off_subtensor < sub_desc.tensor.dim[0]);
            int blockIdx_x = bid.x;  // CJM
            int blockIdx_y = bid.y;
            int blockIdx_z = bid.z;
            tgbody.e("TensorDesc input$;", i);
            tgbody.e("input$.base_ptr = static_cast<char*>($) + $;", i, sub_desc.name);
            tgbody.e("input$.bx = $;", i, blockIdx_x);
            tgbody.e("input$.by = $;", i, blockIdx_y);
            tgbody.e("input$.bz = $;", i, blockIdx_z);
            tgbody.e("input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
            tgbody.e("input$.data_type = $;", i, task_desc.inputs[i].data_type);
            json json_dims = json::array();
            json json_strides = json::array();
            for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
              tgbody.e("input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
              tgbody.e("input$.stride[$] = $;", i, d, sub_desc.tensor.stride[d]);
              json_dims.push_back(task_desc.inputs[i].dim[d]);
              json_strides.push_back(sub_desc.tensor.stride[d]);
            }
            tgbody.e("task_desc.inputs[$] = input$;", i, i);
            json_task["inputs"].push_back(json{
                {"base_ptr", sub_desc.name},
                {"bx", blockIdx_x},
                {"by", blockIdx_y},
                {"bz", blockIdx_z},
                // {"offset", offset * type::get_datatype_size(static_cast<type::DataType>(sub_desc.tensor.data_type))},
                {"data_type", task_desc.inputs[i].data_type},
                {"dims", json_dims},
                {"strides", json_strides}});
          } else {
            int blockIdx_x = bid.x;  // CJM
            int blockIdx_y = bid.y;
            int blockIdx_z = bid.z;
            tgbody.e("TensorDesc input$;", i);
            tgbody.e("input$.base_ptr = static_cast<char*>($);", i, io_desc.name);
            tgbody.e("input$.bx = $;", i, blockIdx_x);
            tgbody.e("input$.by = $;", i, blockIdx_y);
            tgbody.e("input$.bz = $;", i, blockIdx_z);
            tgbody.e("input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
            tgbody.e("input$.data_type = $;", i, task_desc.inputs[i].data_type);
            json json_dims = json::array();
            json json_strides = json::array();
            for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
              tgbody.e("input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
              tgbody.e(
                  "input$.stride[$] = $;", i, d, task_desc.inputs[i].stride[d]);
              json_dims.push_back(task_desc.inputs[i].dim[d]);
              json_strides.push_back(task_desc.inputs[i].stride[d]);
            }
            tgbody.e("task_desc.inputs[$] = input$;", i, i);
            json_task["inputs"].push_back(json{
                {"base_ptr", io_desc.name},
                {"bx", blockIdx_x},
                {"by", blockIdx_y},
                {"bz", blockIdx_z},
                {"data_type", task_desc.inputs[i].data_type},
                {"dims", json_dims},
                {"strides", json_strides}});
          }
        }
        /////////////////////////
        // Output 
        /////////////////////////

        for (int i = 0; i < task_desc.num_outputs; i++) {
          IODesc io_desc = io_configs.find(output_ops[i]->dtensor.guid)->second;
          
          int blockIdx_x = bid.x;  // CJM
          int blockIdx_y = bid.y;
          int blockIdx_z = bid.z;
          tgbody.e("TensorDesc output$;", i);
          tgbody.e("output$.base_ptr = static_cast<char*>($);", i, io_desc.name);
          tgbody.e("input$.bx = $;", i, blockIdx_x);
          tgbody.e("input$.by = $;", i, blockIdx_y);
          tgbody.e("input$.bz = $;", i, blockIdx_z);
          tgbody.e("output$.num_dims = $;", i, task_desc.outputs[i].num_dims);
          tgbody.e("output$.data_type = $;", i, task_desc.outputs[i].data_type);
          json json_dims = json::array();
          json json_strides = json::array();
          for (int d = 0; d < task_desc.outputs[i].num_dims; d++) {
            tgbody.e("output$.dim[$] = $;", i, d, task_desc.outputs[i].dim[d]);
            tgbody.e(
                "output$.stride[$] = $;", i, d, task_desc.outputs[i].stride[d]);
            json_dims.push_back(task_desc.outputs[i].dim[d]);
            json_strides.push_back(task_desc.outputs[i].stride[d]);
          }
          tgbody.e("task_desc.outputs[$] = output$;", i, i);
          json_task["outputs"].push_back(
              json{{"base_ptr", io_desc.name},
                  {"bx", blockIdx_x},
                  {"by", blockIdx_y},
                  {"bz", blockIdx_z},
                  {"data_type", task_desc.outputs[i].data_type},
                  {"dims", json_dims},
                  {"strides", json_strides}});
        }
        tgbody.e("all_tasks.push_back(task_desc);");
        tgbody.e("}");
        json_task_graph["all_tasks"].push_back(json_task);
        task_pos++;
      }
    }

    assert(task_pos == all_tasks.size());
    // Add all events
    for (auto const &event : all_events) {
      tgbody.e(
          "all_events.push_back(EventDesc(static_cast<EventType>($), $, $, $));",
          event.event_type,
          event.num_triggers,
          event.first_task_id,
          event.last_task_id);
      json_task_graph["all_events"].push_back(
          json{{"event_type", event.event_type},
              {"num_triggers", event.num_triggers},
              {"first_task_id", event.first_task_id},
              {"last_task_id", event.last_task_id}});
    }
    // Add first task
    for (auto const &task : first_tasks) {
      tgbody.e("first_tasks.push_back($);", task);
      json_task_graph["first_tasks"].push_back(task);
    }
    if (use_json_format) {
      // Add nullptr for tensors set as None
      code.e("all_tensors[\"nullptr\"] = nullptr;");
      code.e("adjust_params_with_kernel_id(kernel_id, all_tensors);");
      // code.e("if (kernel_id == 1) {");
      // code.e("all_tensors[\"w1\"] = w2;");
      // code.e("}");
      code.e("construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, "
            "first_tasks, all_tensors);");
    } else {
      code.e(tgbody.to_string());
    }
    // ensure cudaMemcpyAsync is completed
    code.e("cudaDeviceSynchronize();");
    code.e("}");
    code.e("");

    // Generate task implementation
    std::map<TaskType, std::string> task_type_to_name;
    task_type_to_name[TASK_EMBEDDING] = "TASK_EMBEDDING";
    task_type_to_name[TASK_RMS_NORM] = "TASK_RMS_NORM";
    task_type_to_name[TASK_RMS_NORM_LINEAR] = "TASK_RMS_NORM_LINEAR";
    task_type_to_name[TASK_ATTENTION_1] = "TASK_ATTENTION_1";
    task_type_to_name[TASK_SILU_MUL] = "TASK_SILU_MUL";
    task_type_to_name[TASK_IDENTITY] = "TASK_IDENTITY";
    task_type_to_name[TASK_SILU_MUL_LINEAR_WITH_RESIDUAL] = "TASK_SILU_MUL_LINEAR_WITH_RESIDUAL";
    task_type_to_name[TASK_LINEAR] = "TASK_LINEAR";
    task_type_to_name[TASK_LINEAR_WITH_RESIDUAL] = "TASK_LINEAR_WITH_RESIDUAL";
    task_type_to_name[TASK_SILU_MUL_LINEAR] = "TASK_SILU_MUL_LINEAR";
    task_type_to_name[TASK_ARGMAX_PARTIAL] = "TASK_ARGMAX_PARTIAL";
    task_type_to_name[TASK_ARGMAX_REDUCE] = "TASK_ARGMAX_REDUCE";
    task_type_to_name[TASK_NVSHMEM_COPY] = "TASK_NVSHMEM_COPY";
    task_type_to_name[TASK_REDUCE] = "TASK_REDUCE";
    task_type_to_name[TASK_FIND_NGRAM_PARTIAL] = "TASK_FIND_NGRAM_PARTIAL";
    task_type_to_name[TASK_FIND_NGRAM_GLOBAL] = "TASK_FIND_NGRAM_GLOBAL";
    task_type_to_name[TASK_TARGET_VERIFY_GREEDY] = "TASK_TARGET_VERIFY_GREEDY";
    task_type_to_name[TASK_SINGLE_BATCH_EXTEND_ATTENTION] = "TASK_SINGLE_BATCH_EXTEND_ATTENTION";
    task_type_to_name[TASK_PAGED_ATTENTION_1] = "TASK_PAGED_ATTENTION_1";
    task_type_to_name[TASK_LINEAR_HOPPER] = "TASK_LINEAR_HOPPER";
    task_type_to_name[TASK_LINEAR_WITH_RESIDUAL_HOPPER] = "TASK_LINEAR_WITH_RESIDUAL_HOPPER";
    task_type_to_name[TASK_PAGED_ATTENTION_HOPPER] = "TASK_PAGED_ATTENTION_HOPPER";
    task_type_to_name[TASK_RMS_NORM_HOPPER] = "TASK_RMS_NORM_HOPPER";
    task_type_to_name[TASK_LINEAR_SWAPAB_HOPPER] = "TASK_LINEAR_SWAPAB_HOPPER";
    task_type_to_name[TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER] = "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER";
    task_type_to_name[TASK_LINEAR_CUTLASS_HOPPER] = "TASK_LINEAR_CUTLASS_HOPPER";
    task_type_to_name[TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER] = "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER";
    task_type_to_name[TASK_SILU_MUL_HOPPER] = "TASK_SILU_MUL_HOPPER";
    task_type_to_name[TASK_EMBEDDING_HOPPER] = "TASK_EMBEDDING_HOPPER";
    task_type_to_name[TASK_LINEAR_SM100] = "TASK_LINEAR_SM100";
    task_type_to_name[TASK_LINEAR_WITH_RESIDUAL_SM100] = "TASK_LINEAR_WITH_RESIDUAL_SM100";
    task_type_to_name[TASK_SPLITK_LINEAR_SM100] = "TASK_SPLITK_LINEAR_SM100";
    task_type_to_name[TASK_ATTN_SM100] = "TASK_ATTN_SM100";
    task_type_to_name[TASK_ARGMAX_PARTIAL_SM100] = "TASK_ARGMAX_PARTIAL_SM100";
    task_type_to_name[TASK_ARGMAX_REDUCE_SM100] = "TASK_ARGMAX_REDUCE_SM100";
    task_type_to_name[TASK_SAMPLING_SM100] = "TASK_SAMPLING_SM100";
    task_type_to_name[TASK_TENSOR_INIT] = "TASK_TENSOR_INIT";
    task_type_to_name[TASK_MOE_TOPK_SOFTMAX_SM100] = "TASK_MOE_TOPK_SOFTMAX_SM100";
    task_type_to_name[TASK_MOE_W13_LINEAR_SM100] = "TASK_MOE_W13_LINEAR_SM100";
    task_type_to_name[TASK_MOE_W2_LINEAR_SM100] = "TASK_MOE_W2_LINEAR_SM100";
    task_type_to_name[TASK_MOE_MUL_SUM_ADD_SM100] = "TASK_MOE_MUL_SUM_ADD_SM100";
    task_type_to_name[TASK_MOE_W13_LINEAR_SM90] = "TASK_MOE_W13_LINEAR_SM90";
    task_type_to_name[TASK_MOE_W2_LINEAR_SM90] = "TASK_MOE_W2_LINEAR_SM90";
    task_type_to_name[TASK_SPLITK_LINEAR_SWAPAB_HOPPER] = "TASK_SPLITK_LINEAR_SWAPAB_HOPPER";
    task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_SM100] = "TASK_PAGED_ATTENTION_SPLIT_KV_SM100";
    task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100] = "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100";
    task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER] = "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER";

    code.e("__device__ __forceinline__");
    code.e("void _execute_task(TaskDesc const* task_desc,");
    code.e("                   RuntimeConfig const &runtime_config) {");
    TaskRegister *task_register = TaskRegister::get_instance();
    bool first_task = true;
    for (auto const &task : task_register->all_task_variants) {
      for (size_t variant_id = 0; variant_id < task.second.size(); variant_id++) {
        std::string cond = first_task ? "if" : "else if";
        assert(task_type_to_name.find(task.first) != task_type_to_name.end());
        code.e("$ (task_desc->task_type == $ && task_desc->variant_id == $) {",
              cond,
              task_type_to_name[task.first],
              variant_id);
        code.e("$", task.second[variant_id]);
        code.e("}");
        first_task = false;
      }
    }
    code.e("}");

    // Write json to output file
    // std::ofstream out("task_graph.json");
    // out << json_task_graph.dump(2);
    // out.close();
    TaskGraphResult result;
    result.cuda_code = code.to_string();
    result.json_file = json_task_graph.dump(2);
    return result;
  }


public:
  std::vector<megakernel::kernel::KNOperator *> operators;
  dim3 gpu_dim;
  // memory allocator
  // device memory offset manager
  off_t dmem_data_offset;
  std::vector<std::pair<off_t, size_t>> allocated_data_tensors;

  // Fields for persistent kernels
  std::map<megakernel::type::GuidType, megakernel::runtime::IODesc> io_config;
  TaskConfigMap task_config;

  using OpType = KNOperator;
  using TensorType = DTensor;
};

} // namespace kernel
} // namespace megakernel
