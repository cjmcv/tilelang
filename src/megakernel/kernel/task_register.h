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

#include "megakernel/persistent_kernel/runtime_header.h"
#include "megakernel/kernel/tb_graph.h"
#include "megakernel/kernel/operator.h"
#include "megakernel/code_keeper.h"

namespace megakernel {
namespace runtime {

namespace kn = megakernel::kernel;
namespace tb = megakernel::threadblock;

class TaskRegister {
public:
  TaskRegister() {}

public:
  static TaskRegister *get_instance() {
    static TaskRegister singleton;
    return &singleton;
  }

  int register_rope_task(tb::TBGraph const &bgraph, std::vector<int> const &params) {
    // assert(params.size() == 1);
    // int sub_kernel_id = params[0];
    // printf("register_rope_task.\n");
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 4;
    int num_outputs = 2;
    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }

    assert(output_ops[0]->output_tensors[0].num_dims == 4);
    int batch_size = output_ops[0]->output_tensors[0].dim[0];
    int seqlen = output_ops[0]->output_tensors[0].dim[1];
    int heads = output_ops[0]->output_tensors[0].dim[2];
    int dim = output_ops[0]->output_tensors[0].dim[3];
    int groups = output_ops[1]->output_tensors[0].dim[2];

    assert(batch_size == 1);
    // assert(input_ops[0]->dtensor.num_dims == 2);
    // assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
    // assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::rope_kernel<bfloat16, $, $, $, $, $, $>(",
      bgraph.thread_num, batch_size, seqlen, heads, groups, dim);
    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->input_ptrs[2],");
    code.e("    task_desc->input_ptrs[3],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    task_desc->output_ptrs[1]);");

    return register_task_variant(TASK_ROPE, code.to_string());
  }

  int register_gqa_decode_task(tb::TBGraph const &bgraph, std::vector<int> const &params) {
    assert(params.size() == 1);
    int sub_kernel_id = params[0];
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 5;
    int num_outputs = 3;
    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }

    assert(output_ops[0]->output_tensors[0].num_dims == 3);
    int batch_size = output_ops[0]->output_tensors[0].dim[0];
    int head = output_ops[0]->output_tensors[0].dim[1];
    int dim = output_ops[0]->output_tensors[0].dim[2];
    int groups = input_ops[1]->output_tensors[0].dim[2];

    assert(batch_size == 1);
    // assert(input_ops[0]->dtensor.num_dims == 2);
    // assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
    // assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::gqa_decode_kernel<bfloat16, $, $, $, $, $, $>(",
      bgraph.thread_num, sub_kernel_id, batch_size, head, groups, dim);
    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->input_ptrs[2],");
    code.e("    task_desc->input_ptrs[3],");
    code.e("    runtime_config.step, // task_desc->input_ptrs[4],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    task_desc->output_ptrs[1],");
    code.e("    task_desc->output_ptrs[2]);");
    
    return register_task_variant(TASK_GQA_DECODE, code.to_string());
  }

  int register_rmsnorm_task(tb::TBGraph const &bgraph, std::vector<int> const &params) {
    assert(params.size() == 0);
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 2;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    int batch_size = output_ops[0]->output_tensors[0].dim[0];
    int hidden_dim = output_ops[0]->output_tensors[0].dim[1];
    // Currently assume that each rmsnorm task processes one token
    assert(batch_size == 1);
    assert(input_ops[0]->dtensor.num_dims == 2);
    assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
    assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::rms_norm_kernel<bfloat16, $, $, $, $, $, $>(", 
      bgraph.thread_num, bgraph.block_dim.x, bgraph.block_dim.y, bgraph.block_dim.z, 
      batch_size, hidden_dim);
    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    1e-12f);");
    return register_task_variant(TASK_RMS_NORM, code.to_string());
  }

  int register_rmsnorm_linear_task(tb::TBGraph const &bgraph, std::vector<int> const &params) {
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 3;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    batch_size = output_ops[0]->output_tensors[0].dim[0];
    output_size = output_ops[0]->output_tensors[0].dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    reduction_size = input_ops[0]->dtensor.dim[1];
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);

    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::norm_linear_task_impl<bfloat16, $, $, $, $>(",
          batch_size,
          output_size,
          reduction_size,
          output_stride);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->input_ptrs[2],");
    code.e("    runtime_config.batch_size,");
    code.e("    1e-6f,");
    code.e("    task_desc->output_ptrs[0]);");
    return register_task_variant(TASK_RMS_NORM_LINEAR, code.to_string());
  }

  int register_linear_task(tb::TBGraph const &bgraph, std::vector<int> const &params, bool with_residual, bool with_silu_mul) {
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = with_residual ? 3 : 2;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    batch_size = output_ops[0]->output_tensors[0].dim[0];
    output_size = output_ops[0]->output_tensors[0].dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    reduction_size = input_ops[0]->dtensor.dim[1];
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);

    if (with_silu_mul) {
      reduction_size /= 2;
    }
    megakernel::transpiler::CodeKeeper code;
    code.e("kernel::linear_kernel<bfloat16, $, $, $, $, $, $, $, $, $, $, $>(",
          bgraph.thread_num, bgraph.block_dim.x, bgraph.block_dim.y, bgraph.block_dim.z, 
          batch_size,
          output_size,
          reduction_size,
          output_stride,
          3,
          with_residual,
          with_silu_mul);

    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    if (with_residual) {
      code.e("    task_desc->input_ptrs[2],");
    } else {
      code.e("    nullptr,");
    }
    code.e("    task_desc->output_ptrs[0],");
    code.e("    runtime_config.batch_size,");
    if (with_residual) {
      code.e("    runtime_config.my_gpu_id == 0);");
    } else {
      code.e("    false/*residual*/);");
    }
    
    if (with_residual) {
      return register_task_variant(TASK_LINEAR_WITH_RESIDUAL, code.to_string());
    } else {
      return register_task_variant(TASK_LINEAR, code.to_string());
    }
  }
  
  int register_silu_mul_task(tb::TBGraph const &bgraph, std::vector<int> const &params){
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, input_stride, output_stride;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 1;
    int num_outputs = 1;
    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    batch_size = output_ops[0]->output_tensors[0].dim[0];
    output_size = output_ops[0]->output_tensors[0].dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
    // get input stride
    assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
    input_stride = input_ops[0]->dtensor.dim[1];
    assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::silu_mul_kernel<bfloat16, $, $, $, $, $, $, $, $>(",
          bgraph.thread_num, bgraph.block_dim.x, bgraph.block_dim.y, bgraph.block_dim.z, 
          batch_size, output_size,
          input_stride, output_stride);
    code.e("    task_desc->bx, task_desc->by, task_desc->bz,"); // CJM
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    runtime_config.batch_size);");
    return register_task_variant(TASK_SILU_MUL, code.to_string());
  }


  int register_silu_mul_linear_with_residual_task(tb::TBGraph const &bgraph, std::vector<int> const &params){
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 3;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    batch_size = output_ops[0]->output_tensors[0].dim[0];
    output_size = output_ops[0]->output_tensors[0].dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    reduction_size = input_ops[0]->dtensor.dim[1] / 2;
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);

    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::silu_mul_linear_task_impl<bfloat16, $, $, $, $>(",
          batch_size,
          output_size,
          reduction_size,
          output_stride);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->input_ptrs[2],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    runtime_config.my_gpu_id == 0);");
    return register_task_variant(TASK_SILU_MUL_LINEAR_WITH_RESIDUAL,
                                code.to_string());
  }
  
  int register_reduce_task(tb::TBGraph const &bgraph, std::vector<int> const &params){
    // Currently, allreduce task is split to two sub-tasks: allgather + reduce
    // params[0]: num_gpus
    // params[1]: my_gpu_id
    assert(params.size() == 2);
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 2;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    // For now, the memory partition of the input[0] results in a strided
    // 2D tensor, which cannot be directly transferred by a single nvshmem
    // memput. So we use for loop to iterate over the first dim and transfer each
    // row. If the upperlayer changes this layout, this "for-loop" method can
    // fail. So we assert it here just in case.
    assert(input_ops[0]->input_map.x == 1 && input_ops[0]->input_map.y == -1 &&
          input_ops[0]->input_map.z == -1);
    // Currently support 2D reduction, buffer has an extra world_size dim
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    assert(input_ops[1]->output_tensors[0].num_dims == 3);
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    int batch_size = input_ops[0]->output_tensors[0].dim[0];
    int output_size = input_ops[0]->output_tensors[0].dim[1];
    // get output stride
    assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
    int input_stride = static_cast<int>(kn_input_op->input_strides[0]);
    kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    int output_stride = static_cast<int>(kn_input_op->input_strides[0]);
    assert(input_stride == output_stride);
    // Register nvshmem copy task (allgather)
    megakernel::transpiler::CodeKeeper c;
    c.inc_indent();
    c.e("size_t event_index = "
        "get_event_position_index(task_desc->trigger_event);");
    c.inc_indent();
    c.e("int gpu_id = static_cast<int>(get_event_gpu_id(task_desc->trigger_event));");
    c.e("assert(gpu_id < runtime_config.num_gpus);");
    c.e("assert(gpu_id != runtime_config.my_gpu_id);");
    c.e("for (int i = 0; i < $; i++) {", batch_size);
    c.e("  nvshmemx_putmem_signal_block(");
    c.e("      reinterpret_cast<char*>(task_desc->output_ptrs[0]) + i * $ * "
        "sizeof(bfloat16),",
        input_stride);
    c.e("      reinterpret_cast<char*>(task_desc->input_ptrs[0]) + i * $ * "
        "sizeof(bfloat16),",
        output_stride);
    c.e("      task_desc->xfer_size_in_bytes / $,", batch_size);
    c.e("      reinterpret_cast<uint64_t "
        "*>(&runtime_config.all_event_counters[event_index]),");
    c.e("      1 /*signal*/,");
    c.e("      NVSHMEM_SIGNAL_ADD,");
    c.e("      gpu_id);");
    c.e("}");
    register_task_variant(TASK_NVSHMEM_COPY, c.to_string());
    // Register reduction kernel
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::reduction_kernel<bfloat16, $, $, $, $, $>(",
          params[0],
          params[1],
          batch_size,
          output_size,
          output_stride);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->output_ptrs[0]);");
    return register_task_variant(TASK_REDUCE, code.to_string());
  }
  
  int register_task_variant(TaskType type, std::string const &code) {
    std::vector<std::string> &variants = all_task_variants[type];
    for (size_t i = 0; i < variants.size(); i++) {
      if (variants[i] == code) {
        return (int)(i);
      }
    }
    // Add a new variant
    variants.push_back(code);
    return (int)(variants.size() - 1);
  }

public:
  std::map<TaskType, std::vector<std::string>> all_task_variants;
};

} // namespace runtime
} // namespace megakernel
