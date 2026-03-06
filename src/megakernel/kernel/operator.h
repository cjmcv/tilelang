/* Copyright 2023-2024 CMU
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

#include "megakernel/kernel/device_tensor.h"
#include "megakernel/kernel/tb_graph.h"
#include <vector>

namespace megakernel {
namespace kernel {


class Graph;

class KNOperator {
public:
  KNOperator(megakernel::type::KNOperatorType _type): op_type(_type) {}
  KNOperator(megakernel::type::KNOperatorType _type, DTensor const &input1):op_type(_type) {
    input_tensors.push_back(input1);
  }

  KNOperator(megakernel::type::KNOperatorType _type,
             DTensor const &input1, DTensor const &input2) : op_type(_type) {
    input_tensors.push_back(input1);
    input_tensors.push_back(input2);
  }

  KNOperator(megakernel::type::KNOperatorType _type,
             std::vector<DTensor> const &inputs) : op_type(_type) {
    for (auto const &i : inputs) {
      input_tensors.push_back(i);
    }
  }

  int get_output_dtensors(DTensor **outputs) {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      outputs[i] = &output_tensors[i];
    }
    return output_tensors.size();
  }
  
  std::vector<DTensor>& get_output_dtensors() {
    return output_tensors;
  }
  virtual ~KNOperator() {}

public:
  megakernel::type::KNOperatorType op_type;
  std::vector<DTensor> input_tensors;
  std::vector<DTensor> output_tensors;
};

class KNInputOp : public KNOperator {
public:
  KNInputOp(std::vector<int> const &dims,
            std::vector<size_t> const &strides,
            megakernel::type::DataType data_type,
            int3 _input_map = {-1, -1, -1})
      : KNOperator(megakernel::type::KN_INPUT_OP), input_strides(strides),
    input_map(_input_map) {
    assert(dims.size() == strides.size());
    DTensor tensor;
    tensor.num_dims = dims.size();
    for (int i = tensor.num_dims - 1; i >= 0; i--) {
      tensor.dim[i] = dims[i];
    }
    tensor.data_type = data_type;
    tensor.owner_op = this;
    tensor.guid = DTensor::next_guid++;
    // kgraph->allocate(tensor);
    output_tensors.push_back(tensor);
  }
  ~KNInputOp() {
    // kgraph->free(output_tensors[0]);
  }

public:
  std::vector<size_t> input_strides;
  int3 input_map;
};

class KNCustomizedOp : public megakernel::kernel::KNOperator {
public:
  KNCustomizedOp(Graph *_kgraph,
                 std::vector<DTensor> const &_inputs,
                 megakernel::threadblock::TBGraph const &_graph)
                 : KNOperator(megakernel::type::KN_CUSTOMIZED_OP, _inputs),
      bgraph(_graph.grid_dim,
             _graph.block_dim,
             _graph.thread_num) {
    size_t input_idx = 0;
    for (auto const &op : _graph.operators) {
      megakernel::threadblock::TBOperator *input_op =
          static_cast<megakernel::threadblock::TBOperator *>(op);
      bgraph.new_input(&_inputs[input_idx++],
                      input_op->input_map);
    }
  }
  virtual ~KNCustomizedOp() {}

public:
  megakernel::threadblock::TBGraph bgraph;
};

} // namespace kernel
} // namespace megakernel
