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
#include "megakernel/threadblock/smem_tensor.h"
#include "megakernel/type.h"
#include <vector>

namespace megakernel {
namespace threadblock {

class TBOperator {
public:
  TBOperator(megakernel::type::TBOperatorType type) : op_type(type) {}
  TBOperator(megakernel::type::TBOperatorType type, STensor const &input1)
    : op_type(type) {
    input_tensors.push_back(input1);
  }

  TBOperator(megakernel::type::TBOperatorType type,
             STensor const &input1,
             STensor const &input2)
             : op_type(type) {
    input_tensors.push_back(input1);
    input_tensors.push_back(input2);
  }

  int get_input_stensors(STensor **inputs) {
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      inputs[i] = &input_tensors[i];
    }
    return input_tensors.size();
  }
  
  int get_output_stensors(STensor **outputs) {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      outputs[i] = &output_tensors[i];
    }
    return output_tensors.size();
  };

  virtual ~TBOperator() {}

  // virtual operator json() const = 0;

public:
  megakernel::type::TBOperatorType op_type;
  std::vector<STensor> input_tensors;
  std::vector<STensor> output_tensors;
};

class TBInputOp : public TBOperator {
public:
  TBInputOp(dim3 grid_dim, off_t smem_offset,
            megakernel::kernel::DTensor const &_dtensor,
            int3 _input_map,
            megakernel::layout::SmemLayout _layout,
            bool store_in_dmem)
      : TBOperator(megakernel::type::TB_INPUT_OP), dtensor(_dtensor),
    input_map(_input_map)  {
    STensor tensor;
    tensor.layout = _layout;
    tensor.num_dims = dtensor.num_dims;
    tensor.data_type = dtensor.data_type;
    for (int i = 0; i < tensor.num_dims; i++) {
      tensor.dim[i] = dtensor.dim[i];
    }

    for (int d = 0; d < 3; d++) {
      int dim_idx = -1;
      int dim_div = 1;
      if (d == 0 && grid_dim.x > 1) {
        dim_idx = input_map.x;
        dim_div = grid_dim.x;
      }
      if (d == 1 && grid_dim.y > 1) {
        dim_idx = input_map.y;
        dim_div = grid_dim.y;
      }
      if (d == 2 && grid_dim.z > 1) {
        dim_idx = input_map.z;
        dim_div = grid_dim.z;
      }
      if (dim_idx >= 0) {
        assert(tensor.dim[dim_idx] > 0);
        // assert(tensor.dim[dim_idx] % dim_div == 0);
        if (tensor.dim[dim_idx] % dim_div != 0) {
          fprintf(stderr, "(tensor.dim[dim_idx] %% dim_div != 0): [tensor.dim[%d]=%d, dim_div=%d]\n", dim_idx, tensor.dim[dim_idx], dim_div);
          abort();
        }
        tensor.dim[dim_idx] /= dim_div;
      }
    }

    tensor.owner_op = this;
    tensor.owner_ts_idx = 0;
    tensor.guid = STensor::next_guid++;
    tensor.after_accum = false;
    tensor.store_in_dmem = store_in_dmem;
    tensor.smem_offset = smem_offset; // bgraph->allocate_fingerprint(tensor);
    output_tensors.push_back(tensor);
  }

  ~TBInputOp() {}

  // operator json() const override;
  size_t get_dtensor_guid() {
    return dtensor.guid;
  }

public:
  megakernel::kernel::DTensor dtensor;
  int3 input_map;
};

} // namespace threadblock
} // namespace megakernel
