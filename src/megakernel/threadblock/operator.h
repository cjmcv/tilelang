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
  TBOperator(dim3 grid_dim, off_t smem_offset,
            megakernel::kernel::DTensor const &_dtensor,
            int3 _input_map,
            bool store_in_dmem)
      : dtensor(_dtensor),
    input_map(_input_map)  {
    STensor tensor;
    tensor.num_dims = dtensor.num_dims;
    tensor.data_type = dtensor.data_type;
    for (int i = 0; i < tensor.num_dims; i++) {
      tensor.dim[i] = dtensor.dim[i];
    }

    tensor.owner_op = this;
    tensor.owner_ts_idx = 0;
    tensor.guid = STensor::next_guid++;
    tensor.store_in_dmem = store_in_dmem;
    tensor.smem_offset = smem_offset; // bgraph->allocate_fingerprint(tensor);
    output_tensors.push_back(tensor);
  }

  ~TBOperator() {}

public:
  megakernel::kernel::DTensor dtensor;
  int3 input_map;

  std::vector<STensor> input_tensors;
  std::vector<STensor> output_tensors;
};

} // namespace threadblock
} // namespace megakernel
