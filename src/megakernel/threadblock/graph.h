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

#include "megakernel/config.h"
#include "megakernel/kernel/device_tensor.h"
#include "megakernel/threadblock/operator.h"
// #include "megakernel/threadblock/smem_tensor.h"
#include <vector>

namespace megakernel {
namespace threadblock {

class Graph {

public:
  Graph()    
    : grid_dim(1, 1, 1), block_dim(1, 1, 1), thread_num(128) {}

  Graph(dim3 _grid_dim, dim3 _block_dim, int _thread_num)
    : grid_dim(_grid_dim), block_dim(_block_dim), thread_num(_thread_num) {
  // A bgraph cannot have more than MAX_NUM_THREADBLOCKS_PER_KERNEL threadblocks
  // otherwise we don't have enough buffers in device memory for saving
  // fingerprints
  assert(grid_dim.x * grid_dim.y * grid_dim.z <=
      megakernel::config::MAX_NUM_THREADBLOCKS_PER_KERNEL);
  }

  ~Graph() {
    while (!operators.empty()) {
      delete operators.back();
      operators.pop_back();
    }
  }

  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator
      
  kernel::DTensor *new_input(megakernel::kernel::DTensor const *dtensor,
                     int3 input_map){
    TBOperator *op = create_input_op(
        dtensor == nullptr ? kernel::DTensor::EMPTY_TENSOR : *dtensor,
        input_map);
    assert(op != nullptr);
    operators.push_back(op);
    return &op->output_tensors[0];
  }
  TBOperator *create_input_op(megakernel::kernel::DTensor const &dtensor,
                              int3 input_map){
    TBOperator *op = new TBOperator(grid_dim, dtensor, input_map);
    return op;
  }

public:
  dim3 grid_dim, block_dim;
  int thread_num;
  std::vector<megakernel::threadblock::TBOperator *> operators;

  using OpType = TBOperator;
};

////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace megakernel
