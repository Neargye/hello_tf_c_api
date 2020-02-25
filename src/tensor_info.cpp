// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2020 Daniil Goncharov <neargye@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <iostream>
#include <vector>
#include <string>

void PrintInputs(TF_Graph*, TF_Operation* op) {
  auto num_inputs = TF_OperationNumInputs(op);

  for (auto i = 0; i < num_inputs; ++i) {
    auto input = TF_Input{op, i};
    auto type = TF_OperationInputType(input);
    std::cout << "Input: " << i << " type: " << tf_utils::DataTypeToString(type) << std::endl;
  }
}

void PrintOutputs(TF_Graph* graph, TF_Operation* op, TF_Status* status) {
  auto num_outputs = TF_OperationNumOutputs(op);

  for (int i = 0; i < num_outputs; ++i) {
    auto output = TF_Output{op, i};
    auto type = TF_OperationOutputType(output);
    auto num_dims = TF_GraphGetTensorNumDims(graph, output, status);

    if (TF_GetCode(status) != TF_OK) {
      std::cout << "Can't get tensor dimensionality" << std::endl;
      continue;
    }

    std::cout << " dims: " << num_dims;

    if (num_dims <= 0) {
      std::cout << " []" << std::endl;;
      continue;
    }

    std::vector<std::int64_t> dims(num_dims);

    std::cout << "Output: " << i << " type: " << tf_utils::DataTypeToString(type);
    TF_GraphGetTensorShape(graph, output, dims.data(), num_dims, status);

    if (TF_GetCode(status) != TF_OK) {
      std::cout << "Can't get get tensor shape" << std::endl;
      continue;
    }

    std::cout << " [";
    for (auto d = 0; d < num_dims; ++d) {
      std::cout << dims[d];
      if (d < num_dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
}

void PrintTensorInfo(TF_Graph* graph, const char* layer_name, TF_Status* status) {
  std::cout << "Tensor: " << layer_name;
  auto op = TF_GraphOperationByName(graph, layer_name);

  if (op == nullptr) {
    std::cout << "Could not get " << layer_name << std::endl;
    return;
  }

  auto num_inputs = TF_OperationNumInputs(op);
  auto num_outputs = TF_OperationNumOutputs(op);
  std::cout << " inputs: " << num_inputs << " outputs: " << num_outputs << std::endl;

  PrintInputs(graph, op);

  PrintOutputs(graph, op, status);
}

int main() {
  auto graph = tf_utils::LoadGraph("graph.pb");
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; // Auto-delete on scope exit.
  if (graph == nullptr) {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.

  PrintTensorInfo(graph, "input_4", status);
  std::cout << std::endl;

  PrintTensorInfo(graph, "output_node0", status);

  return 0;
}
