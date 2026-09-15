// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2026 Daniil Goncharov <neargye@gmail.com>.
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
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

void PrintInputs(TF_Operation* op) {
  auto num_inputs = TF_OperationNumInputs(op);

  for (auto i = 0; i < num_inputs; ++i) {
    auto input = TF_Input{op, i};
    auto type = TF_OperationInputType(input);
    std::cout << "Input: " << i << " type: " << tf_utils::DataTypeToString(type) << std::endl;
  }
}

bool PrintOutputs(TF_Graph* graph, TF_Operation* op, TF_Status* status) {
  auto num_outputs = TF_OperationNumOutputs(op);

  for (int i = 0; i < num_outputs; ++i) {
    auto output = TF_Output{op, i};
    auto type = TF_OperationOutputType(output);
    std::cout << "Output: " << i << " type: " << tf_utils::DataTypeToString(type);
    std::optional<std::vector<std::int64_t>> dims;
    if (tf_utils::GetTensorShape(graph, output, dims, status) != TF_OK) {
      std::cout << " Failed to get tensor shape: " << TF_Message(status) << std::endl;
      return false;
    }

    if (!dims) {
      std::cout << " dims: unknown rank" << std::endl;
      continue;
    }

    std::cout << " dims: " << dims->size() << " [";
    for (std::size_t d = 0; d < dims->size(); ++d) {
      std::cout << (*dims)[d];
      if (d + 1 < dims->size()) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  return true;
}

bool PrintTensorInfo(TF_Graph* graph, const char* layer_name, TF_Status* status) {
  std::cout << "Tensor: " << layer_name;
  auto op = TF_GraphOperationByName(graph, layer_name);

  if (op == nullptr) {
    std::cout << "Could not find operation: " << layer_name << std::endl;
    return false;
  }

  auto num_inputs = TF_OperationNumInputs(op);
  auto num_outputs = TF_OperationNumOutputs(op);
  std::cout << " inputs: " << num_inputs << " outputs: " << num_outputs << std::endl;

  PrintInputs(op);

  return PrintOutputs(graph, op, status);
}

int main() {
  auto graph = tf_utils::LoadGraph("graph.pb");
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); };
  if (graph == nullptr) {
    std::cout << "Failed to load graph" << std::endl;
    return 1;
  }

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  if (!PrintTensorInfo(graph, "input_4", status)) {
    return 2;
  }
  std::cout << std::endl;

  return PrintTensorInfo(graph, "output_node0", status) ? 0 : 2;
}
