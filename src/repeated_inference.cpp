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
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool RunReference(TF_Graph* graph, TF_Output input, TF_Output output,
                  const std::vector<std::int64_t>& input_dims, const std::vector<float>& input_values,
                  std::vector<float>& result,
                  TF_Status* status) {
  auto reference_session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(reference_session); };
  if (reference_session == nullptr) {
    std::cout << "Failed to create reference session: " << TF_Message(status) << std::endl;
    return false;
  }

  auto reference_input = tf_utils::CreateTensor(input_dims, input_values);
  SCOPE_EXIT{ tf_utils::DeleteTensor(reference_input); };
  if (reference_input == nullptr) {
    std::cout << "Failed to create reference input" << std::endl;
    return false;
  }

  TF_Tensor* reference_output = nullptr;
  SCOPE_EXIT{ tf_utils::DeleteTensor(reference_output); };
  if (tf_utils::RunSession(reference_session, &input, &reference_input, 1, &output, &reference_output, 1, status) != TF_OK) {
    std::cout << "Failed to run reference session: " << TF_Message(status) << std::endl;
    return false;
  }

  if (tf_utils::GetTensorData(reference_output, result, status) != TF_OK) {
    std::cout << "Invalid reference output: " << TF_Message(status) << std::endl;
    return false;
  }
  return true;
}

} // namespace

int main() {
  auto graph = tf_utils::LoadGraph("graph.pb");
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); };
  if (graph == nullptr) {
    std::cout << "Failed to load graph" << std::endl;
    return 1;
  }

  const auto input = TF_Output{TF_GraphOperationByName(graph, "input_4"), 0};
  if (input.oper == nullptr) {
    std::cout << "Failed to find input operation" << std::endl;
    return 2;
  }

  const auto output = TF_Output{TF_GraphOperationByName(graph, "output_node0"), 0};
  if (output.oper == nullptr) {
    std::cout << "Failed to find output operation" << std::endl;
    return 3;
  }

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  if (session == nullptr || TF_GetCode(status) != TF_OK) {
    std::cout << "Failed to create session: " << TF_Message(status) << std::endl;
    return 4;
  }

  const std::vector<std::int64_t> input_dims = {1, 5, 12};
  std::vector<float> input_values(60, 0.0f);
  auto input_tensor = tf_utils::CreateTensor(input_dims, input_values);
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor); };
  if (input_tensor == nullptr) {
    std::cout << "Failed to create input tensor" << std::endl;
    return 5;
  }

  const std::vector<TF_Output> inputs = {input};
  const std::vector<TF_Tensor*> input_tensors = {input_tensor};
  const std::vector<TF_Output> outputs = {output};

  std::vector<float> last_result;
  for (int iteration = 0; iteration < 10; ++iteration) {
    for (std::size_t i = 0; i < input_values.size(); ++i) {
      input_values[i] = static_cast<float>(iteration) + static_cast<float>(i) / 100.0f;
    }

    if (!tf_utils::SetTensorData(input_tensor, input_values)) {
      std::cout << "Failed to update input tensor" << std::endl;
      return 6;
    }

    std::vector<TF_Tensor*> output_tensors = {nullptr};
    SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); };

    const auto code = tf_utils::RunSession(session, inputs, input_tensors, outputs, output_tensors, status);
    if (code != TF_OK) {
      std::cout << "Failed to run session: " << TF_Message(status) << std::endl;
      return 7;
    }

    if (tf_utils::GetTensorData(output_tensors[0], last_result, status) != TF_OK) {
      std::cout << "Failed to read output tensor: " << TF_Message(status) << std::endl;
      return 8;
    }
    if (last_result.size() != 4) {
      std::cout << "Unexpected output tensor size" << std::endl;
      return 8;
    }

    // Check reuse against a fresh session and a freshly populated input tensor.
    std::vector<float> expected_result;
    if (!RunReference(graph, input, output, input_dims, input_values, expected_result, status)) {
      return 9;
    }
    if (expected_result.size() != last_result.size()) {
      std::cout << "Invalid reference output: " << TF_Message(status) << std::endl;
      return 9;
    }
    for (std::size_t i = 0; i < last_result.size(); ++i) {
      const auto expected = expected_result[i];
      const auto tolerance = 1.0e-5f * (1.0f + std::abs(expected));
      if (!std::isfinite(expected) || !std::isfinite(last_result[i]) || std::abs(last_result[i] - expected) > tolerance) {
        std::cout << "Unexpected output at iteration " << iteration << ", element " << i << ": expected " << expected << ", got " << last_result[i] << std::endl;
        return 9;
      }
    }
  }

  std::cout << "Ran repeated inference 10 times" << std::endl;
  std::cout << "Last output values: " << last_result[0] << ", " << last_result[1] << ", " << last_result[2] << ", " << last_result[3] << std::endl;

  return 0;
}
