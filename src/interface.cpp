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

int main() {
  auto graph = tf_utils::LoadGraph("graph.pb");
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; // Auto-delete on scope exit.
  if (graph == nullptr) {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  const std::vector<std::int64_t> input_dims = {1, 5, 12};
  const std::vector<float> input_vals = {
    -0.4809832f, -0.3770838f, 0.1743573f, 0.7720509f, -0.4064746f, 0.0116595f, 0.0051413f, 0.9135732f, 0.7197526f, -0.0400658f, 0.1180671f, -0.6829428f,
    -0.4810135f, -0.3772099f, 0.1745346f, 0.7719303f, -0.4066443f, 0.0114614f, 0.0051195f, 0.9135003f, 0.7196983f, -0.0400035f, 0.1178188f, -0.6830465f,
    -0.4809143f, -0.3773398f, 0.1746384f, 0.7719052f, -0.4067171f, 0.0111654f, 0.0054433f, 0.9134697f, 0.7192584f, -0.0399981f, 0.1177435f, -0.6835230f,
    -0.4808300f, -0.3774327f, 0.1748246f, 0.7718700f, -0.4070232f, 0.0109549f, 0.0059128f, 0.9133330f, 0.7188759f, -0.0398740f, 0.1181437f, -0.6838635f,
    -0.4807833f, -0.3775733f, 0.1748378f, 0.7718275f, -0.4073670f, 0.0107582f, 0.0062978f, 0.9131795f, 0.7187147f, -0.0394935f, 0.1184392f, -0.6840039f,
  };

  const std::vector<TF_Output> input_ops = {{TF_GraphOperationByName(graph, "input_4"), 0}};
  const std::vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_FLOAT, input_dims, input_vals)};
  SCOPE_EXIT{ tf_utils::DeleteTensors(input_tensors); }; // Auto-delete on scope exit.

  const std::vector<TF_Output> out_ops = {{TF_GraphOperationByName(graph, "output_node0"), 0}};
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); }; // Auto-delete on scope exit.

  auto session = tf_utils::CreateSession(graph);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); }; // Auto-delete on scope exit.
  if (session == nullptr) {
    std::cout << "Can't create session" << std::endl;
    return 2;
  }

  auto code = tf_utils::RunSession(session, input_ops, input_tensors, out_ops, output_tensors);

  if (code == TF_OK) {
    auto result = tf_utils::GetTensorData<float>(output_tensors[0]);
    std::cout << "Output vals: " << result[0] << ", " << result[1] << ", " << result[2] << ", " << result[3] << std::endl;
  } else {
    std::cout << "Error run session TF_CODE: " << code;
    return code;
  }

  return 0;
}
