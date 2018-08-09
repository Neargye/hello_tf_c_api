// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// Copyright (c) 2018 Daniil Goncharov <neargye@gmail.com>.
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

#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif
#include <c_api.h> // TensorFlow C API header
#include "tf_utils.hpp"
#include <array>
#include <iostream>
#include <vector>

static void DeallocateTensor(void*, size_t, void*) {}

int main() {
  TF_Graph* graph = LoadGraphDef("graph.pb");
  if (graph == nullptr) {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  TF_Output input_op = {TF_GraphOperationByName(graph, "input_4"), 0};
  if (input_op.oper == nullptr) {
    std::cout << "Can't init input_op" << std::endl;
    return 2;
  }

  std::array<std::int64_t, 3> input_dims = {{1, 5, 12}};
  std::vector<float> input_vals = {
    -0.4809832f, -0.3770838f, 0.1743573f, 0.7720509f, -0.4064746f, 0.0116595f, 0.0051413f, 0.9135732f, 0.7197526f, -0.0400658f, 0.1180671f, -0.6829428f,
    -0.4810135f, -0.3772099f, 0.1745346f, 0.7719303f, -0.4066443f, 0.0114614f, 0.0051195f, 0.9135003f, 0.7196983f, -0.0400035f, 0.1178188f, -0.6830465f,
    -0.4809143f, -0.3773398f, 0.1746384f, 0.7719052f, -0.4067171f, 0.0111654f, 0.0054433f, 0.9134697f, 0.7192584f, -0.0399981f, 0.1177435f, -0.6835230f,
    -0.4808300f, -0.3774327f, 0.1748246f, 0.7718700f, -0.4070232f, 0.0109549f, 0.0059128f, 0.9133330f, 0.7188759f, -0.0398740f, 0.1181437f, -0.6838635f,
    -0.4807833f, -0.3775733f, 0.1748378f, 0.7718275f, -0.4073670f, 0.0107582f, 0.0062978f, 0.9131795f, 0.7187147f, -0.0394935f, 0.1184392f, -0.6840039f,
  };

  TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,
                                         input_dims.data(), static_cast<int>(input_dims.size()),
                                         input_vals.data(), input_vals.size() * sizeof(float),
                                         DeallocateTensor, nullptr);

  TF_Output out_op = {TF_GraphOperationByName(graph, "output_node0"), 0};
  if (input_op.oper == nullptr) {
    std::cout << "Can't init out_op" << std::endl;
    return 3;
  }

  TF_Tensor* output_tensor = nullptr;

  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* options = TF_NewSessionOptions();
  TF_Session* sess = TF_NewSession(graph, options, status);
  TF_DeleteSessionOptions(options);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    return 4;
  }

  TF_SessionRun(sess,
                nullptr, // Run options.
                &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                &out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );

  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error run session";
    TF_DeleteStatus(status);
    return 5;
  }

  TF_CloseSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error close session";
    TF_DeleteStatus(status);
    return 6;
  }

  TF_DeleteSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error delete session";
    TF_DeleteStatus(status);
    return 7;
  }

  auto data = static_cast<float*>(TF_TensorData(output_tensor));

  std::cout << "Output vals: " << data[0] << "," << data[1] << "," << data[2] << "," << data[3] << std::endl;

  TF_DeleteTensor(input_tensor);
  TF_DeleteTensor(output_tensor);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(status);

  return 0;
}
