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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

TF_Tensor* CreateScalarFloatTensor(float value) {
  auto tensor = TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(float));
  if (tensor == nullptr || TF_TensorData(tensor) == nullptr) {
    tf_utils::DeleteTensor(tensor);
    return nullptr;
  }

  std::memcpy(TF_TensorData(tensor), &value, sizeof(value));

  return tensor;
}

TF_Operation* FinishOperation(TF_OperationDescription* desc, TF_Status* status) {
  auto op = TF_FinishOperation(desc, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error finish operation: " << TF_Message(status) << std::endl;
    return nullptr;
  }

  return op;
}

TF_Operation* AddImagePlaceholder(TF_Graph* graph, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Placeholder", "input_image");
  TF_SetAttrType(desc, "dtype", TF_UINT8);

  return FinishOperation(desc, status);
}

TF_Operation* AddCastToFloat(TF_Graph* graph, TF_Output input, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Cast", "cast_to_float");
  TF_AddInput(desc, input);
  TF_SetAttrType(desc, "SrcT", TF_UINT8);
  TF_SetAttrType(desc, "DstT", TF_FLOAT);
  TF_SetAttrBool(desc, "Truncate", static_cast<unsigned char>(0));

  return FinishOperation(desc, status);
}

TF_Operation* AddScalarConst(TF_Graph* graph, const char* name, float value, TF_Status* status) {
  auto tensor = CreateScalarFloatTensor(value);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };
  if (tensor == nullptr) {
    return nullptr;
  }

  auto desc = TF_NewOperation(graph, "Const", name);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);
  TF_SetAttrTensor(desc, "value", tensor, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error set const tensor: " << TF_Message(status) << std::endl;
    return nullptr;
  }

  return FinishOperation(desc, status);
}

TF_Operation* AddMul(TF_Graph* graph, const char* name, TF_Output lhs, TF_Output rhs, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Mul", name);
  TF_AddInput(desc, lhs);
  TF_AddInput(desc, rhs);
  TF_SetAttrType(desc, "T", TF_FLOAT);

  return FinishOperation(desc, status);
}

TF_Operation* AddIdentity(TF_Graph* graph, const char* name, TF_Output input, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Identity", name);
  TF_AddInput(desc, input);
  TF_SetAttrType(desc, "T", TF_FLOAT);

  return FinishOperation(desc, status);
}

bool AlmostEqual(float lhs, float rhs) {
  return std::fabs(lhs - rhs) < 1.0e-6f;
}

} // namespace

int main() {
  const std::vector<std::int64_t> image_dims = {1, 2, 2, 3}; // NHWC: batch, height, width, channels.
  const std::vector<std::uint8_t> pixels = {
    0, 127, 255,
    64, 128, 192,
    255, 0, 32,
    16, 240, 80,
  };

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); };

  auto input = AddImagePlaceholder(graph, status);
  if (input == nullptr) {
    return 1;
  }

  auto cast = AddCastToFloat(graph, TF_Output{input, 0}, status);
  if (cast == nullptr) {
    return 2;
  }

  auto scale = AddScalarConst(graph, "scale", 1.0f / 255.0f, status);
  if (scale == nullptr) {
    return 3;
  }

  auto normalized = AddMul(graph, "normalized", TF_Output{cast, 0}, TF_Output{scale, 0}, status);
  if (normalized == nullptr) {
    return 4;
  }

  auto output = AddIdentity(graph, "output_image", TF_Output{normalized, 0}, status);
  if (output == nullptr) {
    return 5;
  }

  auto input_tensor = tf_utils::CreateTensor(TF_UINT8, image_dims, pixels);
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor); };
  if (input_tensor == nullptr) {
    std::cout << "Can't create image tensor" << std::endl;
    return 6;
  }

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  if (session == nullptr || TF_GetCode(status) != TF_OK) {
    std::cout << "Can't create session: " << TF_Message(status) << std::endl;
    return 7;
  }

  const std::vector<TF_Output> inputs = {TF_Output{input, 0}};
  const std::vector<TF_Tensor*> input_tensors = {input_tensor};
  const std::vector<TF_Output> outputs = {TF_Output{output, 0}};
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); };

  auto code = tf_utils::RunSession(session, inputs, input_tensors, outputs, output_tensors, status);
  if (code != TF_OK) {
    std::cout << "Error run session: " << TF_Message(status) << std::endl;
    return 8;
  }

  auto result = tf_utils::GetTensorData<float>(output_tensors[0]);
  if (result.size() != pixels.size()) {
    std::cout << "Wrong output image size" << std::endl;
    return 9;
  }

  for (std::size_t i = 0; i < pixels.size(); ++i) {
    const auto expected = static_cast<float>(pixels[i]) / 255.0f;
    if (!AlmostEqual(result[i], expected)) {
      std::cout << "Wrong normalized value for pixel element: " << i << std::endl;
      return 10;
    }
  }

  std::cout << "Input image tensor NHWC: "
            << image_dims[0] << "x" << image_dims[1] << "x" << image_dims[2] << "x" << image_dims[3] << std::endl;
  std::cout << "First pixel normalized RGB: "
            << result[0] << ", " << result[1] << ", " << result[2] << std::endl;
  std::cout << "Success image processing" << std::endl;

  return 0;
}
