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
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF  ANY KIND, EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT SHALL THE
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

TF_Operation* FinishOperation(TF_OperationDescription* desc, TF_Status* status) {
  auto op = TF_FinishOperation(desc, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error finish operation: " << TF_Message(status) << std::endl;
    return nullptr;
  }

  return op;
}

TF_Operation* AddScalarConst(TF_Graph* graph, const char* name, float value, TF_Status* status) {
  const std::vector<std::int64_t> dims = {};
  const std::vector<float> values = {value};
  auto tensor = tf_utils::CreateTensor(TF_FLOAT, dims, values);
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

TF_Operation* AddVariable(TF_Graph* graph, const char* name, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "VarHandleOp", name);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);
  TF_SetAttrShape(desc, "shape", nullptr, 0);

  return FinishOperation(desc, status);
}

TF_Operation* AddPlaceholder(TF_Graph* graph, const char* name, TF_DataType data_type, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", data_type);

  return FinishOperation(desc, status);
}

TF_Operation* AddAssign(TF_Graph* graph, const char* name, TF_Output variable, TF_Output value, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "AssignVariableOp", name);
  TF_AddInput(desc, variable);
  TF_AddInput(desc, value);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);

  return FinishOperation(desc, status);
}

TF_Operation* AddAssignAdd(TF_Graph* graph, const char* name, TF_Output variable, TF_Output value, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "AssignAddVariableOp", name);
  TF_AddInput(desc, variable);
  TF_AddInput(desc, value);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);

  return FinishOperation(desc, status);
}

TF_Operation* AddReadVariable(TF_Graph* graph, const char* name, TF_Output variable, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "ReadVariableOp", name);
  TF_AddInput(desc, variable);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);

  return FinishOperation(desc, status);
}

bool AlmostEqual(float lhs, float rhs) {
  return std::fabs(lhs - rhs) < 1.0e-6f;
}

float ReadScalar(TF_Session* session, TF_Output output, TF_Status* status) {
  const std::vector<TF_Output> inputs = {};
  const std::vector<TF_Tensor*> input_tensors = {};
  const std::vector<TF_Output> outputs = {output};
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); };

  const auto code = tf_utils::RunSession(session, inputs, input_tensors, outputs, output_tensors, status);
  if (code != TF_OK || output_tensors[0] == nullptr) {
    std::cout << "Error read scalar: " << TF_Message(status) << std::endl;
    return 0.0f;
  }

  const auto values = tf_utils::GetTensorData<float>(output_tensors[0]);
  if (values.size() != 1) {
    std::cout << "Wrong scalar output size" << std::endl;
    return 0.0f;
  }

  return values[0];
}

} // namespace

int main() {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); };

  auto variable = AddVariable(graph, "weight", status);
  if (variable == nullptr) {
    return 1;
  }

  auto initial_value = AddScalarConst(graph, "initial_value", 0.0f, status);
  if (initial_value == nullptr) {
    return 2;
  }

  auto delta = AddPlaceholder(graph, "delta", TF_FLOAT, status);
  if (delta == nullptr) {
    return 3;
  }

  auto init_op = AddAssign(graph, "init_weight", TF_Output{variable, 0}, TF_Output{initial_value, 0}, status);
  if (init_op == nullptr) {
    return 4;
  }

  auto train_op = AddAssignAdd(graph, "train_step", TF_Output{variable, 0}, TF_Output{delta, 0}, status);
  if (train_op == nullptr) {
    return 5;
  }

  auto read_op = AddReadVariable(graph, "read_weight", TF_Output{variable, 0}, status);
  if (read_op == nullptr) {
    return 6;
  }

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  if (session == nullptr || TF_GetCode(status) != TF_OK) {
    std::cout << "Can't create session: " << TF_Message(status) << std::endl;
    return 7;
  }

  const std::vector<TF_Output> empty_inputs = {};
  const std::vector<TF_Tensor*> empty_input_tensors = {};
  const std::vector<TF_Output> empty_outputs = {};
  std::vector<TF_Tensor*> empty_output_tensors = {};

  const std::vector<const TF_Operation*> init_targets = {init_op};
  auto code = tf_utils::RunSession(session,
                                   empty_inputs, empty_input_tensors,
                                   empty_outputs, empty_output_tensors,
                                   init_targets,
                                   status);
  if (code != TF_OK) {
    std::cout << "Error run init target: " << TF_Message(status) << std::endl;
    return 8;
  }

  const auto initial = ReadScalar(session, TF_Output{read_op, 0}, status);
  if (!AlmostEqual(initial, 0.0f)) {
    std::cout << "Wrong initial value: " << initial << std::endl;
    return 9;
  }

  const std::vector<std::int64_t> scalar_dims = {};
  const std::vector<float> delta_value = {1.5f};
  const std::vector<TF_Output> train_inputs = {TF_Output{delta, 0}};
  const std::vector<const TF_Operation*> train_targets = {train_op};

  for (int i = 0; i < 3; ++i) {
    auto delta_tensor = tf_utils::CreateTensor(TF_FLOAT, scalar_dims, delta_value);
    SCOPE_EXIT{ tf_utils::DeleteTensor(delta_tensor); };
    if (delta_tensor == nullptr) {
      std::cout << "Can't create delta tensor" << std::endl;
      return 10;
    }

    const std::vector<TF_Tensor*> train_input_tensors = {delta_tensor};
    code = tf_utils::RunSession(session,
                                train_inputs, train_input_tensors,
                                empty_outputs, empty_output_tensors,
                                train_targets,
                                status);
    if (code != TF_OK) {
      std::cout << "Error run train target: " << TF_Message(status) << std::endl;
      return 11;
    }
  }

  const auto trained = ReadScalar(session, TF_Output{read_op, 0}, status);
  if (!AlmostEqual(trained, 4.5f)) {
    std::cout << "Wrong trained value: " << trained << std::endl;
    return 12;
  }

  std::cout << "Initial value: " << initial << std::endl;
  std::cout << "Value after running train_step target 3 times: " << trained << std::endl;
  std::cout << "Success run target operation" << std::endl;

  return 0;
}
