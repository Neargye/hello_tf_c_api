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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4190)
#endif

#include <tensorflow/c/c_api.h> // TensorFlow C API header.

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

TF_Operation* AddPlaceholder(TF_Graph* graph, const char* name, TF_DataType data_type, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", data_type);

  return TF_FinishOperation(desc, status);
}

TF_Operation* AddIdentity(TF_Graph* graph, const char* name, TF_Output input, TF_DataType data_type, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Identity", name);
  TF_SetAttrType(desc, "T", data_type);
  TF_AddInput(desc, input);

  return TF_FinishOperation(desc, status);
}

TF_Operation* AddFloatConst(TF_Graph* graph, const char* name, float value, TF_Status* status) {
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
    return nullptr;
  }

  return TF_FinishOperation(desc, status);
}

TF_Operation* AddNoOp(TF_Graph* graph, const char* name, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "NoOp", name);

  return TF_FinishOperation(desc, status);
}

TF_Operation* AddVariable(TF_Graph* graph, const char* name, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "VarHandleOp", name);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);
  TF_SetAttrShape(desc, "shape", nullptr, 0);

  return TF_FinishOperation(desc, status);
}

TF_Operation* AddAssign(TF_Graph* graph, const char* name, TF_Output variable, TF_Output value, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "AssignVariableOp", name);
  TF_AddInput(desc, variable);
  TF_AddInput(desc, value);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);

  return TF_FinishOperation(desc, status);
}

TF_Operation* AddReadVariable(TF_Graph* graph, const char* name, TF_Output variable, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "ReadVariableOp", name);
  TF_AddInput(desc, variable);
  TF_SetAttrType(desc, "dtype", TF_FLOAT);

  return TF_FinishOperation(desc, status);
}

} // namespace

TEST_CASE("Hello TF C API") {
  const std::string version("2.21.0");
  CHECK(std::string(TF_Version()).compare(0, version.size(), version) == 0);
}

TEST_CASE("CreateTensor copies numeric data") {
  const std::vector<std::int64_t> dims = {2, 2};
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

  auto tensor = tf_utils::CreateTensor(TF_FLOAT, dims, values);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  CHECK(TF_TensorType(tensor) == TF_FLOAT);
  REQUIRE(TF_NumDims(tensor) == static_cast<int>(dims.size()));
  CHECK(TF_Dim(tensor, 0) == dims[0]);
  CHECK(TF_Dim(tensor, 1) == dims[1]);
  CHECK(TF_TensorByteSize(tensor) == values.size() * sizeof(float));
  CHECK(tf_utils::GetTensorData<float>(tensor) == values);
}

TEST_CASE("CreateTensor rejects mismatched tensor types and byte sizes") {
  const std::vector<std::int64_t> dims = {2};
  const std::vector<std::int32_t> int_values = {1, 2};
  const std::vector<float> short_values = {1.0f};
  const std::vector<float> long_values = {1.0f, 2.0f, 3.0f};

  CHECK(tf_utils::CreateTensor(TF_FLOAT, dims, int_values) == nullptr);
  CHECK(tf_utils::CreateTensor(TF_FLOAT, dims.data(), dims.size(), short_values.data(), short_values.size() * sizeof(float)) == nullptr);
  CHECK(tf_utils::CreateTensor(TF_FLOAT, dims.data(), dims.size(), long_values.data(), long_values.size() * sizeof(float)) == nullptr);
  CHECK(tf_utils::CreateTensor(TF_STRING, dims, int_values) == nullptr);
  CHECK(tf_utils::CreateTensor(TF_STRING, dims.data(), dims.size(), nullptr, 0) == nullptr);
  CHECK(tf_utils::CreateEmptyTensor(TF_STRING, dims, 0) == nullptr);
  CHECK(tf_utils::CreateEmptyTensor(TF_FLOAT, dims, short_values.size() * sizeof(float)) == nullptr);
  CHECK(tf_utils::CreateEmptyTensor(TF_FLOAT, dims, long_values.size() * sizeof(float)) == nullptr);
}

TEST_CASE("SetTensorData validates null tensors and updates tensor data") {
  const std::vector<std::int64_t> dims = {3};
  const std::vector<std::int32_t> values = {7, 8, 9};
  const std::vector<std::int32_t> short_values = {1, 2};
  const std::vector<float> wrong_type_values = {1.0f, 2.0f, 3.0f};

  CHECK_FALSE(tf_utils::SetTensorData(nullptr, values.data(), values.size() * sizeof(std::int32_t)));

  auto tensor = tf_utils::CreateEmptyTensor(TF_INT32, dims, values.size() * sizeof(std::int32_t));
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  CHECK(tf_utils::SetTensorData(tensor, values.data(), values.size() * sizeof(std::int32_t)));
  CHECK(tf_utils::GetTensorData<std::int32_t>(tensor) == values);
  CHECK_FALSE(tf_utils::SetTensorData(tensor, short_values));
  CHECK_FALSE(tf_utils::SetTensorData(tensor, wrong_type_values));
  CHECK(tf_utils::GetTensorData<std::int32_t>(tensor) == values);
}

TEST_CASE("GetTensorData rejects mismatched tensor value types") {
  const std::vector<std::int64_t> dims = {2};
  const std::vector<float> values = {1.0f, 2.0f};

  auto tensor = tf_utils::CreateTensor(TF_FLOAT, dims, values);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  CHECK(tf_utils::GetTensorData<float>(tensor) == values);
  CHECK(tf_utils::GetTensorData<std::int32_t>(tensor).empty());
}

TEST_CASE("Public helpers reject invalid arguments") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  CHECK(tf_utils::LoadGraph(nullptr, status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::LoadGraph("missing_graph.pb", status) == nullptr);
  CHECK(TF_GetCode(status) == TF_NOT_FOUND);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::CreateSession(nullptr, status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::DeleteSession(nullptr, status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::RunSession(nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0, status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);
}

TEST_CASE("RestoreCheckpoint validates session graph and operation names") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  CHECK(tf_utils::RestoreCheckpoint(nullptr, nullptr, "checkpoint", "input", "restore", status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  CHECK(tf_utils::RestoreCheckpoint(session, graph, nullptr, "input", "restore", status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::RestoreCheckpoint(session, graph, "checkpoint", nullptr, "restore", status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::RestoreCheckpoint(session, graph, "checkpoint", "missing_input", "missing_restore", status) == TF_NOT_FOUND);
  CHECK(TF_GetCode(status) == TF_NOT_FOUND);
}

TEST_CASE("RestoreCheckpoint runs restore target in caller-owned session") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto checkpoint_input = AddPlaceholder(graph, "checkpoint_path", TF_STRING, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(checkpoint_input != nullptr);

  auto variable = AddVariable(graph, "weight", status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(variable != nullptr);

  auto restored_value = AddFloatConst(graph, "restored_value", 7.0f, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(restored_value != nullptr);

  auto restore_target = AddAssign(graph, "custom_restore", TF_Output{variable, 0}, TF_Output{restored_value, 0}, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(restore_target != nullptr);

  auto read = AddReadVariable(graph, "read_weight", TF_Output{variable, 0}, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(read != nullptr);

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  CHECK(tf_utils::RestoreCheckpoint(session, graph, "checkpoint-prefix", "checkpoint_path", "custom_restore", status) == TF_OK);
  REQUIRE(TF_GetCode(status) == TF_OK);

  const std::vector<TF_Output> inputs = {};
  const std::vector<TF_Tensor*> input_tensors = {};
  const std::vector<TF_Output> outputs = {TF_Output{read, 0}};
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); };

  CHECK(tf_utils::RunSession(session, inputs, input_tensors, outputs, output_tensors, status) == TF_OK);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(output_tensors[0] != nullptr);
  CHECK(tf_utils::GetTensorData<float>(output_tensors[0]) == std::vector<float>{7.0f});
}

TEST_CASE("RestoreCheckpoint preserves TensorFlow status on restore failure") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto checkpoint_input = AddPlaceholder(graph, "checkpoint_path", TF_FLOAT, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(checkpoint_input != nullptr);

  auto restore_target = AddIdentity(graph, "custom_restore", TF_Output{checkpoint_input, 0}, TF_FLOAT, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(restore_target != nullptr);

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  CHECK(tf_utils::RestoreCheckpoint(session, graph, "checkpoint-prefix", "checkpoint_path", "custom_restore", status) != TF_OK);
  CHECK(TF_GetCode(status) != TF_OK);
}

TEST_CASE("CreateSessionOptions helpers create usable TensorFlow session options") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto gpu_options = tf_utils::CreateSessionOptions(0.25, status);
  SCOPE_EXIT{ tf_utils::DeleteSessionOptions(gpu_options); };
  REQUIRE(gpu_options != nullptr);
  REQUIRE(TF_GetCode(status) == TF_OK);

  auto gpu_session = tf_utils::CreateSession(graph, gpu_options, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(gpu_session); };
  REQUIRE(gpu_session != nullptr);
  CHECK(TF_GetCode(status) == TF_OK);

  auto thread_options = tf_utils::CreateSessionOptions(1, 2, status);
  SCOPE_EXIT{ tf_utils::DeleteSessionOptions(thread_options); };
  REQUIRE(thread_options != nullptr);
  REQUIRE(TF_GetCode(status) == TF_OK);

  auto high_thread_options = tf_utils::CreateSessionOptions(1024, 128, status);
  SCOPE_EXIT{ tf_utils::DeleteSessionOptions(high_thread_options); };
  REQUIRE(high_thread_options != nullptr);
  REQUIRE(TF_GetCode(status) == TF_OK);

  auto thread_session = tf_utils::CreateSession(graph, thread_options, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(thread_session); };
  REQUIRE(thread_session != nullptr);
  CHECK(TF_GetCode(status) == TF_OK);

  auto high_thread_session = tf_utils::CreateSession(graph, high_thread_options, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(high_thread_session); };
  REQUIRE(high_thread_session != nullptr);
  CHECK(TF_GetCode(status) == TF_OK);

  auto owned_status_options = tf_utils::CreateSessionOptions(0, 0);
  SCOPE_EXIT{ tf_utils::DeleteSessionOptions(owned_status_options); };
  REQUIRE(owned_status_options != nullptr);
}

TEST_CASE("CreateSessionOptions rejects invalid gpu memory fractions") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  CHECK(tf_utils::CreateSessionOptions(-0.01, status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::CreateSessionOptions(1.01, status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::CreateSessionOptions(std::numeric_limits<double>::infinity(), status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::CreateSessionOptions(std::numeric_limits<double>::quiet_NaN(), status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);
}

TEST_CASE("CreateSessionOptions rejects negative thread counts") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  CHECK(tf_utils::CreateSessionOptions(-1, 1, status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  CHECK(tf_utils::CreateSessionOptions(1, -1, status) == nullptr);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);
}

TEST_CASE("CreateEmptyTensor supports scalar tensors") {
  const float value = 3.5f;

  auto tensor = tf_utils::CreateEmptyTensor(TF_FLOAT, nullptr, 0);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  CHECK(TF_NumDims(tensor) == 0);
  CHECK(TF_TensorByteSize(tensor) == sizeof(float));
  CHECK(tf_utils::SetTensorData(tensor, &value, sizeof(value)));
  CHECK(tf_utils::GetTensorData<float>(tensor) == std::vector<float>{value});
}

TEST_CASE("CreateEmptyTensor infers and validates fixed-size tensor byte size") {
  const std::vector<std::int64_t> dims = {2, 2};

  auto tensor = tf_utils::CreateEmptyTensor(TF_FLOAT, dims);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  CHECK(TF_TensorByteSize(tensor) == 4 * sizeof(float));
  CHECK(tf_utils::CreateEmptyTensor(TF_FLOAT, dims, sizeof(float)) == nullptr);
  CHECK(tf_utils::CreateEmptyTensor(TF_FLOAT, dims, 5 * sizeof(float)) == nullptr);
}

TEST_CASE("Public helpers reject dimensions and counts that do not fit TensorFlow C API int parameters") {
  const auto too_many = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
  const std::int64_t dim = 1;
  const std::string_view string_value = "value";

  CHECK(tf_utils::CreateEmptyTensor(TF_FLOAT, &dim, too_many, 0) == nullptr);
  CHECK(tf_utils::CreateStringTensor(&dim, too_many, &string_value, 1) == nullptr);

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  TF_Output input{nullptr, 0};
  TF_Tensor* input_tensor = nullptr;
  CHECK(tf_utils::RunSession(session, &input, &input_tensor, too_many, nullptr, nullptr, 0, status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);
}

TEST_CASE("GetTensorShape handles unknown rank safely") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto input = AddPlaceholder(graph, "unknown_rank_input", TF_FLOAT, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(input != nullptr);

  CHECK(tf_utils::GetTensorShape(graph, TF_Output{input, 0}).empty());
  CHECK(tf_utils::GetTensorShape(nullptr, TF_Output{input, 0}).empty());
  CHECK(tf_utils::GetTensorShape(graph, TF_Output{nullptr, 0}).empty());
}

TEST_CASE("RunSession rejects mismatched vector sizes before calling TensorFlow") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto input = AddPlaceholder(graph, "input", TF_FLOAT, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(input != nullptr);

  auto output = AddIdentity(graph, "output", TF_Output{input, 0}, TF_FLOAT, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(output != nullptr);

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  const std::vector<std::int64_t> dims = {1};
  const std::vector<float> values = {42.0f};
  auto input_tensor = tf_utils::CreateTensor(TF_FLOAT, dims, values);
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor); };
  REQUIRE(input_tensor != nullptr);

  const std::vector<TF_Output> inputs = {TF_Output{input, 0}};
  const std::vector<TF_Output> outputs = {TF_Output{output, 0}};
  std::vector<TF_Tensor*> input_tensors = {input_tensor};
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); };

  std::vector<TF_Output> empty_inputs;
  CHECK(tf_utils::RunSession(session, empty_inputs, input_tensors, outputs, output_tensors, status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);

  TF_SetStatus(status, TF_OK, "");
  std::vector<TF_Tensor*> empty_outputs;
  CHECK(tf_utils::RunSession(session, inputs, input_tensors, outputs, empty_outputs, status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);
}

TEST_CASE("RunSession raw overload accepts zero input count with null input arrays") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto value = AddFloatConst(graph, "value", 42.0f, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(value != nullptr);

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  const auto output = TF_Output{value, 0};
  TF_Tensor* output_tensor = nullptr;
  SCOPE_EXIT{ tf_utils::DeleteTensor(output_tensor); };

  CHECK(tf_utils::RunSession(session, nullptr, nullptr, 0, &output, &output_tensor, 1, status) == TF_OK);
  CHECK(TF_GetCode(status) == TF_OK);
  REQUIRE(output_tensor != nullptr);
  CHECK(tf_utils::GetTensorData<float>(output_tensor) == std::vector<float>{42.0f});
}

TEST_CASE("RunSession runs target operations through vector overload") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto target = AddNoOp(graph, "target_noop", status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(target != nullptr);

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  const std::vector<TF_Output> inputs = {};
  const std::vector<TF_Tensor*> input_tensors = {};
  const std::vector<TF_Output> outputs = {};
  std::vector<TF_Tensor*> output_tensors = {};
  const std::vector<const TF_Operation*> targets = {target};

  CHECK(tf_utils::RunSession(session, inputs, input_tensors, outputs, output_tensors, targets, status) == TF_OK);
  CHECK(TF_GetCode(status) == TF_OK);
}

TEST_CASE("RunSession target overload rejects missing target array") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  CHECK(tf_utils::RunSession(session, nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr, 1, status) == TF_INVALID_ARGUMENT);
  CHECK(TF_GetCode(status) == TF_INVALID_ARGUMENT);
}

TEST_CASE("CreateStringTensor validates shape and round-trips embedded nulls") {
  const std::vector<std::int64_t> dims = {2};
  const std::vector<std::string> strings = {"owned string", std::string("a\0b", 3)};

  CHECK(tf_utils::CreateStringTensor({3}, strings) == nullptr);
  CHECK(tf_utils::CreateStringTensor(dims.data(), dims.size(), nullptr, strings.size()) == nullptr);

  auto tensor = tf_utils::CreateStringTensor(dims, strings);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  CHECK(TF_TensorType(tensor) == TF_STRING);
  CHECK(tf_utils::GetStringTensorData(tensor) == strings);
  CHECK(tf_utils::GetStringTensorElement(tensor, strings.size()).empty());
}

TEST_CASE("Raw byte tensor helpers reject string tensor storage") {
  const std::vector<std::int64_t> dims = {1};
  const std::vector<std::string> strings = {"value"};

  auto tensor = tf_utils::CreateStringTensor(dims, strings);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  std::uintptr_t raw_value = 0;
  CHECK_FALSE(tf_utils::SetTensorData(tensor, &raw_value, TF_TensorByteSize(tensor)));
  CHECK(tf_utils::GetStringTensorData(tensor) == strings);
}

TEST_CASE("CreateStringTensor supports scalar and empty strings") {
  const std::vector<std::int64_t> scalar_dims = {};
  const std::vector<std::string> scalar_string = {""};

  auto scalar_tensor = tf_utils::CreateStringTensor(scalar_dims, scalar_string);
  SCOPE_EXIT{ tf_utils::DeleteTensor(scalar_tensor); };

  REQUIRE(scalar_tensor != nullptr);
  CHECK(TF_TensorType(scalar_tensor) == TF_STRING);
  CHECK(TF_NumDims(scalar_tensor) == 0);
  CHECK(tf_utils::GetStringTensorData(scalar_tensor) == scalar_string);

  const std::vector<std::int64_t> dims = {3};
  const std::vector<std::string> strings = {"", "middle", ""};

  auto tensor = tf_utils::CreateStringTensor(dims, strings);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); };

  REQUIRE(tensor != nullptr);
  CHECK(TF_TensorType(tensor) == TF_STRING);
  CHECK(tf_utils::GetStringTensorData(tensor) == strings);
}

TEST_CASE("TF_STRING tensor round-trips through TensorFlow SessionRun") {
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };

  auto input = AddPlaceholder(graph, "input", TF_STRING, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(input != nullptr);

  auto output = AddIdentity(graph, "output", TF_Output{input, 0}, TF_STRING, status);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(output != nullptr);

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(session != nullptr);

  const std::vector<std::int64_t> dims = {2, 2};
  const std::vector<std::string> strings = {"hello", "tensorflow", "c-api", std::string("a\0b", 3)};

  std::vector<TF_Tensor*> input_tensors = {tf_utils::CreateStringTensor(dims, strings)};
  SCOPE_EXIT{ tf_utils::DeleteTensors(input_tensors); };
  REQUIRE(input_tensors[0] != nullptr);

  const std::vector<TF_Output> inputs = {TF_Output{input, 0}};
  const std::vector<TF_Output> outputs = {TF_Output{output, 0}};
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); };

  CHECK(tf_utils::RunSession(session, inputs, input_tensors, outputs, output_tensors, status) == TF_OK);
  REQUIRE(TF_GetCode(status) == TF_OK);
  REQUIRE(output_tensors[0] != nullptr);
  CHECK(TF_TensorType(output_tensors[0]) == TF_STRING);
  CHECK(TF_NumDims(output_tensors[0]) == static_cast<int>(dims.size()));
  CHECK(TF_Dim(output_tensors[0], 0) == dims[0]);
  CHECK(TF_Dim(output_tensors[0], 1) == dims[1]);

  for (std::size_t i = 0; i < strings.size(); ++i) {
    CHECK(tf_utils::GetStringTensorElement(output_tensors[0], i) == strings[i]);
  }
}
