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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>
#include <vector>

namespace tf_utils {

namespace {

static void DeallocateBuffer(void* data, size_t) {
  std::free(data);
}

struct StringTensorDeallocatorArg {
  std::size_t size;
};

struct StringTensorStorage {
  explicit StringTensorStorage(std::size_t size)
      : data(new TF_TString[size]) {}

  ~StringTensorStorage() {
    if (data == nullptr) {
      return;
    }
    for (std::size_t i = 0; i < initialized; ++i) {
      TF_StringDealloc(&data[i]);
    }
    delete[] data;
  }

  TF_TString* get() const {
    return data;
  }

  void mark_initialized() {
    ++initialized;
  }

  TF_TString* release() {
    auto* released = data;
    data = nullptr;
    initialized = 0;
    return released;
  }

  TF_TString* data;
  std::size_t initialized = 0;
};

static void DeallocateStringTensor(void* data, size_t, void* arg) {
  auto strings = static_cast<TF_TString*>(data);
  auto deallocator_arg = static_cast<StringTensorDeallocatorArg*>(arg);

  if (strings != nullptr && deallocator_arg != nullptr) {
    for (std::size_t i = 0; i < deallocator_arg->size; ++i) {
      TF_StringDealloc(&strings[i]);
    }
  }

  delete[] strings;
  delete deallocator_arg;
}

static bool ShapeElementCount(const std::int64_t* dims, std::size_t num_dims, std::size_t& count) {
  if (dims == nullptr && num_dims != 0) {
    return false;
  }

  count = 1;
  for (std::size_t i = 0; i < num_dims; ++i) {
    if (dims[i] < 0) {
      return false;
    }
    const auto dim = static_cast<std::size_t>(dims[i]);
    if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
      return false;
    }
    count *= dim;
  }

  return true;
}

static std::size_t FixedSizeDataTypeByteSize(TF_DataType data_type) {
  return TF_DataTypeSize(data_type);
}

static bool FitsTensorFlowIntParameter(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

static bool IsFixedSizeTensorDataType(TF_DataType data_type) {
  return FixedSizeDataTypeByteSize(data_type) != 0;
}

static void SetStatus(TF_Status* status, TF_Code code, const char* message) {
  if (status != nullptr) {
    TF_SetStatus(status, code, message);
  }
}

static TF_Code InvalidArgument(TF_Status* status, const char* message) {
  SetStatus(status, TF_INVALID_ARGUMENT, message);
  return TF_INVALID_ARGUMENT;
}

static void StoreLittleEndianDouble(double value, std::array<std::uint8_t, sizeof(double)>& output) {
  static_assert(sizeof(double) == sizeof(std::uint64_t), "Unexpected double size.");
  static_assert(std::numeric_limits<double>::is_iec559, "CreateSessionOptions requires IEEE 754 doubles.");

  std::uint64_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] = static_cast<std::uint8_t>((raw >> (i * 8)) & 0xffu);
  }
}

enum class ProtobufWireType : std::uint32_t {
  Varint = 0,
  Fixed64 = 1,
  LengthDelimited = 2,
};

static void AppendProtobufVarint(std::uint32_t value, std::vector<std::uint8_t>& output) {
  while (value >= 0x80u) {
    output.push_back(static_cast<std::uint8_t>((value & 0x7fu) | 0x80u));
    value >>= 7u;
  }
  output.push_back(static_cast<std::uint8_t>(value));
}

static void AppendProtobufKey(std::uint32_t field_number, ProtobufWireType wire_type, std::vector<std::uint8_t>& output) {
  AppendProtobufVarint((field_number << 3u) | static_cast<std::uint32_t>(wire_type), output);
}

static void AppendProtobufInt32Field(std::uint32_t field_number, std::int32_t value, std::vector<std::uint8_t>& output) {
  AppendProtobufKey(field_number, ProtobufWireType::Varint, output);
  AppendProtobufVarint(static_cast<std::uint32_t>(value), output);
}

static void AppendProtobufBoolField(std::uint32_t field_number, bool value, std::vector<std::uint8_t>& output) {
  AppendProtobufKey(field_number, ProtobufWireType::Varint, output);
  output.push_back(value ? std::uint8_t{1} : std::uint8_t{0});
}

static void AppendProtobufFixed64Field(std::uint32_t field_number,
                                        const std::array<std::uint8_t, sizeof(double)>& value,
                                        std::vector<std::uint8_t>& output) {
  AppendProtobufKey(field_number, ProtobufWireType::Fixed64, output);
  output.insert(output.end(), value.begin(), value.end());
}

static void AppendProtobufMessageField(std::uint32_t field_number,
                                       const std::vector<std::uint8_t>& message,
                                       std::vector<std::uint8_t>& output) {
  AppendProtobufKey(field_number, ProtobufWireType::LengthDelimited, output);
  AppendProtobufVarint(static_cast<std::uint32_t>(message.size()), output);
  output.insert(output.end(), message.begin(), message.end());
}

static bool FileSizeForBuffer(const char* file, std::size_t& size) {
  if (file == nullptr) {
    return false;
  }

  std::error_code error;
  const auto file_size = std::filesystem::file_size(file, error);
  if (error || file_size == 0) {
    return false;
  }
  if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) ||
      file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }

  size = static_cast<std::size_t>(file_size);
  return true;
}

static void CleanupSessionAfterCloseFailure(TF_Session* session) {
  auto cleanup_status = TF_NewStatus();
  if (cleanup_status == nullptr) {
    return;
  }

  TF_CloseSession(session, cleanup_status);
  TF_DeleteSession(session, cleanup_status);
  TF_DeleteStatus(cleanup_status);
}

static bool ExpectedTensorByteSize(TF_DataType data_type,
                                   const std::int64_t* dims,
                                   std::size_t num_dims,
                                   std::size_t& byte_size) {
  std::size_t element_count = 0;
  if (!ShapeElementCount(dims, num_dims, element_count)) {
    return false;
  }

  const auto element_size = FixedSizeDataTypeByteSize(data_type);
  if (element_size == 0) {
    return false;
  }
  if (element_count != 0 && element_size > std::numeric_limits<std::size_t>::max() / element_count) {
    return false;
  }

  byte_size = element_count * element_size;
  return true;
}

template <typename GetString>
TF_Tensor* CreateStringTensorImpl(const std::int64_t* dims, std::size_t num_dims, std::size_t num_strings, GetString get_string) {
  if (!FitsTensorFlowIntParameter(num_dims) || num_strings > std::numeric_limits<std::size_t>::max() / sizeof(TF_TString)) {
    return nullptr;
  }

  std::size_t element_count = 0;
  if (!ShapeElementCount(dims, num_dims, element_count) || element_count != num_strings) {
    return nullptr;
  }

  StringTensorStorage storage(num_strings);
  for (std::size_t i = 0; i < num_strings; ++i) {
    auto* data = storage.get();
    const auto str = get_string(i);
    const auto* str_data = str.empty() ? "" : str.data();
    TF_StringInit(&data[i]);
    storage.mark_initialized();
    TF_StringCopy(&data[i], str_data, str.size());
  }

  auto deallocator_arg = std::make_unique<StringTensorDeallocatorArg>(StringTensorDeallocatorArg{num_strings});
  auto tensor = TF_NewTensor(TF_STRING,
                             dims, static_cast<int>(num_dims),
                             storage.get(), num_strings * sizeof(TF_TString),
                             &DeallocateStringTensor, deallocator_arg.get());
  if (tensor != nullptr) {
    storage.release();
    deallocator_arg.release();
  }

  return tensor;
}

static TF_Buffer* ReadBufferFromFile(const char* file) {
  std::size_t file_size = 0;
  if (!FileSizeForBuffer(file, file_size)) {
    return nullptr;
  }

  auto data = static_cast<char*>(std::malloc(file_size));
  if (data == nullptr) {
    return nullptr;
  }

  std::ifstream f(file, std::ios::binary);
  if (!f.is_open()) {
    std::free(data);
    return nullptr;
  }

  if (!f.read(data, static_cast<std::streamsize>(file_size))) {
    std::free(data);
    return nullptr;
  }

  auto buf = TF_NewBuffer();
  if (buf == nullptr) {
    std::free(data);
    return nullptr;
  }

  buf->data = data;
  buf->length = file_size;
  buf->data_deallocator = DeallocateBuffer;

  return buf;
}

TF_Tensor* ScalarStringTensor(const char* str, TF_Status*) {
  const std::string_view value(str);

  return CreateStringTensor(nullptr, 0, &value, 1);
}

bool IsValidGpuMemoryFraction(double gpu_memory_fraction) {
  return std::isfinite(gpu_memory_fraction) && gpu_memory_fraction >= 0.0 && gpu_memory_fraction <= 1.0;
}

bool IsValidThreadCount(std::int32_t thread_count) {
  return thread_count >= 0;
}

std::vector<std::uint8_t> CreateGpuMemorySessionConfig(double gpu_memory_fraction) {
  constexpr std::uint32_t gpu_options_field = 6; // ConfigProto.gpu_options.
  constexpr std::uint32_t allow_soft_placement_field = 7; // ConfigProto.allow_soft_placement.
  constexpr std::uint32_t gpu_memory_fraction_field = 1; // GPUOptions.per_process_gpu_memory_fraction.
  constexpr std::uint32_t gpu_allow_growth_field = 4; // GPUOptions.allow_growth.

  std::array<std::uint8_t, sizeof(double)> percentage_bytes = {};
  StoreLittleEndianDouble(gpu_memory_fraction, percentage_bytes);

  std::vector<std::uint8_t> gpu_options;
  gpu_options.reserve(11);
  AppendProtobufFixed64Field(gpu_memory_fraction_field, percentage_bytes, gpu_options);
  AppendProtobufBoolField(gpu_allow_growth_field, true, gpu_options);

  std::vector<std::uint8_t> config;
  config.reserve(gpu_options.size() + 4);
  AppendProtobufMessageField(gpu_options_field, gpu_options, config);
  AppendProtobufBoolField(allow_soft_placement_field, true, config);

  return config;
}

std::vector<std::uint8_t> CreateThreadSessionConfig(std::int32_t intra_op_parallelism_threads,
                                                    std::int32_t inter_op_parallelism_threads) {
  constexpr std::uint32_t intra_op_parallelism_threads_field = 2; // ConfigProto.intra_op_parallelism_threads.
  constexpr std::uint32_t inter_op_parallelism_threads_field = 5; // ConfigProto.inter_op_parallelism_threads.

  std::vector<std::uint8_t> config;
  config.reserve(12);
  AppendProtobufInt32Field(intra_op_parallelism_threads_field, intra_op_parallelism_threads, config);
  AppendProtobufInt32Field(inter_op_parallelism_threads_field, inter_op_parallelism_threads, config);
  return config;
}

TF_SessionOptions* CreateConfiguredSessionOptions(const std::vector<std::uint8_t>& config, TF_Status* status) {
  const bool owns_status = status == nullptr;
  if (status == nullptr) {
    status = TF_NewStatus();
  }
  if (status == nullptr) {
    return nullptr;
  }
  SCOPE_EXIT{
    if (owns_status) {
      TF_DeleteStatus(status);
    }
  };

  auto options = TF_NewSessionOptions();
  if (options == nullptr) {
    SetStatus(status, TF_RESOURCE_EXHAUSTED, "Failed to create TensorFlow session options.");
    return nullptr;
  }

  TF_SetConfig(options, config.data(), config.size(), status);

  if (TF_GetCode(status) != TF_OK) {
    DeleteSessionOptions(options);
    return nullptr;
  }

  return options;
}

} // namespace

TF_Graph* LoadGraph(const char* graph_path, TF_Status* status) {
  if (graph_path == nullptr) {
    SetStatus(status, TF_INVALID_ARGUMENT, "Graph path must not be null.");
    return nullptr;
  }

  auto buffer = ReadBufferFromFile(graph_path);
  if (buffer == nullptr) {
    SetStatus(status, TF_NOT_FOUND, "Failed to read graph file.");
    return nullptr;
  }

  MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
  if (status == nullptr) {
    status = TF_NewStatus();
  } else {
    delete_status.dismiss();
  }

  auto graph = TF_NewGraph();
  auto opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status) != TF_OK) {
    DeleteGraph(graph);
    return nullptr;
  }

  return graph;
}

void DeleteGraph(TF_Graph* graph) {
  if (graph != nullptr) {
    TF_DeleteGraph(graph);
  }
}

TF_Session* CreateSession(TF_Graph* graph, TF_SessionOptions* options, TF_Status* status) {
  if (graph == nullptr) {
    SetStatus(status, TF_INVALID_ARGUMENT, "Graph must not be null.");
    return nullptr;
  }

  MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
  if (status == nullptr) {
    status = TF_NewStatus();
  } else {
    delete_status.dismiss();
  }

  MAKE_SCOPE_EXIT(delete_options){ DeleteSessionOptions(options); };
  if (options == nullptr) {
    options = TF_NewSessionOptions();
  } else {
    delete_options.dismiss();
  }

  auto session = TF_NewSession(graph, options, status);
  if (TF_GetCode(status) != TF_OK) {
    DeleteSession(session);
    return nullptr;
  }

  return session;
}

TF_Session* CreateSession(TF_Graph* graph, TF_Status* status) {
  return CreateSession(graph, nullptr, status);
}

TF_Code DeleteSession(TF_Session* session, TF_Status* status) {
  if (session == nullptr) {
    return InvalidArgument(status, "Session must not be null.");
  }

  MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
  if (status == nullptr) {
    status = TF_NewStatus();
  } else {
    delete_status.dismiss();
  }

  TF_CloseSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
    const auto code = TF_GetCode(status);
    CleanupSessionAfterCloseFailure(session);
    return code;
  }

  TF_DeleteSession(session, status);
  return TF_GetCode(status);
}

TF_Code RestoreCheckpoint(TF_Session* session,
                          TF_Graph* graph,
                          const char* checkpoint_prefix,
                          const char* checkpoint_prefix_input_operation_name,
                          const char* restore_operation_name,
                          TF_Status* status) {
  if (session == nullptr) {
    return InvalidArgument(status, "Session must not be null.");
  }
  if (graph == nullptr) {
    return InvalidArgument(status, "Graph must not be null.");
  }
  if (checkpoint_prefix == nullptr) {
    return InvalidArgument(status, "Checkpoint prefix must not be null.");
  }
  if (checkpoint_prefix_input_operation_name == nullptr || restore_operation_name == nullptr) {
    return InvalidArgument(status, "Checkpoint restore operation names must not be null.");
  }

  auto checkpoint_tensor = ScalarStringTensor(checkpoint_prefix, status);
  SCOPE_EXIT{ DeleteTensor(checkpoint_tensor); };
  if (checkpoint_tensor == nullptr) {
    SetStatus(status, TF_RESOURCE_EXHAUSTED, "Failed to create checkpoint prefix tensor.");
    return TF_RESOURCE_EXHAUSTED;
  }

  auto input = TF_Output{TF_GraphOperationByName(graph, checkpoint_prefix_input_operation_name), 0};
  if (input.oper == nullptr) {
    SetStatus(status, TF_NOT_FOUND, "Checkpoint prefix input operation not found.");
    return TF_NOT_FOUND;
  }

  auto restore_op = TF_GraphOperationByName(graph, restore_operation_name);
  if (restore_op == nullptr) {
    SetStatus(status, TF_NOT_FOUND, "Checkpoint restore operation not found.");
    return TF_NOT_FOUND;
  }

  const TF_Operation* target_op = restore_op;
  return RunSession(session,
                    &input, &checkpoint_tensor, 1,
                    nullptr, nullptr, 0,
                    &target_op, 1,
                    status);
}

TF_Code RestoreCheckpoint(TF_Session* session,
                          TF_Graph* graph,
                          const char* checkpoint_prefix,
                          TF_Status* status) {
  return RestoreCheckpoint(session, graph, checkpoint_prefix, "save/Const", "save/restore_all", status);
}

TF_Code RunSession(TF_Session* session,
                   const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                   const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs,
                   TF_Status* status) {
  return RunSession(session,
                    inputs, input_tensors, ninputs,
                    outputs, output_tensors, noutputs,
                    nullptr, 0,
                    status);
}

TF_Code RunSession(TF_Session* session,
                   const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                   const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs,
                   const TF_Operation* const* target_opers, std::size_t ntargets,
                   TF_Status* status) {
  if (session == nullptr) {
    return InvalidArgument(status, "Session must not be null.");
  }
  if (ninputs != 0 && inputs == nullptr) {
    return InvalidArgument(status, "Input operation array must not be null when input count is non-zero.");
  }
  if (ninputs != 0 && input_tensors == nullptr) {
    return InvalidArgument(status, "Input tensor array must not be null when input count is non-zero.");
  }
  if (noutputs != 0 && outputs == nullptr) {
    return InvalidArgument(status, "Output operation array must not be null when output count is non-zero.");
  }
  if (noutputs != 0 && output_tensors == nullptr) {
    return InvalidArgument(status, "Output tensor array must not be null when output count is non-zero.");
  }
  if (ntargets != 0 && target_opers == nullptr) {
    return InvalidArgument(status, "Target operation array must not be null when target count is non-zero.");
  }
  if (!FitsTensorFlowIntParameter(ninputs) ||
      !FitsTensorFlowIntParameter(noutputs) ||
      !FitsTensorFlowIntParameter(ntargets)) {
    return InvalidArgument(status, "Input, output, and target counts must fit TensorFlow C API int parameters.");
  }

  MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
  if (status == nullptr) {
    status = TF_NewStatus();
  } else {
    delete_status.dismiss();
  }


  TF_SessionRun(session,
                nullptr, // Run options.
                inputs, input_tensors, static_cast<int>(ninputs), // Input tensors, input tensor values, number of inputs.
                outputs, output_tensors, static_cast<int>(noutputs), // Output tensors, output tensor values, number of outputs.
                target_opers, static_cast<int>(ntargets), // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
  );

  return TF_GetCode(status);
}

TF_Code RunSession(TF_Session* session,
                   const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                   const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors,
                   TF_Status* status) {
  if (inputs.size() != input_tensors.size() || outputs.size() != output_tensors.size()) {
    return InvalidArgument(status, "Input and output tensor counts must match operation counts.");
  }

  return RunSession(session,
                    inputs.data(), input_tensors.data(), input_tensors.size(),
                    outputs.data(), output_tensors.data(), output_tensors.size(),
                    status);
}

TF_Code RunSession(TF_Session* session,
                   const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                   const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors,
                   const std::vector<const TF_Operation*>& target_opers,
                   TF_Status* status) {
  if (inputs.size() != input_tensors.size() || outputs.size() != output_tensors.size()) {
    return InvalidArgument(status, "Input and output tensor counts must match operation counts.");
  }

  return RunSession(session,
                    inputs.data(), input_tensors.data(), input_tensors.size(),
                    outputs.data(), output_tensors.data(), output_tensors.size(),
                    target_opers.data(), target_opers.size(),
                    status);
}

TF_Tensor* CreateStringTensor(const std::int64_t* dims, std::size_t num_dims,
                              const std::string_view* strings, std::size_t num_strings) {
  if (strings == nullptr && num_strings != 0) {
    return nullptr;
  }

  return CreateStringTensorImpl(dims, num_dims, num_strings, [strings](std::size_t i) {
    return strings[i];
  });
}

TF_Tensor* CreateStringTensor(const std::vector<std::int64_t>& dims, const std::vector<std::string_view>& strings) {
  return CreateStringTensor(dims.data(), dims.size(), strings.data(), strings.size());
}

TF_Tensor* CreateStringTensor(const std::vector<std::int64_t>& dims, const std::vector<std::string>& strings) {
  return CreateStringTensorImpl(dims.data(), dims.size(), strings.size(), [&strings](std::size_t i) -> std::string_view {
    return strings[i];
  });
}

std::string GetStringTensorElement(const TF_Tensor* tensor, std::size_t index) {
  if (tensor == nullptr || TF_TensorType(tensor) != TF_STRING) {
    return {};
  }

  const auto byte_size = TF_TensorByteSize(tensor);
  if (byte_size % sizeof(TF_TString) != 0 || index >= byte_size / sizeof(TF_TString)) {
    return {};
  }

  const auto data = static_cast<const TF_TString*>(TF_TensorData(tensor));
  if (data == nullptr) {
    return {};
  }

  const auto* str = &data[index];
  const auto* begin = TF_StringGetDataPointer(str);
  const auto size = TF_StringGetSize(str);
  if (size == 0) {
    return {};
  }
  if (begin == nullptr) {
    return {};
  }

  return {begin, size};
}

std::vector<std::string> GetStringTensorData(const TF_Tensor* tensor) {
  if (tensor == nullptr || TF_TensorType(tensor) != TF_STRING) {
    return {};
  }

  const auto byte_size = TF_TensorByteSize(tensor);
  if (byte_size % sizeof(TF_TString) != 0) {
    return {};
  }

  std::vector<std::string> result;
  const auto size = byte_size / sizeof(TF_TString);
  result.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    result.push_back(GetStringTensorElement(tensor, i));
  }

  return result;
}

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, std::size_t len) {
  if ((dims == nullptr && num_dims != 0) || !FitsTensorFlowIntParameter(num_dims)) {
    return nullptr;
  }

  std::size_t expected_len = 0;
  if (!ExpectedTensorByteSize(data_type, dims, num_dims, expected_len)) {
    return nullptr;
  }

  const auto allocation_len = (len == 0 && expected_len != 0) ? expected_len : len;
  if (allocation_len != expected_len) {
    return nullptr;
  }

  return TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), allocation_len);
}

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t len) {
  return CreateEmptyTensor(data_type, dims.data(), dims.size(), len);
}

TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len) {
  std::size_t expected_len = 0;
  if (!ExpectedTensorByteSize(data_type, dims, num_dims, expected_len) || len != expected_len) {
    return nullptr;
  }

  auto tensor = CreateEmptyTensor(data_type, dims, num_dims, expected_len);
  if (tensor == nullptr) {
    return nullptr;
  }

  auto tensor_data = TF_TensorData(tensor);
  if (expected_len == 0) {
    return tensor;
  }

  if (tensor_data == nullptr || data == nullptr) {
    DeleteTensor(tensor);
    return nullptr;
  }

  std::memcpy(tensor_data, data, expected_len);

  return tensor;
}

void DeleteTensor(TF_Tensor* tensor) {
  if (tensor != nullptr) {
    TF_DeleteTensor(tensor);
  }
}

void DeleteTensors(const std::vector<TF_Tensor*>& tensors) {
  for (auto& t : tensors) {
    DeleteTensor(t);
  }
}

bool SetTensorData(TF_Tensor* tensor, const void* data, std::size_t len) {
  if (tensor == nullptr) {
    return false;
  }
  if (!IsFixedSizeTensorDataType(TF_TensorType(tensor))) {
    return false;
  }

  auto tensor_data = TF_TensorData(tensor);
  if (len != TF_TensorByteSize(tensor)) {
    return false;
  }
  if (len == 0) {
    return true;
  }
  if (tensor_data == nullptr || data == nullptr) {
    return false;
  }

  std::memcpy(tensor_data, data, len);
  return true;
}

std::vector<std::int64_t> GetTensorShape(TF_Graph* graph, const TF_Output& output) {
  if (graph == nullptr || output.oper == nullptr) {
    return {};
  }

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto num_dims = TF_GraphGetTensorNumDims(graph, output, status);
  if (TF_GetCode(status) != TF_OK || num_dims < 0) {
    return {};
  }

  std::vector<std::int64_t> result(num_dims);
  TF_GraphGetTensorShape(graph, output, result.data(), num_dims, status);
  if (TF_GetCode(status) != TF_OK) {
    return {};
  }

  return result;
}

std::vector<std::vector<std::int64_t>> GetTensorsShape(TF_Graph* graph, const std::vector<TF_Output>& outputs) {
  std::vector<std::vector<std::int64_t>> result;
  result.reserve(outputs.size());

  for (const auto& o : outputs) {
    result.push_back(GetTensorShape(graph, o));
  }

  return result;
}

TF_SessionOptions* CreateSessionOptions(double gpu_memory_fraction, TF_Status* status) {
  // See https://github.com/Neargye/hello_tf_c_api/issues/21 for details.
  if (!IsValidGpuMemoryFraction(gpu_memory_fraction)) {
    SetStatus(status, TF_INVALID_ARGUMENT, "gpu_memory_fraction must be finite and in the range [0.0, 1.0].");
    return nullptr;
  }

  return CreateConfiguredSessionOptions(CreateGpuMemorySessionConfig(gpu_memory_fraction), status);
}

TF_SessionOptions* CreateSessionOptions(std::int32_t intra_op_parallelism_threads, std::int32_t inter_op_parallelism_threads, TF_Status* status) {
  // See https://github.com/tensorflow/tensorflow/issues/13853 for details.
  if (!IsValidThreadCount(intra_op_parallelism_threads) || !IsValidThreadCount(inter_op_parallelism_threads)) {
    SetStatus(status, TF_INVALID_ARGUMENT, "Thread counts must be non-negative.");
    return nullptr;
  }

  return CreateConfiguredSessionOptions(CreateThreadSessionConfig(intra_op_parallelism_threads, inter_op_parallelism_threads), status);
}

void DeleteSessionOptions(TF_SessionOptions* options) {
  if (options != nullptr) {
    TF_DeleteSessionOptions(options);
  }
}

const char* DataTypeToString(TF_DataType data_type) {
  switch (data_type) {
    case TF_FLOAT:
      return "TF_FLOAT";
    case TF_DOUBLE:
      return "TF_DOUBLE";
    case TF_INT32:
      return "TF_INT32";
    case TF_UINT8:
      return "TF_UINT8";
    case TF_INT16:
      return "TF_INT16";
    case TF_INT8:
      return "TF_INT8";
    case TF_STRING:
      return "TF_STRING";
    case TF_COMPLEX64:
      return "TF_COMPLEX64";
    case TF_INT64:
      return "TF_INT64";
    case TF_BOOL:
      return "TF_BOOL";
    case TF_QINT8:
      return "TF_QINT8";
    case TF_QUINT8:
      return "TF_QUINT8";
    case TF_QINT32:
      return "TF_QINT32";
    case TF_BFLOAT16:
      return "TF_BFLOAT16";
    case TF_QINT16:
      return "TF_QINT16";
    case TF_QUINT16:
      return "TF_QUINT16";
    case TF_UINT16:
      return "TF_UINT16";
    case TF_COMPLEX128:
      return "TF_COMPLEX128";
    case TF_HALF:
      return "TF_HALF";
    case TF_RESOURCE:
      return "TF_RESOURCE";
    case TF_VARIANT:
      return "TF_VARIANT";
    case TF_UINT32:
      return "TF_UINT32";
    case TF_UINT64:
      return "TF_UINT64";
    default:
      return "Unknown";
  }
}

const char* CodeToString(TF_Code code) {
  switch (code) {
    case TF_OK:
      return "TF_OK";
    case TF_CANCELLED:
      return "TF_CANCELLED";
    case TF_UNKNOWN:
      return "TF_UNKNOWN";
    case TF_INVALID_ARGUMENT:
      return "TF_INVALID_ARGUMENT";
    case TF_DEADLINE_EXCEEDED:
      return "TF_DEADLINE_EXCEEDED";
    case TF_NOT_FOUND:
      return "TF_NOT_FOUND";
    case TF_ALREADY_EXISTS:
      return "TF_ALREADY_EXISTS";
    case TF_PERMISSION_DENIED:
      return "TF_PERMISSION_DENIED";
    case TF_UNAUTHENTICATED:
      return "TF_UNAUTHENTICATED";
    case TF_RESOURCE_EXHAUSTED:
      return "TF_RESOURCE_EXHAUSTED";
    case TF_FAILED_PRECONDITION:
      return "TF_FAILED_PRECONDITION";
    case TF_ABORTED:
      return "TF_ABORTED";
    case TF_OUT_OF_RANGE:
      return "TF_OUT_OF_RANGE";
    case TF_UNIMPLEMENTED:
      return "TF_UNIMPLEMENTED";
    case TF_INTERNAL:
      return "TF_INTERNAL";
    case TF_UNAVAILABLE:
      return "TF_UNAVAILABLE";
    case TF_DATA_LOSS:
      return "TF_DATA_LOSS";
    default:
      return "Unknown";
  }
}

} // namespace tf_utils
