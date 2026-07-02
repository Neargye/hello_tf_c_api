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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

namespace tf_utils {

namespace {

static void DeallocateBuffer(void* data, size_t) {
  std::free(data);
}

struct StringTensorDeallocatorArg {
  std::size_t size;
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

static std::size_t DataTypeByteSize(TF_DataType data_type) {
  if (data_type == TF_STRING) {
    return sizeof(TF_TString);
  }

  return TF_DataTypeSize(data_type);
}

static bool FitsTensorFlowIntParameter(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

static bool ExpectedTensorByteSize(TF_DataType data_type,
                                   const std::int64_t* dims,
                                   std::size_t num_dims,
                                   std::size_t& byte_size) {
  std::size_t element_count = 0;
  if (!ShapeElementCount(dims, num_dims, element_count)) {
    return false;
  }

  const auto element_size = DataTypeByteSize(data_type);
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

  auto data = new TF_TString[num_strings];
  std::size_t initialized = 0;
  for (; initialized < num_strings; ++initialized) {
    const auto str = get_string(initialized);
    const auto* str_data = str.empty() ? "" : str.data();
    TF_StringInit(&data[initialized]);
    TF_StringCopy(&data[initialized], str_data, str.size());
  }

  auto deallocator_arg = new StringTensorDeallocatorArg{num_strings};
  auto tensor = TF_NewTensor(TF_STRING,
                             dims, static_cast<int>(num_dims),
                             data, num_strings * sizeof(TF_TString),
                             &DeallocateStringTensor, deallocator_arg);
  if (tensor == nullptr) {
    for (std::size_t i = 0; i < initialized; ++i) {
      TF_StringDealloc(&data[i]);
    }
    delete[] data;
    delete deallocator_arg;
  }

  return tensor;
}

static TF_Buffer* ReadBufferFromFile(const char* file) {
  std::ifstream f(file, std::ios::binary);
  SCOPE_EXIT{ f.close(); };
  if (f.fail() || !f.is_open()) {
    return nullptr;
  }

  if (f.seekg(0, std::ios::end).fail()) {
    return nullptr;
  }
  auto fsize = f.tellg();
  if (f.seekg(0, std::ios::beg).fail()) {
    return nullptr;
  }

  if (fsize <= 0) {
    return nullptr;
  }

  auto data = static_cast<char*>(std::malloc(fsize));
  if (data == nullptr) {
    return nullptr;
  }

  if (f.read(data, fsize).fail()) {
    std::free(data);
    return nullptr;
  }

  auto buf = TF_NewBuffer();
  if (buf == nullptr) {
    std::free(data);
    return nullptr;
  }

  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  return buf;
}

TF_Tensor* ScalarStringTensor(const char* str, TF_Status*) {
  const std::string_view value(str);

  return CreateStringTensor(nullptr, 0, &value, 1);
}

} // namespace

TF_Graph* LoadGraph(const char* graph_path, const char* checkpoint_prefix, TF_Status* status) {
  if (graph_path == nullptr) {
    return nullptr;
  }

  auto buffer = ReadBufferFromFile(graph_path);
  if (buffer == nullptr) {
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

  if (checkpoint_prefix == nullptr) {
    return graph;
  }

  auto checkpoint_tensor = ScalarStringTensor(checkpoint_prefix, status);
  SCOPE_EXIT{ DeleteTensor(checkpoint_tensor); };
  if (checkpoint_tensor == nullptr || TF_GetCode(status) != TF_OK) {
    DeleteGraph(graph);
    return nullptr;
  }

  auto input = TF_Output{TF_GraphOperationByName(graph, "save/Const"), 0};
  auto restore_op = TF_GraphOperationByName(graph, "save/restore_all");
  if (input.oper == nullptr || restore_op == nullptr) {
    DeleteGraph(graph);
    return nullptr;
  }

  auto session = CreateSession(graph);
  SCOPE_EXIT{ DeleteSession(session); };
  if (session == nullptr) {
    DeleteGraph(graph);
    return nullptr;
  }

  TF_SessionRun(session,
                nullptr, // Run options.
                &input, &checkpoint_tensor, 1, // Input tensors, input tensor values, number of inputs.
                nullptr, nullptr, 0, // Output tensors, output tensor values, number of outputs.
                &restore_op, 1, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
  );

  if (TF_GetCode(status) != TF_OK) {
    DeleteGraph(graph);
    return nullptr;
  }

  return graph;
}

TF_Graph* LoadGraph(const char* graph_path, TF_Status* status) {
  return LoadGraph(graph_path, nullptr, status);
}

void DeleteGraph(TF_Graph* graph) {
  if (graph != nullptr) {
    TF_DeleteGraph(graph);
  }
}

TF_Session* CreateSession(TF_Graph* graph, TF_SessionOptions* options, TF_Status* status) {
  if (graph == nullptr) {
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
    return TF_INVALID_ARGUMENT;
  }

  MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
  if (status == nullptr) {
    status = TF_NewStatus();
  } else {
    delete_status.dismiss();
  }

  TF_CloseSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
    SCOPE_EXIT{ TF_CloseSession(session, status); };
    SCOPE_EXIT{ TF_DeleteSession(session, status); };
    return TF_GetCode(status);
  }

  TF_DeleteSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
    SCOPE_EXIT{ TF_DeleteSession(session, status); };
    return TF_GetCode(status);
  }

  return TF_OK;
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
  if (session == nullptr ||
      (ninputs != 0 && (inputs == nullptr || input_tensors == nullptr)) ||
      (noutputs != 0 && (outputs == nullptr || output_tensors == nullptr)) ||
      (ntargets != 0 && target_opers == nullptr) ||
      !FitsTensorFlowIntParameter(ninputs) ||
      !FitsTensorFlowIntParameter(noutputs) ||
      !FitsTensorFlowIntParameter(ntargets)) {
    return TF_INVALID_ARGUMENT;
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
    return TF_INVALID_ARGUMENT;
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
    return TF_INVALID_ARGUMENT;
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

  return TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
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

  MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
  if (status == nullptr) {
    status = TF_NewStatus();
  } else {
    delete_status.dismiss();
  }

  auto options = TF_NewSessionOptions();

  // This matches the following Python configuration:
  // config = tf.ConfigProto( allow_soft_placement = True )
  // config.gpu_options.allow_growth = True
  // config.gpu_options.per_process_gpu_memory_fraction = percentage
  // Create the serialized ProtoConfig byte array with the fixed bytes already set.
  std::array<std::uint8_t, 15> config = {{0x32, 0xb, 0x9, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x20, 0x1, 0x38, 0x1}};

  // Convert the desired percentage into bytes.
  auto bytes = reinterpret_cast<std::uint8_t*>(&gpu_memory_fraction);

  // Store the percentage bytes in positions 3 through 10.
  for (std::size_t i = 0; i < sizeof(gpu_memory_fraction); ++i) {
    config[i + 3] = bytes[i];
  }

  TF_SetConfig(options, config.data(), config.size(), status);

  if (TF_GetCode(status) != TF_OK) {
    DeleteSessionOptions(options);
    return nullptr;
  }

  return options;
}

TF_SessionOptions* CreateSessionOptions(std::uint8_t intra_op_parallelism_threads, std::uint8_t inter_op_parallelism_threads, TF_Status* status) {
  // See https://github.com/tensorflow/tensorflow/issues/13853 for details.

  MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
  if (status == nullptr) {
    status = TF_NewStatus();
  } else {
    delete_status.dismiss();
  }

  auto options = TF_NewSessionOptions();
  std::array<std::uint8_t, 4> config = {{0x10, intra_op_parallelism_threads, 0x28, inter_op_parallelism_threads}};
  TF_SetConfig(options, config.data(), config.size(), status);

  if (TF_GetCode(status) != TF_OK) {
    DeleteSessionOptions(options);
    return nullptr;
  }

  return options;
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
