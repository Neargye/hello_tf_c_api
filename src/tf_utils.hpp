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

#pragma once

#if defined(_MSC_VER)
#  if !defined(COMPILER_MSVC)
#    define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#  endif
#  pragma warning(push)
#  pragma warning(disable : 4190)
#endif

#include <tensorflow/c/c_api.h> // TensorFlow C API header.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tf_utils {

namespace detail {

template <typename T>
struct TensorDataType {
  static constexpr bool supported = false;
};

template <>
struct TensorDataType<float> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_FLOAT;
};

template <>
struct TensorDataType<double> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_DOUBLE;
};

template <>
struct TensorDataType<std::int8_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_INT8;
};

template <>
struct TensorDataType<std::uint8_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_UINT8;
};

template <>
struct TensorDataType<std::int16_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_INT16;
};

template <>
struct TensorDataType<std::uint16_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_UINT16;
};

template <>
struct TensorDataType<std::int32_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_INT32;
};

template <>
struct TensorDataType<std::uint32_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_UINT32;
};

template <>
struct TensorDataType<std::int64_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_INT64;
};

template <>
struct TensorDataType<std::uint64_t> {
  static constexpr bool supported = true;
  static constexpr TF_DataType value = TF_UINT64;
};

template <typename T>
using TensorValueType = typename std::remove_cv<T>::type;

template <typename T>
constexpr bool IsSupportedTensorValueType() {
  return TensorDataType<TensorValueType<T>>::supported;
}

template <typename T>
constexpr TF_DataType TensorDataTypeValue() {
  static_assert(IsSupportedTensorValueType<T>(), "Unsupported TensorFlow tensor value type.");
  return TensorDataType<TensorValueType<T>>::value;
}

} // namespace detail

TF_Graph* LoadGraph(const char* graph_path, const char* checkpoint_prefix, TF_Status* status = nullptr);

TF_Graph* LoadGraph(const char* graph_path, TF_Status* status = nullptr);

void DeleteGraph(TF_Graph* graph);

TF_Session* CreateSession(TF_Graph* graph, TF_SessionOptions* options, TF_Status* status = nullptr);

TF_Session* CreateSession(TF_Graph* graph, TF_Status* status = nullptr);

TF_Code DeleteSession(TF_Session* session, TF_Status* status = nullptr);

TF_Code RunSession(TF_Session* session,
                   const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                   const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs,
                   TF_Status* status = nullptr);

TF_Code RunSession(TF_Session* session,
                   const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                   const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs,
                   const TF_Operation* const* target_opers, std::size_t ntargets,
                   TF_Status* status = nullptr);

TF_Code RunSession(TF_Session* session,
                   const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                   const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors,
                   TF_Status* status = nullptr);

TF_Code RunSession(TF_Session* session,
                   const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                   const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors,
                   const std::vector<const TF_Operation*>& target_opers,
                   TF_Status* status = nullptr);

TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len);

template <typename T>
TF_Tensor* CreateTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, const std::vector<T>& data) {
  static_assert(detail::IsSupportedTensorValueType<T>(), "Use CreateStringTensor for TF_STRING and supported arithmetic types for numeric tensors.");
  if (data_type != detail::TensorDataTypeValue<T>()) {
    return nullptr;
  }

  return CreateTensor(data_type,
                      dims.data(), dims.size(),
                      data.data(), data.size() * sizeof(T));
}

TF_Tensor* CreateStringTensor(const std::int64_t* dims, std::size_t num_dims,
                              const std::string_view* strings, std::size_t num_strings);

TF_Tensor* CreateStringTensor(const std::vector<std::int64_t>& dims, const std::vector<std::string_view>& strings);

TF_Tensor* CreateStringTensor(const std::vector<std::int64_t>& dims, const std::vector<std::string>& strings);

std::string GetStringTensorElement(const TF_Tensor* tensor, std::size_t index);

std::vector<std::string> GetStringTensorData(const TF_Tensor* tensor);

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, std::size_t len = 0);

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t len = 0);

void DeleteTensor(TF_Tensor* tensor);

void DeleteTensors(const std::vector<TF_Tensor*>& tensors);

bool SetTensorData(TF_Tensor* tensor, const void* data, std::size_t len);

template <typename T>
bool SetTensorData(TF_Tensor* tensor, const std::vector<T>& data) {
  static_assert(detail::IsSupportedTensorValueType<T>(), "Unsupported TensorFlow tensor value type.");
  if (tensor == nullptr || TF_TensorType(tensor) != detail::TensorDataTypeValue<T>()) {
    return false;
  }

  return SetTensorData(tensor, data.data(), data.size() * sizeof(T));
}

template <typename T>
std::vector<T> GetTensorData(const TF_Tensor* tensor) {
  static_assert(detail::IsSupportedTensorValueType<T>(), "Use GetStringTensorData for TF_STRING and supported arithmetic types for numeric tensors.");
  if (tensor == nullptr) {
    return {};
  }
  if (TF_TensorType(tensor) != detail::TensorDataTypeValue<T>()) {
    return {};
  }

  const auto byte_size = TF_TensorByteSize(tensor);
  if (byte_size % sizeof(T) != 0) {
    return {};
  }

  auto data = static_cast<const T*>(TF_TensorData(tensor));
  auto size = byte_size / sizeof(T);
  if (size == 0) {
    return {};
  }
  if (data == nullptr) {
    return {};
  }

  return {data, data + size};
}

template <typename T>
std::vector<std::vector<T>> GetTensorsData(const std::vector<TF_Tensor*>& tensors) {
  std::vector<std::vector<T>> data;
  data.reserve(tensors.size());
  for (auto t : tensors) {
    data.push_back(GetTensorData<T>(t));
  }

  return data;
}

std::vector<std::int64_t> GetTensorShape(TF_Graph* graph, const TF_Output& output);

std::vector<std::vector<std::int64_t>> GetTensorsShape(TF_Graph* graph, const std::vector<TF_Output>& output);

TF_SessionOptions* CreateSessionOptions(double gpu_memory_fraction, TF_Status* status = nullptr);

TF_SessionOptions* CreateSessionOptions(std::uint8_t intra_op_parallelism_threads, std::uint8_t inter_op_parallelism_threads, TF_Status* status = nullptr);

void DeleteSessionOptions(TF_SessionOptions* options);

const char* DataTypeToString(TF_DataType data_type);

const char* CodeToString(TF_Code code);

} // namespace tf_utils

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
