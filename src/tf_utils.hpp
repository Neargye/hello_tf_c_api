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
#include <vector>

namespace tf_utils {

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
                   const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                   const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors,
                   TF_Status* status = nullptr);

TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len);

template <typename T>
TF_Tensor* CreateTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, const std::vector<T>& data) {
  return CreateTensor(data_type,
                      dims.data(), dims.size(),
                      data.data(), data.size() * sizeof(T));
}

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, std::size_t len = 0);

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t len = 0);

void DeleteTensor(TF_Tensor* tensor);

void DeleteTensors(const std::vector<TF_Tensor*>& tensors);

bool SetTensorData(TF_Tensor* tensor, const void* data, std::size_t len);

template <typename T>
void SetTensorData(TF_Tensor* tensor, const std::vector<T>& data) {
  SetTensorsData(tensor, data.data(), data.size() * sizeof(T));
}

template <typename T>
std::vector<T> GetTensorData(const TF_Tensor* tensor) {
  if (tensor == nullptr) {
    return {};
  }
  auto data = static_cast<T*>(TF_TensorData(tensor));
  auto size = TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor));
  if (data == nullptr || size <= 0) {
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
