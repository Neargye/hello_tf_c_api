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

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4996)
#endif

#include "tf_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace tf_utils {

namespace {
static void DeallocateBuffer(void* data, size_t) {
  std::free(data);
}

static TF_Buffer* ReadBufferFromFile(const char* file) {
  const auto f = std::fopen(file, "rb");
  if (f == nullptr) {
    return nullptr;
  }

  std::fseek(f, 0, SEEK_END);
  const auto fsize = ftell(f);
  std::fseek(f, 0, SEEK_SET);

  if (fsize < 1) {
    std::fclose(f);
    return nullptr;
  }

  const auto data = std::malloc(fsize);
  std::fread(data, fsize, 1, f);
  std::fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  return buf;
}

} // namespace tf_utils::

TF_Graph* LoadGraphDef(const char* file) {
  if (file == nullptr) {
    return nullptr;
  }

  TF_Buffer* buffer = ReadBufferFromFile(file);
  if (buffer == nullptr) {
    return nullptr;
  }

  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteGraph(graph);
    graph = nullptr;
  }

  TF_DeleteStatus(status);

  return graph;
}

bool RunSession(TF_Graph* graph,
                const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs) {
  if (graph == nullptr ||
      inputs == nullptr || input_tensors == nullptr ||
      outputs == nullptr || output_tensors == nullptr) {
    return false;
  }

  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* options = TF_NewSessionOptions();
  TF_Session* sess = TF_NewSession(graph, options, status);
  TF_DeleteSessionOptions(options);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    return false;
  }

  TF_SessionRun(sess,
                nullptr, // Run options.
                inputs, input_tensors, static_cast<int>(ninputs), // Input tensors, input tensor values, number of inputs.
                outputs, output_tensors, static_cast<int>(noutputs), // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
  );

  if (TF_GetCode(status) != TF_OK) {
    TF_CloseSession(sess, status);
    TF_DeleteSession(sess, status);
    TF_DeleteStatus(status);
    return false;
  }

  TF_CloseSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_CloseSession(sess, status);
    TF_DeleteSession(sess, status);
    TF_DeleteStatus(status);
    return false;
  }

  TF_DeleteSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    return false;
  }

  TF_DeleteStatus(status);

  return true;
}

bool RunSession(TF_Graph* graph,
                const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors) {
  return RunSession(graph,
                    inputs.data(), input_tensors.data(), input_tensors.size(),
                    outputs.data(), output_tensors.data(), output_tensors.size());
}

TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len) {
  if (dims == nullptr || data == nullptr) {
    return nullptr;
  }

  TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
  if (tensor == nullptr) {
    return nullptr;
  }

  void* tensor_data = TF_TensorData(tensor);
  if (tensor_data == nullptr) {
    TF_DeleteTensor(tensor);
    return nullptr;
  }

  std::memcpy(tensor_data, data, std::min(len, TF_TensorByteSize(tensor)));

  return tensor;
}

void DeleteTensor(TF_Tensor* tensor) {
  if (tensor == nullptr) {
    return;
  }
  TF_DeleteTensor(tensor);
}

void DeleteTensors(const std::vector<TF_Tensor*>& tensors) {
  for (auto t : tensors) {
    TF_DeleteTensor(t);
  }
}

} // namespace tf_utils

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
