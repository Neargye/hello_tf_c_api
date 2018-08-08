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

#include "tf_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4996)
#endif

static void DeallocateBuffer(void* data, size_t) {
  free(data);
}

static TF_Buffer* ReadBufferFromFile(const char* file) {
  FILE* f = fopen(file, "rb");
  if (f == nullptr) {
    return nullptr;
  }

  fseek(f, 0, SEEK_END);
  const long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (fsize < 1) {
    fclose(f);
    return nullptr;
  }

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  return buf;
}

TF_Graph* LoadGraphDef(const char* file) {
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
                TF_Output* input, TF_Tensor** input_tensor, int ninputs,
                TF_Output* output, TF_Tensor** output_tensor, int noutputs) {
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
                input, input_tensor, ninputs, // Input tensors, input tensor values, number of inputs.
                output, output_tensor, noutputs, // Output tensors, output tensor values, number of outputs.
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

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
