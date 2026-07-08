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

#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4190)
#endif

#include <tensorflow/c/c_api.h> // TensorFlow C API header.
#include <scope_guard.hpp>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <system_error>

static void DeallocateBuffer(void* data, size_t) {
  std::free(data);
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

int main() {
  auto buffer = ReadBufferFromFile("graph.pb");
  if (buffer == nullptr) {
    std::cout << "Failed to read graph.pb" << std::endl;
    return 1;
  }

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ TF_DeleteGraph(graph); };
  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };
  auto opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Failed to import GraphDef" << std::endl;
    return 2;
  }

  std::cout << "Loaded graph successfully" << std::endl;

  return 0;
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
