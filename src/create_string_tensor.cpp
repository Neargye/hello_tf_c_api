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
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct StringTensorDeallocatorArg {
  std::size_t size;
};

static void DeallocateStringTensor(void* data, std::size_t, void* arg) {
  auto strings = static_cast<TF_TString*>(data);
  auto deallocator_arg = static_cast<StringTensorDeallocatorArg*>(arg);

  if (strings != nullptr && deallocator_arg != nullptr) {
    for (std::size_t i = 0; i < deallocator_arg->size; ++i) {
      TF_StringDealloc(&strings[i]);
    }
  }

  delete[] strings;
  delete deallocator_arg;
  std::cout << "Deallocate string tensor" << std::endl;
}

static std::int64_t ShapeElementCount(const std::vector<std::int64_t>& dims) {
  return std::accumulate(dims.begin(), dims.end(), std::int64_t{1}, std::multiplies<std::int64_t>{});
}

static TF_Tensor* CreateStringTensor(const std::vector<std::int64_t>& dims, const std::vector<const char*>& strings) {
  if (ShapeElementCount(dims) != static_cast<std::int64_t>(strings.size())) {
    return nullptr;
  }

  auto data = new TF_TString[strings.size()];
  for (std::size_t i = 0; i < strings.size(); ++i) {
    TF_StringInit(&data[i]);
    TF_StringCopy(&data[i], strings[i], std::strlen(strings[i]));
  }

  auto deallocator_arg = new StringTensorDeallocatorArg{strings.size()};
  return TF_NewTensor(TF_STRING,
                      dims.data(), static_cast<int>(dims.size()),
                      data, strings.size() * sizeof(TF_TString),
                      DeallocateStringTensor, deallocator_arg);
}

static std::string GetStringTensorElement(const TF_Tensor* tensor, std::size_t index) {
  const auto data = static_cast<const TF_TString*>(TF_TensorData(tensor));
  const auto* str = &data[index];
  const auto* begin = TF_StringGetDataPointer(str);

  return {begin, begin + TF_StringGetSize(str)};
}

} // namespace

int main() {
  const std::int64_t batch_size = 2;
  const std::int64_t max_length = 3;
  const std::vector<std::int64_t> dims = {batch_size, max_length};

  const std::vector<const char*> tokens = {
    "hello", "tensorflow", "c-api",
    "string", "tensor", "example",
  };

  auto tensor = CreateStringTensor(dims, tokens);
  SCOPE_EXIT{ TF_DeleteTensor(tensor); }; // Auto-delete on scope exit.

  if (tensor == nullptr) {
    std::cout << "Wrong create string tensor" << std::endl;
    return 1;
  }

  if (TF_TensorType(tensor) != TF_STRING) {
    std::cout << "Wrong tensor type" << std::endl;
    return 2;
  }

  if (TF_NumDims(tensor) != static_cast<int>(dims.size())) {
    std::cout << "Wrong number of dimensions" << std::endl;
    return 3;
  }

  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (TF_Dim(tensor, static_cast<int>(i)) != dims[i]) {
      std::cout << "Wrong dimension size for dim: " << i << std::endl;
      return 4;
    }
  }

  if (TF_TensorByteSize(tensor) != tokens.size() * sizeof(TF_TString)) {
    std::cout << "Wrong tensor byte size" << std::endl;
    return 5;
  }

  if (TF_TensorData(tensor) == nullptr) {
    std::cout << "Wrong string tensor data" << std::endl;
    return 6;
  }

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto value = GetStringTensorElement(tensor, i);
    if (value != tokens[i]) {
      std::cout << "Element: " << i << " does not match" << std::endl;
      return 7;
    }

    std::cout << "p " << i << " == " << value << " len = " << value.size() << std::endl;
  }

  std::cout << "Success create string tensor" << std::endl;

  return 0;
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
