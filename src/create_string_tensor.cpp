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
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int Run() {
  const std::int64_t batch_size = 2;
  const std::int64_t max_length = 3;
  const std::vector<std::int64_t> dims = {batch_size, max_length};

  const std::vector<std::string> owned_tokens = {
    "hello", "tensorflow", "c-api",
    "string", "tensor", "example",
  };

  auto tensor = tf_utils::CreateStringTensor(dims, owned_tokens);
  SCOPE_EXIT{ tf_utils::DeleteTensor(tensor); }; // Auto-delete on scope exit.

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

  if (TF_TensorByteSize(tensor) != owned_tokens.size() * sizeof(TF_TString)) {
    std::cout << "Wrong tensor byte size" << std::endl;
    return 5;
  }

  if (TF_TensorData(tensor) == nullptr) {
    std::cout << "Wrong string tensor data" << std::endl;
    return 6;
  }

  for (std::size_t i = 0; i < owned_tokens.size(); ++i) {
    const auto value = tf_utils::GetStringTensorElement(tensor, i);
    if (value != owned_tokens[i]) {
      std::cout << "Element: " << i << " does not match" << std::endl;
      return 7;
    }

    std::cout << "p " << i << " == " << value << " len = " << value.size() << std::endl;
  }

  std::cout << "Success create string tensor" << std::endl;

  return 0;
}

} // namespace

int main() {
  return Run();
}
