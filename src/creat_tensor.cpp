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

#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif
#include <c_api.h> // TensorFlow C API header
#include <array>
#include <iostream>
#include <vector>

void tensor_deallocation(void* data, std::size_t, void*) {
  free(data);
}

static void DeallocateTensor(void* data, std::size_t, void*) {
  free(data);
  std::cout << "Deallocate tensor" << std::endl;
}

int main() {
  const std::array<std::int64_t, 3> dims = {{1, 5, 12}};
  std::size_t size = sizeof(float);
  for (auto i : dims) {
    size *= i;
  }
  auto data = static_cast<float*>(malloc(size));
  std::vector<float> vals = {
    -0.4809832f, -0.3770838f, 0.1743573f, 0.7720509f, -0.4064746f, 0.0116595f, 0.0051413f, 0.9135732f, 0.7197526f, -0.0400658f, 0.1180671f, -0.6829428f,
    -0.4810135f, -0.3772099f, 0.1745346f, 0.7719303f, -0.4066443f, 0.0114614f, 0.0051195f, 0.9135003f, 0.7196983f, -0.0400035f, 0.1178188f, -0.6830465f,
    -0.4809143f, -0.3773398f, 0.1746384f, 0.7719052f, -0.4067171f, 0.0111654f, 0.0054433f, 0.9134697f, 0.7192584f, -0.0399981f, 0.1177435f, -0.6835230f,
    -0.4808300f, -0.3774327f, 0.1748246f, 0.7718700f, -0.4070232f, 0.0109549f, 0.0059128f, 0.9133330f, 0.7188759f, -0.0398740f, 0.1181437f, -0.6838635f,
    -0.4807833f, -0.3775733f, 0.1748378f, 0.7718275f, -0.4073670f, 0.0107582f, 0.0062978f, 0.9131795f, 0.7187147f, -0.0394935f, 0.1184392f, -0.6840039f,
  };

  std::copy(vals.begin(), vals.end(), data); // init input_vals.

  TF_Tensor* tensor = TF_NewTensor(TF_FLOAT,
                                   dims.data(), static_cast<int>(dims.size()),
                                   data, size,
                                   DeallocateTensor, nullptr);


  if (TF_TensorType(tensor) != TF_FLOAT) {
    std::cout << "Wrong tensor type" << std::endl;
  }

  if (TF_NumDims(tensor) != dims.size())

  {
    std::cout << "Wrong number of dimensions" << std::endl;
  }

  for (std::size_t i = 0; i < dims.size(); i++) {
    if (TF_Dim(tensor, static_cast<int>(i)) != dims[i]) {
      std::cout << "Wrong dimension size for dim: " << i << std::endl;
    }
  }
  if (TF_TensorByteSize(tensor) != size) {
    std::cout << "Wrong tensor byte size" << std::endl;
  }

  auto tf_data = static_cast<float*>(TF_TensorData(tensor));

  for (std::size_t i = 0; i < vals.size(); ++i) {
    if (tf_data[i] != vals[i]) {
      std::cout << "Element: " << i << " does not match" << std::endl;
    }
  }

  std::cout << "Success creat tensor" << std::endl;

  TF_DeleteTensor(tensor);

  return 0;
}
