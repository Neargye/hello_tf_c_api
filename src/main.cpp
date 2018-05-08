#include <iostream>
#define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#include <c_api.h> // TensorFlow C API header

int main() {
  std::cout<< "TensorFlow Version: " << TF_Version() << std::endl;
  return 0;
}
