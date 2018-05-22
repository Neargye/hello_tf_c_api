#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif
#include <c_api.h> // TensorFlow C API header
#include <iostream>

int main() {
  std::cout<< "TensorFlow Version: " << TF_Version() << std::endl;
  return 0;
}
