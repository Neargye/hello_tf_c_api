#include <iostream>
#define COMPILER_MSVC
#include "tensorflow/include/c_api.h"

int main() {
  std::cout<< "TensorFlow Version: " << TF_Version() << std::endl;
  return 0;
}
