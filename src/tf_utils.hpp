#pragma once

#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif
#include <c_api.h> // TensorFlow C API header

TF_Graph* LoadGraphDef(const char* file);

bool RunSession(TF_Graph* graph,
                TF_Output* input, TF_Tensor** input_tensor, int ninputs,
                TF_Output* output, TF_Tensor** output_tensor, int noutputs);
