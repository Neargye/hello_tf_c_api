#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif
#include <c_api.h> // TensorFlow C API header
#include "tf_utils.hpp"
#include <iostream>
#include <vector>
#include <string>

const char *TFDataTypeToString(TF_DataType data_type) {
  switch (data_type) {
  case TF_FLOAT:
    return "TF_FLOAT";
  case TF_DOUBLE:
    return "TF_DOUBLE";
  case TF_INT32:
    return "TF_INT32";
  case TF_UINT8:
    return "TF_UINT8";
  case TF_INT16:
    return "TF_INT16";
  case TF_INT8:
    return "TF_INT8";
  case TF_STRING:
    return "TF_STRING";
  case TF_COMPLEX64:
    return "TF_COMPLEX64";
  case TF_INT64:
    return "TF_INT64";
  case TF_BOOL:
    return "TF_BOOL";
  case TF_QINT8:
    return "TF_QINT8";
  case TF_QUINT8:
    return "TF_QUINT8";
  case TF_QINT32:
    return "TF_QINT32";
  case TF_BFLOAT16:
    return "TF_BFLOAT16";
  case TF_QINT16:
    return "TF_QINT16";
  case TF_QUINT16:
    return "TF_QUINT16";
  case TF_UINT16:
    return "TF_UINT16";
  case TF_COMPLEX128:
    return "TF_COMPLEX128";
  case TF_HALF:
    return "TF_HALF";
  case TF_RESOURCE:
    return "TF_RESOURCE";
  case TF_VARIANT:
    return "TF_VARIANT";
  case TF_UINT32:
    return "TF_UINT32";
  case TF_UINT64:
    return "TF_UINT64";
  default:
    return "Unknown";
  }
}

void PrintInputs(TF_Graph*, TF_Operation* op) {
  const int num_inputs = TF_OperationNumInputs(op);

  for (int i = 0; i < num_inputs; ++i) {
    const TF_Input input = {op, i};
    const TF_DataType type = TF_OperationInputType(input);
    std::cout << "Input: " << i << " type: " << TFDataTypeToString(type) << std::endl;
  }
}

void PrintOutputs(TF_Graph* graph, TF_Operation* op) {
  const int num_outputs = TF_OperationNumOutputs(op);
  TF_Status *status = TF_NewStatus();

  for (int i = 0; i < num_outputs; i++) {
    const TF_Output output = {op, i};
    const TF_DataType type = TF_OperationOutputType(output);
    const int num_dims = TF_GraphGetTensorNumDims(graph, output, status);

    if (TF_GetCode(status) != TF_OK) {
      std::cout << "Can't get tensor dimensionality" << std::endl;
      continue;
    }

    std::vector<std::int64_t> dims(num_dims);

    std::cout << "Output: " << i << " type: " << TFDataTypeToString(type);
    TF_GraphGetTensorShape(graph, output, dims.data(), num_dims, status);

    if (TF_GetCode(status) != TF_OK) {
      std::cout << "Can't get get tensor shape" << std::endl;
      continue;
    }

    std::cout << " dims: " << num_dims << " [";
    for (int d = 0; d < num_dims; ++d) {
      std::cout << dims[d];
      if (d < num_dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  TF_DeleteStatus(status);
}

void PrintTensorInfo(TF_Graph *graph, const char *layer_name) {
  std::cout << "Tensor: " << layer_name;
  TF_Operation *op = TF_GraphOperationByName(graph, layer_name);

  if (op == nullptr) {
    std::cout << "Could not get " << layer_name << std::endl;
    return;
  }

  const int num_inputs = TF_OperationNumInputs(op);
  const int num_outputs = TF_OperationNumOutputs(op);
  std::cout << " inputs: " << num_inputs << " outputs: " << num_outputs << std::endl;

  PrintInputs(graph, op);

  PrintOutputs(graph, op);
}

int main() {
  TF_Graph *graph = LoadGraphDef("graph.pb");
  if (graph == nullptr) {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  PrintTensorInfo(graph, "input_4");
  std::cout << std::endl;

  PrintTensorInfo(graph, "output_node0");

  TF_DeleteGraph(graph);

  return 0;
}
