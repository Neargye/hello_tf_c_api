#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif
#include <c_api.h> // TensorFlow C API header
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

int main() {
  TF_Buffer* buffer = ReadBufferFromFile("graph.pb");
  if (buffer == nullptr) {
    std::cout << "Can't read buffer from file" << std::endl;
    return 1;
  }

  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);
    std::cout << "Can't import GraphDef" << std::endl;
    return 2;
  }

  std::cout << "Load draph success" << std::endl;

  TF_DeleteStatus(status);

  return 0;
}
