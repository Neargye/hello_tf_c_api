# TensorFlow C API Examples

![TensorFlow C API Examples logo](logo.png)

A small cross-platform set of TensorFlow C API examples for Windows, Linux, and macOS.

## Requirements

* CMake 3.20 or newer.
* C++17 compiler.
* Python with pip. CI uses Python 3.12.
* 64-bit target platform.

## [Examples](src/)

* [Hello TF](src/hello_tf.cpp)
* [Load graph](src/load_graph.cpp)
* [Create Tensor](src/create_tensor.cpp)
* [Create String Tensor](src/create_string_tensor.cpp)
* [Image processing](src/image_example.cpp)
* [Run target operation](src/target_operation.cpp)
* [OpenCV image file processing](src/opencv_image_file_example.cpp) (optional, requires OpenCV)
* [Allocate Tensor](src/allocate_tensor.cpp)
* [Run session](src/session_run.cpp)
* [Repeated inference](src/repeated_inference.cpp)
* [Interface](src/interface.cpp)
* [Batch Interface](src/batch_interface.cpp)
* [Tensor Info](src/tensor_info.cpp)
* [Graph Info](src/graph_info.cpp)

## Build and test

### Windows

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake -A x64 ..
cmake --build . --config Release
ctest --output-on-failure -C Release
```

### Linux

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 4
ctest --output-on-failure
```

### macOS

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure -C Release
```

### Notes

* CMake downloads TensorFlow 2.21.0 from the Python wheel into the build-local `<build>/_deps/tensorflow/python` cache by default.
* To use an existing local TensorFlow wheel extraction, configure with `-DTENSORFLOW_ROOT=/path/to/tensorflow`. Auto-fetch only writes to the default build-local TensorFlow cache; it refuses to overwrite an external `TENSORFLOW_ROOT`. To require a pre-existing extraction and disable downloads during configure, add `-DHELLO_TF_FETCH_TENSORFLOW=OFF`.
* Python with pip is required during CMake configure.
* The small GraphDef used by graph and session examples is committed as `models/graph.pb`; no external model download is required.
* To regenerate the example GraphDef, run `python tools/create_example_graph.py` from a Python environment where the full TensorFlow package is available.
* OpenCV is optional. If CMake finds it, the OpenCV image-file example is built and tested.
* On Windows, CMake copies the required TensorFlow runtime DLLs into the build output directories.
* Tests use [doctest](test/3rdparty/doctest/doctest.h). CI also runs an ASan/UBSan test job on Ubuntu.
* To configure only the helper library without example executables, add `-DHELLO_TF_BUILD_EXAMPLES=OFF`.
* Tests follow CMake's standard `BUILD_TESTING` option. To configure without tests, add `-DBUILD_TESTING=OFF`.

## TensorFlow library

This project uses the TensorFlow 2.21.0 Python wheel and links the C API headers and native libraries from the local `<TENSORFLOW_ROOT>/python` directory. The CMake file creates an imported `tensorflow` target, a `hello_tf_utils` helper library target, and copies required runtime libraries where needed.

`tf_utils::LoadGraph` only imports a GraphDef. If a graph needs checkpoint restore operations, create the session first and call `tf_utils::RestoreCheckpoint(session, graph, ...)` on that session. TensorFlow variable state belongs to `TF_Session`, not to `TF_Graph`.

If you want to link TensorFlow manually, use the headers from:

```text
<TENSORFLOW_ROOT>/python/tensorflow/include
```

and the native libraries from:

```text
<TENSORFLOW_ROOT>/python/tensorflow
<TENSORFLOW_ROOT>/python/tensorflow/python
```

You can also build the TensorFlow library version you need from source, with CPU or GPU support.

### Project-local CMake targets

#### CMakeLists.txt

Examples that use the helper API link the `hello_tf_utils` target:

```text
target_link_libraries(<target> PRIVATE hello_tf_utils)
```

Examples that demonstrate only the raw TensorFlow C API use:

```text
target_link_tensorflow(<target>)
```

If another project needs a small part of this repository, copy the relevant example or helper source and wire it to that project's TensorFlow target explicitly. This repository is maintained as local examples plus tests, not as a packaged dependency.

### Use TensorFlow in Visual Studio

Open "Project" -> "Properties" -> "Configuration Properties" -> "C/C++" -> "Additional Include Directories" and add the TensorFlow include path.

Open "Project" -> "Properties" -> "Configuration Properties" -> "Linker" -> "Additional Dependencies" and add the TensorFlow import library path.

Make sure that the TensorFlow DLLs are in the output directory or in a directory contained by the `%PATH%` environment variable.

### [Prepare models](doc/prepare_models.md)

This repository already includes the demo `models/graph.pb` used by the examples. For your own models, prefer a TensorFlow 2 SavedModel export, or use a small inference-only GraphDef when you want the same import path as these examples.

### [Optimize models and examples](doc/optimizing.md)

### [Create a Windows import library from a TensorFlow DLL](doc/create_lib_file_from_dll_for_windows.md)

### Further reading

* https://www.tensorflow.org/guide/saved_model
* https://www.tensorflow.org/lite/performance/model_optimization

## Licensed under the [MIT License](LICENSE)
