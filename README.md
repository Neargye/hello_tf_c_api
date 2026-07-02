# Example TensorFlow C API

![Example TensorFlow C API Logo](logo.png)

Example how to run TensorFlow C API on Windows, Linux and macOS (Darwin).

## Requirements

* CMake 3.20 or newer.
* C++17 compiler.
* Python with pip. CI uses Python 3.12.
* 64-bit target platform.

## [Example](src/)

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

## Build example

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

### macOS (Darwin)

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure -C Release
```

### Remarks

* CMake downloads TensorFlow 2.21.0 from the Python wheel into hello_tf_c_api/tensorflow/python. It does not install TensorFlow into the system Python.
* Python with pip is required during CMake configure.
* The small GraphDef used by graph and session examples is committed as `models/graph.pb`; no external model download is required.
* To regenerate the example GraphDef, run `python tools/create_example_graph.py` from a virtual environment with a full TensorFlow Python package installed.
* OpenCV is optional. If CMake finds it, the OpenCV image-file example is built and tested.
* On Windows, CMake copies the required TensorFlow runtime DLLs into the build output directories.
* Tests use [doctest](test/3rdparty/doctest/doctest.h). CI also runs an ASan/UBSan test job on Ubuntu.

## TensorFlow library

This project uses the TensorFlow 2.21.0 Python wheel and links the C API headers and native libraries from the local tensorflow/python directory. The CMake file creates an imported `tensorflow` target and copies required runtime libraries where needed.

If you want to link TensorFlow manually, use the headers from:

```text
tensorflow/python/tensorflow/include
```

and the native libraries from:

```text
tensorflow/python/tensorflow
tensorflow/python/tensorflow/python
```

For standalone C API packages, you can also download the TensorFlow C library from https://www.tensorflow.org/install/lang_c.

Or build the library version you need from sources, with CPU or GPU support.

### Link TensorFlow lib

#### CMakeLists.txt

Inside this project, examples use:

```text
target_link_tensorflow(<target>)
```

For a separate CMake project, add the TensorFlow include directory and link the native library or imported target you define:

```text
target_include_directories(<target> PRIVATE path/to/tensorflow/include)
target_link_libraries(<target> PRIVATE path/to/tensorflow/library)
```

#### Visual Studio

"Project" -> "Properties" -> "Configuration Properties" -> "C/C++" -> "Additional Include Directories" and add the TensorFlow include path.

"Project" -> "Properties" -> "Configuration Properties" -> "Linker" -> "Additional Dependencies" and add the TensorFlow import library path.

Make sure that the TensorFlow DLLs are in the output directory or in a directory contained by the `%PATH%` environment variable.

### [Here’s an example how to prepare models](doc/prepare_models.md)

This repository already includes the demo `models/graph.pb` used by the examples. For your own models, prefer a TensorFlow 2 SavedModel export, or use a small inference-only GraphDef when you want the same import path as these examples.

### [Here’s an example how to optimize models and examples](doc/optimizing.md)

### [Here’s a fallback for creating a tensorflow.lib file from tensorflow.dll for Windows](doc/create_lib_file_from_dll_for_windows.md)

### __Few articles with details__

* https://www.tensorflow.org/install/lang_c
* https://www.tensorflow.org/guide/saved_model
* https://www.tensorflow.org/lite/performance/model_optimization


## Licensed under the [MIT License](LICENSE)
