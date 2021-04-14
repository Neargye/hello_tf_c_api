# Example TensorFlow C API

![Example TensorFlow C API Logo](logo.png)

Example how to run TensorFlow lib C API on Windows, Linux and macOS(Darwin).

## [Example](src/)

* [Hello TF](src/hello_tf.cpp)
* [Load graph](src/load_graph.cpp)
* [Create Tensor](src/create_tensor.cpp)
* [Allocate Tensor](src/allocate_tensor.cpp)
* [Run session](src/session_run.cpp)
* [Interface](src/interface.cpp)
* [Tensor Info](src/tensor_info.cpp)
* [Graph Info](src/graph_info.cpp)
* [Image processing](https://github.com/Xonxt/hello_tf_c_api/blob/master/src/image_example.cpp)

## Build example

### Windows

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake -G "Visual Studio 15 2017" -A x64 ..
cmake --build . --config Debug
```

### Linux

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake -G "Unix Makefiles" ..
cmake --build .
```

### macOS(Darwin)

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake -G "XCode" ..
cmake --build .
```

### Remarks

* After the build, you can find the TensorFlow lib in the folder hello_tf_c_api/tensorflow/lib, and header in hello_tf_c_api/tensorflow/include.
* The tensorflow in the repository is compiled in x64 mode. Make sure that project target 64-bit platforms.
* Make sure that the tensorflow lib is in Output Directory or either in the directory contained by the %PATH% environment variable.

## Get tensorflow lib

For x64 CPU, you can download the tensorflow.so, tensorflow.dll and tensorflow.lib from <https://github.com/neargye-forks/tensorflow/releases>.

Or build lib which version you need from the sources, with CPU or GPU support.

### Link tensorflow lib

#### CMakeLists.txt

```text
link_directories(yourpath/to/tensorflow) # path to tensorflow lib
... # other
target_link_libraries(<target> <PRIVATE|PUBLIC|INTERFACE> tensorflow)
```

#### Visual Studio

"Project"->"Properties"->Configuration Properties"->"Linker"->"Additional Dependencies" and add path to your tensorflow.lib as a next line.

Make sure that the tensorflow.dll is in Output Directory (by default, this is Debug\Release under your project's folder) or either in the directory contained by the %PATH% environment variable.

### [Here’s an example how prepare models](doc/prepare_models.md)

To generated the graph.pb file need takes a graph definition and a set of checkpoints and freezes them together into a single file.

### [Here’s an example how create tensorflow.lib file from tensorflow.dll for windows](doc/create_lib_file_from_dll_for_windows.md)

### __Few articles with details__

* https://www.tensorflow.org/install/lang_c
* https://medium.com/@vladislavsd/undocumented-tensorflow-c-api-b527c0b4ef6
* https://medium.com/analytics-vidhya/deploying-tensorflow-2-1-as-c-c-executable-1d090845055c


## Licensed under the [MIT License](LICENSE)
