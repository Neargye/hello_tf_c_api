# Example TensorFlow C API

![Example TensorFlow C API logo](logo.png)

[![Build status](https://ci.appveyor.com/api/projects/status/vmp61qk96clboeds/branch/master?svg=true)](https://ci.appveyor.com/project/Neargye/hello-tf-win-c-api/branch/master)
[![License](https://img.shields.io/github/license/Neargye/hello_tf_win_c_api.svg)](LICENSE)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/65a8401ec7da4ff49a9d4603dfbb600a)](https://www.codacy.com/app/Neargye/hello_tf_win_c_api?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Neargye/hello_tf_win_c_api&amp;utm_campaign=Badge_Grade)

Example how to run TensorFlow lib C API on Windows, Linux and macOS(darwin).

## [Example](src/)

* [Hello TF](src/hello_tf.cpp)
* [Creat Tensor](src/creat_tensor.cpp)
* [Run session](src/session_run.cpp)
* [Load graph](src/load_graph.cpp)
* [Interface](src/interface.cpp)
* [Tensor Info](src/tensor_info.cpp)
* [Graph Info](src/graph_info.cpp)

## Build example

### Windows

```text
mkdir build
cd build
cmake .. -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Linux and macOS(darwin)

```text
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Remarks

* The tensorflow in the repository is compiled in x64 mode. Make sure that project target 64-bit platforms.
* Make sure that the tensorflow lib is in Output Directory or either in the directory contained by the %PATH% environment variable.

## Get tensorflow lib

For x64 CPU, you can download the tensorflow.so, tensorflow.dll and tensorflow.lib from <https://github.com/Neargye/tensorflow/releases>.

Or build lib which version you need from the sources, with CPU or GPU support.

### Create .lib file from .dll for windows

Open the Visual Studio Command Prompt, you find its shortcut in "Start"->"Programs"->"Microsoft Visual Studio"->"Tools". Now run the dumpbin command to get a list of all exported functions of your dll:

```text
dumpbin /exports yourpath/tensorflow.dll
```

This will print quite a bit of text to the console. However we are only interested in the functions:

```text
    ordinal hint RVA      name

          1    0 028D4AB8 ?DEVICE_CPU@tensorflow@@3QEBDEB
          2    1 028D4AC0 ?DEVICE_GPU@tensorflow@@3QEBDEB
          3    2 028D4AC8 ?DEVICE_SYCL@tensorflow@@3QEBDEB
          4    3 028E1380 ?kDatasetGraphKey@GraphDatasetBase@tensorflow@@2QBDB
          5    4 028E1390 ?kDatasetGraphOutputNodeKey@GraphDatasetBase@tensorflow@@2QBDB
          6    5 03242488 ?tracing_engine_@Tracing@port@tensorflow@@0U?$atomic@PEAVEngine@Tracing@port@tensorflow@@@std@@A
          7    6 001996C0 TFE_ContextAddFunction
          8    7 00199710 TFE_ContextAddFunctionDef
          9    8 001997D0 TFE_ContextAsyncClearError
         10    9 001997E0 TFE_ContextAsyncWait
         11    A 00199830 TFE_ContextClearCaches
...
```

Now copy all those function names (only the names!) and paste them into a new textfile. Name the nextfile tensorflow.def and put the line “EXPORTS” at its top. My tensorflow.def file looks like this:

```test
EXPORTS
?DEVICE_CPU@tensorflow@@3QEBDEB
?DEVICE_GPU@tensorflow@@3QEBDEB
?DEVICE_SYCL@tensorflow@@3QEBDEB
?kDatasetGraphKey@GraphDatasetBase@tensorflow@@2QBDB
?kDatasetGraphOutputNodeKey@GraphDatasetBase@tensorflow@@2QBDB
?tracing_engine_@Tracing@port@tensorflow@@0U?$atomic@PEAVEngine@Tracing@port@tensorflow@@@std@@A
TFE_ContextAddFunction
TFE_ContextAddFunctionDef
TFE_ContextAsyncClearError
TFE_ContextAsyncWait
TFE_ContextClearCaches
...
```

Now from that definition file, we can finally create the .lib file. We use the “lib” tool for this, so run this command in your Visual Studio Command Prompt:

```text
lib /def:yourpath/tensorflow.def /OUT:yourpath/tensorflow.lib /MACHINE:X64
```

/MACHINE:X64 - fow x64 build, and /MACHINE:X86 for x32 build.

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

## Licensed under the [MIT License](LICENSE)
