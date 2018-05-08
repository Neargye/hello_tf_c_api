# Hello TensorFlow C API

A small example of how to run TensorFlow C API on Windows.

## Link tensorflow.dll

For CPU, you can download the .dll from:
<https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-*.*.*.zip>, where *.*.* - version of TensorFlow which you need.

Or build dll whichversion  you need from the sources.

For example:

I used v1.8.0, downloaded dll from here <https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.8.0.zip>

Create .lib file from .dll

Open the Visual Studio Command Prompt, you find its shortcut in "Start"->"Programs"->"Microsoft Visual Studio"->"Tools". Now run the dumpbin command to get a list of all exported functions of your dll:

```bash
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

```bash
lib /def:yourpath/tensorflow.def /OUT:yourpath/tensorflow.lib
```

Now you can link the library.

## Licensed under the [Unlicense](LICENSE)
