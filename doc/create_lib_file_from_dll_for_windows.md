# Create a Windows import library

The normal CMake build handles import libraries and runtime DLLs. Follow these steps only when linking a standalone TensorFlow DLL without a matching `.lib`.

In the Visual Studio Developer Command Prompt, list the DLL exports:

```bat
dumpbin /exports path\to\tensorflow.dll
```

Create `tensorflow.def` with `EXPORTS` on the first line, followed by the exported function names. Omit the ordinal, hint, and address columns. For example:

```text
EXPORTS
TF_Version
TF_NewStatus
TF_DeleteStatus
```

Include all exports your program needs; the example above shows only three.

For an x64 DLL, generate the import library:

```bat
lib /def:path\to\tensorflow.def /OUT:path\to\tensorflow.lib /MACHINE:X64
```

Link the `.lib` and keep the matching DLL and its dependencies beside the executable or on `%PATH%`.
