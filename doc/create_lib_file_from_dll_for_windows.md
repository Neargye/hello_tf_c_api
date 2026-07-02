# Create a .lib file from a .dll on Windows

This project normally does not need this step. CMake finds the TensorFlow import
library from the TensorFlow Python wheel and copies the required runtime DLLs to
the target output directory.

Use this document only when you have a standalone TensorFlow DLL but no matching
import library.

Open the Visual Studio Developer Command Prompt and list exported functions:

```text
dumpbin /exports path\to\tensorflow.dll
```

Copy only the exported function names into a definition file. The file must
start with `EXPORTS`:

```text
EXPORTS
TF_Version
TF_NewStatus
TF_DeleteStatus
TF_NewGraph
TF_DeleteGraph
...
```

Create the import library with the Visual Studio `lib` tool:

```text
lib /def:path\to\tensorflow.def /OUT:path\to\tensorflow.lib /MACHINE:X64
```

Use the generated `.lib` during linking and keep the matching `.dll` in the
executable output directory or in a directory listed in `%PATH%`.
