# Prepare models

## Repository example graph

The C API examples in this repository use `models/graph.pb`, which is committed to the repository. No external model download is required.

The example graph is intentionally small and has the operation names used by the sample programs:

- input: `input_4`
- output: `output_node0`

To regenerate this demo graph, use a normal Python environment where TensorFlow is available and run:

```text
python tools/create_example_graph.py --output models/graph.pb
```

This script is only for the repository demo graph. The regular CMake build uses the committed `models/graph.pb` and does not require a full Python TensorFlow runtime.

## GraphDef and SavedModel

The examples load a serialized `GraphDef` (`.pb`) with `TF_GraphImportGraphDef` and execute it with `TF_SessionRun`. This keeps the C API examples small and makes the input/output operation names explicit.

Modern TensorFlow training code usually exports a `SavedModel`. For a real project, choose one of these routes:

- Use `TF_LoadSessionFromSavedModel` and adapt the C++ code to the SavedModel tags and signature names.
- Export a small inference-only `GraphDef` when you want to keep using the simple `TF_GraphImportGraphDef` path shown in this repository.

For new application code, prefer a clear SavedModel export unless you have a specific reason to ship a raw GraphDef.

## Input and output names

TensorFlow tools often show tensor names such as `input_4:0` and `output_node0:0`. The C API call `TF_GraphOperationByName` takes the operation name without the output index, so the examples use `input_4` and `output_node0`.

Useful ways to inspect a model:

- Run the `graph_info` and `tensor_info` examples against a GraphDef.
- Inspect the model in Python before export.
- Use TensorBoard for larger graphs.

## Export notes

Keep the inference artifact small and predictable:

- Export only the inference path.
- Avoid training-only operations in the runtime graph.
- Keep preprocessing requirements explicit. If preprocessing is done in C++, feed already-normalized tensors into TensorFlow.
- Keep input shapes and data types documented next to the C++ call site.

## References

- TensorFlow SavedModel guide: https://www.tensorflow.org/guide/saved_model
