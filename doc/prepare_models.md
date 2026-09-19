# Prepare models

## Included graph

The examples load the bundled `models/graph.pb`. CMake copies it to the build directory; no model download is needed.

The inference examples use these operation names:

- `input_4`: float input, shape `[batch, 5, 12]`.
- `output_node0`: float output, shape `[batch, 4]`.

Tensor names include an output index, such as `input_4:0`. Pass only `input_4` to `TF_GraphOperationByName`, then use `TF_Output{operation, 0}`. The [graph_info](../src/graph_info.cpp) example lists operations and shapes.

## Generate a test graph

With TensorFlow installed in your Python environment, run from the repository root:

```sh
python tools/create_example_graph.py --output build/generated-model/graph.pb
```

This creates a replacement graph with the same input/output interface. It returns the input mean multiplied by 1, 2, 3, and 4; it does **not** reproduce the bundled model's values. The command above leaves `models/graph.pb` unchanged.

To try it, run the built example from `build/generated-model`: `../Release/repeated_inference.exe` on Windows or `../repeated_inference` on Linux/macOS.

## Your own model

`LoadGraph` imports a serialized GraphDef. Update the operation names, tensor types, shapes, and preprocessing in the example to match your model.

If the graph needs a checkpoint, create a session and call `RestoreCheckpoint(session, graph, ...)` before inference. Pass the graph's checkpoint-path input and restore-operation names; the short overload assumes `save/Const` and `save/restore_all`. Restored values belong to that session.

A SavedModel directory needs `TF_LoadSessionFromSavedModel`, with the appropriate tags and input/output tensors. The helpers do not wrap this loader yet. See the [TensorFlow SavedModel guide](https://www.tensorflow.org/guide/saved_model).
