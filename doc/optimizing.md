# Optimizing models and examples

This repository demonstrates TensorFlow C API usage on desktop Windows, Linux, and macOS. It links the native libraries from the TensorFlow Python wheel, so the main optimization target here is the model and runtime usage, not rebuilding a custom mobile TensorFlow runtime.

## Model format

For the examples in this repository, `models/graph.pb` is a small GraphDef loaded with `TF_GraphImportGraphDef`.

For production models:

- Prefer `SavedModel` when you want a standard TensorFlow 2 export format with signatures and assets.
- Use a raw GraphDef when you need a compact single-file inference graph and can control the input/output operation names.
- Use TensorFlow Lite for mobile and edge deployments where binary size, startup time, and model quantization are primary requirements.

## Runtime usage

Keep TensorFlow objects alive and reuse them:

- Load the graph once.
- Create the session once.
- Reuse input/output operation handles.
- Batch requests when latency requirements allow it.
- Avoid repeated tensor allocation in hot paths when tensor shapes are stable.

The examples keep each program small, so they create and destroy resources in `main`. A long-running application should move graph/session setup into its initialization path.

`TF_SessionRun` owns neither input tensors nor output tensors forever. The caller must keep input tensors alive for the call and must delete every output tensor returned by TensorFlow with `TF_DeleteTensor`. In a loop, delete output tensors on every iteration. The `repeated_inference` example shows this pattern while reusing the graph, session, operation handles, and input tensor.

## Tensor shape and data layout

Most runtime issues come from mismatched tensor shape, type, or layout. Keep these details close to the call site:

- `TF_DataType`
- dimensions
- element count
- byte size
- channel order for image tensors
- string tensor encoding rules for `TF_STRING`

The helper functions in `tf_utils.hpp` are intentionally strict about element counts and byte sizes so mistakes fail early.

## Image preprocessing

If preprocessing is done in C++, keep it explicit and deterministic:

- Decode image files outside TensorFlow unless your model intentionally contains decode operations.
- Resize to the model's expected width and height.
- Convert channel order if needed.
- Normalize values the same way as during training.

The `image_example` target shows tensor construction without external image dependencies. The optional `opencv_image_file_example` target shows file-based image preprocessing when OpenCV is available.

## Measuring performance

Measure the workload you actually ship:

- Include preprocessing time when it is part of request latency.
- Warm up the session before measuring steady-state inference.
- Test representative batch sizes.
- Test Release builds.
- Measure on the same OS and CPU architecture as the deployment target.

For lower-level TensorFlow benchmarking, use tools from the TensorFlow source tree or TensorFlow Lite tooling that matches your deployment format.

## References

- TensorFlow C API install notes: https://www.tensorflow.org/install/lang_c
- TensorFlow SavedModel guide: https://www.tensorflow.org/guide/saved_model
- TensorFlow Lite model optimization: https://www.tensorflow.org/lite/performance/model_optimization
