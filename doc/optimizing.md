# Runtime and performance

## Reuse resources

Load the graph and create the session once. Keep operation handles and reuse input tensors when their shapes stay the same; update their contents with `SetTensorData`.

Keep input tensors alive during `RunSession`. The caller owns returned output tensors: delete each one after use and reset its slot to `nullptr` before the next call. `DeleteTensors` deletes the tensors but does not clear the pointers.

The [repeated_inference](../src/repeated_inference.cpp) example demonstrates reuse. It also runs a fresh reference session on each iteration to check the result, so its total runtime is not an inference benchmark.

## Match the model's input

Check element type, shape, and data layout. For images, use the same resize, channel order, and normalization as during training. See [image_example](../src/image_example.cpp) for tensor construction and [opencv_image_file_example](../src/opencv_image_file_example.cpp) for loading an image file.

Use `CreateStringTensor` for `TF_STRING`; do not copy raw character bytes into string tensor storage.

## Measure

- Use a Release build on the target hardware.
- Warm up the session before timing repeated calls.
- Measure preprocessing separately, and include it when reporting request latency.
- Compare batch sizes and thread counts using representative inputs. Larger batches and more threads are not always faster.

`CreateSessionOptions(intra_threads, inter_threads)` controls TensorFlow thread counts. Pass the options when creating the session.
