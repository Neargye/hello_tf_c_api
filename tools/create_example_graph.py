#!/usr/bin/env python3
"""Generate the small GraphDef used by the C API examples.

This script is intentionally separate from the CMake build. The examples use
the committed models/graph.pb file so builds do not need a full TensorFlow
Python runtime, only the TensorFlow wheel files used for C API headers/libs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def remove_repo_root_from_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    clean_path = []
    for entry in sys.path:
        resolved = Path(entry or ".").resolve()
        if resolved != repo_root:
            clean_path.append(entry)
    sys.path[:] = clean_path


def generate_graph(output: Path) -> None:
    remove_repo_root_from_import_path()

    try:
        import tensorflow as tf
    except ImportError as error:
        raise SystemExit(
            "TensorFlow Python package is required to regenerate the example "
            "graph. Install it in a virtual environment and rerun this script."
        ) from error

    if not hasattr(tf, "compat") or not hasattr(tf.compat, "v1"):
        raise SystemExit(
            "A full TensorFlow Python package is required to regenerate the "
            "example graph. The imported 'tensorflow' module does not expose "
            "tf.compat.v1."
        )

    tf.compat.v1.disable_eager_execution()

    graph = tf.Graph()
    with graph.as_default():
        input_tensor = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 5, 12], name="input_4"
        )
        mean = tf.reduce_mean(input_tensor, axis=[1, 2], name="mean")
        tf.stack([mean, mean * 2.0, mean * 3.0, mean * 4.0], axis=1, name="output_node0")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(graph.as_graph_def().SerializeToString())
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the small GraphDef used by hello_tf_c_api examples."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "graph.pb",
        help="Output GraphDef path. Defaults to models/graph.pb.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_graph(args.output)


if __name__ == "__main__":
    main()
