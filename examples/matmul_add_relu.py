"""
Milestone 3 example: MatMul → Add → ReLU computation graph.

Demonstrates graph capture, validation, topological execution, and the
full path from user API to MLX GPU execution.

Usage:
    python3 examples/matmul_add_relu.py
"""

import mlx.core as mx
import numpy as np

from tensorforge.graph import ComputationGraph
from tensorforge.tensor import Tensor
from tensorforge.runtime import Runtime


def main():
    print("=== TensorForge Example: MatMul → Add → ReLU ===\n")

    # --- Build the graph ---
    g = ComputationGraph()

    # Input: (4, 8) feature matrix
    x = g.add_input("x", shape=(4, 8), dtype=mx.float32)
    # Weight: (8, 4) constant
    w = g.add_constant("w", mx.random.normal(shape=(8, 4), dtype=mx.float32))
    # Bias: (1, 4) constant — broadcast to (4, 4)
    bias = g.add_constant("bias", mx.zeros((1, 4), dtype=mx.float32))

    mm = g.op("matmul", [x, w])
    added = g.op("add", [mm, bias])
    output = g.op("relu", [added], name="output")
    g.set_outputs([output])

    print(g.summary())
    print()
    print(g.dump_ir())
    print()

    # --- Execute ---
    rt = Runtime()
    x_data = mx.random.normal(shape=(4, 8), dtype=mx.float32)
    results = g.execute({"x": x_data}, runtime=rt)

    mx.eval(results["output"].data)
    out_np = np.array(results["output"].data)

    print(f"Output shape : {results['output'].shape}")
    print(f"Output dtype : {results['output'].dtype}")
    print(f"Output min   : {out_np.min():.4f}  (should be >= 0 after ReLU)")
    print(f"Output max   : {out_np.max():.4f}")
    print()

    # --- Trace ---
    print("Execution trace:")
    for entry in rt.trace:
        print(f"  {entry['op']:12s}  backend={entry['backend']}  "
              f"in={entry['input_shapes']}  out={entry['output_shape']}")


if __name__ == "__main__":
    main()
