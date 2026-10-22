"""
Milestone 5: Add → ReLU fusion pass.

Detects Add immediately followed by ReLU and rewrites to a single
add_relu_fused operation backed by the custom Metal fused kernel.

Conditions for fusion:
  1. The Add output has exactly one consumer (the ReLU op).
  2. The Add output is not a declared graph output.
  3. Both Add inputs have known shape and dtype.
  4. Both Add inputs are the same shape (fused Metal kernel constraint).
  5. Both Add inputs are float32 (fused Metal kernel constraint).

If any condition is not met the pair is left unchanged.

Returns the number of Add+ReLU pairs fused.
"""

import mlx.core as mx
from collections import defaultdict
from tensorforge.graph import ComputationGraph


def fuse_add_relu(graph: ComputationGraph) -> int:
    """Rewrite eligible Add→ReLU pairs to add_relu_fused in place.

    Returns:
        Number of pairs fused.
    """
    graph_output_ids = set(graph._output_ids)

    # Build: value_id -> list of ops that consume it
    consumers: dict = defaultdict(list)
    for op in graph._operations:
        for inp in op.input_values:
            consumers[inp.id].append(op)

    ops_to_remove = []
    fused = 0

    for op in list(graph._operations):
        if op.op_type != "add":
            continue

        add_out_id = op.output.id

        # Condition 2: not a graph output
        if add_out_id in graph_output_ids:
            continue

        # Condition 1: exactly one consumer, and it must be ReLU
        consuming = consumers.get(add_out_id, [])
        if len(consuming) != 1:
            continue
        relu_op = consuming[0]
        if relu_op.op_type != "relu":
            continue

        # Conditions 3–5: known shape, same shape, float32
        a, b = op.input_values[0], op.input_values[1]
        if a.shape is None or b.shape is None:
            continue
        if a.shape != b.shape:
            continue
        if a.dtype is None or b.dtype is None:
            continue
        if a.dtype != mx.float32 or b.dtype != mx.float32:
            continue

        # Rewrite: turn the Add into add_relu_fused, redirect its output to
        # the ReLU's output (so downstream consumers see the right value).
        op.op_type = "add_relu_fused"
        op.output = relu_op.output

        ops_to_remove.append(relu_op)
        fused += 1

    for op in ops_to_remove:
        if op in graph._operations:
            graph._operations.remove(op)

    return fused
