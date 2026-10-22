"""
Milestone 5: Constant folding pass.

Evaluates operations whose inputs are all constants at graph-construction
time, replaces the operation's output with a new GraphConstant, and removes
the operation from the graph.

Supported foldable ops: add, multiply, relu.

Operations with non-constant inputs, or ops not in the supported set, are
left unchanged. The pass is safe to run multiple times (idempotent given the
same constant values).

Returns the number of operations folded.
"""

import mlx.core as mx
from tensorforge.graph import ComputationGraph, GraphConstant
from tensorforge.tensor import Tensor


def fold_constants(graph: ComputationGraph, runtime=None) -> int:
    """Evaluate constant-only operations and replace with embedded constants.

    Returns:
        Number of operations folded.
    """
    from tensorforge.runtime import Runtime

    if runtime is None:
        runtime = Runtime()

    topo = graph._topological_sort()

    # Track which value IDs are constants (inputs or previously folded).
    constant_ids: set = set()
    for v in graph._constants.values():
        constant_ids.add(v.id)

    # Map value_id -> Tensor for evaluation
    const_data: dict = {}
    for v in graph._constants.values():
        const_data[v.id] = Tensor(v.data, op_record=f"const:{v.name}")

    ops_to_remove = []
    folded = 0

    for op in topo:
        if op.op_type not in ("add", "multiply", "relu"):
            continue
        if not all(v.id in constant_ids for v in op.input_values):
            continue

        inputs = [const_data[v.id] for v in op.input_values]

        try:
            if op.op_type == "add":
                result_tensor = runtime.add(inputs[0], inputs[1])
            elif op.op_type == "multiply":
                result_tensor = runtime.multiply(inputs[0], inputs[1])
            elif op.op_type == "relu":
                result_tensor = runtime.relu(inputs[0])
            else:
                continue
        except Exception:
            continue

        mx.eval(result_tensor.data)

        # Create a new GraphConstant with the folded value.
        # Preserve the original output's name so output-dict keys stay stable.
        original_name = op.output.name
        folded_name = f"_folded_{original_name}"
        new_const = graph.add_constant(folded_name, result_tensor.data)
        new_const.name = original_name  # output key lookup uses .name, not the _constants key

        old_id = op.output.id
        new_id = new_const.id

        # Remap all consumers of the old output to the new constant.
        for other_op in graph._operations:
            for i, inp in enumerate(other_op.input_values):
                if inp.id == old_id:
                    other_op.input_values[i] = new_const

        # Remap graph outputs.
        for i, oid in enumerate(graph._output_ids):
            if oid == old_id:
                graph._output_ids[i] = new_id

        constant_ids.add(new_id)
        const_data[new_id] = Tensor(new_const.data, op_record=f"folded:{folded_name}")

        ops_to_remove.append(op)
        folded += 1

    for op in ops_to_remove:
        if op in graph._operations:
            graph._operations.remove(op)

    return folded
