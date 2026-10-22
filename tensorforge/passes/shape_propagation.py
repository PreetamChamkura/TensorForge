"""
Milestone 5: Shape propagation pass.

Walks the graph in topological order and infers the output shape and dtype
for each operation, attaching the results to each operation's output Value.

Supported operations:
  add, multiply   — broadcast shape inference (numpy rules)
  relu            — output shape == input shape
  matmul          — rank-2 only: (M, K) @ (K, N) -> (M, N)
  add_relu_fused  — same as add (broadcast)
  reshape         — shape from op attrs
  transpose       — rank-2 only: (M, N) -> (N, M)

If any input shape is unknown, the output shape is left as None.
"""

from typing import Dict, Optional, Tuple
from tensorforge.graph import ComputationGraph


def _broadcast_shape(
    shape_a: Optional[Tuple], shape_b: Optional[Tuple]
) -> Optional[Tuple]:
    """Compute numpy-style broadcast output shape, or None if unknown/incompatible."""
    if shape_a is None or shape_b is None:
        return None
    a = list(reversed(shape_a))
    b = list(reversed(shape_b))
    result = []
    for i in range(max(len(a), len(b))):
        da = a[i] if i < len(a) else 1
        db = b[i] if i < len(b) else 1
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            return None  # incompatible dimensions
    return tuple(reversed(result))


def propagate_shapes(graph: ComputationGraph) -> None:
    """Infer and attach shape/dtype to all operation output Values in-place."""
    shape_cache: Dict[int, Optional[Tuple]] = {}
    dtype_cache: Dict[int, object] = {}

    # Seed with inputs and constants
    for v in graph._inputs.values():
        shape_cache[v.id] = v.shape
        dtype_cache[v.id] = v.dtype
    for v in graph._constants.values():
        shape_cache[v.id] = v.shape
        dtype_cache[v.id] = v.dtype

    for op in graph._topological_sort():
        in_shapes = [shape_cache.get(v.id) for v in op.input_values]
        in_dtypes = [dtype_cache.get(v.id) for v in op.input_values]
        out_dtype = in_dtypes[0] if in_dtypes else None

        t = op.op_type
        if t in ("add", "multiply"):
            if len(in_shapes) == 2:
                out_shape = _broadcast_shape(in_shapes[0], in_shapes[1])
            else:
                out_shape = None
        elif t == "relu":
            out_shape = in_shapes[0] if in_shapes else None
        elif t == "matmul":
            a, b = (in_shapes + [None, None])[:2]
            if (a and b and len(a) == 2 and len(b) == 2 and a[1] == b[0]):
                out_shape = (a[0], b[1])
            else:
                out_shape = None
        elif t == "add_relu_fused":
            if len(in_shapes) == 2:
                out_shape = _broadcast_shape(in_shapes[0], in_shapes[1])
            else:
                out_shape = None
        elif t == "reshape":
            out_shape = tuple(op.attrs.get("shape", ()))
        elif t == "transpose":
            s = in_shapes[0] if in_shapes else None
            out_shape = (s[1], s[0]) if (s and len(s) == 2) else None
        else:
            out_shape = None

        op.output.shape = out_shape
        op.output.dtype = out_dtype
        shape_cache[op.output.id] = out_shape
        dtype_cache[op.output.id] = out_dtype
