"""
Milestone 3: Computation Graph and Intermediate Representation (IR).

The graph captures the structure of a computation before execution, enabling
analysis, optimization, and scheduling. This is the same pattern used by
ML frameworks like JAX (jaxpr), TensorFlow (tf.Graph), and MLIR.

Key types:
  Value           — a named node representing a tensor in the graph
  GraphInput      — a placeholder for a runtime-supplied tensor
  GraphConstant   — a node with embedded constant data (mx.array)
  GraphOperation  — a single op: N inputs → 1 output Value
  ComputationGraph — holds all nodes; validates, sorts, and executes them
"""

import itertools
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
import mlx.core as mx


# Module-level counter ensures globally unique Value IDs within a process.
_value_id_counter = itertools.count(1)


def _next_id() -> int:
    return next(_value_id_counter)


# ------------------------------------------------------------------ #
#  Node types                                                          #
# ------------------------------------------------------------------ #

class Value:
    """A named node in the computation graph.

    Shape and dtype are optional metadata, attached by the shape-propagation
    pass or by the caller when the information is statically known.
    """

    def __init__(self, name: str, shape=None, dtype=None):
        self.id = _next_id()
        self.name = name
        self.shape = shape    # tuple of ints, or None
        self.dtype = dtype    # mx.Dtype, or None

    def __repr__(self) -> str:
        s = str(self.shape) if self.shape is not None else "?"
        return f"Value(id={self.id}, name={self.name!r}, shape={s})"


class GraphInput(Value):
    """Placeholder for a tensor supplied at execution time via feed_dict."""

    def __init__(self, name: str, shape=None, dtype=None):
        super().__init__(name, shape, dtype)


class GraphConstant(Value):
    """A constant tensor embedded in the graph."""

    def __init__(self, name: str, data: mx.array):
        super().__init__(name, shape=tuple(data.shape), dtype=data.dtype)
        self.data = data


class GraphOperation:
    """A single operation: maps a list of input Values to one output Value."""

    def __init__(
        self,
        op_type: str,
        input_values: List[Value],
        output: Value,
        **attrs,
    ):
        self.op_type = op_type
        self.input_values = input_values  # mutable — passes may remap entries
        self.output = output
        self.attrs = attrs  # op-specific attributes (e.g. reshape target shape)

    def __repr__(self) -> str:
        in_ids = [v.id for v in self.input_values]
        return f"GraphOperation({self.op_type!r}, inputs={in_ids} -> {self.output.id})"


# ------------------------------------------------------------------ #
#  Computation graph                                                   #
# ------------------------------------------------------------------ #

class ComputationGraph:
    """Captures a computation as a DAG of Values and GraphOperations.

    Workflow:
        g = ComputationGraph()
        x = g.add_input("x", shape=(4, 4), dtype=mx.float32)
        w = g.add_constant("w", mx.ones((4, 4)))
        mm = g.op("matmul", [x, w])
        g.set_outputs([mm])
        result = g.execute({"x": some_array})
    """

    def __init__(self):
        self._inputs: Dict[str, GraphInput] = {}
        self._constants: Dict[str, GraphConstant] = {}
        self._operations: List[GraphOperation] = []
        self._values: Dict[int, Value] = {}
        self._output_ids: List[int] = []
        self._exec_trace: List[dict] = []

    # ------------------------------------------------------------------ #
    #  Graph construction                                                  #
    # ------------------------------------------------------------------ #

    def add_input(self, name: str, shape=None, dtype=None) -> GraphInput:
        v = GraphInput(name, shape, dtype)
        self._inputs[name] = v
        self._values[v.id] = v
        return v

    def add_constant(self, name: str, data: mx.array) -> GraphConstant:
        v = GraphConstant(name, data)
        self._constants[name] = v
        self._values[v.id] = v
        return v

    def op(
        self,
        op_type: str,
        inputs: List[Value],
        name: Optional[str] = None,
        **attrs,
    ) -> Value:
        """Append an operation and return its output Value."""
        if name is None:
            name = f"{op_type}_out_{len(self._operations)}"
        output = Value(name)
        operation = GraphOperation(op_type, list(inputs), output, **attrs)
        self._operations.append(operation)
        self._values[output.id] = output
        return output

    def set_outputs(self, values: List[Value]) -> None:
        """Declare which Values are the graph outputs."""
        self._output_ids = [v.id for v in values]

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        defined = set(self._values.keys())

        for operation in self._operations:
            for inp in operation.input_values:
                if inp.id not in defined:
                    raise ValueError(
                        f"Operation {operation.op_type!r} references "
                        f"undefined value id={inp.id} (name={inp.name!r})"
                    )

        for oid in self._output_ids:
            if oid not in defined:
                raise ValueError(f"Declared output id={oid} is not a defined Value")

        if not self._output_ids:
            raise ValueError("Graph has no declared outputs — call set_outputs()")

    # ------------------------------------------------------------------ #
    #  Cycle detection (DFS, three-color marking)                         #
    # ------------------------------------------------------------------ #

    def _detect_cycles(self) -> None:
        produced_by: Dict[int, GraphOperation] = {}
        for op in self._operations:
            produced_by[op.output.id] = op

        # 0=white (unvisited), 1=gray (in stack), 2=black (done)
        color: Dict[int, int] = {}

        def dfs(vid: int) -> None:
            c = color.get(vid, 0)
            if c == 2:
                return
            if c == 1:
                raise ValueError(
                    f"Cycle detected: value id={vid} "
                    f"(name={self._values[vid].name!r}) is on the DFS stack"
                )
            color[vid] = 1
            if vid in produced_by:
                for inp in produced_by[vid].input_values:
                    dfs(inp.id)
            color[vid] = 2

        for vid in list(self._values.keys()):
            if color.get(vid, 0) == 0:
                dfs(vid)

    # ------------------------------------------------------------------ #
    #  Topological sort (Kahn's algorithm — deterministic BFS order)      #
    # ------------------------------------------------------------------ #

    def _topological_sort(self) -> List[GraphOperation]:
        produced_ids = {op.output.id for op in self._operations}

        # For each op, count how many of its inputs are produced by other ops.
        in_degree: Dict[int, int] = {}
        for op in self._operations:
            in_degree[id(op)] = sum(
                1 for inp in op.input_values if inp.id in produced_ids
            )

        # Build: value_id -> ops that consume it
        consumers: Dict[int, List[GraphOperation]] = defaultdict(list)
        for op in self._operations:
            for inp in op.input_values:
                consumers[inp.id].append(op)

        queue = deque(op for op in self._operations if in_degree[id(op)] == 0)
        result: List[GraphOperation] = []

        while queue:
            op = queue.popleft()
            result.append(op)
            for consumer in consumers.get(op.output.id, []):
                in_degree[id(consumer)] -= 1
                if in_degree[id(consumer)] == 0:
                    queue.append(consumer)

        if len(result) != len(self._operations):
            raise ValueError(
                "Topological sort incomplete — possible cycle or disconnected node"
            )
        return result

    # ------------------------------------------------------------------ #
    #  Execution                                                           #
    # ------------------------------------------------------------------ #

    def execute(
        self,
        feed_dict: Dict[str, Any],
        runtime=None,
        backend: str = "mlx",
    ) -> Dict[str, "Tensor"]:
        """Execute the graph and return a dict mapping output name -> Tensor.

        Args:
            feed_dict: Maps input names to mx.array or Tensor values.
            runtime:   A Runtime instance (created fresh if None).
            backend:   'mlx' or 'metal'. Affects how ops are dispatched for
                       operations that have custom Metal paths.
        """
        from tensorforge.runtime import Runtime
        from tensorforge.tensor import Tensor

        if runtime is None:
            runtime = Runtime()

        self._validate()
        self._detect_cycles()
        topo = self._topological_sort()

        value_table: Dict[int, Tensor] = {}

        # Bind inputs
        for name, inp_val in self._inputs.items():
            if name not in feed_dict:
                raise ValueError(
                    f"Missing input: {name!r}. "
                    f"feed_dict has: {list(feed_dict.keys())}"
                )
            raw = feed_dict[name]
            if isinstance(raw, Tensor):
                t = raw
            elif isinstance(raw, mx.array):
                t = Tensor(raw)
            else:
                t = Tensor(mx.array(raw))

            if inp_val.shape is not None and tuple(t.shape) != tuple(inp_val.shape):
                raise ValueError(
                    f"Input {name!r}: expected shape {inp_val.shape}, got {t.shape}"
                )
            if inp_val.dtype is not None and t.dtype != inp_val.dtype:
                raise ValueError(
                    f"Input {name!r}: expected dtype {inp_val.dtype}, got {t.dtype}"
                )
            value_table[inp_val.id] = t

        # Bind constants
        for const_val in self._constants.values():
            value_table[const_val.id] = Tensor(
                const_val.data, op_record=f"constant:{const_val.name}"
            )

        self._exec_trace = []

        # Execute in topological order
        for op in topo:
            inputs = [value_table[v.id] for v in op.input_values]
            result = self._dispatch(op, inputs, runtime, backend)
            value_table[op.output.id] = result
            self._exec_trace.append({
                "op": op.op_type,
                "output_id": op.output.id,
                "output_name": op.output.name,
                "output_shape": result.shape,
            })

        return {
            self._values[oid].name: value_table[oid]
            for oid in self._output_ids
        }

    def _dispatch(
        self,
        op: GraphOperation,
        inputs: List["Tensor"],
        runtime: Any,
        backend: str,
    ) -> "Tensor":
        t = op.op_type
        if t == "add":
            if backend == "metal":
                return runtime.metal_add(inputs[0], inputs[1])
            return runtime.add(inputs[0], inputs[1])
        elif t == "multiply":
            return runtime.multiply(inputs[0], inputs[1])
        elif t == "relu":
            if backend == "metal":
                return runtime.metal_relu(inputs[0])
            return runtime.relu(inputs[0])
        elif t == "matmul":
            return runtime.matmul(inputs[0], inputs[1])
        elif t == "reshape":
            return runtime.reshape(inputs[0], op.attrs["shape"])
        elif t == "transpose":
            return runtime.transpose(inputs[0])
        elif t == "add_relu_fused":
            return runtime.metal_add_relu(inputs[0], inputs[1])
        else:
            raise ValueError(f"Unknown op type: {t!r}")

    # ------------------------------------------------------------------ #
    #  Inspection                                                          #
    # ------------------------------------------------------------------ #

    @property
    def operations(self) -> List[GraphOperation]:
        return list(self._operations)

    @property
    def exec_trace(self) -> List[dict]:
        return list(self._exec_trace)

    def summary(self) -> str:
        lines = [
            "ComputationGraph",
            f"  inputs    : {list(self._inputs.keys())}",
            f"  constants : {list(self._constants.keys())}",
            f"  ops       : {len(self._operations)}",
        ]
        for op in self._operations:
            in_names = ", ".join(v.name for v in op.input_values)
            shape_str = str(op.output.shape) if op.output.shape is not None else "?"
            dtype_str = str(op.output.dtype) if op.output.dtype is not None else "?"
            lines.append(
                f"    {op.op_type}({in_names}) -> {op.output.name} "
                f"shape={shape_str} dtype={dtype_str}"
            )
        out_names = [self._values[oid].name for oid in self._output_ids]
        lines.append(f"  outputs   : {out_names}")
        return "\n".join(lines)

    def dump_ir(self) -> str:
        lines = ["=== TensorForge IR ==="]
        lines.append("INPUTS:")
        for v in self._inputs.values():
            lines.append(
                f"  %{v.id:<4} = input({v.name!r}, shape={v.shape}, dtype={v.dtype})"
            )
        if self._constants:
            lines.append("CONSTANTS:")
            for v in self._constants.values():
                lines.append(
                    f"  %{v.id:<4} = const({v.name!r}, shape={v.shape}, dtype={v.dtype})"
                )
        lines.append("OPS:")
        for op in self._operations:
            in_refs = ", ".join(f"%{v.id}" for v in op.input_values)
            shape_str = str(op.output.shape) if op.output.shape is not None else "?"
            lines.append(
                f"  %{op.output.id:<4} = {op.op_type}({in_refs})"
                f"  # {op.output.name} shape={shape_str}"
            )
        lines.append("OUTPUTS:")
        for oid in self._output_ids:
            v = self._values[oid]
            lines.append(f"  %{v.id:<4}  ({v.name})")
        return "\n".join(lines)
