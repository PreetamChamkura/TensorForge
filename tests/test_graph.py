"""Tests for ComputationGraph and IR (Milestone 3)."""

import pytest
import numpy as np
import mlx.core as mx

from tensorforge.graph import (
    ComputationGraph, Value, GraphInput, GraphConstant, GraphOperation
)
from tensorforge.tensor import Tensor
from tensorforge.runtime import Runtime


def f32(data) -> mx.array:
    return mx.array(data, dtype=mx.float32)


# ------------------------------------------------------------------ #
#  Construction                                                        #
# ------------------------------------------------------------------ #

def test_add_input_returns_graph_input():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 2), dtype=mx.float32)
    assert isinstance(x, GraphInput)
    assert x.name == "x"
    assert x.shape == (2, 2)

def test_add_constant_returns_graph_constant():
    g = ComputationGraph()
    data = f32([[1.0, 0.0], [0.0, 1.0]])
    c = g.add_constant("eye", data)
    assert isinstance(c, GraphConstant)
    assert c.shape == (2, 2)

def test_op_returns_value():
    g = ComputationGraph()
    x = g.add_input("x")
    y = g.add_input("y")
    out = g.op("add", [x, y])
    assert isinstance(out, Value)
    assert "add" in out.name

def test_set_outputs_stores_ids():
    g = ComputationGraph()
    x = g.add_input("x")
    out = g.op("relu", [x])
    g.set_outputs([out])
    assert out.id in g._output_ids

# ------------------------------------------------------------------ #
#  Validation                                                          #
# ------------------------------------------------------------------ #

def test_missing_output_raises():
    g = ComputationGraph()
    g.add_input("x")
    with pytest.raises(ValueError, match="outputs"):
        g.execute({"x": f32([1.0])})

def test_missing_input_raises():
    g = ComputationGraph()
    x = g.add_input("x")
    out = g.op("relu", [x])
    g.set_outputs([out])
    with pytest.raises(ValueError, match="Missing input"):
        g.execute({})  # "x" not provided

def test_input_shape_mismatch_raises():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 2), dtype=mx.float32)
    out = g.op("relu", [x])
    g.set_outputs([out])
    with pytest.raises(ValueError, match="shape"):
        g.execute({"x": f32([[1.0, 2.0, 3.0]])})  # (1,3) not (2,2)

def test_input_dtype_mismatch_raises():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2,), dtype=mx.float32)
    out = g.op("relu", [x])
    g.set_outputs([out])
    with pytest.raises(ValueError, match="dtype"):
        g.execute({"x": mx.array([1, 2], dtype=mx.int32)})

# ------------------------------------------------------------------ #
#  Cycle detection                                                     #
# ------------------------------------------------------------------ #

def test_no_cycle_valid_graph():
    g = ComputationGraph()
    x = g.add_input("x")
    out = g.op("relu", [x])
    g.set_outputs([out])
    # Should not raise
    g._detect_cycles()

def test_cycle_raises():
    g = ComputationGraph()
    x = g.add_input("x")
    # Manually create a self-referential op (artificial cycle)
    v_a = g.op("relu", [x])
    v_b = g.op("relu", [v_a])
    # Force a cycle by making v_a's op depend on v_b
    g._operations[0].input_values.append(v_b)
    with pytest.raises(ValueError, match="[Cc]ycle"):
        g._detect_cycles()

# ------------------------------------------------------------------ #
#  Execution                                                           #
# ------------------------------------------------------------------ #

def test_execute_relu():
    g = ComputationGraph()
    x = g.add_input("x")
    out = g.op("relu", [x], name="result")
    g.set_outputs([out])
    results = g.execute({"x": f32([-1.0, 0.0, 2.0])})
    mx.eval(results["result"].data)
    assert np.allclose(np.array(results["result"].data), [0.0, 0.0, 2.0])

def test_execute_add():
    g = ComputationGraph()
    x = g.add_input("x")
    y = g.add_input("y")
    out = g.op("add", [x, y], name="sum")
    g.set_outputs([out])
    results = g.execute({"x": f32([1.0, 2.0]), "y": f32([3.0, 4.0])})
    mx.eval(results["sum"].data)
    assert np.allclose(np.array(results["sum"].data), [4.0, 6.0])

def test_execute_matmul_add_relu():
    """Representative M→M→A→R graph from Milestone 3."""
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 3), dtype=mx.float32)
    w = g.add_constant("w", f32([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))  # (3,2)
    b = g.add_constant("b", f32([[-5.0, -5.0]]))  # (1,2)

    mm = g.op("matmul", [x, w])
    add_out = g.op("add", [mm, b])
    relu_out = g.op("relu", [add_out], name="output")
    g.set_outputs([relu_out])

    x_data = f32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    results = g.execute({"x": x_data})
    mx.eval(results["output"].data)

    # x @ w = [[1,2], [4,5]], add b: [[-4,-3], [-1,0]], relu: [[0,0],[0,0]] — wait let me recalc
    # x @ w: row0 = [1*1+2*0+3*0, 1*0+2*1+3*0] = [1, 2]
    #        row1 = [4*1+5*0+6*0, 4*0+5*1+6*0] = [4, 5]
    # add b: [[1-5, 2-5], [4-5, 5-5]] = [[-4,-3],[-1,0]]
    # relu:  [[0, 0], [0, 0]]
    expected = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(np.array(results["output"].data), expected)

def test_execute_multiple_outputs():
    g = ComputationGraph()
    x = g.add_input("x")
    r = g.op("relu", [x], name="relu_out")
    d = g.op("add", [x, x], name="doubled")
    g.set_outputs([r, d])
    results = g.execute({"x": f32([-1.0, 2.0])})
    mx.eval(results["relu_out"].data, results["doubled"].data)
    assert np.allclose(np.array(results["relu_out"].data), [0.0, 2.0])
    assert np.allclose(np.array(results["doubled"].data), [-2.0, 4.0])

def test_execute_with_constant():
    g = ComputationGraph()
    x = g.add_input("x")
    bias = g.add_constant("bias", f32([1.0, 2.0, 3.0]))
    out = g.op("add", [x, bias], name="result")
    g.set_outputs([out])
    results = g.execute({"x": f32([0.0, 0.0, 0.0])})
    mx.eval(results["result"].data)
    assert np.allclose(np.array(results["result"].data), [1.0, 2.0, 3.0])

def test_exec_trace_populated():
    g = ComputationGraph()
    x = g.add_input("x")
    y = g.op("relu", [x])
    g.set_outputs([y])
    g.execute({"x": f32([1.0])})
    assert len(g.exec_trace) == 1
    assert g.exec_trace[0]["op"] == "relu"

# ------------------------------------------------------------------ #
#  Summary and IR dump                                                 #
# ------------------------------------------------------------------ #

def test_summary_contains_op_names():
    g = ComputationGraph()
    x = g.add_input("x")
    g.op("relu", [x])
    g.op("add", [x, x])
    s = g.summary()
    assert "relu" in s
    assert "add" in s

def test_dump_ir_contains_inputs_and_ops():
    g = ComputationGraph()
    x = g.add_input("x")
    out = g.op("relu", [x], name="out")
    g.set_outputs([out])
    ir = g.dump_ir()
    assert "INPUTS" in ir
    assert "OPS" in ir
    assert "relu" in ir
