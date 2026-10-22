"""Tests for optimization passes (Milestone 5)."""

import pytest
import numpy as np
import mlx.core as mx

from tensorforge.graph import ComputationGraph
from tensorforge.passes.shape_propagation import propagate_shapes
from tensorforge.passes.constant_folding import fold_constants
from tensorforge.passes.fusion import fuse_add_relu
from tensorforge.runtime import Runtime


def f32(data) -> mx.array:
    return mx.array(data, dtype=mx.float32)


# ------------------------------------------------------------------ #
#  Shape propagation                                                   #
# ------------------------------------------------------------------ #

def test_shape_relu_preserves_shape():
    g = ComputationGraph()
    x = g.add_input("x", shape=(4, 4), dtype=mx.float32)
    out = g.op("relu", [x])
    g.set_outputs([out])
    propagate_shapes(g)
    assert out.shape == (4, 4)

def test_shape_add_same_shape():
    g = ComputationGraph()
    x = g.add_input("x", shape=(3, 3), dtype=mx.float32)
    y = g.add_input("y", shape=(3, 3), dtype=mx.float32)
    out = g.op("add", [x, y])
    g.set_outputs([out])
    propagate_shapes(g)
    assert out.shape == (3, 3)

def test_shape_add_broadcast():
    g = ComputationGraph()
    x = g.add_input("x", shape=(4, 1), dtype=mx.float32)
    y = g.add_input("y", shape=(1, 4), dtype=mx.float32)
    out = g.op("add", [x, y])
    g.set_outputs([out])
    propagate_shapes(g)
    assert out.shape == (4, 4)

def test_shape_matmul():
    g = ComputationGraph()
    x = g.add_input("x", shape=(8, 16), dtype=mx.float32)
    w = g.add_input("w", shape=(16, 4), dtype=mx.float32)
    out = g.op("matmul", [x, w])
    g.set_outputs([out])
    propagate_shapes(g)
    assert out.shape == (8, 4)

def test_shape_add_relu_fused():
    g = ComputationGraph()
    x = g.add_input("x", shape=(2, 2), dtype=mx.float32)
    y = g.add_input("y", shape=(2, 2), dtype=mx.float32)
    out = g.op("add_relu_fused", [x, y])
    g.set_outputs([out])
    propagate_shapes(g)
    assert out.shape == (2, 2)

def test_shape_chained_propagation():
    """Shape propagates through a multi-op chain."""
    g = ComputationGraph()
    x = g.add_input("x", shape=(4, 8), dtype=mx.float32)
    w = g.add_input("w", shape=(8, 2), dtype=mx.float32)
    mm = g.op("matmul", [x, w])
    r = g.op("relu", [mm])
    g.set_outputs([r])
    propagate_shapes(g)
    assert mm.shape == (4, 2)
    assert r.shape == (4, 2)

def test_shape_unknown_input_propagates_none():
    """Without shape info on an input, output shape stays None."""
    g = ComputationGraph()
    x = g.add_input("x")  # no shape
    out = g.op("relu", [x])
    g.set_outputs([out])
    propagate_shapes(g)
    assert out.shape is None


# ------------------------------------------------------------------ #
#  Constant folding                                                    #
# ------------------------------------------------------------------ #

def test_constant_folding_add():
    g = ComputationGraph()
    a = g.add_constant("a", f32([1.0, 2.0, 3.0]))
    b = g.add_constant("b", f32([4.0, 5.0, 6.0]))
    out = g.op("add", [a, b], name="result")
    g.set_outputs([out])
    n_folded = fold_constants(g)
    assert n_folded == 1
    assert len(g._operations) == 0  # op was removed

def test_constant_folding_relu():
    g = ComputationGraph()
    a = g.add_constant("a", f32([-1.0, 0.0, 2.0]))
    out = g.op("relu", [a], name="result")
    g.set_outputs([out])
    fold_constants(g)
    assert len(g._operations) == 0

def test_constant_folding_preserves_output_value():
    g = ComputationGraph()
    a = g.add_constant("a", f32([3.0]))
    b = g.add_constant("b", f32([4.0]))
    out = g.op("add", [a, b], name="result")
    g.set_outputs([out])

    rt = Runtime()
    fold_constants(g, runtime=rt)

    results = g.execute({"x": f32([0.0])} if False else {})  # no inputs needed
    mx.eval(results["result"].data)
    assert np.allclose(np.array(results["result"].data), [7.0])

def test_constant_folding_skips_non_constant():
    g = ComputationGraph()
    x = g.add_input("x")
    a = g.add_constant("a", f32([1.0]))
    out = g.op("add", [x, a], name="result")
    g.set_outputs([out])
    n_folded = fold_constants(g)
    assert n_folded == 0
    assert len(g._operations) == 1  # op stays


# ------------------------------------------------------------------ #
#  Fusion                                                              #
# ------------------------------------------------------------------ #

def test_fusion_add_relu():
    """Basic Add → ReLU fusion."""
    g = ComputationGraph()
    x = g.add_input("x", shape=(4,), dtype=mx.float32)
    y = g.add_input("y", shape=(4,), dtype=mx.float32)
    propagate_shapes(g)

    add_out = g.op("add", [x, y])
    relu_out = g.op("relu", [add_out], name="out")
    g.set_outputs([relu_out])
    propagate_shapes(g)

    n_fused = fuse_add_relu(g)
    assert n_fused == 1
    assert len(g._operations) == 1
    assert g._operations[0].op_type == "add_relu_fused"

def test_fusion_skips_when_add_is_output():
    """If Add output is a graph output, do not fuse."""
    g = ComputationGraph()
    x = g.add_input("x", shape=(4,), dtype=mx.float32)
    y = g.add_input("y", shape=(4,), dtype=mx.float32)
    propagate_shapes(g)

    add_out = g.op("add", [x, y], name="add_out")
    relu_out = g.op("relu", [add_out], name="relu_out")
    g.set_outputs([add_out, relu_out])  # add_out is also an output
    propagate_shapes(g)

    n_fused = fuse_add_relu(g)
    assert n_fused == 0  # not fused

def test_fusion_skips_multiple_consumers():
    """Add output used by two ops — don't fuse."""
    g = ComputationGraph()
    x = g.add_input("x", shape=(4,), dtype=mx.float32)
    y = g.add_input("y", shape=(4,), dtype=mx.float32)
    propagate_shapes(g)

    add_out = g.op("add", [x, y])
    relu_out = g.op("relu", [add_out], name="r1")
    also_uses = g.op("relu", [add_out], name="r2")  # second consumer
    g.set_outputs([relu_out, also_uses])
    propagate_shapes(g)

    n_fused = fuse_add_relu(g)
    assert n_fused == 0

def test_fused_graph_produces_correct_output():
    """After fusion, execution via Runtime gives same result as unfused."""
    # Unfused reference
    g_ref = ComputationGraph()
    x_ref = g_ref.add_input("x", shape=(4,), dtype=mx.float32)
    y_ref = g_ref.add_input("y", shape=(4,), dtype=mx.float32)
    add_ref = g_ref.op("add", [x_ref, y_ref])
    relu_ref = g_ref.op("relu", [add_ref], name="out")
    g_ref.set_outputs([relu_ref])
    propagate_shapes(g_ref)

    # Fused graph
    g_fused = ComputationGraph()
    x_f = g_fused.add_input("x", shape=(4,), dtype=mx.float32)
    y_f = g_fused.add_input("y", shape=(4,), dtype=mx.float32)
    add_f = g_fused.op("add", [x_f, y_f])
    relu_f = g_fused.op("relu", [add_f], name="out")
    g_fused.set_outputs([relu_f])
    propagate_shapes(g_fused)
    fuse_add_relu(g_fused)

    x_data = f32([-1.0, 2.0, -3.0, 4.0])
    y_data = f32([0.5, -3.0, 4.0, -5.0])

    ref_result = g_ref.execute({"x": x_data, "y": y_data})
    fused_result = g_fused.execute({"x": x_data, "y": y_data})

    mx.eval(ref_result["out"].data, fused_result["out"].data)
    assert np.allclose(
        np.array(ref_result["out"].data),
        np.array(fused_result["out"].data),
        atol=1e-5,
    )
