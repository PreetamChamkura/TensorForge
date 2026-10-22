"""Tests for Runtime dispatch (Milestone 2)."""

import pytest
import numpy as np
import mlx.core as mx

from tensorforge.tensor import Tensor
from tensorforge.runtime import Runtime


def make(data, dtype=mx.float32) -> Tensor:
    return Tensor(mx.array(data, dtype=dtype))


# ------------------------------------------------------------------ #
#  MLX paths — basic correctness                                       #
# ------------------------------------------------------------------ #

def test_add_basic():
    rt = Runtime()
    a = make([[1.0, 2.0], [3.0, 4.0]])
    b = make([[10.0, 20.0], [30.0, 40.0]])
    c = rt.add(a, b)
    mx.eval(c.data)
    assert np.allclose(np.array(c.data), [[11, 22], [33, 44]])

def test_add_broadcast():
    rt = Runtime()
    a = make([[1.0], [2.0]])  # (2,1)
    b = make([10.0, 20.0])    # (2,)  broadcasts to (2,2)
    c = rt.add(a, b)
    mx.eval(c.data)
    assert c.shape == (2, 2)

def test_multiply_basic():
    rt = Runtime()
    a = make([2.0, 3.0, 4.0])
    b = make([1.0, 2.0, 3.0])
    c = rt.multiply(a, b)
    mx.eval(c.data)
    assert np.allclose(np.array(c.data), [2.0, 6.0, 12.0])

def test_relu_zeros_negative():
    rt = Runtime()
    a = make([-2.0, -1.0, 0.0, 1.0, 2.0])
    r = rt.relu(a)
    mx.eval(r.data)
    assert np.allclose(np.array(r.data), [0.0, 0.0, 0.0, 1.0, 2.0])

def test_relu_all_positive():
    rt = Runtime()
    a = make([1.0, 2.0, 3.0])
    r = rt.relu(a)
    mx.eval(r.data)
    assert np.allclose(np.array(r.data), [1.0, 2.0, 3.0])

def test_matmul_basic():
    rt = Runtime()
    a = Tensor(mx.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(mx.array([[1.0, 0.0], [0.0, 1.0]]))  # identity
    c = rt.matmul(a, b)
    mx.eval(c.data)
    assert np.allclose(np.array(c.data), [[1.0, 2.0], [3.0, 4.0]])

def test_matmul_rectangular():
    rt = Runtime()
    a = Tensor(mx.array([[1.0, 2.0, 3.0]]))      # (1, 3)
    b = Tensor(mx.array([[1.0], [1.0], [1.0]]))  # (3, 1)
    c = rt.matmul(a, b)
    mx.eval(c.data)
    assert c.shape == (1, 1)
    assert np.allclose(np.array(c.data), [[6.0]])

def test_matmul_wrong_rank_raises():
    rt = Runtime()
    a = Tensor(mx.array([1.0, 2.0]))  # rank 1
    b = Tensor(mx.array([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="rank"):
        rt.matmul(a, b)

def test_matmul_inner_dim_mismatch():
    rt = Runtime()
    a = Tensor(mx.array([[1.0, 2.0]]))     # (1, 2)
    b = Tensor(mx.array([[1.0, 2.0]]))     # (1, 2) — inner dims don't match
    with pytest.raises(ValueError):
        rt.matmul(a, b)

def test_reshape():
    rt = Runtime()
    a = make([1.0, 2.0, 3.0, 4.0])
    b = rt.reshape(a, (2, 2))
    assert b.shape == (2, 2)

def test_transpose():
    rt = Runtime()
    a = make([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b = rt.transpose(a)
    assert b.shape == (3, 2)

def test_eval_forces_computation():
    rt = Runtime()
    a = Tensor.random_normal((10, 10))
    b = Tensor.random_normal((10, 10))
    c = rt.add(a, b)
    rt.eval(c)  # should not raise

# ------------------------------------------------------------------ #
#  Trace                                                               #
# ------------------------------------------------------------------ #

def test_trace_records_op():
    rt = Runtime()
    a = make([1.0, 2.0])
    b = make([3.0, 4.0])
    rt.add(a, b)
    assert len(rt.trace) == 1
    assert rt.trace[0]["op"] == "add"

def test_trace_backend_mlx():
    rt = Runtime()
    a = make([1.0])
    rt.relu(a)
    assert rt.trace[-1]["backend"] == "mlx"

def test_trace_accumulates():
    rt = Runtime()
    a = make([1.0, 2.0])
    b = make([0.5, 0.5])
    rt.add(a, b)
    rt.relu(a)
    rt.multiply(a, b)
    assert len(rt.trace) == 3

def test_trace_clear():
    rt = Runtime()
    a = make([1.0])
    rt.relu(a)
    rt.clear_trace()
    assert rt.trace == []

def test_trace_shapes():
    rt = Runtime()
    a = make([[1.0, 2.0], [3.0, 4.0]])
    b = make([[0.0, 0.0], [0.0, 0.0]])
    rt.add(a, b)
    entry = rt.trace[0]
    assert entry["input_shapes"] == [[2, 2], [2, 2]]
    assert entry["output_shape"] == [2, 2]
