"""Tests for Tensor wrapper (Milestone 2)."""

import pytest
import numpy as np
import mlx.core as mx

from tensorforge.tensor import Tensor


# ------------------------------------------------------------------ #
#  Creation                                                            #
# ------------------------------------------------------------------ #

def test_from_array_list():
    t = Tensor.from_array([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert t.dtype == mx.float32

def test_from_array_dtype_override():
    t = Tensor.from_array([1, 2, 3], dtype=mx.float16)
    assert t.dtype == mx.float16

def test_zeros():
    t = Tensor.zeros((3, 4))
    assert t.shape == (3, 4)
    assert t.dtype == mx.float32
    mx.eval(t.data)
    assert np.allclose(np.array(t.data), 0.0)

def test_ones():
    t = Tensor.ones((2, 5))
    mx.eval(t.data)
    assert np.allclose(np.array(t.data), 1.0)

def test_full():
    t = Tensor.full((2, 3), 7.0)
    mx.eval(t.data)
    assert np.allclose(np.array(t.data), 7.0)

def test_random_uniform_shape():
    t = Tensor.random_uniform((4, 4))
    assert t.shape == (4, 4)
    assert t.dtype == mx.float32

def test_random_normal_shape():
    t = Tensor.random_normal((8,))
    assert t.shape == (8,)

def test_wrap_mx_array():
    arr = mx.array([1.0, 2.0, 3.0])
    t = Tensor(arr)
    assert t.shape == (3,)

def test_wrong_type_raises():
    with pytest.raises(TypeError):
        Tensor([1.0, 2.0])  # plain list, not mx.array

# ------------------------------------------------------------------ #
#  Metadata properties                                                 #
# ------------------------------------------------------------------ #

def test_rank_2d():
    t = Tensor.zeros((4, 4))
    assert t.rank == 2

def test_rank_1d():
    t = Tensor.zeros((10,))
    assert t.rank == 1

def test_rank_3d():
    t = Tensor.zeros((2, 3, 4))
    assert t.rank == 3

def test_size():
    t = Tensor.zeros((3, 4))
    assert t.size == 12

def test_estimated_bytes_float32():
    t = Tensor.zeros((4, 4))
    assert t.estimated_bytes == 64  # 16 elements * 4 bytes

def test_estimated_bytes_float16():
    t = Tensor.zeros((4, 4), dtype=mx.float16)
    assert t.estimated_bytes == 32  # 16 elements * 2 bytes

# ------------------------------------------------------------------ #
#  Evaluation and conversion                                           #
# ------------------------------------------------------------------ #

def test_eval_returns_self():
    t = Tensor.ones((2, 2))
    result = t.eval()
    assert result is t

def test_to_numpy():
    t = Tensor.from_array([[1.0, 2.0], [3.0, 4.0]])
    arr = t.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]])

def test_repr_contains_shape():
    t = Tensor.zeros((3, 4))
    r = repr(t)
    assert "(3, 4)" in r

def test_op_record_stored():
    t = Tensor.zeros((2, 2))
    assert t.op_record == "zeros"

def test_default_op_record():
    t = Tensor(mx.array([1.0]))
    assert t.op_record == "created"
