"""Tests for custom Metal kernels (Milestones 4 and 5)."""

import pytest
import numpy as np
import mlx.core as mx

from tensorforge.tensor import Tensor
from tensorforge.runtime import Runtime
from tensorforge.kernels.metal_relu import MetalReluKernel
from tensorforge.kernels.metal_add import MetalAddKernel
from tensorforge.kernels.metal_add_relu import MetalAddReluKernel


ATOL = 1e-5  # tolerance for float32 comparisons


def f32(data) -> mx.array:
    return mx.array(data, dtype=mx.float32)


def tensor_f32(data) -> Tensor:
    return Tensor(f32(data))


# ------------------------------------------------------------------ #
#  MetalReluKernel                                                     #
# ------------------------------------------------------------------ #

def test_metal_relu_correctness_1d():
    kernel = MetalReluKernel()
    a = f32([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = kernel(a)
    mx.eval(result)
    assert np.allclose(np.array(result), [0.0, 0.0, 0.0, 1.0, 2.0], atol=ATOL)

def test_metal_relu_correctness_2d():
    kernel = MetalReluKernel()
    a = f32([[-3.0, 2.0], [0.0, -1.0]])
    result = kernel(a)
    mx.eval(result)
    expected = np.array([[0.0, 2.0], [0.0, 0.0]])
    assert result.shape == a.shape
    assert np.allclose(np.array(result), expected, atol=ATOL)

def test_metal_relu_parity_with_mlx():
    """Custom Metal ReLU must match mx.maximum(a, 0)."""
    kernel = MetalReluKernel()
    a = f32([-1.0, 0.5, -0.3, 2.1, -0.0])
    metal_out = kernel(a)
    mlx_out = mx.maximum(a, mx.zeros_like(a))
    mx.eval(metal_out, mlx_out)
    assert np.allclose(np.array(metal_out), np.array(mlx_out), atol=ATOL)

def test_runtime_metal_relu_dtype_error():
    rt = Runtime()
    a = Tensor(mx.array([1.0, 2.0], dtype=mx.float16))
    with pytest.raises(TypeError, match="float32"):
        rt.metal_relu(a)

def test_runtime_metal_relu_backend_label():
    rt = Runtime()
    a = tensor_f32([1.0, -1.0])
    rt.metal_relu(a)
    assert rt.trace[-1]["backend"] == "metal"


# ------------------------------------------------------------------ #
#  MetalAddKernel                                                      #
# ------------------------------------------------------------------ #

def test_metal_add_correctness():
    kernel = MetalAddKernel()
    a = f32([1.0, 2.0, 3.0])
    b = f32([10.0, 20.0, 30.0])
    result = kernel(a, b)
    mx.eval(result)
    assert np.allclose(np.array(result), [11.0, 22.0, 33.0], atol=ATOL)

def test_metal_add_2d():
    kernel = MetalAddKernel()
    a = f32([[1.0, 2.0], [3.0, 4.0]])
    b = f32([[0.1, 0.2], [0.3, 0.4]])
    result = kernel(a, b)
    mx.eval(result)
    expected = np.array([[1.1, 2.2], [3.3, 4.4]])
    assert np.allclose(np.array(result), expected, atol=ATOL)

def test_metal_add_shape_mismatch_raises():
    rt = Runtime()
    a = tensor_f32([1.0, 2.0])
    b = tensor_f32([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same shape"):
        rt.metal_add(a, b)

def test_metal_add_parity_with_mlx():
    kernel = MetalAddKernel()
    a = f32([1.5, -2.3, 0.0, 4.4])
    b = f32([-0.5, 1.3, 2.0, -4.4])
    metal_out = kernel(a, b)
    mlx_out = mx.add(a, b)
    mx.eval(metal_out, mlx_out)
    assert np.allclose(np.array(metal_out), np.array(mlx_out), atol=ATOL)


# ------------------------------------------------------------------ #
#  MetalAddReluKernel (fused)                                         #
# ------------------------------------------------------------------ #

def test_metal_add_relu_correctness():
    kernel = MetalAddReluKernel()
    a = f32([-3.0, 1.0, -1.0, 2.0])
    b = f32([4.0, -2.0, 1.0, -3.0])
    result = kernel(a, b)
    mx.eval(result)
    # a+b = [1, -1, 0, -1], relu = [1, 0, 0, 0]
    assert np.allclose(np.array(result), [1.0, 0.0, 0.0, 0.0], atol=ATOL)

def test_metal_add_relu_parity_with_unfused():
    """Fused result must match separate add then relu."""
    kernel_fused = MetalAddReluKernel()
    a = f32([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
    b = f32([0.5, -3.0, 4.0, -5.0, 6.0, -7.0])
    fused_out = kernel_fused(a, b)
    unfused_out = mx.maximum(mx.add(a, b), mx.zeros_like(a))
    mx.eval(fused_out, unfused_out)
    assert np.allclose(np.array(fused_out), np.array(unfused_out), atol=ATOL)

def test_runtime_metal_add_relu_backend_label():
    rt = Runtime()
    a = tensor_f32([1.0, -1.0])
    b = tensor_f32([-0.5, 2.0])
    rt.metal_add_relu(a, b)
    assert rt.trace[-1]["backend"] == "metal"
    assert rt.trace[-1]["op"] == "add_relu_fused"


# ------------------------------------------------------------------ #
#  Metal matmul (naive + tiled)                                        #
# ------------------------------------------------------------------ #

from tensorforge.kernels.metal_matmul import (
    MetalMatmulNaiveKernel, MetalMatmulTiledKernel,
)

MATMUL_SHAPES = [(1, 1, 1), (5, 7, 3), (16, 16, 16), (33, 50, 17), (128, 64, 96)]


@pytest.mark.parametrize("kernel_cls", [MetalMatmulNaiveKernel, MetalMatmulTiledKernel])
@pytest.mark.parametrize("M,K,N", MATMUL_SHAPES)
def test_metal_matmul_parity_with_mlx(kernel_cls, M, K, N):
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))
    out = kernel_cls()(a, b)
    ref = mx.matmul(a, b)
    mx.eval(out, ref)
    assert out.shape == (M, N)
    assert np.allclose(np.array(out), np.array(ref), atol=1e-3, rtol=1e-4)

def test_runtime_metal_matmul_backend_and_variants():
    rt = Runtime()
    a = Tensor(mx.random.normal((20, 30)))
    b = Tensor(mx.random.normal((30, 10)))
    for variant in ("tiled", "naive"):
        out = rt.metal_matmul(a, b, variant=variant)
        out.eval()
        assert np.allclose(np.array(out.data), np.array(mx.matmul(a.data, b.data)), atol=1e-3)
    assert rt.trace[-1]["backend"] == "metal"
    assert rt.trace[-1]["op"] == "matmul"

def test_runtime_metal_matmul_errors():
    rt = Runtime()
    a = Tensor(mx.ones((2, 3)))
    with pytest.raises(ValueError, match="inner dimensions"):
        rt.metal_matmul(a, Tensor(mx.ones((4, 2))))
    with pytest.raises(ValueError, match="rank-2"):
        rt.metal_matmul(Tensor(mx.ones((3,))), a)
    with pytest.raises(TypeError, match="float32"):
        rt.metal_matmul(Tensor(mx.ones((2, 3), dtype=mx.float16)), Tensor(mx.ones((3, 2), dtype=mx.float16)))
    with pytest.raises(ValueError, match="variant"):
        rt.metal_matmul(a, Tensor(mx.ones((3, 2))), variant="bogus")
