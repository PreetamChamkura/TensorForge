"""
Milestone 2 & 4: Tensor Runtime.

Dispatches tensor operations to either built-in MLX paths or explicit custom
Metal paths. Each call is recorded in a trace for inspection and debugging.

The custom Metal paths are opt-in. The default paths (add, multiply, relu,
matmul) all use built-in MLX operations and are not routed through custom
Metal kernels unless the caller explicitly requests metal_relu / metal_add /
metal_add_relu.
"""

from typing import List, Optional
import mlx.core as mx

from tensorforge.tensor import Tensor


class Runtime:
    """Dispatches tensor operations and records an execution trace."""

    def __init__(self):
        self._trace: List[dict] = []
        # Kernel objects are built lazily on first use.
        self._metal_relu_kernel = None
        self._metal_add_kernel = None
        self._metal_add_relu_kernel = None
        self._metal_matmul_kernels = {}

    # ------------------------------------------------------------------ #
    #  Built-in MLX paths                                                  #
    # ------------------------------------------------------------------ #

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        """Elementwise add with broadcasting via MLX."""
        result = mx.add(a.data, b.data)
        self._log("add", [a, b], result, backend="mlx")
        return Tensor(result, op_record="add")

    def multiply(self, a: Tensor, b: Tensor) -> Tensor:
        """Elementwise multiply with broadcasting via MLX."""
        result = mx.multiply(a.data, b.data)
        self._log("multiply", [a, b], result, backend="mlx")
        return Tensor(result, op_record="multiply")

    def relu(self, a: Tensor) -> Tensor:
        """Elementwise ReLU via MLX: max(a, 0)."""
        result = mx.maximum(a.data, mx.zeros_like(a.data))
        self._log("relu", [a], result, backend="mlx")
        return Tensor(result, op_record="relu")

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Rank-2 matrix multiplication via MLX.

        Validates rank-2 and inner-dimension compatibility before dispatch.
        """
        if a.rank != 2 or b.rank != 2:
            raise ValueError(
                f"matmul requires rank-2 tensors, got rank {a.rank} and {b.rank}"
            )
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"matmul inner dimensions must match: "
                f"{a.shape} @ {b.shape} — {a.shape[1]} != {b.shape[0]}"
            )
        result = mx.matmul(a.data, b.data)
        self._log("matmul", [a, b], result, backend="mlx")
        return Tensor(result, op_record="matmul")

    def reshape(self, a: Tensor, shape) -> Tensor:
        result = mx.reshape(a.data, shape)
        self._log("reshape", [a], result, backend="mlx")
        return Tensor(result, op_record="reshape")

    def transpose(self, a: Tensor) -> Tensor:
        result = mx.transpose(a.data)
        self._log("transpose", [a], result, backend="mlx")
        return Tensor(result, op_record="transpose")

    def eval(self, *tensors: Tensor) -> None:
        """Force evaluation of one or more tensors (flushes MLX lazy graph)."""
        mx.eval(*[t.data for t in tensors])

    # ------------------------------------------------------------------ #
    #  Explicit custom Metal paths                                         #
    # ------------------------------------------------------------------ #

    def metal_relu(self, a: Tensor) -> Tensor:
        """ReLU via a custom Metal kernel. float32 only."""
        if a.dtype != mx.float32:
            raise TypeError(
                f"metal_relu requires float32, got {a.dtype}"
            )
        kernel = self._get_metal_relu_kernel()
        result = kernel(a.data)
        self._log("relu", [a], result, backend="metal")
        return Tensor(result, op_record="metal_relu")

    def metal_add(self, a: Tensor, b: Tensor) -> Tensor:
        """Same-shape elementwise add via a custom Metal kernel. float32 only."""
        if a.dtype != mx.float32 or b.dtype != mx.float32:
            raise TypeError(
                f"metal_add requires float32, got {a.dtype}, {b.dtype}"
            )
        if a.shape != b.shape:
            raise ValueError(
                f"metal_add requires same shape: {a.shape} vs {b.shape}"
            )
        kernel = self._get_metal_add_kernel()
        result = kernel(a.data, b.data)
        self._log("add", [a, b], result, backend="metal")
        return Tensor(result, op_record="metal_add")

    def metal_add_relu(self, a: Tensor, b: Tensor) -> Tensor:
        """Fused add+relu via a custom Metal kernel. float32, same shape only."""
        if a.dtype != mx.float32 or b.dtype != mx.float32:
            raise TypeError(
                f"metal_add_relu requires float32, got {a.dtype}, {b.dtype}"
            )
        if a.shape != b.shape:
            raise ValueError(
                f"metal_add_relu requires same shape: {a.shape} vs {b.shape}"
            )
        kernel = self._get_metal_add_relu_kernel()
        result = kernel(a.data, b.data)
        self._log("add_relu_fused", [a, b], result, backend="metal")
        return Tensor(result, op_record="metal_add_relu_fused")

    def metal_matmul(self, a: Tensor, b: Tensor, variant: str = "tiled") -> Tensor:
        """Rank-2 matmul via a custom Metal kernel. float32 only.

        variant: "tiled" (threadgroup-memory tiling) or "naive".
        """
        if variant not in ("tiled", "naive"):
            raise ValueError(f"variant must be 'tiled' or 'naive', got {variant!r}")
        if a.dtype != mx.float32 or b.dtype != mx.float32:
            raise TypeError(
                f"metal_matmul requires float32, got {a.dtype}, {b.dtype}"
            )
        if a.rank != 2 or b.rank != 2:
            raise ValueError(
                f"matmul requires rank-2 tensors, got rank {a.rank} and {b.rank}"
            )
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"matmul inner dimensions must match: "
                f"{a.shape} @ {b.shape} — {a.shape[1]} != {b.shape[0]}"
            )
        kernel = self._get_metal_matmul_kernel(variant)
        result = kernel(a.data, b.data)
        self._log("matmul", [a, b], result, backend="metal")
        return Tensor(result, op_record=f"metal_matmul_{variant}")

    # ------------------------------------------------------------------ #
    #  Trace                                                               #
    # ------------------------------------------------------------------ #

    @property
    def trace(self) -> List[dict]:
        """Read-only copy of the execution trace."""
        return list(self._trace)

    def clear_trace(self) -> None:
        self._trace = []

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _log(self, op: str, inputs: List[Tensor], result: mx.array,
             backend: str) -> None:
        self._trace.append({
            "op": op,
            "backend": backend,
            "input_shapes": [list(t.shape) for t in inputs],
            "output_shape": list(result.shape),
            "dtype": str(inputs[0].dtype),
        })

    def _get_metal_relu_kernel(self):
        if self._metal_relu_kernel is None:
            from tensorforge.kernels.metal_relu import MetalReluKernel
            self._metal_relu_kernel = MetalReluKernel()
        return self._metal_relu_kernel

    def _get_metal_add_kernel(self):
        if self._metal_add_kernel is None:
            from tensorforge.kernels.metal_add import MetalAddKernel
            self._metal_add_kernel = MetalAddKernel()
        return self._metal_add_kernel

    def _get_metal_add_relu_kernel(self):
        if self._metal_add_relu_kernel is None:
            from tensorforge.kernels.metal_add_relu import MetalAddReluKernel
            self._metal_add_relu_kernel = MetalAddReluKernel()
        return self._metal_add_relu_kernel

    def _get_metal_matmul_kernel(self, variant: str):
        if variant not in self._metal_matmul_kernels:
            from tensorforge.kernels.metal_matmul import (
                MetalMatmulNaiveKernel, MetalMatmulTiledKernel,
            )
            cls = MetalMatmulTiledKernel if variant == "tiled" else MetalMatmulNaiveKernel
            self._metal_matmul_kernels[variant] = cls()
        return self._metal_matmul_kernels[variant]
