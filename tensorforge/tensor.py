"""
Milestone 2: Tensor wrapper around MLX arrays.

Adds shape/dtype metadata, educational memory accounting, operation
recording, and creation helpers on top of MLX's lazy-evaluated arrays.
"""

from typing import Optional, Tuple
import mlx.core as mx


# Bytes per element for each supported dtype
_DTYPE_BYTES = {
    mx.float32: 4,
    mx.float16: 2,
    mx.bfloat16: 2,
    mx.int32: 4,
    mx.int16: 2,
    mx.int8: 1,
    mx.uint32: 4,
    mx.uint16: 2,
    mx.uint8: 1,
    mx.bool_: 1,
}


class Tensor:
    """Educational tensor wrapping an MLX array.

    MLX arrays are lazy-evaluated. Values are not computed until mx.eval()
    is called or Python forces a read (e.g. printing). This wrapper preserves
    that laziness while adding metadata useful for understanding runtime
    behavior.
    """

    def __init__(self, data: mx.array, op_record: Optional[str] = None):
        if not isinstance(data, mx.array):
            raise TypeError(
                f"Tensor requires an mx.array, got {type(data).__name__}. "
                "Use Tensor.from_array() to convert Python lists or numpy arrays."
            )
        self._data = data
        self.op_record = op_record or "created"

    # --- core properties ---

    @property
    def data(self) -> mx.array:
        """The underlying MLX array (may be unevaluated)."""
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self._data.shape)

    @property
    def dtype(self) -> mx.Dtype:
        return self._data.dtype

    @property
    def size(self) -> int:
        """Total number of elements."""
        n = 1
        for d in self._data.shape:
            n *= d
        return n

    @property
    def estimated_bytes(self) -> int:
        """Estimated memory footprint in bytes based on dtype and element count."""
        return self.size * _DTYPE_BYTES.get(self._data.dtype, 4)

    # --- evaluation ---

    def eval(self) -> "Tensor":
        """Force evaluation of the underlying MLX computation graph."""
        mx.eval(self._data)
        return self

    def to_numpy(self):
        """Evaluate and return a numpy array."""
        import numpy as np
        mx.eval(self._data)
        return np.array(self._data)

    # --- creation helpers ---

    @staticmethod
    def zeros(shape, dtype=mx.float32) -> "Tensor":
        return Tensor(mx.zeros(shape, dtype=dtype), op_record="zeros")

    @staticmethod
    def ones(shape, dtype=mx.float32) -> "Tensor":
        return Tensor(mx.ones(shape, dtype=dtype), op_record="ones")

    @staticmethod
    def full(shape, fill_value, dtype=mx.float32) -> "Tensor":
        return Tensor(mx.full(shape, fill_value, dtype=dtype), op_record="full")

    @staticmethod
    def from_array(data, dtype=None) -> "Tensor":
        """Create from a Python list or numpy array."""
        arr = mx.array(data, dtype=dtype) if dtype is not None else mx.array(data)
        return Tensor(arr, op_record="from_array")

    @staticmethod
    def random_uniform(shape, low: float = 0.0, high: float = 1.0,
                       dtype=mx.float32) -> "Tensor":
        arr = mx.random.uniform(low=low, high=high, shape=shape, dtype=dtype)
        return Tensor(arr, op_record="random_uniform")

    @staticmethod
    def random_normal(shape, dtype=mx.float32) -> "Tensor":
        arr = mx.random.normal(shape=shape, dtype=dtype)
        return Tensor(arr, op_record="random_normal")

    # --- display ---

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"~{self.estimated_bytes}B, op={self.op_record!r})"
        )
