"""
Milestone 4: Reusable custom Metal same-shape elementwise Add kernel.

Limitation: float32 only, same shape only (no broadcasting).
Inputs must be the same shape. This constraint is enforced before launch.
"""

import mlx.core as mx


class MetalAddKernel:
    """Custom Metal elementwise add: out[i] = a[i] + b[i]. Same shape only."""

    def __init__(self):
        self._kernel = self._build()

    def _build(self):
        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = a[elem] + b[elem];
        """
        return mx.fast.metal_kernel(
            name="tensorforge_add",
            input_names=["a", "b"],
            output_names=["out"],
            source=source,
        )

    def __call__(self, a: mx.array, b: mx.array) -> mx.array:
        """Elementwise add. Both inputs must have the same shape."""
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(
                f"metal_add requires same shape: {tuple(a.shape)} vs {tuple(b.shape)}"
            )
        flat_a = mx.flatten(a)
        flat_b = mx.flatten(b)
        n = flat_a.size
        if n == 0:
            return a
        threadgroup = min(n, 256)
        results = self._kernel(
            inputs=[flat_a, flat_b],
            grid=(n, 1, 1),
            threadgroup=(threadgroup, 1, 1),
            output_shapes=[flat_a.shape],
            output_dtypes=[mx.float32],
        )
        return mx.reshape(results[0], a.shape)
