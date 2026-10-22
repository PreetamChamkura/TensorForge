"""
Milestone 5: Fused custom Metal Add+ReLU kernel.

Fuses two operations into one kernel pass:
    out[i] = max(a[i] + b[i], 0.0)

This eliminates a round-trip through memory between the Add output and
the ReLU input, which is the primary benefit of kernel fusion for
elementwise chains.

Limitation: float32 only, same shape only.
"""

import mlx.core as mx


class MetalAddReluKernel:
    """Fused Metal add+relu: out[i] = max(a[i] + b[i], 0.0). Same shape only."""

    def __init__(self):
        self._kernel = self._build()

    def _build(self):
        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = fmax(a[elem] + b[elem], 0.0f);
        """
        return mx.fast.metal_kernel(
            name="tensorforge_add_relu_fused",
            input_names=["a", "b"],
            output_names=["out"],
            source=source,
        )

    def __call__(self, a: mx.array, b: mx.array) -> mx.array:
        """Fused add+relu. Both inputs must have the same shape."""
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(
                f"metal_add_relu requires same shape: {tuple(a.shape)} vs {tuple(b.shape)}"
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
