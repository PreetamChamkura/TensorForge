"""
Milestone 4: Reusable custom Metal ReLU kernel.

The kernel is built once and reused across calls. MLX compiles the Metal
source on first use and caches the compiled kernel internally.

Limitation: float32 only. Inputs are flattened for a 1-D dispatch then
reshaped to the original shape on output.
"""

import mlx.core as mx


class MetalReluKernel:
    """Custom Metal elementwise ReLU: out[i] = max(inp[i], 0.0)."""

    def __init__(self):
        self._kernel = self._build()

    def _build(self):
        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = fmax(inp[elem], 0.0f);
        """
        return mx.fast.metal_kernel(
            name="tensorforge_relu",
            input_names=["inp"],
            output_names=["out"],
            source=source,
        )

    def __call__(self, a: mx.array) -> mx.array:
        """Apply ReLU. Returns an array with the same shape as `a`."""
        flat = mx.flatten(a)
        n = flat.size
        if n == 0:
            return a
        threadgroup = min(n, 256)
        results = self._kernel(
            inputs=[flat],
            grid=(n, 1, 1),
            threadgroup=(threadgroup, 1, 1),
            output_shapes=[flat.shape],
            output_dtypes=[mx.float32],
        )
        return mx.reshape(results[0], a.shape)
