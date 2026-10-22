"""
Custom Metal matrix multiplication: naive and tiled.

Both kernels compute C = A @ B for rank-2 float32 inputs, with A (M, K),
B (K, N), C (M, N). Row-major contiguous layout is assumed (MLX arrays are).

Naive: one thread per output element, each reads a full row of A and column
of B straight from device memory (K loads from each per output element).

Tiled: one TILE x TILE threadgroup per output tile. Each iteration the group
cooperatively stages a TILE x TILE block of A and of B in threadgroup
(on-chip) memory, so every loaded value is reused TILE times. Edges are
zero-padded, so any M, K, N works.
"""

import mlx.core as mx

TILE = 16


def _check(a: mx.array, b: mx.array, name: str) -> None:
    if a.dtype != mx.float32 or b.dtype != mx.float32:
        raise TypeError(f"{name} requires float32, got {a.dtype}, {b.dtype}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"{name} requires rank-2 inputs, got {a.ndim} and {b.ndim}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"{name} inner dimensions must match: {tuple(a.shape)} @ {tuple(b.shape)}"
        )


class MetalMatmulNaiveKernel:
    """C[m, n] = sum_k A[m, k] * B[k, n], one thread per output element."""

    def __init__(self):
        source = """
            uint col = thread_position_in_grid.x;
            uint row = thread_position_in_grid.y;
            uint M = a_shape[0];
            uint K = a_shape[1];
            uint N = b_shape[1];
            if (row >= M || col >= N) return;
            float acc = 0.0f;
            for (uint k = 0; k < K; k++) {
                acc += a[row * K + k] * b[k * N + col];
            }
            out[row * N + col] = acc;
        """
        self._kernel = mx.fast.metal_kernel(
            name="tensorforge_matmul_naive",
            input_names=["a", "b"],
            output_names=["out"],
            source=source,
        )

    def __call__(self, a: mx.array, b: mx.array) -> mx.array:
        _check(a, b, "metal_matmul_naive")
        M, N = a.shape[0], b.shape[1]
        if M == 0 or N == 0 or a.shape[1] == 0:
            return mx.zeros((M, N), dtype=mx.float32)
        return self._kernel(
            inputs=[a, b],
            grid=(N, M, 1),
            threadgroup=(TILE, TILE, 1),
            output_shapes=[(M, N)],
            output_dtypes=[mx.float32],
        )[0]


class MetalMatmulTiledKernel:
    """Shared-memory tiled matmul, TILE x TILE threadgroups."""

    def __init__(self):
        source = f"""
            const uint TILE = {TILE};
            threadgroup float As[{TILE}][{TILE}];
            threadgroup float Bs[{TILE}][{TILE}];

            uint M = a_shape[0];
            uint K = a_shape[1];
            uint N = b_shape[1];
            uint tx = thread_position_in_threadgroup.x;
            uint ty = thread_position_in_threadgroup.y;
            uint col = threadgroup_position_in_grid.x * TILE + tx;
            uint row = threadgroup_position_in_grid.y * TILE + ty;

            float acc = 0.0f;
            for (uint t = 0; t < K; t += TILE) {{
                uint ak = t + tx;
                uint bk = t + ty;
                As[ty][tx] = (row < M && ak < K) ? a[row * K + ak] : 0.0f;
                Bs[ty][tx] = (bk < K && col < N) ? b[bk * N + col] : 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 0; k < TILE; k++) {{
                    acc += As[ty][k] * Bs[k][tx];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            if (row < M && col < N) {{
                out[row * N + col] = acc;
            }}
        """
        self._kernel = mx.fast.metal_kernel(
            name="tensorforge_matmul_tiled",
            input_names=["a", "b"],
            output_names=["out"],
            source=source,
        )

    def __call__(self, a: mx.array, b: mx.array) -> mx.array:
        _check(a, b, "metal_matmul_tiled")
        M, N = a.shape[0], b.shape[1]
        if M == 0 or N == 0 or a.shape[1] == 0:
            return mx.zeros((M, N), dtype=mx.float32)
        # Grid is in threads and must be a whole number of tiles so that every
        # thread reaches the barriers; the kernel masks out-of-range rows/cols.
        gx = ((N + TILE - 1) // TILE) * TILE
        gy = ((M + TILE - 1) // TILE) * TILE
        return self._kernel(
            inputs=[a, b],
            grid=(gx, gy, 1),
            threadgroup=(TILE, TILE, 1),
            output_shapes=[(M, N)],
            output_dtypes=[mx.float32],
        )[0]
