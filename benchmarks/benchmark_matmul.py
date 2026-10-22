"""
Matmul benchmark: custom Metal kernels vs PyTorch (MPS) vs MLX built-in.

Square float32 matmuls. Each timing calls the op `ITERS` times with a sync
(mx.eval / torch.mps.synchronize) after every call, after `WARMUP` untimed
calls, and reports the median. Correctness is checked against PyTorch.

Usage:
    python3 benchmarks/benchmark_matmul.py
"""

import time
import statistics
import numpy as np
import mlx.core as mx
import torch

from tensorforge.kernels.metal_matmul import (
    MetalMatmulNaiveKernel, MetalMatmulTiledKernel,
)

SIZES = [256, 512, 1024, 2048]
WARMUP = 5
ITERS = 20


def median_ms(fn, sync) -> float:
    for _ in range(WARMUP):
        fn()
    sync()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS backend unavailable")
    naive, tiled = MetalMatmulNaiveKernel(), MetalMatmulTiledKernel()
    print(f"{'N':>5} | {'method':<14} {'ms':>9} {'GFLOP/s':>9} {'vs torch':>9} {'max err':>9}")
    print("-" * 62)
    for n in SIZES:
        a_np = np.random.randn(n, n).astype(np.float32)
        b_np = np.random.randn(n, n).astype(np.float32)
        ta, tb = torch.from_numpy(a_np).to("mps"), torch.from_numpy(b_np).to("mps")
        ma, mb = mx.array(a_np), mx.array(b_np)
        mx.eval(ma, mb)
        ref = torch.matmul(ta, tb).cpu().numpy()

        cases = {
            "torch (mps)": (lambda: torch.matmul(ta, tb), torch.mps.synchronize, None),
            "mlx builtin": (lambda: mx.eval(mx.matmul(ma, mb)), lambda: None, lambda: mx.matmul(ma, mb)),
            "metal naive": (lambda: mx.eval(naive(ma, mb)), lambda: None, lambda: naive(ma, mb)),
            "metal tiled": (lambda: mx.eval(tiled(ma, mb)), lambda: None, lambda: tiled(ma, mb)),
        }
        torch_ms = None
        flops = 2 * n ** 3
        for name, (fn, sync, out_fn) in cases.items():
            ms = median_ms(fn, sync)
            if torch_ms is None:
                torch_ms = ms
            if out_fn is None:
                err = 0.0
            else:
                out = out_fn()
                mx.eval(out)
                err = float(np.max(np.abs(np.array(out) - ref)))
            print(f"{n:>5} | {name:<14} {ms:>9.3f} {flops / ms / 1e6:>9.1f} "
                  f"{ms / torch_ms:>8.2f}x {err:>9.1e}")
        print()
    print("'vs torch' = time relative to PyTorch MPS (lower is faster, 1.00x = parity).")


if __name__ == "__main__":
    main()
