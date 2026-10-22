"""
Milestone 5 benchmark: elementwise kernel and fusion comparison.

Compares:
  1. Direct MLX  (mx.maximum, mx.add — no graph overhead)
  2. Runtime MLX  (through TensorForge Runtime)
  3. Separate custom Metal Add + ReLU
  4. Fused custom Metal AddReLU

Measures cold and warm passes separately. Cold pass includes MLX kernel
compilation; warm passes measure steady-state dispatch latency.

Usage:
    python3 benchmarks/benchmark_m5.py
"""

import time
import numpy as np
import mlx.core as mx

from tensorforge.tensor import Tensor
from tensorforge.runtime import Runtime

SHAPE = (1024, 1024)
WARM_ITERS = 20
N = SHAPE[0] * SHAPE[1]


def timed(fn, label: str, warmup: bool = False) -> float:
    mx.eval(fn())  # run once to trigger compilation
    if not warmup:
        return 0.0
    t0 = time.perf_counter()
    for _ in range(WARM_ITERS):
        r = fn()
        mx.eval(r)
    t1 = time.perf_counter()
    return (t1 - t0) / WARM_ITERS * 1000  # ms per iteration


def main():
    print(f"=== Milestone 5 Benchmark ===")
    print(f"Shape: {SHAPE}  ({N:,} elements)  dtype: float32\n")

    a_arr = mx.random.uniform(low=-1.0, high=1.0, shape=SHAPE, dtype=mx.float32)
    b_arr = mx.random.uniform(low=-1.0, high=1.0, shape=SHAPE, dtype=mx.float32)
    mx.eval(a_arr, b_arr)

    a_t = Tensor(a_arr)
    b_t = Tensor(b_arr)

    rt = Runtime()
    # Warm up kernel objects
    rt.metal_add(a_t, b_t).eval()
    rt.metal_relu(a_t).eval()
    rt.metal_add_relu(a_t, b_t).eval()

    results = {}

    # 1. Direct MLX
    def direct_mlx():
        return mx.maximum(mx.add(a_arr, b_arr), mx.zeros_like(a_arr))
    results["direct_mlx"] = timed(direct_mlx, "direct MLX", warmup=True)

    # 2. Runtime MLX
    def runtime_mlx():
        return rt.add(a_t, b_t).data
    rt2 = Runtime()
    results["runtime_mlx"] = timed(
        lambda: mx.maximum(rt2.add(a_t, b_t).data, mx.zeros_like(a_arr)),
        "runtime MLX", warmup=True
    )

    # 3. Separate Metal Add + ReLU
    def separate_metal():
        added = rt.metal_add(a_t, b_t)
        return rt.metal_relu(added).data
    results["metal_add_relu_sep"] = timed(separate_metal, "metal sep", warmup=True)

    # 4. Fused Metal AddReLU
    def fused_metal():
        return rt.metal_add_relu(a_t, b_t).data
    results["metal_add_relu_fused"] = timed(fused_metal, "metal fused", warmup=True)

    # --- Print results ---
    print(f"{'Method':<30}  {'Warm ms/iter':>12}")
    print("-" * 46)
    for name, ms in results.items():
        print(f"{name:<30}  {ms:>12.4f}")

    print()
    print("Notes:")
    print("  - Cold-pass timing (includes kernel compilation) is not shown here.")
    print("  - MLX kernels are highly optimized; custom Metal paths are educational.")
    print("  - 'warm' excludes first-run compilation overhead.")

    # --- Correctness check ---
    ref = direct_mlx()
    fused = fused_metal()
    mx.eval(ref, fused)
    diff = float(mx.max(mx.abs(ref - fused)).item())
    print(f"\nMax absolute difference (direct MLX vs fused Metal): {diff:.2e}")
    print("Correctness: PASS" if diff < 1e-4 else "Correctness: FAIL")


if __name__ == "__main__":
    main()
