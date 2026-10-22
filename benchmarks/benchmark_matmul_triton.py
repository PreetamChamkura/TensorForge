"""
Triton matmul vs PyTorch (cuBLAS) on an NVIDIA GPU.

Requires CUDA + triton; will not run on Apple Silicon (use
benchmark_matmul.py there). Compares float32 with TF32 enabled on both sides,
float32 strict, and float16, using triton.testing.do_bench.

Usage:
    python3 benchmarks/benchmark_matmul_triton.py
"""

import importlib.util
import os

import torch
import triton.testing

# Load the kernel file directly so the tensorforge package (which imports MLX,
# unavailable on NVIDIA machines) is never imported.
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "tensorforge", "kernels", "triton_matmul.py")
_spec = importlib.util.spec_from_file_location("triton_matmul", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
triton_matmul = _mod.triton_matmul

SIZES = [256, 512, 1024, 2048, 4096]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: this benchmark needs an NVIDIA GPU")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'N':>5} {'mode':<12} {'method':<8} {'ms':>8} {'TFLOP/s':>8} {'vs torch':>9} {'max err':>9}")
    print("-" * 66)
    major, minor = torch.cuda.get_device_capability(0)
    ampere_plus = major >= 8
    modes = [("fp32 strict", torch.float32, False)]
    if ampere_plus:
        modes.insert(0, ("fp32+tf32", torch.float32, True))
        modes.append(("fp16", torch.float16, True))
    else:
        print(f"Compute capability {major}.{minor} (pre-Ampere): skipping TF32 (no hardware "
              "support) and fp16 (Triton does not use tensor cores here, so numbers are "
              "misleadingly slow).")
    for n in SIZES:
        for mode, dtype, tf32 in modes:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            a = torch.randn((n, n), device="cuda", dtype=dtype)
            b = torch.randn((n, n), device="cuda", dtype=dtype)
            ref = torch.matmul(a.double(), b.double())
            flops = 2 * n ** 3
            t_torch = triton.testing.do_bench(lambda: torch.matmul(a, b))
            t_tri = triton.testing.do_bench(lambda: triton_matmul(a, b, allow_tf32=tf32))
            err_t = (torch.matmul(a, b).double() - ref).abs().max().item()
            err_k = (triton_matmul(a, b, allow_tf32=tf32).double() - ref).abs().max().item()
            for name, ms, err in (("torch", t_torch, err_t), ("triton", t_tri, err_k)):
                print(f"{n:>5} {mode:<12} {name:<8} {ms:>8.3f} {flops / ms / 1e9:>8.2f} "
                      f"{ms / t_torch:>8.2f}x {err:>9.1e}")
        print()
    print("Errors are vs a float64 reference. 'vs torch' < 1.00x means Triton is faster.")


if __name__ == "__main__":
    main()
