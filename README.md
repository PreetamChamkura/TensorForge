# TensorForge

An educational GPU tensor compiler and runtime built on Apple Silicon using Python and MLX.

TensorForge is a personal learning project that builds every layer of a minimal ML runtime from scratch: tensor abstraction, computation graph, IR, graph optimization passes, and custom Metal GPU kernels.

## Goal

Understand how ML frameworks work internally — not by reading their source, but by building equivalent mechanisms at educational scale. The target is depth of understanding, not production performance.

## Environment

- Apple Silicon Mac (M-series)
- Python 3.10+
- [MLX](https://ml-explore.github.io/mlx/) (Apple's ML framework for Apple Silicon)
- Metal (Apple's GPU compute API, accessed through MLX custom kernels)

## Setup

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

## Running Examples

```bash
python3 examples/matmul_add_relu.py
```

## Running Benchmarks

```bash
python3 benchmarks/benchmark_m5.py
PYTHONPATH=. python3 benchmarks/benchmark_matmul.py          # Metal matmul vs PyTorch MPS (needs torch)
PYTHONPATH=. python3 benchmarks/benchmark_matmul_triton.py   # Triton matmul vs PyTorch CUDA (NVIDIA GPU + triton only)
```

`tensorforge/kernels/triton_matmul.py` is a standalone CUDA/Triton port of matmul. It is not used by the Apple Silicon runtime and is untested on this repo's dev machine (no NVIDIA GPU).

## Architecture

```
User Tensor API
      │
      ▼
TensorForge Tensor          ← wraps mx.array, adds metadata
      │
      ▼
Computation Graph / IR      ← captures structure before execution
  ├─ GraphInput              ← runtime-supplied values
  ├─ GraphConstant           ← embedded constants
  └─ GraphOperation          ← op_type + inputs → output Value
      │
      ▼
Optimization Passes
  ├─ Shape Propagation       ← infers output shapes
  ├─ Constant Folding        ← evaluates constant-only subgraphs
  └─ Add→ReLU Fusion         ← rewrites to fused kernel
      │
      ▼
Runtime Dispatch
  ├─ MLX path                ← built-in MLX operations (default)
  └─ Custom Metal path       ← explicit opt-in, float32 same-shape
      │
      ▼
MLX Lazy Evaluation
      │
      ▼
Apple Silicon GPU (via Metal)
      │
      ▼
mx.eval() — observation boundary
```

## Milestones

| Milestone | Topic |
|-----------|-------|
| 1 | MLX and GPU fundamentals |
| 2 | Tensor runtime |
| 3 | Computation graph and IR |
| 4 | Custom Metal kernels (ReLU, Add) |
| 5 | Optimization passes and kernel fusion |
| 6 | Custom Metal matrix multiplication (naive + tiled) |
| 7 | Production-style runtime and backend systems |
| 8 | Integrated MVP, evaluation, and final closeout |

## Supported Operations

| Op | MLX path | Custom Metal path |
|----|----------|-------------------|
| Add | ✓ (broadcast) | ✓ (same-shape, float32) |
| Multiply | ✓ (broadcast) | — |
| ReLU | ✓ | ✓ (float32) |
| MatMul | ✓ | ✓ (naive + tiled, float32, rank-2) |
| Reshape | ✓ | — |
| Transpose | ✓ | — |
| Add+ReLU fused | — | ✓ (same-shape, float32) |

## What This Is Not

TensorForge is not production-ready. It does not claim to compete with MLX, PyTorch, or JAX. It has no concurrency guarantees, no broad dtype coverage, no multi-device support, and no stable public API. It is an educational artifact.
