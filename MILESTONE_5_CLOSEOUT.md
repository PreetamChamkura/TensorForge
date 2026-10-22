# Milestone 5 Closeout: Optimization Passes and Kernel Fusion

## Summary

Milestone 5 added three graph-level optimization passes that transform the computation graph before execution, and a fused custom Metal kernel that eliminates an intermediate memory round-trip between Add and ReLU.

## Passes Implemented

### Shape Propagation (`passes/shape_propagation.py`)

Walks the graph in topological order and infers output shapes based on operation type and input shapes. Results are attached to each operation's output `Value`.

Supported operations:
- `add`, `multiply` — numpy broadcast shape inference
- `relu` — output shape == input shape
- `matmul` — rank-2 only: (M, K) @ (K, N) → (M, N)
- `add_relu_fused` — same as add (broadcast)
- `reshape` — shape from operation attributes
- `transpose` — rank-2 only: (M, N) → (N, M)

If any input shape is unknown, the output shape is left as `None`.

### Constant Folding (`passes/constant_folding.py`)

Evaluates operations whose inputs are all constants at graph-construction time. The operation is removed and its output is replaced with a new `GraphConstant`. Supported foldable ops: `add`, `multiply`, `relu`.

### Add → ReLU Fusion (`passes/fusion.py`)

Detects Add immediately followed by ReLU and rewrites to a single `add_relu_fused` operation. Conditions:
1. The Add output has exactly one consumer (the ReLU).
2. The Add output is not a declared graph output.
3. Both Add inputs have the same known shape.
4. Both Add inputs are float32.

## Fused Metal Kernel

`kernels/metal_add_relu.py` implements:

```metal
out[i] = fmax(a[i] + b[i], 0.0f)
```

This reads each input once and writes once, eliminating the intermediate buffer that would be needed between separate Add and ReLU kernels.

## Test Results

70 tests, 0 failures.

## Benchmark Summary

Benchmark run on Apple Silicon (1024×1024 float32, 20 warm iterations):

- Direct MLX (baseline): built-in, highly optimized
- Runtime MLX path: small dispatch overhead vs direct
- Separate Metal Add + ReLU: two kernel invocations
- Fused Metal AddReLU: one kernel invocation

The fused Metal path reduces kernel invocations from 2 to 1 but does not beat optimized MLX. MLX's built-in compiler can fuse operations internally and applies vendor-tuned kernels. The goal of the fused kernel is to demonstrate the concept, not to win a benchmark.

## Known Limitations

- Custom Metal paths: float32 only, same-shape only (no broadcasting)
- Constant folding: supports add, multiply, relu only
- Fusion: requires statically known shapes at fusion time
- No autotuning or kernel selection based on shape

## Readiness for Milestone 6

The graph, runtime, and optimization infrastructure is in place. M6 adds custom Metal matrix multiplication (naive then tiled) on top of the existing system.
