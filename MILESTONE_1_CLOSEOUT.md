# Milestone 1 Closeout: MLX and GPU Fundamentals

## Summary

Milestone 1 established the foundational understanding of MLX, Apple Silicon GPU execution, and the performance characteristics of tensor operations.

## Concepts Covered

### MLX Fundamentals
- Tensor creation, shapes, dtypes
- Elementwise operations and vectorization
- Lazy evaluation: operations are recorded, not computed
- `mx.eval()` as the observation boundary that flushes the graph

### Apple Silicon Unified Memory
- CPU and GPU share the same physical memory
- No explicit host-to-device copy
- Implications for bandwidth and memory pressure

### Hardware Execution Hierarchy
- Python → MLX → Metal → Apple Silicon GPU
- MLX handles kernel selection, compilation, dispatch
- Metal is Apple's low-level GPU compute API

### Benchmarking Methodology
- Separate setup cost from operation cost
- Always call `mx.eval()` before stopping the timer
- Measure warm (steady-state) behavior separately from cold (first-run) behavior
- Report relative comparisons honestly; do not claim speedup without evidence

### MatMul Dominance
- Arithmetic intensity: multiply-accumulate operations scale as O(N³)
- Memory bandwidth: input data scales as O(N²)
- At large N, compute time dominates bandwidth — GPU advantage is largest here
- Elementwise ops are bandwidth-bound; matmul benefits most from GPU hardware

### Memory Constraints
- Float32 uses 4 bytes/element
- Large tensors can exhaust unified memory
- OOM behavior on Apple Silicon differs from discrete GPU
