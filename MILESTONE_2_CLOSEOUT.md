# Milestone 2 Closeout: Tensor Runtime

## Summary

Milestone 2 built the Tensor wrapper and Runtime dispatcher that form the foundation of TensorForge.

## Tensor Wrapper

`Tensor` wraps `mx.array` and adds:
- `shape`, `rank`, `dtype`, `size`, `estimated_bytes` properties
- Creation helpers: `zeros`, `ones`, `full`, `from_array`, `random_uniform`, `random_normal`
- `eval()` — forces evaluation of the MLX lazy graph
- `to_numpy()` — evaluates and converts
- `op_record` — records how the tensor was created

MLX arrays are lazy-evaluated. The wrapper preserves this behavior; values are not computed until `mx.eval()` is called.

## Runtime

`Runtime` dispatches operations to MLX:
- `add`, `multiply` — elementwise with broadcasting
- `relu` — `mx.maximum(a, 0)`
- `matmul` — rank-2 validated
- `reshape`, `transpose`
- `eval(*tensors)` — force evaluation

Each call is recorded in an execution trace with op name, backend, input shapes, output shape, and dtype.

## Memory Accounting

`estimated_bytes` gives an educational estimate of tensor memory footprint based on dtype size. This is approximate — it does not account for alignment, padding, or MLX internal representation.
