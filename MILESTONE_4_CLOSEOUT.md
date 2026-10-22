# Milestone 4 Closeout: First Custom Metal Execution Path

## Summary

Milestone 4 added the first custom Metal kernels through MLX's `mx.fast.metal_kernel` API, establishing the pattern used for all custom kernels in the project.

## Kernels Implemented

### ReLU (`kernels/metal_relu.py`)

```metal
out[elem] = fmax(inp[elem], 0.0f)
```

- Float32 only
- Any shape (flattened for 1-D dispatch, reshaped on output)
- Reusable kernel object built once, reused across calls

### Same-Shape Add (`kernels/metal_add.py`)

```metal
out[elem] = a[elem] + b[elem]
```

- Float32 only
- Same-shape restriction enforced before launch (no broadcasting)

## Runtime Integration

`Runtime.metal_relu()` and `Runtime.metal_add()` are explicit opt-in paths. The default `Runtime.relu()` and `Runtime.add()` continue to use built-in MLX.

Backend labels distinguish `mlx` from `metal` in the execution trace.

## MLX Custom Kernel API

MLX compiles Metal source as the **body** of a kernel function. The function signature (input/output buffers, thread position variables) is generated automatically.

```python
kernel = mx.fast.metal_kernel(
    name="name",
    input_names=["inp"],
    output_names=["out"],
    source="uint elem = thread_position_in_grid.x; out[elem] = ...;",
)
outputs = kernel(
    inputs=[array],
    grid=(n, 1, 1),
    threadgroup=(min(n, 256), 1, 1),
    output_shapes=[shape],
    output_dtypes=[mx.float32],
)
```

## Known Limitations

- Float32 only
- Custom Add: same shape only (no broadcasting)
- MLX built-in kernels are faster; custom kernels are for learning
