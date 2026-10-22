# Milestone 3 Closeout: Computation Graph and IR

## Summary

Milestone 3 built a computation graph that captures the structure of a computation before any execution occurs, enabling analysis, optimization, and reproducible scheduling.

## Types Implemented

- `Value` — named node with optional shape and dtype metadata
- `GraphInput` — placeholder for a runtime-supplied tensor
- `GraphConstant` — embedded constant mx.array
- `GraphOperation` — op_type + input Values → output Value
- `ComputationGraph` — holds all nodes, validates, sorts, executes

## Key Mechanisms

### Dependency Validation
All input values referenced by operations must be defined in the graph. All declared output IDs must exist.

### Cycle Detection
DFS with three-color marking (white/gray/black). Gray nodes are on the current DFS path; a gray node encountered again means a cycle.

### Topological Sort
Kahn's BFS algorithm. Produces a deterministic execution order regardless of the order operations were added to the graph.

### Execution
Inputs are bound from `feed_dict`, constants from embedded data. Operations execute in topological order through the Runtime. Results are collected for declared outputs.

## Representative Graph

MatMul → Add → ReLU (the standard neural network linear+activation pattern):

```
input x (4,8) ──┐
                 matmul ──┐
const w (8,4) ──┘         add ──> relu ──> output
const bias (1,4) ──────────┘
```

## Test Results

46 tests, 0 failures (M1–M3 coverage).
