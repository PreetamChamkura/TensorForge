"""
Triton matmul kernel (CUDA). C = A @ B for rank-2 float32/float16/bfloat16.

NOT part of the Apple Silicon / MLX path: requires an NVIDIA GPU, PyTorch with
CUDA, and `pip install triton`. Deliberately not imported by
tensorforge.kernels.__init__ so the rest of the package works without Triton.

Same idea as the Metal tiled kernel, in Triton's block-level model: each
program instance computes one BLOCK_M x BLOCK_N tile of C, looping over K in
BLOCK_K steps. Triton handles shared-memory staging and tensor-core use.
Program ids are remapped in GROUP_M-row groups so concurrent programs reuse
the same A/B tiles in L2.

Float32 inputs use TF32 tensor cores by default (allow_tf32=True, i.e. input_precision="tf32"; needs Triton >= 3.0), so expect
~1e-3 relative error vs a strict fp32 reference; pass allow_tf32=False for
full fp32 accumulation.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                      num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                      num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                      num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                      num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering for L2 reuse.
    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    group_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_m)
    pid_n = (pid % group_size) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor, allow_tf32: bool = True) -> torch.Tensor:
    """Rank-2 matmul on CUDA tensors via the Triton kernel."""
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("triton_matmul requires CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"triton_matmul requires rank-2 inputs, got {a.ndim} and {b.ndim}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions must match: {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"expected matching float32/float16/bfloat16, got {a.dtype}, {b.dtype}")
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0 or K == 0:
        return c.zero_()
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        INPUT_PRECISION="tf32" if allow_tf32 else "ieee",
    )
    return c
