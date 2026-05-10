"""
CPU/PyTorch fake-quantization reference and autograd Function.

fake_quantize:
    x_q  = clamp(round(x / scale), -128, 127)   [int8 range]
    x_dq = x_q.float() * scale                   [dequantized]

STE mask:
    mask = (x_q > -128) & (x_q < 127)
    Gradient passes through unclipped values only.
"""

import torch
from torch import Tensor


def _compute_scale(x: Tensor) -> Tensor:
    return x.abs().max().clamp(min=1e-8) / 127.0


def fake_quantize(x: Tensor, scale: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (x_dequant, scale, mask). mask is True where x_q ∈ (-128, 127)."""
    if scale is None:
        scale = _compute_scale(x)
    x_q = x.div(scale).round().clamp(-128, 127).to(torch.int8)
    x_dq = x_q.float() * scale
    mask = (x_q > -128) & (x_q < 127)
    return x_dq, scale, mask


def ref_quant_matmul(
    A: Tensor,
    B: Tensor,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
) -> Tensor:
    """Reference fake-quant matmul on any device. Used to validate CUDA kernels."""
    A_dq, _, _ = fake_quantize(A, scale_a)
    B_dq, _, _ = fake_quantize(B, scale_b)
    return A_dq @ B_dq


class QuantMatmul(torch.autograd.Function):
    """
    Fake-quant matmul with STE backward.

    Forward:  Y = dequant(A) @ dequant(B)
    Backward: dA = (dY @ B_dq.T) * mask_A
              dB = (A_dq.T @ dY) * mask_B

    Dispatch:
      - CUDA inputs + extension built → CUDA kernels
      - otherwise                     → pure PyTorch
    """

    @staticmethod
    def forward(
        ctx,
        A: Tensor,
        B: Tensor,
        scale_a: Tensor | None = None,
        scale_b: Tensor | None = None,
        tile_size: int = 32,
    ) -> Tensor:
        A_dq, scale_a, mask_A = fake_quantize(A, scale_a)
        B_dq, scale_b, mask_B = fake_quantize(B, scale_b)

        # Store for backward regardless of dispatch path
        ctx.save_for_backward(A_dq, B_dq, mask_A, mask_B)
        ctx.use_cuda_backward = False

        if A.is_cuda:
            from qgemm._ext import cuda_forward_available, cuda_forward
            if cuda_forward_available():
                A_q = A.div(scale_a).round().clamp(-128, 127).to(torch.int8)
                B_q = B.div(scale_b).round().clamp(-128, 127).to(torch.int8)
                ctx.use_cuda_backward = True
                return cuda_forward(A_q, B_q, scale_a.item(), scale_b.item(), tile_size)

        return A_dq @ B_dq

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        A_dq, B_dq, mask_A, mask_B = ctx.saved_tensors

        if ctx.use_cuda_backward:
            from qgemm._ext import cuda_backward
            grad_A, grad_B = cuda_backward(grad_output, A_dq, B_dq, mask_A, mask_B)
        else:
            grad_A = (grad_output @ B_dq.T) * mask_A.float()
            grad_B = (A_dq.T @ grad_output) * mask_B.float()

        return grad_A, grad_B, None, None, None
