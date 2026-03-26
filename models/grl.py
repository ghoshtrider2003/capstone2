"""Gradient reversal layer for domain-adversarial training."""

from __future__ import annotations

import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity forward pass and sign-flipped gradients in backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for gradient reversal."""

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambd)
