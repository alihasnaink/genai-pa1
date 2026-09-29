"""Cross-entropy loss and global gradient-norm clipping."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... vocab"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """Mean of ``logsumexp(z) - z_y`` over every leading position.

    Float16 logits (the LM head output under autocast) are upcast to float32
    first, so the log-sum-exp reduction never runs in half precision.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match logits leading shape {tuple(logits.shape[:-1])}"
        )
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    log_normalizer = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1)
    return (log_normalizer - target_logits).mean()


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> float:
    """Scale all gradients in place by one factor so their global L2 norm is ``<= max_l2_norm``.

    Returns the pre-clipping global norm as a Python float. A non-finite norm
    (a float16 overflow step that the loss scaler will skip) is returned as-is
    and the gradients are left untouched.
    """
    if not max_l2_norm > 0:
        raise ValueError(f"max_l2_norm must be positive, got {max_l2_norm}")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    per_tensor_norms = torch.stack(
        [torch.linalg.vector_norm(gradient, ord=2, dtype=torch.float32) for gradient in gradients]
    )
    total_norm = float(torch.linalg.vector_norm(per_tensor_norms, ord=2))
    if math.isfinite(total_norm) and total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for gradient in gradients:
            gradient.mul_(scale)
    return total_norm
