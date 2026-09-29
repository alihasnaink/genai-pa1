"""Mixed-precision policy: float16 compute on tensor cores, float32 everywhere else.

With ``precision="fp16"`` on CUDA:

* parameters, gradients, and AdamW state stay float32 (master weights);
* ``torch.autocast`` runs matmuls (Linear, attention scores, attention @ V) in float16;
* RMSNorm, RoPE, attention softmax, and cross-entropy upcast to float32 explicitly;
* the residual stream stays float32 because ``fp32 + fp16`` promotes to float32;
* a dynamic loss scaler (``torch.amp.GradScaler``) prevents float16 gradient
  underflow and skips any update whose gradients overflowed.

``bf16`` is supported for Ampere+ GPUs (no scaler needed); the T4 has no native
bf16, so fp16 is the right choice there.
"""

from __future__ import annotations

import contextlib

import torch

PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")
_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_precision(precision: str, device: torch.device) -> str:
    if precision not in PRECISION_CHOICES:
        raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {precision!r}")
    if precision == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if precision != "fp32" and device.type != "cuda":
        raise ValueError(f"precision={precision} requires a CUDA device; use fp32 on {device.type}")
    return precision


def autocast_context(precision: str, device: torch.device):
    """Autocast context for the forward pass and loss (never for backward/optimizer)."""
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPES[precision])


def make_grad_scaler(precision: str, device: torch.device, init_scale: float = 2.0**14) -> torch.amp.GradScaler:
    """Dynamic loss scaler; a transparent no-op unless training in float16."""
    return torch.amp.GradScaler(
        device.type,
        init_scale=init_scale,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=(precision == "fp16"),
    )
