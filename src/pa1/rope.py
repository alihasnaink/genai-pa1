"""Rotary positional embeddings using the adjacent-pair convention."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotate each adjacent coordinate pair ``(x_{2k-1}, x_{2k})`` by ``i * omega_k``.

    The ``cos``/``sin`` tables are float32 buffers of shape
    ``[context_length, head_dim / 2]``. Rotation runs in at least float32 and
    the result is cast back to the input dtype, so a float16 query stays
    float16 under autocast instead of being silently promoted.
    """

    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        context_length: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be positive and even, got {head_dim}")
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        self.rope_theta = float(rope_theta)
        self.head_dim = head_dim
        self.context_length = context_length

        # Angles are computed in float64 and rounded once into float32 tables.
        pair_offsets = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device)
        inverse_frequencies = self.rope_theta ** (-pair_offsets / head_dim)
        positions = torch.arange(context_length, dtype=torch.float64, device=device)
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cos", angles.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin", angles.sin().to(torch.float32), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... sequence head_dim"],
        token_positions: Int[Tensor, "... sequence"],
        *,
        validate_positions: bool = True,
    ) -> Float[Tensor, "... sequence head_dim"]:
        """Rotate ``x``; ``token_positions`` leading dims align with ``x``'s leading dims.

        ``validate_positions=False`` skips the range check, which needs a GPU
        synchronization; callers use it only for positions they built themselves.
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"expected final dimension head_dim={self.head_dim}, got {x.shape[-1]}")
        if x.ndim < 2:
            raise ValueError("RoPE input must have shape (..., sequence_length, head_dim)")
        if token_positions.is_floating_point() or token_positions.is_complex() or token_positions.dtype == torch.bool:
            raise TypeError(f"token_positions must be an integer tensor, got {token_positions.dtype}")
        if token_positions.ndim < 1 or token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                f"token_positions final dimension {tuple(token_positions.shape)} does not match "
                f"sequence length {x.shape[-2]}"
            )
        extra_dims = (x.ndim - 2) - (token_positions.ndim - 1)
        if extra_dims < 0:
            raise ValueError("token_positions has more batch-like dimensions than the input")
        if validate_positions and token_positions.numel() > 0:
            if bool((token_positions < 0).any()) or bool((token_positions >= self.context_length).any()):
                raise ValueError(f"token positions outside [0, {self.context_length})")

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # Leading position dims align with the leading dims of x; insert
        # singleton axes (e.g. attention heads) between them and the sequence axis.
        table_shape = (*token_positions.shape[:-1], *((1,) * extra_dims), x.shape[-2], self.head_dim // 2)
        cos = cos.reshape(table_shape)
        sin = sin.reshape(table_shape)

        compute_dtype = torch.promote_types(x.dtype, torch.float32)
        pairs = x.to(compute_dtype).unflatten(-1, (self.head_dim // 2, 2))
        even, odd = pairs[..., 0], pairs[..., 1]
        cos = cos.to(compute_dtype)
        sin = sin.to(compute_dtype)
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).to(x.dtype)

    def extra_repr(self) -> str:
        return f"rope_theta={self.rope_theta}, head_dim={self.head_dim}, context_length={self.context_length}"
