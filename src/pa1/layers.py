"""Primitive layers: Linear, Embedding, RMSNorm, SiLU, and SwiGLU.

Mixed-precision notes
---------------------
Parameters are always created in the requested dtype (float32 for training).
Under ``torch.autocast(dtype=torch.float16)`` the matrix multiplication in
``Linear`` runs in float16 on tensor cores while the float32 master weights
stay untouched. ``RMSNorm`` always normalizes in at least float32.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

_LOW_PRECISION = (torch.float16, torch.bfloat16)


def _trunc_normal_(tensor: Tensor, std: float) -> Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


class Linear(nn.Module):
    """Bias-free linear map ``y = x W^T`` with ``W`` stored as ``[out, in]``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        _trunc_normal_(self.weight, std=math.sqrt(2.0 / (in_features + out_features)))

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        # matmul is on autocast's float16 list, so this is the tensor-core path.
        return torch.matmul(x, self.weight.transpose(0, 1))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Embedding(nn.Module):
    """Token-embedding lookup into a ``[num_embeddings, embedding_dim]`` table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        _trunc_normal_(self.weight, std=1.0)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        if token_ids.is_floating_point() or token_ids.is_complex() or token_ids.dtype == torch.bool:
            raise TypeError(f"token ids must be an integer tensor, got {token_ids.dtype}")
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final axis with a learned gain."""

    def __init__(
        self,
        d_model: int,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        if in_dtype in _LOW_PRECISION:
            x = x.to(torch.float32)
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        result = x * inverse_rms * self.weight.to(x.dtype)
        return result.to(in_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, norm_eps={self.norm_eps}"


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish: ``x * sigmoid(x)``."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Gated feed-forward network ``W_down (SiLU(W_gate x) * W_up x)``."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.down(silu(self.gate(x)) * self.up(x))
