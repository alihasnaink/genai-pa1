"""Softmax, scaled dot-product attention, and causal grouped-query self-attention."""

from __future__ import annotations

import math

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from pa1.layers import Linear
from pa1.rope import RotaryPositionalEmbedding


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along ``dim`` (max-shifted before ``exp``)."""
    shifted = x - x.amax(dim=dim, keepdim=True).detach()
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    queries: Float[Tensor, "... n_q d_k"],
    keys: Float[Tensor, "... n_kv d_k"],
    values: Float[Tensor, "... n_kv d_v"],
    mask: Bool[Tensor, "... n_q n_kv"] | None = None,
    *,
    validate_mask: bool = True,
) -> Float[Tensor, "... n_q d_v"]:
    """``softmax(Q K^T / sqrt(d_k)) V`` where ``mask == True`` marks allowed pairs.

    The score matmul and the value matmul run in the input (or autocast) dtype,
    but masking and softmax always run in at least float32. ``validate_mask``
    checks that every query row keeps one allowed key; it needs a GPU sync, so
    internal callers that pass a known-valid causal mask disable it.
    """
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            f"query/key feature dimensions differ: d_k={queries.shape[-1]} vs {keys.shape[-1]}"
        )
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            f"key/value sequence lengths differ: {keys.shape[-2]} vs {values.shape[-2]}"
        )
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"attention mask must be a boolean tensor, got {mask.dtype}")
        if mask.ndim < 2 or mask.shape[-2:] != (queries.shape[-2], keys.shape[-2]):
            raise ValueError(
                f"mask final dimensions {tuple(mask.shape[-2:])} must equal "
                f"(n_q, n_kv)=({queries.shape[-2]}, {keys.shape[-2]})"
            )
        if validate_mask and queries.shape[-2] > 0 and not bool(mask.any(dim=-1).all()):
            raise ValueError("attention mask leaves a query with no permitted key; softmax is undefined")

    # Scaling Q before the matmul keeps float16 scores well inside their range.
    queries = queries * (1.0 / math.sqrt(queries.shape[-1]))
    scores = torch.matmul(queries, keys.transpose(-1, -2))
    scores = scores.to(torch.promote_types(scores.dtype, torch.float32))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = softmax(scores, dim=-1)
    return torch.matmul(probabilities.to(values.dtype), values)


class GroupedQuerySelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and grouped key/value heads.

    Covers MHA (``n_kv_heads == n_q_heads``), GQA, and MQA (``n_kv_heads == 1``).
    Query head ``a = h * g + r`` (0-indexed) uses key/value head ``h``. Each
    key/value head is shared by its ``g`` query heads without repeating K or V:
    the group axis is folded into the query-position axis so that one batched
    matmul per KV head computes all ``g`` heads at once.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        context_length: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if n_q_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("head counts must be positive")
        if d_model % n_q_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_q_heads={n_q_heads}")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}")
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.context_length = context_length

        self.q_proj = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length, device=device)
        causal = torch.ones(context_length, context_length, dtype=torch.bool, device=device).tril()
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "batch sequence d_model"],
        token_positions: Int[Tensor, "... sequence"] | None = None,
    ) -> Float[Tensor, "batch sequence d_model"]:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension d_model={self.d_model}, got {x.shape[-1]}")
        sequence_length = x.shape[-2]
        if not 1 <= sequence_length <= self.context_length:
            raise ValueError(f"sequence length {sequence_length} outside [1, {self.context_length}]")

        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device)
            validate_positions = False
        else:
            validate_positions = True

        q = rearrange(
            self.q_proj(x), "... s (h g d) -> ... h g s d", h=self.n_kv_heads, g=self.group_size
        )
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.n_kv_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.n_kv_heads)

        # RoPE aligns the position batch dims with x's batch dims and broadcasts
        # over the inserted head axes, so (sequence,) and (batch, sequence) both work.
        q = self.rope(q, token_positions, validate_positions=validate_positions)
        k = self.rope(k, token_positions, validate_positions=validate_positions)

        # [..., h, g, s, d] -> [..., h, g*s, d]: every query head of a group
        # attends to the same (unrepeated) key/value head.
        q = q.flatten(-3, -2)
        causal = self.causal_mask[:sequence_length, :sequence_length].repeat(self.group_size, 1)
        out = scaled_dot_product_attention(q, k, v, causal, validate_mask=False)
        out = out.unflatten(-2, (self.group_size, sequence_length))
        out = rearrange(out, "... h g s d -> ... s (h g d)")
        return self.out_proj(out)
