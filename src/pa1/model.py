"""Pre-norm Transformer block and the full Transformer language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from pa1.attention import GroupedQuerySelfAttention
from pa1.layers import Embedding, Linear, RMSNorm, SwiGLU


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters; defaults are the fixed TinyStories model (Section 3.1)."""

    vocab_size: int = 8192
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    n_q_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 1344
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerBlock(nn.Module):
    """``u = x + Attn(RMSNorm(x)); y = u + SwiGLU(RMSNorm(u))``."""

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.attention = GroupedQuerySelfAttention(
            d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device=device, dtype=dtype
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch sequence d_model"],
        token_positions: Int[Tensor, "... sequence"] | None = None,
    ) -> Float[Tensor, "batch sequence d_model"]:
        # Under autocast the sublayer outputs are float16 while x is float32,
        # so each residual addition keeps the residual stream in float32.
        x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """Token embedding -> ``num_layers`` pre-norm blocks -> RMSNorm -> untied LM head."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
        )
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_q_heads,
                n_kv_heads,
                d_ff,
                context_length,
                rope_theta,
                norm_eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TransformerLM":
        return cls(**config.to_dict(), device=device, dtype=dtype)

    def forward(
        self,
        token_ids: Int[Tensor, "batch sequence"],
        token_positions: Int[Tensor, "... sequence"] | None = None,
    ) -> Float[Tensor, "batch sequence vocab_size"]:
        if token_ids.ndim < 1:
            raise ValueError("token_ids must have a sequence dimension")
        sequence_length = token_ids.shape[-1]
        if not 1 <= sequence_length <= self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} must be in [1, context_length={self.context_length}]"
            )
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
        return self.lm_head(self.final_norm(x))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
