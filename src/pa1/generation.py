"""Autoregressive decoding with temperature scaling and nucleus (top-p) sampling."""

from __future__ import annotations

import torch

from pa1.attention import softmax


def top_p_filter(probabilities: torch.Tensor, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(filtered_sorted_probs, sorted_token_ids)`` for the smallest descending
    prefix whose cumulative probability reaches ``top_p``; the prefix is renormalized."""
    sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Keep an entry iff the mass strictly before it is still below top_p.
    # The top entry is therefore always kept.
    keep = (cumulative - sorted_probs) < top_p
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    return filtered / filtered.sum(dim=-1, keepdim=True), sorted_ids


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a completion and return ``prompt + completion`` as a 1-D long tensor.

    Every step feeds only the most recent ``context_length`` tokens (no KV cache),
    applies temperature and top-p, and samples a token. Decoding stops after
    ``max_new_tokens`` or immediately after emitting ``eot_token_id``.
    """
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must be a non-empty one-dimensional tensor")
    if prompt_ids.dtype != torch.long:
        raise TypeError(f"prompt_ids must be torch.long, got {prompt_ids.dtype}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    model_context = getattr(model, "context_length", None)
    if model_context is None:
        raise AttributeError("model must expose a context_length attribute")
    if context_length != model_context:
        raise ValueError(f"context_length={context_length} does not match model.context_length={model_context}")
    if max_new_tokens == 0:
        return prompt_ids.clone()

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            sequence = prompt_ids.clone()
            for _ in range(max_new_tokens):
                window = sequence[-context_length:]
                logits = model(window.unsqueeze(0))[0, -1].float()
                probabilities = softmax(logits / temperature, dim=-1)
                filtered, sorted_ids = top_p_filter(probabilities, top_p)
                if generator is not None and generator.device != filtered.device:
                    filtered = filtered.to(generator.device)
                    sorted_ids = sorted_ids.to(generator.device)
                rank = torch.multinomial(filtered, num_samples=1, generator=generator)
                next_token = sorted_ids[rank].to(sequence.device)
                sequence = torch.cat([sequence, next_token])
                if eot_token_id is not None and int(next_token) == eot_token_id:
                    break
            return sequence
    finally:
        model.train(was_training)
