"""Validation loss and the standardized final evaluation (Section 7.3)."""

from __future__ import annotations

import math

import numpy as np
import torch

from pa1.data import get_sharded_batch
from pa1.distributed import DistContext, all_reduce_mean
from pa1.nn_utils import cross_entropy
from pa1.precision import autocast_context


def evaluate_validation(
    model: torch.nn.Module,
    tokens: np.ndarray,
    *,
    batch_size: int,
    sequence_length: int,
    num_batches: int,
    device: torch.device,
    generator: torch.Generator,
    precision: str = "fp32",
    ctx: DistContext | None = None,
) -> float:
    """Mean cross-entropy over ``num_batches`` batches sampled with ``generator``.

    Runs under ``model.eval()`` and ``torch.inference_mode()`` and restores the
    model's previous mode afterwards. With data parallelism every rank samples
    the same batches (keeping ``generator`` in lockstep) and evaluates its shard;
    shard means are averaged, which equals the full-batch mean.
    """
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}")
    rank, world_size = (ctx.rank, ctx.world_size) if ctx is not None else (0, 1)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            total = torch.zeros((), dtype=torch.float64, device=device)
            for _ in range(num_batches):
                x, y = get_sharded_batch(tokens, batch_size, sequence_length, device, generator, rank, world_size)
                with autocast_context(precision, device):
                    logits = model(x)
                total += cross_entropy(logits, y).double()
            mean = total / num_batches
            if ctx is not None:
                mean = all_reduce_mean(mean, ctx)
            return float(mean)
    finally:
        model.train(was_training)


def standardized_evaluation(
    model: torch.nn.Module,
    tokens: np.ndarray,
    device: torch.device,
    *,
    seed: int = 42,
    num_batches: int = 100,
    batch_size: int = 16,
    sequence_length: int = 256,
    precision: str = "fp32",
) -> dict[str, float]:
    """Fresh generator seeded 42, 100 batches of 16 x 256; PPL = exp(mean CE)."""
    generator = torch.Generator().manual_seed(seed)
    mean_ce = evaluate_validation(
        model,
        tokens,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_batches=num_batches,
        device=device,
        generator=generator,
        precision=precision,
    )
    return {
        "mean_cross_entropy": mean_ce,
        "perplexity": math.exp(mean_ce),
        "seed": seed,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }
