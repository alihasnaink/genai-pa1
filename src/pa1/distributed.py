"""Minimal data parallelism over ``torch.distributed`` (e.g. Kaggle 2x T4).

The assignment forbids ``torch.nn`` implementations beyond Parameter,
containers, and init, so ``DistributedDataParallel`` is not used. Instead each
rank computes gradients on its shard of every microbatch and gradients are
averaged with one flat all-reduce per optimizer step (not per microbatch).

Launch with ``torchrun --nproc_per_node=2 src/train.py ...``; a plain
``python src/train.py`` run is the single-process case (world size 1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_distributed(device: str = "auto") -> DistContext:
    """Initialize the process group when launched by ``torchrun``; otherwise single process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        resolved = _default_device() if device == "auto" else torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        return DistContext(rank=0, local_rank=0, world_size=1, device=resolved)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device in ("auto", "cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        resolved = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        resolved = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return DistContext(rank=rank, local_rank=local_rank, world_size=world_size, device=resolved)


def cleanup_distributed(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    if ctx.enabled:
        dist.barrier()


@torch.no_grad()
def broadcast_module(module: torch.nn.Module, ctx: DistContext) -> None:
    """Make every rank start from rank 0's parameters.

    Buffers (RoPE tables, causal mask) are deterministic functions of the
    config, so they are identical already and are not broadcast.
    """
    if not ctx.enabled:
        return
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)


@torch.no_grad()
def all_reduce_gradients_and_scalars(
    parameters: list[torch.nn.Parameter],
    scalars: torch.Tensor,
    ctx: DistContext,
) -> torch.Tensor:
    """Average all gradients (and a small tensor of scalars) across ranks in one collective.

    Gradients are still loss-scaled at this point; an inf/NaN on either rank
    propagates to both, so both ranks' GradScalers skip the same step and the
    replicas stay bit-identical.
    """
    if not ctx.enabled:
        return scalars
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    flat = torch.cat([gradient.reshape(-1).float() for gradient in gradients] + [scalars.reshape(-1).float()])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(ctx.world_size)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset : offset + count].view_as(gradient))
        offset += count
    return flat[offset:].view_as(scalars).to(scalars.dtype)


def all_reduce_mean(value: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    if not ctx.enabled:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / ctx.world_size


def broadcast_flag(flag: bool, ctx: DistContext) -> bool:
    """Rank 0's decision (e.g. 'time limit reached') applied identically on every rank."""
    if not ctx.enabled:
        return flag
    tensor = torch.tensor([1 if flag else 0], device=ctx.device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())
