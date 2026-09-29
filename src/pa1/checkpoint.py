"""Resumable training checkpoints and the fp16 model-only export."""

from __future__ import annotations

import os
import typing
from pathlib import Path

import torch

PathOrFile = str | os.PathLike | typing.BinaryIO | typing.IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    out: PathOrFile,
    *,
    extra_state: dict | None = None,
) -> None:
    """Serialize model, optimizer, ``next_step``, and both generator states.

    ``extra_state`` holds optional run metadata such as the float16 loss-scaler
    state, so a resumed mixed-precision run continues with the same scale.
    Path destinations are written atomically (temp file + rename) so an
    interrupted Kaggle session never leaves a truncated checkpoint.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": int(next_step),
        "train_generator_state": train_generator.get_state(),
        "val_generator_state": val_generator.get_state(),
        "extra_state": extra_state or {},
    }
    if isinstance(out, (str, os.PathLike)):
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    else:
        torch.save(payload, out)


def load_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    *,
    extra_state: dict | None = None,
) -> int:
    """Restore everything written by :func:`save_checkpoint` and return ``next_step``.

    If ``extra_state`` is a dict, it is updated in place with the saved extra state.
    """
    payload = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    train_generator.set_state(payload["train_generator_state"])
    val_generator.set_state(payload["val_generator_state"])
    if extra_state is not None:
        extra_state.update(payload.get("extra_state", {}))
    return int(payload["next_step"])


def read_model_state(src: PathOrFile) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Return ``(state_dict, model_config_or_None)`` from a training checkpoint or a
    plain model-only state dict such as ``final_model.pt``."""
    payload = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload and "optimizer" in payload:
        return payload["model"], payload.get("extra_state", {}).get("model_config")
    return payload, None


def export_fp16_state_dict(state_dict: dict[str, torch.Tensor], out: PathOrFile) -> dict[str, torch.Tensor]:
    """Write the Section 7.4 artifact: a plain CPU state dict with float16 floating tensors."""
    exported = {
        name: tensor.detach().cpu().to(torch.float16) if tensor.is_floating_point() else tensor.detach().cpu()
        for name, tensor in state_dict.items()
    }
    for name, tensor in exported.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} is not finite after conversion to float16")
    torch.save(exported, out)
    return exported
