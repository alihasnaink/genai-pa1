"""Memory-mapped token streams and random next-token batch sampling."""

from __future__ import annotations

import numbers
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

TOKEN_DTYPE = np.dtype("<u2")


def load_token_array(path: str | os.PathLike) -> np.memmap:
    """Open a headerless little-endian uint16 token file as a read-only memmap."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"token file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"token file is empty: {path}")
    if size % TOKEN_DTYPE.itemsize != 0:
        raise ValueError(f"token file has an odd byte length ({size} bytes): {path}")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def _check_positive_int(name: str, value) -> None:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def get_batch(
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    sequence_length: int,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` windows of ``sequence_length + 1`` tokens.

    Starts are drawn uniformly from ``[0, len(dataset) - sequence_length)`` with
    ``generator``; only the sampled slices are read and converted to int64.
    Returns ``(x, y)``: inputs and one-token-shifted targets on ``device``.
    """
    _check_positive_int("batch_size", batch_size)
    _check_positive_int("sequence_length", sequence_length)
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional token array")
    if len(dataset) < sequence_length + 1:
        raise ValueError(
            f"dataset has {len(dataset)} tokens; need at least sequence_length + 1 = {sequence_length + 1}"
        )

    starts = torch.randint(0, len(dataset) - sequence_length, (batch_size,), generator=generator)
    windows = np.stack([dataset[start : start + sequence_length + 1] for start in starts.tolist()])
    windows = torch.from_numpy(windows.astype(np.int64))

    windows = _to_device(windows, torch.device(device))
    return windows[:, :-1], windows[:, 1:]


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def get_sharded_batch(
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    sequence_length: int,
    device: str | torch.device,
    generator: torch.Generator,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Data-parallel sampling: every rank draws the *same* full batch, then keeps rows ``rank::world_size``.

    Keeping the generators in lockstep means the checkpointed generator state is
    identical on all ranks and the run sees exactly the data a single-GPU run
    with the same seed would see.
    """
    if world_size == 1:
        return get_batch(dataset, batch_size, sequence_length, device, generator)
    if batch_size % world_size != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by world_size={world_size}")
    x, y = get_batch(dataset, batch_size, sequence_length, "cpu", generator)
    device = torch.device(device)
    return _to_device(x[rank::world_size].contiguous(), device), _to_device(y[rank::world_size].contiguous(), device)
