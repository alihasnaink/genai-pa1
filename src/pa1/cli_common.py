"""Helpers shared by the inference-side command-line tools."""

from __future__ import annotations

import argparse

import torch

from pa1.checkpoint import read_model_state
from pa1.distributed import init_distributed
from pa1.model import ModelConfig, TransformerLM


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="final_model.pt",
        help="final_model.pt (fp16 state dict) or a full training checkpoint",
    )
    parser.add_argument("--device", default="auto")


def resolve_device(name: str) -> torch.device:
    return init_distributed(name).device


def load_model(path: str, device: torch.device) -> TransformerLM:
    """Build the fixed architecture (or the checkpoint's config) in float32 and load weights.

    A float16 ``final_model.pt`` is upcast into float32 parameters, which is what
    a grader loading it through ``get_transformer_lm`` sees.
    """
    state, config_dict = read_model_state(path)
    config = ModelConfig(**config_dict) if config_dict else ModelConfig()
    model = TransformerLM.from_config(config, device=device, dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    return model
