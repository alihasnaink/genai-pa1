"""AdamW optimizer and the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable

import torch


def _validate_hyperparameters(lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
    if not lr >= 0.0:
        raise ValueError(f"invalid learning rate: {lr}")
    if not eps >= 0.0:
        raise ValueError(f"invalid eps: {eps}")
    if not weight_decay >= 0.0:
        raise ValueError(f"invalid weight_decay: {weight_decay}")
    if len(betas) != 2:
        raise ValueError(f"betas must contain two values, got {betas}")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"invalid beta1: {beta1}")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"invalid beta2: {beta2}")


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay and per-parameter bias correction.

    State is kept in the parameter dtype (float32 master weights during
    mixed-precision training), so float16 is never used for optimizer math.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_hyperparameters(lr, betas, eps, weight_decay)
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            _validate_hyperparameters(group["lr"], group["betas"], group["eps"], group["weight_decay"])

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, betas, eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            _validate_hyperparameters(lr, betas, eps, weight_decay)
            beta1, beta2 = betas
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                parameter.mul_(1.0 - lr * weight_decay)
                denominator = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-lr / bias_correction1)
        return loss


def _is_integer(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def get_lr_cosine_schedule(
    step: int,
    learning_rate_max: float,
    learning_rate_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """Linear warmup to ``learning_rate_max`` over ``[0, warmup_steps)``, cosine decay to
    ``learning_rate_min`` at ``cosine_steps``, then a constant floor."""
    if not _is_integer(step):
        raise TypeError(f"step must be an integer, got {type(step).__name__}")
    if not _is_integer(warmup_steps) or not _is_integer(cosine_steps):
        raise TypeError("warmup_steps and cosine_steps must be integers")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if warmup_steps < 0 or warmup_steps >= cosine_steps:
        raise ValueError(f"require 0 <= warmup_steps < cosine_steps, got {warmup_steps}, {cosine_steps}")
    if not 0.0 <= learning_rate_min <= learning_rate_max:
        raise ValueError(f"require 0 <= lr_min <= lr_max, got {learning_rate_min}, {learning_rate_max}")

    if step < warmup_steps:
        return step / warmup_steps * learning_rate_max
    if step <= cosine_steps:
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
        return learning_rate_min + 0.5 * (1.0 + math.cos(math.pi * progress)) * (
            learning_rate_max - learning_rate_min
        )
    return float(learning_rate_min)
