"""Verify the fp16 mixed-precision policy and benchmark fp32 vs. fp16 throughput.

    uv run python src/precision_check.py                    # dtype + parity checks, then benchmark
    uv run python src/precision_check.py --batch-sizes 16 32 --steps 20 --out report_assets/precision_benchmark.json

Checks (CUDA only; on CPU only the fp32 benchmark runs):
  1. dtype audit under autocast: Linear/LM-head outputs float16, RMSNorm and
     residual stream float32, RoPE preserves float16, loss float32, params and
     grads float32;
  2. fp32 vs. fp16 parity on one batch: loss difference and gradient cosine similarity;
  3. per-optimizer-step time, tokens/s and peak memory for each precision / microbatch size.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pa1.cli_common import resolve_device
from pa1.layers import Linear, RMSNorm
from pa1.model import ModelConfig, TransformerLM
from pa1.nn_utils import cross_entropy, gradient_clipping
from pa1.optim import AdamW
from pa1.precision import autocast_context, make_grad_scaler
from pa1.rope import RotaryPositionalEmbedding


def dtype_audit(model: TransformerLM, tokens: torch.Tensor, device: torch.device) -> dict[str, list[str]]:
    seen: dict[str, set[str]] = {}

    def record(kind: str):
        def hook(_module, _inputs, output):
            seen.setdefault(kind, set()).add(str(output.dtype).removeprefix("torch."))
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            handles.append(module.register_forward_hook(record("lm_head" if name == "lm_head" else "linear")))
        elif isinstance(module, RMSNorm):
            handles.append(module.register_forward_hook(record("rmsnorm")))
        elif isinstance(module, RotaryPositionalEmbedding):
            handles.append(module.register_forward_hook(record("rope")))
    handles.append(model.blocks[-1].register_forward_hook(record("residual_stream")))
    try:
        with autocast_context("fp16", device):
            logits = model(tokens[:, :-1])
            loss = cross_entropy(logits, tokens[:, 1:])
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    seen["loss"] = {str(loss.dtype).removeprefix("torch.")}
    seen["parameters"] = {str(p.dtype).removeprefix("torch.") for p in model.parameters()}
    seen["gradients"] = {str(p.grad.dtype).removeprefix("torch.") for p in model.parameters()}
    model.zero_grad(set_to_none=True)
    return {key: sorted(value) for key, value in seen.items()}


def parity(model: TransformerLM, tokens: torch.Tensor, device: torch.device) -> dict[str, float]:
    results = {}
    gradients = {}
    for precision in ("fp32", "fp16"):
        model.zero_grad(set_to_none=True)
        with autocast_context(precision, device):
            loss = cross_entropy(model(tokens[:, :-1]), tokens[:, 1:])
        loss.backward()
        results[f"loss_{precision}"] = float(loss.detach())
        gradients[precision] = torch.cat([p.grad.reshape(-1).float() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    results["abs_loss_difference"] = abs(results["loss_fp32"] - results["loss_fp16"])
    reference, mixed = gradients["fp32"], gradients["fp16"]
    results["grad_cosine_similarity"] = float((reference @ mixed) / (reference.norm() * mixed.norm()))
    results["grad_relative_l2_error"] = float(
        (gradients["fp32"] - gradients["fp16"]).norm() / gradients["fp32"].norm()
    )
    return results


def benchmark(
    config: ModelConfig,
    device: torch.device,
    precision: str,
    batch_size: int,
    sequence_length: int,
    grad_accum: int,
    steps: int,
    warmup: int,
    compile_model: bool,
) -> dict:
    torch.manual_seed(0)
    model = TransformerLM.from_config(config, device=device)
    forward_model = torch.compile(model) if compile_model else model
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = make_grad_scaler(precision, device)
    tokens = torch.randint(0, config.vocab_size, (batch_size, sequence_length + 1), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with autocast_context(precision, device):
                loss = cross_entropy(forward_model(tokens[:, :-1]), tokens[:, 1:])
            scaler.scale(loss / grad_accum).backward()
        scaler.unscale_(optimizer)
        gradient_clipping(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds_per_step = (time.perf_counter() - start) / steps
    tokens_per_step = batch_size * sequence_length * grad_accum
    return {
        "precision": precision,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "compile": compile_model,
        "sec_per_step": seconds_per_step,
        "tokens_per_sec": tokens_per_step / seconds_per_step,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--small", action="store_true", help="tiny model (CPU smoke test)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=2, help="microbatches per timed step")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", action="store_true", help="also benchmark torch.compile")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    config = ModelConfig(d_model=64, num_layers=2, n_q_heads=4, n_kv_heads=2, d_ff=192) if args.small else ModelConfig()
    report: dict = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        report["capability"] = torch.cuda.get_device_capability(device)
        torch.manual_seed(0)
        model = TransformerLM.from_config(config, device=device)
        tokens = torch.randint(0, config.vocab_size, (4, args.sequence_length + 1), device=device)
        audit = dtype_audit(model, tokens, device)
        report["dtype_audit"] = audit
        expected = {
            "linear": ["float16"],
            "lm_head": ["float16"],
            "rmsnorm": ["float32"],
            "residual_stream": ["float32"],
            "rope": ["float16"],
            "loss": ["float32"],
            "parameters": ["float32"],
            "gradients": ["float32"],
        }
        mismatches = {key: audit.get(key) for key, value in expected.items() if audit.get(key) != value}
        report["dtype_audit_ok"] = not mismatches
        print("dtype audit:", json.dumps(audit), "OK" if not mismatches else f"MISMATCH {mismatches}")
        report["parity"] = parity(model, tokens, device)
        print("parity:", json.dumps(report["parity"]))
        del model

    precisions = ["fp32", "fp16"] if device.type == "cuda" else ["fp32"]
    runs = []
    for batch_size in args.batch_sizes:
        for precision in precisions:
            for compile_model in ([False, True] if args.compile else [False]):
                try:
                    result = benchmark(config, device, precision, batch_size, args.sequence_length,
                                       args.grad_accum, args.steps, args.warmup, compile_model)
                except torch.OutOfMemoryError:
                    result = {"precision": precision, "batch_size": batch_size, "compile": compile_model, "oom": True}
                    torch.cuda.empty_cache()
                runs.append(result)
                print("benchmark:", json.dumps(result), flush=True)
    report["benchmark"] = runs
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
