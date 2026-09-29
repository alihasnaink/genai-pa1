"""Train the Transformer LM on memory-mapped token streams.

Single GPU / CPU:
    uv run python src/train.py --run-dir runs/debug --num-steps 100

Two GPUs (e.g. Kaggle 2x T4), float16 mixed precision:
    uv run torchrun --standalone --nproc_per_node=2 src/train.py --run-dir runs/final

Per global step s the canonical order (Section 5.3.2) is:
    train mode -> lr_s -> zero grads -> n_acc microbatches (autocast fwd, scaled bwd)
    -> [all-reduce grads] -> unscale -> clip -> optimizer step (skipped on fp16 overflow)
    -> validation -> logging -> checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from pa1.checkpoint import export_fp16_state_dict, load_checkpoint, save_checkpoint
from pa1.data import get_sharded_batch, load_token_array
from pa1.distributed import (
    all_reduce_gradients_and_scalars,
    barrier,
    broadcast_flag,
    broadcast_module,
    cleanup_distributed,
    init_distributed,
)
from pa1.evaluation import evaluate_validation
from pa1.model import ModelConfig, TransformerLM, count_parameters
from pa1.nn_utils import cross_entropy, gradient_clipping
from pa1.optim import AdamW, get_lr_cosine_schedule
from pa1.precision import PRECISION_CHOICES, autocast_context, make_grad_scaler, resolve_precision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", default="data/tinystories/data/train.bin")
    data.add_argument("--val-data", default="data/tinystories/data/validation.bin")

    arch = parser.add_argument_group("model architecture (fixed for the final run)")
    defaults = ModelConfig()
    arch.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    arch.add_argument("--context-length", type=int, default=defaults.context_length)
    arch.add_argument("--d-model", type=int, default=defaults.d_model)
    arch.add_argument("--num-layers", type=int, default=defaults.num_layers)
    arch.add_argument("--n-q-heads", type=int, default=defaults.n_q_heads)
    arch.add_argument("--n-kv-heads", type=int, default=defaults.n_kv_heads)
    arch.add_argument("--d-ff", type=int, default=defaults.d_ff)
    arch.add_argument("--rope-theta", type=float, default=defaults.rope_theta)
    arch.add_argument("--norm-eps", type=float, default=defaults.norm_eps)

    opt = parser.add_argument_group("optimization")
    opt.add_argument("--num-steps", type=int, default=10_000, help="optimizer updates to run (global step cap)")
    opt.add_argument("--sequence-length", type=int, default=256)
    opt.add_argument(
        "--batch-size", type=int, default=16, help="sequences per microbatch, summed over all GPUs"
    )
    opt.add_argument("--grad-accum", type=int, default=16, help="microbatches per optimizer update (n_acc)")
    opt.add_argument("--lr-max", type=float, default=3e-4)
    opt.add_argument("--lr-min", type=float, default=3e-5)
    opt.add_argument("--warmup-steps", type=int, default=200)
    opt.add_argument("--cosine-steps", type=int, default=9_999, help="s_c; keep 9999 even for short runs")
    opt.add_argument("--beta1", type=float, default=0.9)
    opt.add_argument("--beta2", type=float, default=0.95)
    opt.add_argument("--adam-eps", type=float, default=1e-8)
    opt.add_argument("--weight-decay", type=float, default=0.1)
    opt.add_argument(
        "--no-decay-1d",
        action="store_true",
        help="exclude 1-D parameters (RMSNorm gains) from weight decay",
    )
    opt.add_argument("--max-grad-norm", type=float, default=1.0)

    run = parser.add_argument_group("evaluation, logging, checkpointing")
    run.add_argument("--eval-interval", type=int, default=250)
    run.add_argument("--log-interval", type=int, default=25)
    run.add_argument("--checkpoint-interval", type=int, default=500)
    run.add_argument("--num-val-batches", type=int, default=20)
    run.add_argument("--val-batch-size", type=int, default=None, help="defaults to --batch-size")
    run.add_argument("--run-dir", default="runs/default")
    run.add_argument("--checkpoint", default=None, help="checkpoint path (default: <run-dir>/checkpoint.pt)")
    run.add_argument(
        "--resume",
        default="auto",
        help="'auto' resumes from --checkpoint if it exists, 'none' starts fresh, or a path",
    )
    run.add_argument(
        "--time-limit-hours",
        type=float,
        default=None,
        help="checkpoint and exit cleanly once this much wall time is used (Kaggle session limits)",
    )

    seeds = parser.add_argument_group("seeds")
    seeds.add_argument("--model-seed", type=int, default=0)
    seeds.add_argument("--train-seed", type=int, default=1)
    seeds.add_argument("--val-seed", type=int, default=2)

    system = parser.add_argument_group("system / precision")
    system.add_argument("--device", default="auto")
    system.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="auto = fp16 on CUDA")
    system.add_argument("--init-loss-scale", type=float, default=2.0**14)
    system.add_argument("--compile", action="store_true", help="torch.compile the model forward")
    system.add_argument(
        "--overfit-single-batch",
        action="store_true",
        help="debug: reuse one fixed training batch every microbatch",
    )
    return parser


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    for name in ("num_steps", "batch_size", "grad_accum", "eval_interval", "log_interval",
                 "checkpoint_interval", "num_val_batches"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.sequence_length > args.context_length:
        raise ValueError("--sequence-length cannot exceed --context-length")
    if args.batch_size % world_size or args.val_batch_size % world_size:
        raise ValueError(f"batch sizes must be divisible by the number of GPUs ({world_size})")


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _param_groups(model: torch.nn.Module, weight_decay: float, no_decay_1d: bool) -> list[dict]:
    if not no_decay_1d:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ctx = init_distributed(args.device)
    device = ctx.device
    args.val_batch_size = args.val_batch_size or args.batch_size
    _validate_args(args, ctx.world_size)
    precision = resolve_precision(args.precision, device)

    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.jsonl"
    if ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        if ctx.is_main:
            print(message, flush=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # no-op on T4 (Turing); helps on Ampere+
        torch.backends.cudnn.allow_tf32 = True

    train_tokens = load_token_array(args.train_data)
    val_tokens = load_token_array(args.val_data)

    config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
    )
    torch.manual_seed(args.model_seed)
    model = TransformerLM.from_config(config, device=device, dtype=torch.float32)
    broadcast_module(model, ctx)
    optimizer = AdamW(
        _param_groups(model, args.weight_decay, args.no_decay_1d),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(precision, device, args.init_loss_scale)
    train_generator = torch.Generator().manual_seed(args.train_seed)
    val_generator = torch.Generator().manual_seed(args.val_seed)

    next_step = 0
    skipped_steps = 0
    resume_path = None
    if args.resume == "auto":
        resume_path = checkpoint_path if checkpoint_path.exists() else None
    elif args.resume != "none":
        resume_path = Path(args.resume)
    if resume_path is not None:
        extra: dict = {}
        next_step = load_checkpoint(resume_path, model, optimizer, train_generator, val_generator, extra_state=extra)
        if "grad_scaler" in extra and scaler.is_enabled():
            scaler.load_state_dict(extra["grad_scaler"])
        skipped_steps = int(extra.get("skipped_steps", 0))
        log(f"resumed from {resume_path} at next_step={next_step}")

    forward_model = torch.compile(model) if args.compile else model
    tokens_per_step = args.batch_size * args.sequence_length * args.grad_accum
    if ctx.is_main:
        run_config = {
            **vars(args),
            "model_config": config.to_dict(),
            "resolved_precision": precision,
            "world_size": ctx.world_size,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "num_parameters": count_parameters(model),
            "tokens_per_step": tokens_per_step,
            "total_tokens": tokens_per_step * args.num_steps,
            "torch_version": torch.__version__,
        }
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    log(
        f"params={count_parameters(model):,} precision={precision} world_size={ctx.world_size} "
        f"device={device} batch={args.batch_size}x{args.sequence_length} grad_accum={args.grad_accum} "
        f"tokens/step={tokens_per_step:,}"
    )

    fixed_batch = None
    if args.overfit_single_batch:
        fixed_batch = get_sharded_batch(
            train_tokens, args.batch_size, args.sequence_length, device, train_generator, ctx.rank, ctx.world_size
        )

    def next_train_batch() -> tuple[torch.Tensor, torch.Tensor]:
        if fixed_batch is not None:
            return fixed_batch
        return get_sharded_batch(
            train_tokens, args.batch_size, args.sequence_length, device, train_generator, ctx.rank, ctx.world_size
        )

    def checkpoint(next_step_value: int) -> None:
        if ctx.is_main:
            save_checkpoint(
                model,
                optimizer,
                next_step_value,
                train_generator,
                val_generator,
                checkpoint_path,
                extra_state={
                    "grad_scaler": scaler.state_dict() if scaler.is_enabled() else {},
                    "skipped_steps": skipped_steps,
                    "model_config": config.to_dict(),
                    "precision": precision,
                },
            )
        barrier(ctx)

    parameters = list(model.parameters())
    session_start_step = next_step
    session_start = time.perf_counter()
    interval_start = time.perf_counter()
    interval_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(next_step, args.num_steps):
        if args.time_limit_hours is not None:
            out_of_time = (time.perf_counter() - session_start) / 3600 >= args.time_limit_hours
            if broadcast_flag(out_of_time, ctx):
                checkpoint(step)
                log(f"time limit reached; checkpointed next_step={step}. Re-run the same command to resume.")
                break

        model.train()
        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(args.grad_accum):
            x, y = next_train_batch()
            with autocast_context(precision, device):
                logits = forward_model(x)
                loss = cross_entropy(logits, y)  # float32 even under autocast
            scaler.scale(loss / args.grad_accum).backward()
            loss_sum += loss.detach()

        loss_sum = all_reduce_gradients_and_scalars(parameters, loss_sum, ctx)
        scaler.unscale_(optimizer)  # true gradients before clipping; records inf/NaN
        grad_norm = gradient_clipping(parameters, args.max_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)  # skipped when unscale_ found inf/NaN
        scaler.update()
        step_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
        skipped_steps += int(step_skipped)
        train_loss = float(loss_sum) / args.grad_accum
        interval_steps += 1

        completed_steps = step + 1
        final_step = completed_steps == args.num_steps
        should_validate = final_step or completed_steps % args.eval_interval == 0
        should_log = should_validate or completed_steps % args.log_interval == 0
        should_checkpoint = final_step or completed_steps % args.checkpoint_interval == 0

        if not math.isfinite(train_loss):
            log(f"step {completed_steps}: non-finite training loss {train_loss}")

        val_loss = None
        if should_validate:
            val_loss = evaluate_validation(
                forward_model,
                val_tokens,
                batch_size=args.val_batch_size,
                sequence_length=args.sequence_length,
                num_batches=args.num_val_batches,
                device=device,
                generator=val_generator,
                precision=precision,
                ctx=ctx,
            )

        if should_log:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            elapsed = now - interval_start
            # ETA uses this session's average step time, which includes validation overhead.
            session_elapsed = now - session_start
            eta = session_elapsed / (completed_steps - session_start_step) * (args.num_steps - completed_steps)
            record = {
                "step": completed_steps,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "loss_scale": scaler.get_scale() if scaler.is_enabled() else None,
                "step_skipped": step_skipped,
                "skipped_steps": skipped_steps,
                "tokens_seen": completed_steps * tokens_per_step,
                "tokens_per_sec": interval_steps * tokens_per_step / elapsed,
                "sec_per_step": elapsed / interval_steps,
                "session_elapsed_sec": session_elapsed,
                "eta_sec": eta,
                "wall_time": time.time(),
                "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
            }
            if ctx.is_main:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            val_text = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
            scale_text = f" scale={record['loss_scale']:.0f} skipped={skipped_steps}" if scaler.is_enabled() else ""
            log(
                f"step {completed_steps}/{args.num_steps} train_loss={train_loss:.4f}{val_text} "
                f"lr={lr:.2e} grad_norm={grad_norm:.3f}{scale_text} "
                f"tok/s={record['tokens_per_sec']:,.0f} s/step={record['sec_per_step']:.3f} "
                f"elapsed={_format_duration(session_elapsed)} eta={_format_duration(eta)}"
            )
            interval_start = time.perf_counter()
            interval_steps = 0

        if should_checkpoint:
            checkpoint(completed_steps)

        if final_step and ctx.is_main:
            export_fp16_state_dict(model.state_dict(), run_dir / "final_model.pt")
            log(f"training complete; wrote {run_dir / 'final_model.pt'}")

    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
