"""Plot training metrics from one or more ``metrics.jsonl`` files into report figures.

Single run (loss / lr / grad norm / loss scale / throughput panels):
    uv run python src/plot_metrics.py runs/final --out report_assets/final_run.png

Compare runs (validation loss vs. step and vs. wall-clock time):
    uv run python src/plot_metrics.py runs/fp32 runs/fp16 --labels fp32 fp16 \
        --compare --out report_assets/precision_ablation.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (never cycled); text stays in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def read_metrics(run: str | Path) -> list[dict]:
    """Read ``metrics.jsonl``; after a resume, the latest record for each step wins."""
    path = Path(run)
    if path.is_dir():
        path = path / "metrics.jsonl"
    records: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["step"]] = record
    return [records[step] for step in sorted(records)]


def _style(ax, title: str, xlabel: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK_PRIMARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def _series(records: list[dict], key: str, x_key: str = "step") -> tuple[list, list]:
    xs, ys = [], []
    for record in records:
        value = record.get(key)
        if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
            xs.append(record[x_key])
            ys.append(value)
    return xs, ys


def _training_minutes(records: list[dict]) -> list[float]:
    """Cumulative measured training time per record (excludes gaps between resumed sessions)."""
    minutes, total, previous_step = [], 0.0, 0
    for record in records:
        total += record["sec_per_step"] * (record["step"] - previous_step)
        previous_step = record["step"]
        minutes.append(total / 60)
    return minutes


def plot_single_run(records: list[dict], out: Path, title: str) -> None:
    has_scale = any(record.get("loss_scale") for record in records)
    panels = 5 if has_scale else 4
    fig, axes = plt.subplots(panels, 1, figsize=(8, 2.6 * panels), sharex=True, facecolor=SURFACE)

    ax = axes[0]
    xs, ys = _series(records, "train_loss")
    ax.plot(xs, ys, color=SERIES[0], linewidth=1.5, label=f"train (final {ys[-1]:.3f})" if ys else "train")
    xs, ys = _series(records, "val_loss")
    ax.plot(xs, ys, color=SERIES[1], linewidth=2, marker="o", markersize=4,
            label=f"validation (final {ys[-1]:.3f})" if ys else "validation")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _style(ax, f"{title}: cross-entropy", "", "nats / token")

    ax = axes[1]
    xs, ys = _series(records, "lr")
    ax.plot(xs, ys, color=SERIES[0], linewidth=1.5)
    _style(ax, "Learning rate", "", "lr")

    ax = axes[2]
    xs, ys = _series(records, "grad_norm")
    ax.plot(xs, ys, color=SERIES[0], linewidth=1.2)
    _style(ax, "Pre-clipping global gradient norm", "", "L2 norm")

    ax = axes[3]
    xs, ys = _series(records, "tokens_per_sec")
    ax.plot(xs, ys, color=SERIES[0], linewidth=1.2)
    _style(ax, "Throughput", "", "tokens / s")

    if has_scale:
        ax = axes[4]
        xs, ys = _series(records, "loss_scale")
        ax.step(xs, ys, color=SERIES[0], linewidth=1.5, where="post")
        ax.set_yscale("log", base=2)
        skipped = records[-1].get("skipped_steps", 0)
        _style(ax, f"fp16 dynamic loss scale ({skipped} skipped updates)", "", "scale")

    axes[-1].set_xlabel("optimizer step", color=INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_comparison(runs: list[list[dict]], labels: list[str], out: Path, metric: str) -> None:
    if len(runs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs per comparison figure; facet the rest")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE)
    for index, (records, label) in enumerate(zip(runs, labels)):
        color = SERIES[index]
        minutes = _training_minutes(records)
        points = [(record["step"], minute, record.get(metric)) for record, minute in zip(records, minutes)]
        points = [(step, minute, value) for step, minute, value in points
                  if isinstance(value, (int, float)) and math.isfinite(value)]
        final = f" (final {points[-1][2]:.3f})" if points else ""
        axes[0].plot([p[0] for p in points], [p[2] for p in points], color=color, linewidth=2,
                     marker="o", markersize=3, label=f"{label}{final}")
        axes[1].plot([p[1] for p in points], [p[2] for p in points], color=color, linewidth=2,
                     marker="o", markersize=3, label=f"{label}{final}")
    name = metric.replace("_", " ")
    _style(axes[0], f"{name} vs. optimizer step", "optimizer step", "nats / token")
    _style(axes[1], f"{name} vs. measured training time", "minutes", "nats / token")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories or metrics.jsonl files")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--metric", default="val_loss", help="metric for --compare (val_loss or train_loss)")
    parser.add_argument("--title", default="Training run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    runs = [read_metrics(run) for run in args.runs]
    labels = args.labels or [Path(run).name for run in args.runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels must match the number of runs")
    if args.compare or len(runs) > 1:
        plot_comparison(runs, labels, Path(args.out), args.metric)
    else:
        plot_single_run(runs[0], Path(args.out), args.title)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
