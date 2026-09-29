"""Standardized final validation (Section 7.3): seed 42, 100 batches of 16 x 256.

    uv run python src/evaluate.py --model final_model.pt --out report_assets/final_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pa1.cli_common import add_model_arguments, load_model, resolve_device
from pa1.data import load_token_array
from pa1.evaluation import standardized_evaluation
from pa1.precision import PRECISION_CHOICES, resolve_precision


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--val-data", default="data/tinystories/data/validation.bin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default="fp32",
        help="compute precision; fp32 (default) matches a grader loading final_model.pt",
    )
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    model = load_model(args.model, device)
    result = standardized_evaluation(
        model,
        load_token_array(args.val_data),
        device,
        seed=args.seed,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        precision=precision,
    )
    result.update(model=str(args.model), precision=precision, device=str(device))
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
