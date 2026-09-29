"""Export ``final_model.pt`` (Section 7.4) from a training checkpoint.

    uv run python src/export_model.py --checkpoint runs/final/checkpoint.pt --out final_model.pt

Writes a plain CPU state dict whose floating tensors are float16 and checks
that it holds exactly the expected number of values.
"""

from __future__ import annotations

import argparse

from pa1.checkpoint import export_fp16_state_dict, read_model_state

EXPECTED_VALUES = 19_272_192


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="final_model.pt")
    parser.add_argument("--expected-values", type=int, default=EXPECTED_VALUES)
    args = parser.parse_args(argv)

    state, _ = read_model_state(args.checkpoint)
    exported = export_fp16_state_dict(state, args.out)
    total = sum(tensor.numel() for tensor in exported.values())
    if args.expected_values and total != args.expected_values:
        raise SystemExit(f"exported {total:,} values, expected {args.expected_values:,}")
    print(f"wrote {args.out}: {len(exported)} tensors, {total:,} values, float16 on CPU")


if __name__ == "__main__":
    main()
