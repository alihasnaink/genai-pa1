"""Generate text with temperature / top-p sampling, optionally over a decoding grid.

Single sample:
    uv run python src/generate.py --model final_model.pt --prompt "Once upon a time"

Decoding study for REPORT.md (every temperature x top-p pair, several samples each):
    uv run python src/generate.py --model final_model.pt --prompt "Once upon a time" \
        --temperatures 0.3 0.7 1.0 1.3 --top-ps 0.5 0.9 1.0 --num-samples 3 \
        --out report_assets/decoding_samples.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pa1.cli_common import add_model_arguments, load_model, resolve_device
from pa1.generation import generate

EOT_TOKEN = "<|endoftext|>"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--tokenizer", default="data/tinystories/tokenizer/tokenizer.json")
    parser.add_argument("--prompt", action="append", default=None, help="repeatable; default 'Once upon a time'")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    parser.add_argument("--top-ps", type=float, nargs="+", default=[1.0])
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-eot-stop", action="store_true", help="do not stop at <|endoftext|>")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model = load_model(args.model, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot_id = None if args.no_eot_stop else tokenizer.token_to_id(EOT_TOKEN)
    prompts = args.prompt or ["Once upon a time"]

    results = []
    for prompt, temperature, top_p in itertools.product(prompts, args.temperatures, args.top_ps):
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False).ids, dtype=torch.long, device=device
        )
        for sample_index in range(args.num_samples):
            # One seed per (setting, sample) so every configuration is reproducible on its own.
            generator = torch.Generator().manual_seed(args.seed + sample_index)
            output_ids = generate(
                model,
                prompt_ids,
                args.max_new_tokens,
                model.context_length,
                temperature=temperature,
                top_p=top_p,
                eot_token_id=eot_id,
                generator=generator,
            )
            new_ids = output_ids[len(prompt_ids) :].tolist()
            text = tokenizer.decode(output_ids.tolist(), skip_special_tokens=False)
            record = {
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "seed": args.seed + sample_index,
                "num_new_tokens": len(new_ids),
                "stopped_at_eot": eot_id is not None and bool(new_ids) and new_ids[-1] == eot_id,
                "text": text,
            }
            results.append(record)
            print(f"--- prompt={prompt!r} T={temperature} top_p={top_p} seed={record['seed']} "
                  f"tokens={record['num_new_tokens']} eot={record['stopped_at_eot']}")
            print(text, flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
