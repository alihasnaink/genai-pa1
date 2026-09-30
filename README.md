# CS 5326 Programming Assignment 1

## The Modern Transformer LM

Implement and train a decoder-only Transformer language model from scratch
using low-level PyTorch tensor operations. The fixed model uses bias-free
projections, pre-RMSNorm, adjacent-pair RoPE, SwiGLU, and grouped-query
attention.

Read `PA1.pdf` before beginning. It is the authoritative implementation,
testing, training, reporting, and submission contract.

## Repository workflow

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --frozen
uv run pytest
```

The locked environment supplies PyTorch, NumPy, einops, jaxtyping, tokenizers,
huggingface-hub, matplotlib, wrapt, and pytest.

Write your implementation inside the supplied `src/` directory. Connect it to
the tests by completing `tests/adapters.py`; adapters are glue code and must not
contain the assignment mathematics. Do not edit the public test files.

Grading may also use hidden tests through the same adapter interface. Hidden
tests cover only behavior documented in the assignment manual.

## Download TinyStories

The course dataset contains a fixed 8,192-token byte-level BPE tokenizer and the
complete pretokenized TinyStories train and validation splits:

```bash
uv run hf download alooboii/pa1-tinystories \
  metadata.json tokenizer/tokenizer.json \
  data/train.bin data/validation.bin \
  --repo-type dataset \
  --local-dir data/tinystories
```

The `.bin` files are flat little-endian `uint16` token streams. Open them with
`numpy.memmap`; do not convert the complete corpus into an in-memory `int64`
array. Public tests use synthetic data and require neither a network connection
nor the downloaded corpus.

## Suggested test order

```bash
uv run pytest tests/test_data.py
uv run pytest tests/test_layers.py
uv run pytest tests/test_rope.py
uv run pytest tests/test_attention.py
uv run pytest tests/test_model.py
uv run pytest tests/test_optim.py
uv run pytest tests/test_checkpoint.py
uv run pytest tests/test_restrictions.py
uv run pytest
```

## Submission

Complete `REPORT.md`, include at least one figure or visualization under
`report_assets/`, and export the final model's tensor-only FP16 CPU state
dictionary as `final_model.pt`. Then run:

```bash
bash make_submission.sh
```

The script prints the ordinary public-test results and creates
`submission.zip`, even if some tests fail. Rename the archive to
`<roll_number_pa1>.zip`, replacing `<roll_number>` with your roll number, and
upload it to the LMS. The archive contains only `src/`, `tests/adapters.py`,
`REPORT.md`, `report_assets/`, and `final_model.pt`; downloaded data, caches,
and full training checkpoints are excluded.

## Implementation and command-line tools

The implementation lives in `src/pa1/`; `tests/adapters.py` only imports it.

| Module | Contents |
|---|---|
| `layers.py` | `Linear`, `Embedding`, `RMSNorm`, `silu`, `SwiGLU` |
| `rope.py` | adjacent-pair `RotaryPositionalEmbedding` |
| `attention.py` | `softmax`, `scaled_dot_product_attention`, `GroupedQuerySelfAttention` |
| `model.py` | `TransformerBlock`, `TransformerLM`, `ModelConfig` |
| `nn_utils.py` / `optim.py` | `cross_entropy`, `gradient_clipping` / `AdamW`, `get_lr_cosine_schedule` |
| `data.py` / `checkpoint.py` | memmap loading, batch sampling / checkpoints, fp16 export |
| `precision.py` / `distributed.py` | fp16 mixed-precision policy / 2-GPU data parallelism |
| `generation.py` / `evaluation.py` | temperature + top-p decoding / validation and standardized eval |

**Mixed precision.** With `--precision fp16` (the default on CUDA), parameters, gradients,
and AdamW state stay float32; `torch.autocast` runs every matmul in float16; RMSNorm, RoPE,
the attention softmax, and cross-entropy compute in float32; the residual stream stays float32;
and `torch.amp.GradScaler` applies dynamic loss scaling (skipped overflow steps are logged).

**Multi-GPU.** `torchrun --nproc_per_node=2` splits every microbatch across GPUs and averages
gradients with one all-reduce per optimizer update (no `DistributedDataParallel`, which the
assignment's `torch.nn` restriction rules out). All ranks sample identical batches and keep the
shard for their rank, so a 2-GPU run sees exactly the data of a 1-GPU run with the same seeds.

```bash
# train (single GPU/CPU, or both Kaggle T4s); resumes automatically from <run-dir>/checkpoint.pt
uv run python src/train.py --run-dir runs/debug --num-steps 1000
uv run torchrun --standalone --nproc_per_node=2 src/train.py --run-dir runs/final \
    --batch-size 64 --grad-accum 4 --compile --time-limit-hours 11

uv run python src/precision_check.py --out report_assets/precision_benchmark.json  # dtype audit + fp32/fp16 benchmark
uv run python src/export_model.py --checkpoint runs/final/checkpoint.pt --out final_model.pt
uv run python src/evaluate.py --model final_model.pt --out report_assets/final_eval.json
uv run python src/generate.py --model final_model.pt --temperatures 0.7 1.0 --top-ps 0.9 1.0 --num-samples 3
uv run python src/plot_metrics.py runs/final --out report_assets/final_run.png
```

Every tool supports `--help`. `notebooks/kaggle_pa1.ipynb` runs the whole pipeline on Kaggle
(GPU T4 x2): environment setup, tests, precision check, overfit check, parallel ablations,
the final 2-GPU run with resume support, and post-processing.
