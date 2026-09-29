"""Adapter boundary between the PA1 tests and student-written code.

Students may use any package layout, module names, or class names. Complete each
adapter by importing your implementation, constructing it with the supplied
arguments, injecting the supplied weights where applicable, and returning the
requested value. Keep the mathematics in your implementation, not in this file.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import sys

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pa1 import attention, checkpoint, data, layers, model, nn_utils, optim, rope  # noqa: E402


def run_load_token_array(path: str | Path) -> np.memmap:
    """Open ``path`` as a read-only, little-endian uint16 token stream.

    ``path`` names a raw flat token stream with no header. The returned object
    must remain an ``np.memmap`` opened with mode ``"r"`` and dtype ``<u2``;
    do not eagerly load or convert the complete file. The adapter should only
    import and call the student's loader.

    Returns:
        A one-dimensional, read-only memory map of little-endian uint16 IDs.
    """
    return data.load_token_array(path)


def run_get_batch(
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    sequence_length: int,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[Int[Tensor, "batch sequence"], Int[Tensor, "batch sequence"]]:
    """Sample next-token input/target windows from a flat token stream.

    Sample ``batch_size`` starts uniformly from
    ``[0, len(dataset) - sequence_length)`` using ``generator`` as the sole
    source of randomness. Each sampled slice contains ``sequence_length + 1``
    IDs. Return its first and last ``sequence_length`` IDs as ``x`` and ``y``.
    Both outputs must be ``torch.long`` tensors on ``device`` with shape
    ``[batch_size, sequence_length]``. The adapter should only forward arguments
    to the student's batching function.
    """
    return data.get_batch(dataset, batch_size, sequence_length, device, generator)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    """Run a student bias-free linear layer with an injected weight matrix.

    ``weights`` has canonical shape ``[d_out, d_in]`` and ``in_features`` may
    have arbitrary leading dimensions. Instantiate the student's module and
    copy ``weights`` into its sole matrix. The adapter must not perform the
    matrix multiplication itself.

    Returns:
        The module output with shape ``[..., d_out]``.
    """
    module = layers.Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({"weight": weights})
    return module(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, "vocab d_model"],
    token_ids: Int[Tensor, "..."],
) -> Float[Tensor, "... d_model"]:
    """Run a student embedding layer with the supplied embedding table.

    ``weights`` uses canonical shape ``[vocab_size, d_model]``. Instantiate the
    student's embedding, inject that table, and call it on integer
    ``token_ids`` of any shape. Do not perform indexing in the adapter.

    Returns:
        Embedded token vectors with shape ``[*token_ids.shape, d_model]``.
    """
    module = layers.Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({"weight": weights})
    return module(token_ids)


def run_rmsnorm(
    d_model: int,
    norm_eps: float,
    weights: Float[Tensor, "d_model"],
    in_features: Float[Tensor, "... d_model"],
) -> Float[Tensor, "... d_model"]:
    """Run RMSNorm using ``weights`` as its learned gain vector.

    Instantiate the student's RMSNorm with width ``d_model`` and numerical
    constant ``norm_eps``, then inject the ``[d_model]`` gain vector. Normalization
    belongs in the student module and acts only over the final dimension.

    Returns:
        A tensor with the same shape and floating dtype as ``in_features``.
    """
    module = layers.RMSNorm(d_model, norm_eps, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({"weight": weights})
    return module(in_features)


def run_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Apply the student's SiLU function elementwise.

    The adapter must call the student function rather than write the sigmoid
    formula itself. Return a tensor with the same shape as ``in_features``.
    """
    return layers.silu(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    gate_weight: Float[Tensor, "d_ff d_model"],
    down_weight: Float[Tensor, "d_model d_ff"],
    up_weight: Float[Tensor, "d_ff d_model"],
    in_features: Float[Tensor, "... d_model"],
) -> Float[Tensor, "... d_model"]:
    """Run SwiGLU with all three bias-free projection matrices injected.

    ``gate_weight`` and ``up_weight`` have shape ``[d_ff, d_model]``;
    ``down_weight`` has shape ``[d_model, d_ff]``. Instantiate the student's
    module, inject all three matrices, and call it. All projections are
    bias-free and the gated computation belongs in the module, not the adapter.

    Returns:
        A tensor with shape ``[..., d_model]``.
    """
    module = layers.SwiGLU(d_model, d_ff, device=gate_weight.device, dtype=gate_weight.dtype)
    module.load_state_dict({"gate.weight": gate_weight, "up.weight": up_weight, "down.weight": down_weight})
    return module(in_features)


def run_rope(
    head_dim: int,
    rope_theta: float,
    context_length: int,
    in_query_or_key: Float[Tensor, "... sequence head_dim"],
    token_positions: Int[Tensor, "... sequence"],
) -> Float[Tensor, "... sequence head_dim"]:
    """Apply adjacent-pair RoPE to a query or key tensor.

    ``in_query_or_key`` uses adjacent real/imaginary pairs in its final axis.
    ``token_positions`` may be one-dimensional or have batch dimensions that
    broadcast across leading head dimensions. Instantiate the student's RoPE
    object with the supplied cache limit and base, then call it; do not rotate
    values in the adapter.

    Returns:
        The rotated tensor with unchanged shape and floating dtype.
    """
    module = rope.RotaryPositionalEmbedding(rope_theta, head_dim, context_length, device=in_query_or_key.device)
    return module(in_query_or_key, token_positions)


def run_softmax(
    in_features: Float[Tensor, "..."], dim: int
) -> Float[Tensor, "..."]:
    """Apply the student's numerically stable softmax along ``dim``.

    ``dim`` may be positive or negative. The adapter simply forwards the tensor
    and dimension and returns an output of the same shape.
    """
    return attention.softmax(in_features, dim)


def run_scaled_dot_product_attention(
    queries: Float[Tensor, "... queries d_k"],
    keys: Float[Tensor, "... keys d_k"],
    values: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """Run scaled dot-product attention.

    A boolean ``True`` mask entry means attention is allowed. Query/key feature
    width is ``d_k``; values may use a different width ``d_v``. Leading
    dimensions and ``mask`` must broadcast, and query/key sequence lengths may
    differ. The adapter calls the student's attention function without
    computing scores itself.

    Returns:
        Attention values with shape ``[..., queries, d_v]``.
    """
    return attention.scaled_dot_product_attention(queries, keys, values, mask)


def run_grouped_query_self_attention(
    d_model: int,
    n_q_heads: int,
    n_kv_heads: int,
    context_length: int,
    rope_theta: float,
    q_proj_weight: Float[Tensor, "h_q_times_d_h d_model"],
    k_proj_weight: Float[Tensor, "h_kv_times_d_h d_model"],
    v_proj_weight: Float[Tensor, "h_kv_times_d_h d_model"],
    output_proj_weight: Float[Tensor, "d_model h_q_times_d_h"],
    in_features: Float[Tensor, "batch sequence d_model"],
    token_positions: Int[Tensor, "... sequence"] | None = None,
) -> Float[Tensor, "batch sequence d_model"]:
    """Run causal, RoPE-enabled grouped-query self-attention.

    Let ``head_dim = d_model // n_q_heads``. Projection shapes are
    ``q=[n_q_heads*head_dim,d_model]``,
    ``k/v=[n_kv_heads*head_dim,d_model]``, and
    ``output=[d_model,n_q_heads*head_dim]``. Instantiate the student's module,
    inject those matrices, and forward ``in_features`` and optional positions.
    The module must apply RoPE to Q/K, use a causal mask, compute grouped heads
    directly without materialized K/V repeats, and support both MHA and MQA
    boundary cases.

    Returns:
        Attention output with shape ``[batch, sequence, d_model]``.
    """
    module = attention.GroupedQuerySelfAttention(
        d_model,
        n_q_heads,
        n_kv_heads,
        context_length,
        rope_theta,
        device=q_proj_weight.device,
        dtype=q_proj_weight.dtype,
    )
    module.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "out_proj.weight": output_proj_weight,
        }
    )
    return module(in_features, token_positions=token_positions)


def run_transformer_block(
    d_model: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_ff: int,
    context_length: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence d_model"],
    token_positions: Int[Tensor, "... sequence"] | None = None,
    norm_eps: float = 1e-5,
) -> Float[Tensor, "batch sequence d_model"]:
    """Run one pre-RMSNorm Transformer block with frozen fixture weights.

    Canonical keys are ``attention.{q,k,v,out}_proj.weight``,
    ``attention_norm.weight``, ``ffn_norm.weight``, and
    ``ffn.{gate,up,down}.weight``. Instantiate a block using the explicit
    architectural arguments, translate these names if the student's state-dict
    differs, inject every tensor, and call the block. The adapter must not
    reproduce either residual path.

    Returns:
        Block output with shape ``[batch, sequence, d_model]``.
    """
    reference = next(iter(weights.values()))
    block = model.TransformerBlock(
        d_model,
        n_q_heads,
        n_kv_heads,
        d_ff,
        context_length,
        rope_theta,
        norm_eps,
        device=reference.device,
        dtype=reference.dtype,
    )
    # The student state-dict names match the canonical fixture names.
    block.load_state_dict(weights)
    return block(in_features, token_positions=token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    token_ids: Int[Tensor, "batch sequence"],
    token_positions: Int[Tensor, "... sequence"] | None = None,
    norm_eps: float = 1e-5,
) -> Float[Tensor, "batch sequence vocab"]:
    """Run the complete Transformer LM with frozen fixture weights.

    Canonical keys are ``token_embedding.weight``, ``blocks.{i}.<block key>``,
    ``final_norm.weight``, and ``lm_head.weight``. Construct the student model
    using the explicit dimensions, translate fixture keys if necessary, inject
    every weight, and call its forward method. Input/output embedding matrices
    must be distinct parameters.

    Returns:
        Unnormalized logits with shape ``[batch, sequence, vocab_size]``.
    """
    reference = next(iter(weights.values()))
    lm = model.TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        n_q_heads,
        n_kv_heads,
        d_ff,
        rope_theta,
        norm_eps,
        device=reference.device,
        dtype=reference.dtype,
    )
    lm.load_state_dict(weights)
    return lm(token_ids, token_positions=token_positions)


def get_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_ff: int,
    rope_theta: float,
    *,
    norm_eps: float = 1e-5,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Construct and return the student's complete Transformer LM module.

    Map the explicit architecture, device, and dtype arguments
    into whatever constructor or configuration type the student chose. Return a
    real ``torch.nn.Module`` so tests can inspect parameters and gradients; do
    not wrap or reimplement its forward computation in the adapter.
    """
    return model.TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        n_q_heads,
        n_kv_heads,
        d_ff,
        rope_theta,
        norm_eps,
        device=device,
        dtype=dtype,
    )


def run_cross_entropy(
    logits: Float[Tensor, "... vocab"], targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    """Return stable mean cross-entropy over every target position.

    ``targets`` matches all leading dimensions of ``logits`` and contains class
    IDs for the final vocabulary axis. The adapter calls the student's loss
    function; it must not delegate to PyTorch cross-entropy here.
    """
    return nn_utils.cross_entropy(logits, targets)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """Return the student's custom AdamW optimizer class, not an instance.

    The returned class must accept PyTorch-style parameter iterables and groups,
    expose serializable optimizer state, and implement the assignment equations
    without wrapping ``torch.optim.AdamW``.
    """
    return optim.AdamW


def run_get_lr_cosine_schedule(
    step: int,
    learning_rate_max: float,
    learning_rate_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """Evaluate linear warmup followed by cosine decay at ``step``.

    Forward all five scalar arguments to the student's schedule. It warms from
    zero to ``learning_rate_max``, decays to ``learning_rate_min`` by
    ``cosine_steps``, then remains at that floor. Return a Python float.
    """
    return optim.get_lr_cosine_schedule(
        step, learning_rate_max, learning_rate_min, warmup_steps, cosine_steps
    )


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> float:
    """Clip all available gradients by one global factor.

    Pass the iterable through to the student's implementation. Parameters with
    no gradient are ignored; all remaining gradients share one scale factor.

    Returns:
        The global L2 norm before clipping as a Python float.
    """
    return nn_utils.gradient_clipping(parameters, max_l2_norm)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    out: str | Path | BinaryIO,
) -> None:
    """Save the model, optimizer, next step, and both generator states.

    ``out`` may be a path or a writable binary stream. The adapter should only
    call the student's checkpoint function; it must not assemble the payload.
    """
    checkpoint.save_checkpoint(model, optimizer, next_step, train_generator, val_generator, out)


def run_load_checkpoint(
    src: str | Path | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
) -> int:
    """Restore a PA1 checkpoint and return the saved ``next_step``.

    ``src`` may be a path or a readable binary stream. The adapter should only
    call the student's loader; state restoration belongs in that function.
    """
    return checkpoint.load_checkpoint(src, model, optimizer, train_generator, val_generator)
