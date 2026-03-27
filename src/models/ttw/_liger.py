"""Liger Fused Cross-Entropy integration for TTW warmup.

This module provides Liger-specific loss functions and helpers, decoupled from
the worker pattern. The worker pattern lives in `_worker.py`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torch.optim import AdamW
from transformers.processing_utils import ProcessorMixin

from src.models.ttw._batch import build_training_batch, format_chat
from src.utils import get_logger

log = get_logger(__name__, rank_zero_only=True)


def try_create_liger_loss() -> torch.nn.Module | None:
    """Build ``LigerFusedLinearCrossEntropyLoss``, or return *None* if import/init fails.

    Returns
    -------
        LigerFusedLinearCrossEntropyLoss if available, None otherwise.

    """
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100,
            reduction="sum",
            accum_dtype=torch.float32,
        )
    except Exception as exc:
        log.warning(
            "Liger fused linear CE unavailable (%s); falling back to HF loss.",
            exc,
        )
        return None


def liger_forward_backward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    total_tokens: int,
    liger_loss_fn: torch.nn.Module,
) -> None:
    """Perform one forward + backward pass using Liger fused CE.

    This avoids computing full vocab logits by using Liger's fused implementation.

    Args:
    ----
        model: The model to train (must have ``get_output_embeddings()``).
        batch: Batch dict with ``input_ids``, ``attention_mask``, ``labels``.
        total_tokens: Number of non-masked tokens for normalization.
        liger_loss_fn: Liger fused loss function.

    Returns:
    -------
        None

    """
    lm_head = model.get_output_embeddings()
    batch_for_forward = {k: v for k, v in batch.items() if k != "labels"}
    outputs = model(
        **batch_for_forward,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    shift_hidden = hidden[..., :-1, :].contiguous().view(-1, hidden.shape[-1])
    shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
    loss = liger_loss_fn(lm_head.weight, shift_hidden, shift_labels)
    loss = loss / total_tokens
    loss.backward()


def run_liger_training(
    model: torch.nn.Module,
    processor: ProcessorMixin,
    image: Image.Image,
    warmup_captions: list[tuple[str, str]],
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: bool,
    liger_loss_fn: torch.nn.Module,
    format_chat_fn: Callable | None = None,
    device: str | torch.device = "cuda",
) -> None:
    """Run the training loop using Liger fused CE loss.

    Args:
    ----
        model: The model to train.
        processor: The model processor for tokenization.
        image: Input image for all captions.
        warmup_captions: List of (prompt, caption) pairs.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size for grouping captions.
        grad_accum: If True, accumulate gradients over micro-batches.
        liger_loss_fn: Liger loss function.
        format_chat_fn: Chat formatting function.
        device: Device to use for training.

    Returns:
    -------
        None

    """
    if format_chat_fn is None:
        format_chat_fn = format_chat

    model_dtype = next(model.parameters()).dtype
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, foreach=False)

    model.train()
    with torch.enable_grad(), torch.autocast("cuda", dtype=model_dtype):
        for _ in range(epochs):
            for start in range(0, len(warmup_captions), batch_size):
                batch_captions = warmup_captions[start : start + batch_size]

                # Build full batch once to compute total tokens
                batch_full = build_training_batch(
                    processor, image, batch_captions, device, format_chat_fn
                )
                total_tokens = (batch_full["labels"] != -100).sum().item()
                if total_tokens == 0:
                    raise ValueError(
                        "total_tokens is 0. "
                        "This indicates an issue with the caption data or batch building. "
                        "Check that labels contain valid tokens (not all -100)."
                    )

                if grad_accum:
                    # Gradient accumulation: process each caption separately
                    for caption_pair in batch_captions:
                        micro_batch = build_training_batch(
                            processor, image, [caption_pair], device, format_chat_fn
                        )
                        liger_forward_backward(model, micro_batch, total_tokens, liger_loss_fn)
                else:
                    # Full batch forward
                    liger_forward_backward(model, batch_full, total_tokens, liger_loss_fn)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    model.eval()
    del optimizer, trainable_params
