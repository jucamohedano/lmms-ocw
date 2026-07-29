"""Shared training-batch construction for TTW warmup.

Single implementation used by both the sequential path (``TTWModel``) and the
concurrent-worker path (``_worker.py``), eliminating the previous code
duplication between the former monolithic wrapper and worker batch builders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from transformers.processing_utils import ProcessorMixin


def format_chat(
    image: Image.Image,
    prompt: str,
    caption: str | None = None,
) -> list[dict[str, Any]]:
    """Qwen2-VL chat format (mirrors ``Qwen2VL.ttw_format_chat``).

    Used as the default ``format_chat_fn`` by :func:`build_training_batch` and
    as the worker-side chat formatter (workers don't have access to the base
    model instance).
    """
    msg: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if caption is not None:
        msg.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            }
        )
    return msg


def build_training_batch(
    processor: ProcessorMixin,
    image: Image.Image,
    warmup_captions: list[tuple[str, str]],
    device: torch.device | str,
    format_chat_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, torch.Tensor]:
    """Build a training batch with prompt-masked labels.

    Model-agnostic: uses ``format_chat_fn`` to handle prompt formatting, and
    ``apply_chat_template`` to find the boundary between prompt tokens (masked
    to -100) and caption tokens (loss computed on these).

    Args:
    ----
        processor: The model processor (tokenizer + image processor).
        image: The input image for all caption pairs.
        warmup_captions: List of ``(prompt, caption)`` pairs.
        device: Target device for the batch tensors.
        format_chat_fn: ``(image, prompt[, caption]) -> messages``.
            Defaults to :func:`format_chat` (Qwen2-VL format).

    Returns:
    -------
        dict[str, torch.Tensor]: Batch with input_ids, attention_mask, labels,
            and image tensors.

    """
    if format_chat_fn is None:
        format_chat_fn = format_chat

    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    image_tensors: dict[str, torch.Tensor] = {}

    for count, (prompt_text, caption_text) in enumerate(warmup_captions):
        # Step 1: Tokenize prompt-only (with assistant marker appended).
        # This tells us how many tokens are "prompt" (should be masked).
        prompt_msg = format_chat_fn(image, prompt_text)
        prompt_only_text = processor.apply_chat_template(
            prompt_msg, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = processor(text=[prompt_only_text], images=[image], return_tensors="pt")
        prompt_len = prompt_tokens.input_ids.shape[1]

        # Step 2: Tokenize full conversation (prompt + assistant response).
        full_msg = format_chat_fn(image, prompt_text, caption_text)
        full_text = processor.apply_chat_template(
            full_msg, tokenize=False, add_generation_prompt=False
        )
        full_text_tokens = processor(text=[full_text], images=[image], return_tensors="pt")

        # Extract non-text batch keys (pixel_values, image_grid_thw) from the
        # first item.  They are identical for all candidates (same image).
        if count == 0:
            for key, tensor in full_text_tokens.items():
                if key not in ("input_ids", "attention_mask", "labels"):
                    image_tensors[key] = tensor

        # Step 3: Build labels — mask prompt tokens to -100.
        labels = full_text_tokens.input_ids.clone()
        labels[:, :prompt_len] = -100

        all_input_ids.append(full_text_tokens.input_ids)
        all_attention_masks.append(full_text_tokens.attention_mask)
        all_labels.append(labels)

    # Pad and stack into a batch.
    max_len = max(ids.shape[1] for ids in all_input_ids)
    pad_id = processor.tokenizer.pad_token_id or 0

    padded_ids, padded_masks, padded_labels = [], [], []
    for ids, mask, lab in zip(all_input_ids, all_attention_masks, all_labels, strict=True):
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((1, pad_len), pad_id, dtype=ids.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros(1, pad_len, dtype=mask.dtype)], dim=1)
            lab = torch.cat([lab, torch.full((1, pad_len), -100, dtype=lab.dtype)], dim=1)
        padded_ids.append(ids)
        padded_masks.append(mask)
        padded_labels.append(lab)

    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.cat(padded_ids, dim=0).to(device),
        "attention_mask": torch.cat(padded_masks, dim=0).to(device),
        "labels": torch.cat(padded_labels, dim=0).to(device),
    }

    # Collect non-text tensors (pixel_values, image_grid_thw), expanding
    # across the batch.
    n_samples = len(all_input_ids)
    for key, tensor in image_tensors.items():
        repeat_dims = [1] * tensor.ndim
        repeat_dims[0] = n_samples
        batch[key] = tensor.repeat(*repeat_dims).to(device)

    return batch
