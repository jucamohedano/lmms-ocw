"""Training helpers for TTW warmup.

Contains the data collator, gradient checkpointing utilities, trainer
configuration builders, and dataset construction — all pieces that do not
depend on ``TTWModel`` instance state and can be used as standalone functions.

The actual training loop orchestration (``_run_warmup_optimization`` and
``_run_trainer_warmup_optimization``) remains in ``_wrapper.py`` because it
is tightly coupled to ``TTWModel`` state (hyperparams, base model, logging).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from transformers.processing_utils import ProcessorMixin

# ── Data collator ────────────────────────────────────────────────────────


class TTWVisionDataCollator:
    """Project-local VLM collator that preserves TTW prompt masking semantics."""

    def __init__(self, processor: ProcessorMixin) -> None:
        self.processor = processor

    @staticmethod
    def _prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return messages up to (but not including) the assistant response."""
        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1]
        return messages

    @staticmethod
    def _extract_image(messages: list[dict[str, Any]]) -> Image.Image:
        """Extract the PIL image from a TTW warmup message list."""
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                    return item["image"]
        raise ValueError("TTW warmup example is missing its image content.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages_batch = [feature["messages"] for feature in features]
        images = [self._extract_image(messages) for messages in messages_batch]

        prompt_texts = [
            self.processor.apply_chat_template(
                self._prompt_messages(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]
        prompt_tokens = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        prompt_lens = prompt_tokens.attention_mask.sum(dim=1).tolist()

        full_texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in messages_batch
        ]
        batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row, prompt_len in enumerate(prompt_lens):
            labels[row, : int(prompt_len)] = -100
        batch["labels"] = labels
        return batch


# ── Gradient checkpointing ───────────────────────────────────────────────


def enable_gradient_checkpointing(model: torch.nn.Module) -> dict[str, Any]:
    """Enable checkpointing in the FSDP-safe non-reentrant mode."""
    from src.utils import get_logger

    _log = get_logger(__name__, rank_zero_only=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        _log.info("TTW: gradient checkpointing enabled (use_reentrant=False)")


def disable_gradient_checkpointing(model: torch.nn.Module) -> dict[str, Any]:
    """Disable gradient checkpointing after TTW warmup."""
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()


# ── Dataset construction ─────────────────────────────────────────────────


def build_warmup_training_dataset(
    image: Image.Image,
    warmup_captions: list[tuple[str, str]],
    format_chat_fn: Callable[..., list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Convert TTW caption pairs into trainer-ready chat examples.

    Args:
    ----
        image: The input image.
        warmup_captions: List of ``(prompt, caption)`` pairs.
        format_chat_fn: ``(image, prompt, caption) -> messages``.

    Returns:
    -------
        list[dict[str, Any]]: List of dictionaries with chat messages.

    """
    return [
        {"messages": format_chat_fn(image, prompt_text, caption_text)}
        for prompt_text, caption_text in warmup_captions
    ]


# ── Trainer configuration ────────────────────────────────────────────────


def build_trainer_config(
    output_dir: str,
    model_dtype: torch.dtype,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    lora_backend: str = "peft",
) -> Any:  # noqa: ANN401
    """Build TRL SFTConfig for TTW warmup.

    Uses a minimal config for PEFT (matches original TTW: plain AdamW).
    Adds Unsloth-specific args only when ``lora_backend == "unsloth"``.

    Returns
    -------
        SFTConfig: The TRL SFTConfig instance.

    """
    try:
        from trl import SFTConfig
    except ImportError as exc:
        raise ImportError(
            "TTW trainer warmup requires TRL. Install it with `pip install trl`."
        ) from exc

    bf16_enabled = model_dtype == torch.bfloat16
    fp16_enabled = model_dtype == torch.float16
    config_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": batch_size,
        "gradient_checkpointing": True,
        # The default reentrant mode (use_reentrant=True) re-runs the forward pass
        # inside the backward, triggering an unexpected FSDP parameter gather that
        # conflicts with FSDP's internal state and causes assertion errors or deadlocks.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},  # FSDP-safe non-reentrant mode
        "learning_rate": lr,
        "weight_decay": 0.0,
        "bf16": bf16_enabled,
        "fp16": fp16_enabled,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "dataloader_num_workers": 8,
        "max_length": None,
        "optim": "adamw_torch",
    }
    if lora_backend == "unsloth":
        config_kwargs.update(
            {
                "optim": "adamw_torch_fused",
                "average_tokens_across_devices": False,
            }
        )
    return SFTConfig(**config_kwargs)


def build_trainer_kwargs(
    trainer_class: type,
    model: torch.nn.Module,
    processor: ProcessorMixin,
    collator: TTWVisionDataCollator,
    train_dataset: Any,  # noqa: ANN401  # typing.Dataset is complex with generic params
    config: Any,  # noqa: ANN401  # SFTConfig but importing at runtime to avoid hard dependency
) -> dict[str, Any]:
    """Support TRL versions that renamed ``tokenizer`` → ``processing_class``."""
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "data_collator": collator,
        "args": config,
    }
    init_params = inspect.signature(trainer_class.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in init_params:
        trainer_kwargs["tokenizer"] = processor
    return trainer_kwargs


def get_unsloth_response_templates(model_name: str) -> tuple[str, str] | None:
    """Return assistant boundary markers for Unsloth response-only masking."""
    if "Qwen" in model_name:
        return "<|im_start|>user\n", "<|im_start|>assistant\n"
    return None
