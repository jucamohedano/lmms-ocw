"""Concurrent TTW warmup worker for ProcessPoolExecutor.

Each worker process loads its own model + LoRA on ``cuda:0`` (the only visible
GPU, inherited from the DDP rank that spawned it).  Workers are fully isolated:
own CUDA context, own optimizer, no NCCL, no DDP.

Top-level functions must be picklable for ``spawn`` start method.
"""

from __future__ import annotations

import fcntl
import io
import os
import tempfile
from typing import Any

import torch
from PIL import Image
from torch.optim import AdamW
from transformers.processing_utils import ProcessorMixin

from src.utils import get_logger

log = get_logger(__name__, rank_zero_only=True)

# ── Worker-local globals (set once by _init_worker, reused across calls) ─────

_worker_model = None
_worker_processor = None
_worker_initial_lora_state: dict[str, torch.Tensor] | None = None
_worker_liger_loss_fn = None
_worker_gpu_id = 0


def _init_worker(
    model_path: str,
    lora_config_dict: dict[str, Any],
    model_dtype_str: str,
    gpu_id: int = 0,
) -> None:
    """Initialize the worker once at ``ProcessPoolExecutor`` creation.

    Loads the base model + PEFT LoRA on ``cuda:<gpu_id>``.  Stores initial LoRA
    weights so they can be reset between images.
    """
    global _worker_model, _worker_processor, _worker_initial_lora_state
    global _worker_liger_loss_fn, _worker_gpu_id

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    _worker_gpu_id = int(gpu_id)
    torch.cuda.set_device(_worker_gpu_id)
    dtype = getattr(torch, model_dtype_str)

    # Serialize loading so only one worker loads at a time per physical GPU.
    lock_path = os.path.join(tempfile.gettempdir(), f"ttw_warmup_init_cuda{_worker_gpu_id}.lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _worker_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map={"": _worker_gpu_id},
            )
            _worker_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if hasattr(_worker_model, "enable_input_require_grads"):
                _worker_model.enable_input_require_grads()

            _worker_model = get_peft_model(_worker_model, LoraConfig(**lora_config_dict))
            _worker_processor = AutoProcessor.from_pretrained(model_path)
            _worker_liger_loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="sum",
                accum_dtype=torch.float32,
            )

            _worker_initial_lora_state = {
                k: v.cpu().clone() for k, v in _worker_model.state_dict().items() if "lora_" in k
            }
            log.info(
                "TTW worker initialised on cuda:%d (%d LoRA params snapshot)",
                _worker_gpu_id,
                len(_worker_initial_lora_state),
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ── Public API ───────────────────────────────────────────────────────────────


def run_warmup(
    image_bytes: bytes,
    warmup_captions: list[tuple[str, str]],
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: bool = True,
) -> dict[str, torch.Tensor]:
    """Train LoRA on one image and return adapter state dict on CPU.

    Called per image from the main process via ``pool.submit``.

    Args:
    ----
        image_bytes: The serialized image bytes.
        warmup_captions: The list of warmup captions.
        epochs: The number of epochs to train for.
        lr: The learning rate to use for training.
        batch_size: The batch size to use for training.
        grad_accum: If True, process one caption per forward (low VRAM). If False,
            process batch_size captions in one forward (faster, higher VRAM).

    """
    global _worker_model, _worker_processor, _worker_initial_lora_state
    global _worker_liger_loss_fn
    assert _worker_model is not None, "Worker not initialised — call _init_worker first"
    assert _worker_liger_loss_fn is not None, "Liger loss not initialised in worker"

    _reset_lora_weights()

    image = _deserialize_image(image_bytes)
    model = _worker_model
    processor = _worker_processor
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, foreach=False)

    model.train()
    with torch.enable_grad(), torch.autocast("cuda", dtype=model_dtype):
        for _ in range(epochs):
            for start in range(0, len(warmup_captions), batch_size):
                batch_captions = warmup_captions[start : start + batch_size]
                total_tokens = (
                    (
                        _build_training_batch(processor, image, batch_captions, device)["labels"]
                        != -100
                    )
                    .sum()
                    .item()
                )
                if total_tokens == 0:
                    continue
                lm_head = model.get_output_embeddings()

                if grad_accum:
                    # One caption per forward to keep peak VRAM low.
                    for caption_pair in batch_captions:
                        micro_batch = _build_training_batch(
                            processor, image, [caption_pair], device
                        )
                        _forward_and_backward(model, micro_batch, lm_head, total_tokens)
                else:
                    # Full batch in one forward (batch_size captions, no grad accum).
                    batch = _build_training_batch(processor, image, batch_captions, device)
                    _forward_and_backward(model, batch, lm_head, total_tokens)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

    model.eval()
    del optimizer, trainable_params
    torch.cuda.empty_cache()

    adapter_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    return adapter_state


def _forward_and_backward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    lm_head: torch.nn.Module,
    total_tokens: int,
) -> None:
    """Run one forward + backward with Liger fused CE (no full logits)."""
    batch_for_forward = {k: v for k, v in batch.items() if k != "labels"}
    outputs = model(
        **batch_for_forward,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    shift_hidden = hidden[..., :-1, :].contiguous().view(-1, hidden.shape[-1])
    shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
    loss = _worker_liger_loss_fn(lm_head.weight, shift_hidden, shift_labels)
    loss = loss / total_tokens
    loss.backward()


# ── Helpers ──────────────────────────────────────────────────────────────────


def serialize_image(image: Image.Image) -> bytes:
    """PIL Image -> PNG bytes (picklable across spawn boundary)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _deserialize_image(data: bytes) -> Image.Image:
    """PNG bytes -> PIL Image (unpicklable across spawn boundary)."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def _reset_lora_weights() -> None:
    """Restore LoRA parameters to their initial (pre-warmup) values."""
    global _worker_model, _worker_initial_lora_state
    assert _worker_initial_lora_state is not None
    with torch.no_grad():
        state = _worker_model.state_dict()
        for k, v in _worker_initial_lora_state.items():
            if k in state:
                state[k].copy_(v.to(state[k].device))


def _format_chat(
    image: Image.Image,
    prompt: str,
    caption: str | None = None,
) -> list[dict[str, Any]]:
    """Qwen2-VL chat format (mirrors ``Qwen2VL.ttw_format_chat``)."""
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


def _build_training_batch(
    processor: ProcessorMixin,
    image: Image.Image,
    warmup_captions: list[tuple[str, str]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a training batch with prompt-masked labels.

    Mirrors ``TTWModel._ttw_build_training_batch`` but is self-contained
    (no reference to ``self._base``).
    """
    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    image_tensors: dict[str, torch.Tensor] = {}

    for count, (prompt_text, caption_text) in enumerate(warmup_captions):
        prompt_msg = _format_chat(image, prompt_text)
        prompt_only_text = processor.apply_chat_template(
            prompt_msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = processor(
            text=[prompt_only_text],
            images=[image],
            return_tensors="pt",
        )
        prompt_len = prompt_tokens.input_ids.shape[1]

        full_msg = _format_chat(image, prompt_text, caption_text)
        full_text = processor.apply_chat_template(
            full_msg,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_text_tokens = processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
        )

        if count == 0:
            for key, tensor in full_text_tokens.items():
                if key not in ("input_ids", "attention_mask", "labels"):
                    image_tensors[key] = tensor

        labels = full_text_tokens.input_ids.clone()
        labels[:, :prompt_len] = -100

        all_input_ids.append(full_text_tokens.input_ids)
        all_attention_masks.append(full_text_tokens.attention_mask)
        all_labels.append(labels)

    max_len = max(ids.shape[1] for ids in all_input_ids)
    pad_id = processor.tokenizer.pad_token_id or 0

    padded_ids, padded_masks, padded_labels = [], [], []
    for ids, mask, lab in zip(all_input_ids, all_attention_masks, all_labels, strict=True):
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((1, pad_len), pad_id, dtype=ids.dtype)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros(1, pad_len, dtype=mask.dtype)],
                dim=1,
            )
            lab = torch.cat(
                [lab, torch.full((1, pad_len), -100, dtype=lab.dtype)],
                dim=1,
            )
        padded_ids.append(ids)
        padded_masks.append(mask)
        padded_labels.append(lab)

    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.cat(padded_ids, dim=0).to(device),
        "attention_mask": torch.cat(padded_masks, dim=0).to(device),
        "labels": torch.cat(padded_labels, dim=0).to(device),
    }

    n_samples = len(all_input_ids)
    for key, tensor in image_tensors.items():
        repeat_dims = [1] * tensor.ndim
        repeat_dims[0] = n_samples
        batch[key] = tensor.repeat(*repeat_dims).to(device)

    return batch
