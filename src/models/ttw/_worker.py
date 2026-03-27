"""Concurrent TTW warmup worker for ``ProcessPoolExecutor``.

Unified worker that supports both LoRA and SVF adaptation methods.
Each worker process loads its own base model + adaptation on ``cuda:<gpu_id>``.
Workers are isolated: own CUDA context, own optimizer, no NCCL, no DDP.

Top-level functions must be picklable for the ``spawn`` start method.
"""

from __future__ import annotations

import fcntl
import io
import os
import tempfile
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from torch.optim import AdamW
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

from src.models.ttw._batch import build_training_batch, format_chat
from src.models.ttw._liger import run_liger_training, try_create_liger_loss
from src.utils import get_logger


def image_bytes_to_rgb(image_bytes: bytes) -> Image.Image:
    """PNG bytes -> RGB PIL Image for build_training_batch."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


log = get_logger(__name__, rank_zero_only=True)

# ── Worker-local globals (set once by init_worker, reused across calls) ─────

_worker_model = None
_worker_processor = None
_worker_initial_state: dict[str, torch.Tensor] | None = None
_worker_liger_loss_fn = None
_worker_gpu_id = 0
_worker_filter_fn: Callable[[str], bool] | None = None


# ── State filtering helpers ────────────────────────────────────────────────


def _lora_filter(key: str) -> bool:
    """Filter keys for LoRA trainable parameters."""
    return "lora_" in key


def _svf_filter(key: str) -> bool:
    """Filter keys for SVF trainable parameters (S matrices + connector)."""
    return "trainable_svf_S" in key or "visual.merger." in key


def _get_filter_fn(method: str) -> Callable[[str], bool]:
    """Return the appropriate state-dict filter for the given method."""
    if method == "lora":
        return _lora_filter
    if method == "svf":
        return _svf_filter
    raise ValueError(f"Unknown TTW method: {method}")


# ── Model-specific initialization ──────────────────────────────────────────


def _init_lora(model: torch.nn.Module, lora_config_dict: dict[str, Any]) -> None:
    """Apply LoRA adaptation to the model using PEFT and unfreeze connector."""
    from peft import LoraConfig, get_peft_model

    global _worker_model, _worker_initial_state
    _worker_model = get_peft_model(model, LoraConfig(**lora_config_dict))

    # Unfreeze visual.merger (connector) - same as SVF path
    owner = model.model if hasattr(model, "model") else model
    for p in owner.visual.merger.parameters():
        p.requires_grad = True

    # Snapshot initial LoRA weights for reset between images
    _worker_initial_state = {
        k: v.cpu().clone() for k, v in _worker_model.state_dict().items() if _worker_filter_fn(k)
    }
    log.info(
        "TTW LoRA worker: snapshot %d tensors for reset",
        len(_worker_initial_state),
    )


def _init_svf(model: torch.nn.Module, svf_rank: int) -> list:
    """Apply SVF adaptation to the model and unfreeze connector."""
    from src.models.apply_svf_to_llm import apply_svf_to_llm

    global _worker_model, _worker_initial_state
    svf_layers = apply_svf_to_llm(model, rank=int(svf_rank))

    # Freeze all, then unfreeze SVF S matrices + connector
    for p in model.parameters():
        p.requires_grad = False
    for svf in svf_layers:
        svf.S.requires_grad_(True)

    # Unfreeze visual.merger (Qwen2-VL style)
    owner = model.model if hasattr(model, "model") else model
    for p in owner.visual.merger.parameters():
        p.requires_grad = True

    # Snapshot initial SVF state for reset between images
    _worker_initial_state = {
        k: v.cpu().clone() for k, v in _worker_model.state_dict().items() if _worker_filter_fn(k)
    }
    log.info(
        "TTW SVF worker: snapshot %d tensors for reset",
        len(_worker_initial_state),
    )


def init_worker(
    model_path: str,
    method: str,
    config: dict[str, Any],
    model_dtype_str: str,
    gpu_id: int = 0,
) -> None:
    """Initialize the worker once at ``ProcessPoolExecutor`` creation.

    Loads the base model on ``cuda:<gpu_id>`` and applies the specified
    adaptation method (LoRA or SVF).

    Args:
    ----
        model_path: Path to the pretrained model.
        method: Adaptation method (``"lora"`` or ``"svf"``).
        config: Method-specific config dict:
            - LoRA: ``{"lora_config": dict}`` from ``get_lora_config_dict()``
            - SVF: ``{"svf_rank": int}``
        model_dtype_str: Model dtype string (e.g., ``"bfloat16"``).
        gpu_id: CUDA device ID for this worker.

    Returns:
    -------
        None

    """
    global _worker_model, _worker_processor, _worker_liger_loss_fn
    global _worker_gpu_id, _worker_filter_fn

    _worker_gpu_id = int(gpu_id)
    _worker_filter_fn = _get_filter_fn(method)

    torch.cuda.set_device(_worker_gpu_id)
    dtype = getattr(torch, model_dtype_str)

    # Serialize loading so only one worker loads at a time per physical GPU.
    lock_path = os.path.join(tempfile.gettempdir(), f"ttw_warmup_init_cuda{_worker_gpu_id}.lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Load base model
            from transformers import Qwen2VLForConditionalGeneration

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

            # Apply adaptation method
            if method == "lora":  # using PEFT
                _init_lora(_worker_model, config["lora_config"])
            elif method == "svf":
                _init_svf(_worker_model, config["svf_rank"])
            else:
                raise ValueError(f"Unknown TTW method: {method}")

            _worker_processor = AutoProcessor.from_pretrained(model_path)

            # Initialize Liger loss (optional for both methods)
            _worker_liger_loss_fn = try_create_liger_loss()
            if _worker_liger_loss_fn is not None:
                log.info("TTW %s worker: using Liger fused linear CE", method)
            else:
                log.info("TTW %s worker: using HF outputs.loss fallback", method)

            log.info("TTW %s worker initialized on cuda:%d", method, _worker_gpu_id)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ── Public API ───────────────────────────────────────────────────────────────


def serialize_image(image: Image.Image) -> bytes:
    """PIL Image -> PNG bytes (picklable across spawn boundary)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def run_warmup(
    image_bytes: bytes,
    warmup_captions: list[tuple[str, str]],
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: bool = True,
) -> dict[str, torch.Tensor]:
    """Train adaptation parameters on one image and return state dict on CPU.

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

    Returns:
    -------
        Dict of adapted state tensors (filtered by method) on CPU.

    """
    global _worker_model, _worker_processor, _worker_initial_state
    global _worker_liger_loss_fn, _worker_filter_fn

    assert _worker_model is not None, "Worker not initialised — call init_worker first"
    assert _worker_filter_fn is not None, "Filter function not set"

    _reset_adaptation_weights()

    image = image_bytes_to_rgb(image_bytes)
    model = _worker_model
    processor = _worker_processor
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    if _worker_liger_loss_fn is not None:
        # Liger path (works for both LoRA and SVF)
        run_liger_training(
            model,
            processor,
            image,
            warmup_captions,
            epochs,
            lr,
            batch_size,
            grad_accum,
            _worker_liger_loss_fn,
            format_chat,
        )
    else:
        # Unified HF fallback path (works for both LoRA and SVF)
        _run_hf_fallback_training(
            model,
            processor,
            image,
            warmup_captions,
            epochs,
            lr,
            batch_size,
            grad_accum,
            device,
            model_dtype,
        )

    torch.cuda.empty_cache()

    # Return filtered state dict on CPU
    return {k: v.cpu() for k, v in model.state_dict().items() if _worker_filter_fn(k)}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _reset_adaptation_weights() -> None:
    """Restore adaptation parameters to their initial (pre-warmup) values."""
    global _worker_model, _worker_initial_state, _worker_filter_fn
    assert _worker_initial_state is not None
    with torch.no_grad():
        state = _worker_model.state_dict()
        for k, v in _worker_initial_state.items():
            if k in state:
                state[k].copy_(v.to(state[k].device))


def _run_hf_fallback_training(
    model: torch.nn.Module,
    processor: ProcessorMixin,
    image: Image.Image,
    warmup_captions: list[tuple[str, str]],
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: bool,
    device: torch.device,
    model_dtype: torch.dtype,
) -> None:
    """Unified HF fallback training loop (when Liger unavailable)."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, foreach=False)

    model.train()
    with torch.enable_grad(), torch.autocast("cuda", dtype=model_dtype):
        for _ in range(epochs):
            for start in range(0, len(warmup_captions), batch_size):
                batch_captions = warmup_captions[start : start + batch_size]

                # Build full batch once to compute total tokens for normalization
                batch_full = build_training_batch(
                    processor, image, batch_captions, device, format_chat
                )
                total_tokens = (batch_full["labels"] != -100).sum().item()
                if total_tokens == 0:
                    continue

                if grad_accum:
                    for caption_pair in batch_captions:
                        micro_batch = build_training_batch(
                            processor, image, [caption_pair], device, format_chat
                        )
                        out = model(**micro_batch)
                        loss = out.loss / total_tokens  # Normalize by TOTAL tokens
                        loss.backward()
                else:
                    out = model(**batch_full)
                    loss = out.loss  # Already batch-averaged by HF
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    model.eval()
    del optimizer, trainable_params
