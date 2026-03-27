# ruff: noqa: I001
# src/models/ttw/_wrapper.py
"""Test-Time Warmup (TTW) wrapper.

Wraps any Model to add per-image TTW adaptation before inference.
Intercepts `generate_until` to:
  1. Save original model weights
  2. For each image:
    2.1 generate captions
    2.2 CLIP-filter
    2.3 warmup train
    2.4 infer
  3. Restore original weights.
"""

import csv
import gc
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import CLIPModel, CLIPProcessor
from transformers.processing_utils import ProcessorMixin
from trl import SFTConfig

from src.data.tasks import TaskInstance
from src.models._base import Model
from src.models.ttw._batch import build_training_batch
from src.models.ttw._config import TTW_AUXILIARY_PROMPTS, get_lora_config_dict
from src.models.ttw._strategy import (
    RestoreContext,
    configure_trainable_params,
    resolve_warmup_execution_profile,
    restore_model_state,
    snapshot_restore_state,
)
from src.models.ttw._training import TTWVisionDataCollator as _TTWVisionDataCollator
from src.models.ttw._training import (
    build_trainer_config,
    build_trainer_kwargs,
    build_warmup_training_dataset,
    disable_gradient_checkpointing,
    enable_gradient_checkpointing,
    get_unsloth_response_templates,
)
from src.utils import get_logger

log = get_logger(__name__, rank_zero_only=True)

# Per-process CSV file and writer for GPU memory logging (lazily opened)
_gpu_csv_file = None
_gpu_csv_writer = None


# TTW_AUXILIARY_PROMPTS is now imported from src.models.ttw._config

# ── GPU memory logging (CSV = detail, console = fallback) ─────────────────────
#
# Default ``TTW_GPU_MEMORY_CSV=1``: per-GPU rows only (no console spam, avoids ``log.log`` /
# formatter issues with ``get_logger``). Open ``logs/gpu/ttw_gpu_memory_*.csv`` for analysis.
#
# ``TTW_GPU_MEMORY_CSV=0``: one compact console line per call via ``log.info`` / ``log.debug``
# (peak_used, max_used_frac, proc_sum, device count) — no per-GPU text.
#
# Call-site ``level``: DEBUG vs INFO only affects console when CSV is off.


def _gpu_memory_csv_enabled() -> bool:
    """Check if GPU memory CSV logging is enabled via environment variable."""
    return os.environ.get("TTW_GPU_MEMORY_CSV", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _ensure_gpu_csv_open(rank: int) -> None:
    """Open the GPU memory CSV file on first use."""
    global _gpu_csv_file, _gpu_csv_writer
    if _gpu_csv_file is not None:
        return
    log_dir = "logs/gpu"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = os.environ.get("SLURM_JOB_NAME", "local")
    job_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_name)
    if job_safe:
        path = os.path.join(log_dir, f"ttw_gpu_memory_{job_safe}_{timestamp}_rank{rank}.csv")
    else:
        path = os.path.join(log_dir, f"ttw_gpu_memory_{timestamp}_rank{rank}.csv")
    with open(path, newline="", encoding="utf-8") as _gpu_csv_file:
        _gpu_csv_writer = csv.writer(_gpu_csv_file)
        _gpu_csv_writer.writerow(
            [
                "timestamp",
                "rank",
                "doc_id",
                "phase",
                "gpu_id",
                "alloc_gib",
                "reserved_gib",
                "total_gib",
                "total_gpu_used_gib",
            ]
        )
        _gpu_csv_file.flush()
    log.info("TTW GPU memory CSV enabled: %s (disable with TTW_GPU_MEMORY_CSV=0)", path)


def _get_total_gpu_memory_gib() -> list[float] | None:
    """Get total used memory per GPU (all processes) via nvidia-smi. Returns None on failure."""
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return None
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,  # noqa: S603
        )
        if out.returncode != 0:
            return None
        # memory.used is in MiB
        return [float(x.strip()) / 1024 for x in out.stdout.strip().split("\n") if x.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _log_gpu_memory(
    label: str,
    doc_id: int | None = None,
    rank: int | None = None,
    *,
    level: int = logging.INFO,
) -> None:
    """Record GPU memory: detailed per-GPU rows in CSV (default), else one compact console line.

    Uses nvidia-smi for ``total_gpu_used_gib`` when available. When CSV is enabled, skips
    console lines for this call (detail is in the file). When CSV is off, uses ``log.info``
    or ``log.debug`` only (compatible with ``get_logger`` rank/format wrappers).

    See module header and ``TTW_GPU_MEMORY_CSV``.
    """
    if not torch.cuda.is_available():
        return
    r = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
    ts = datetime.now().isoformat()
    total_gpu_per_device = _get_total_gpu_memory_gib()
    n_devices = torch.cuda.device_count()
    csv_on = _gpu_memory_csv_enabled()

    proc_alloc_sum = 0.0
    max_used_ratio = 0.0
    peak_used_gib = 0.0

    for i in range(n_devices):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        proc_alloc_sum += alloc
        total_used = (
            total_gpu_per_device[i]
            if total_gpu_per_device and i < len(total_gpu_per_device)
            else None
        )
        if total_used is not None and total > 0:
            max_used_ratio = max(max_used_ratio, total_used / total)
            peak_used_gib = max(peak_used_gib, total_used)

        if csv_on:
            try:
                _ensure_gpu_csv_open(r)
                _gpu_csv_writer.writerow(
                    [
                        ts,
                        r,
                        doc_id if doc_id is not None else "",
                        label,
                        i,
                        f"{alloc:.2f}",
                        f"{reserved:.2f}",
                        f"{total:.2f}",
                        f"{total_used:.2f}" if total_used is not None else "",
                    ]
                )
                _gpu_csv_file.flush()
            except Exception as e:
                log.debug("Failed to write GPU CSV: %s", e)

    if csv_on:
        return

    msg = (
        "[TTW GPU] rank=%s doc=%s | %s | peak_used=%.2fGiB max_used_frac=%.2f "
        "proc_sum=%.2fGiB (%d devices)"
    )
    args = (
        r,
        str(doc_id) if doc_id is not None else "-",
        label,
        peak_used_gib,
        max_used_ratio if max_used_ratio > 0 else 0.0,
        proc_alloc_sum,
        n_devices,
    )
    if level <= logging.DEBUG:
        log.debug(msg, *args)
    else:
        log.info(msg, *args)


def _ttw_log_loss_to_wandb(
    loss: float,
    rank: int,
    world_size: int,
    global_step: int,
    device: torch.device | None = None,
) -> None:
    """Log TTW warmup loss (averaged across ranks) to WandB when enabled (--wandb_args).

    Only rank 0 logs. In distributed mode, loss is all-reduced and averaged first.
    All ranks must participate in all_reduce (collective) to avoid deadlock.
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if dist.is_initialized() and world_size > 1:
        loss_tensor = torch.tensor([loss], dtype=torch.float32, device=dev)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_to_log = loss_tensor.item() / world_size
    else:
        loss_to_log = loss

    if rank != 0:
        return

    try:
        import wandb

        if wandb.run is not None:
            wandb.log({"ttw_warmup/loss": loss_to_log}, step=global_step)
    except ImportError:
        log.warning("WandB not found, skipping loss logging.")


def _execute_concurrent_ttw_batches(
    requests: Sequence[TaskInstance],
    n_pool_workers: int,
    prepared_model: torch.nn.Module,
    restore_ctx: RestoreContext,
    *,
    pool: ProcessPoolExecutor,
    run_pool_warmup: Callable[..., dict[str, Any] | None],
    merge_worker_state: Callable[[torch.nn.Module, dict[str, torch.Tensor]], None],
    get_first_request_image: Callable[[TaskInstance], tuple[Image.Image | None, Any, int | None]],
    get_warmup_captions: Callable[..., list[tuple[str, str]]],
    serialize_image: Callable[[Image.Image], bytes],
    generate_single_request: Callable[[Any, TaskInstance], list[Any]],
    restore_model_state_fn: Callable[[torch.nn.Module, RestoreContext], None],
    ttw_epochs: int,
    ttw_lr: float,
    ttw_batch_size: int,
    ttw_grad_accum: bool,
    log_gpu_memory: Callable[..., None],
    get_total_gpu_memory_gib: Callable[[], list[float] | None],
    rank: int,
) -> list[Any]:
    """ProcessPool warmup in batches of ``n_pool_workers``; infer sequentially on main."""
    n = int(n_pool_workers)
    res: list[Any] = []
    for i in range(0, len(requests), n):
        batch = list(requests[i : i + n])
        items: list[tuple[bytes | None, list[tuple[str, str]], int | None]] = []
        for req in batch:
            image, task, doc_id = get_first_request_image(req)
            if image is not None:
                captions = get_warmup_captions(
                    image,
                    task_name=task,
                    doc_id=doc_id,
                )
                items.append((serialize_image(image), captions, doc_id))
            else:
                items.append((None, [], doc_id))

        futures: list[Any] = []
        for img_b, caps, _ in items:
            if img_b is not None and caps:
                futures.append(
                    pool.submit(
                        run_pool_warmup,
                        img_b,
                        caps,
                        ttw_epochs,
                        ttw_lr,
                        ttw_batch_size,
                        ttw_grad_accum,
                    )
                )
            else:
                futures.append(None)

        pending = {f for f in futures if f is not None}
        log_interval_count = 0
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            if pending and get_total_gpu_memory_gib() is not None:
                log_interval_count += 1
                log_gpu_memory(
                    f"concurrent warmup: logging #{log_interval_count}",
                    doc_id=items[0][2] if items else None,
                    rank=rank,
                    level=logging.DEBUG,
                )

        worker_states = [f.result() if f is not None else None for f in futures]

        for req, wstate, (_, _, doc_id) in zip(batch, worker_states, items, strict=True):
            if wstate is not None:
                merge_worker_state(prepared_model, wstate)
            log_gpu_memory(
                "generate_until: after warmup (before inference)",
                doc_id=doc_id,
                rank=rank,
            )
            single_result = generate_single_request(req)
            res.extend(single_result)
            log_gpu_memory("generate_until: after inference", doc_id=doc_id, rank=rank)
            restore_model_state_fn(prepared_model, restore_ctx)
            log_gpu_memory("generate_until: after restore", doc_id=doc_id, rank=rank)
            torch.cuda.empty_cache()
    return res


class TTWModel:
    """Wrap any Model to add Test-Time Warmup."""

    # Original TTW paper default hyperparameters:
    # - Learning rate: 1e-6 (AdamW)
    # - Batch size: 5
    # - Epochs: 2 per test image
    # - Candidates per prompt: 10
    # - Generation temperature: 0.75
    def __init__(
        self,
        base_model: Model,
        ttw_lr: float = 1e-6,  # 1e-4 if using LoRA
        ttw_epochs: int = 2,  # 5 if using LoRA
        ttw_batch_size: int = 5,  # half the 10 captions → 2 gradient steps per epoch
        ttw_num_candidates: int = 10,  # N candidates per prompt, CLIP picks best
        ttw_caption_temperature: float = 0.75,
        ttw_max_new_tokens: int = 128,
        clip_model_name: str | None = None,
        offline_caption_dir: str | None = None,
        ttw_finetune_method: str = "full",  # "full" or "svf" or "lora"
        ttw_lora_backend: str = "peft",  # "peft" or "unsloth"
        ttw_svf_rank: int = -1,  # SVD truncation rank (-1 = full)
        ttw_grad_accum: bool = False,  # micro-batch gradient accumulation (fallback for OOM)
        ttw_concurrent_warmups: int = 1,  # >1 enables ProcessPool workers (requires MPS)
    ) -> None:
        self._base = base_model
        self.ttw_lr = ttw_lr
        self.ttw_epochs = ttw_epochs
        self.ttw_batch_size = ttw_batch_size
        self.ttw_num_candidates = ttw_num_candidates
        self.ttw_caption_temperature = ttw_caption_temperature
        self.ttw_max_new_tokens = ttw_max_new_tokens
        self.offline_caption_dir = offline_caption_dir
        self._offline_captions: dict[str, dict[int, list[tuple[str, str]]]] = {}
        self._clip_model_name = clip_model_name or "openai/clip-vit-large-patch14-336"
        self._clip_model = None
        self._clip_processor = None
        self.ttw_finetune_method = ttw_finetune_method
        self.ttw_lora_backend = ttw_lora_backend
        self.ttw_svf_rank = ttw_svf_rank
        self.ttw_grad_accum = ttw_grad_accum
        self.ttw_concurrent_warmups = int(ttw_concurrent_warmups)
        self._warmup_pool = None
        self._svf_warmup_pool = None

        # SVF and LoRA are applied in Qwen2VL._transform_model_before_prepare (before FSDP/DDP).
        # For SVF, the base model stores _svf_layers for warmup.
        self._svf_layers = getattr(self._base, "_svf_layers", None)

    def _get_connector_owner(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the model object that owns visual.merger (connector)."""
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            return model.base_model.model
        if hasattr(model, "model"):
            return model.model
        return model

    def _unfreeze_connector(self, model: torch.nn.Module) -> None:
        """Unfreeze the vision-language connector (guaranteed in our multimodal setups)."""
        connector_owner = self._get_connector_owner(model)
        # TODO: change for other models, this is qwen2-vl specific
        for param in connector_owner.visual.merger.parameters():
            param.requires_grad = True

    def _init_clip(self) -> None:
        """Lazy load CLIP model only when needed to save memory."""
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(self._clip_model_name).to("cuda").eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)

    def _get_model_after_ddp_wrapping(self) -> torch.nn.Module:
        """Return the wrapped ``Model``'s prepared module.

        (same as :meth:`Model._get_prepared_model`).
        """
        return self._base._get_prepared_model()

    # ── Snapshot/restore – delegated to src.models.ttw._strategy ───────

    def _snapshot_restore_state(
        self, model: torch.nn.Module, *, use_lora_zero_restore: bool
    ) -> RestoreContext:
        """Capture the minimal state needed to restore the model after TTW."""
        return snapshot_restore_state(
            model,
            self.ttw_finetune_method,
            use_lora_zero_restore=use_lora_zero_restore,
        )

    def _restore_model_state(self, model: torch.nn.Module, ctx: RestoreContext) -> None:
        """Restore the model state from the TTW snapshot/reset strategy."""
        restore_model_state(model, ctx)

    def __getattr__(self, name: str) -> object:
        """Delegate everything not defined on TTWModel to the base model."""
        return getattr(self._base, name)

    def __delattr__(self, name: str) -> None:
        """Delegate attribute deletion to the base model if not on TTWModel."""
        if name in self.__dict__:
            super().__delattr__(name)
        else:
            delattr(self._base, name)

    def _set_requires_grad(self, model: torch.nn.Module, requires_grad: bool) -> None:
        """Set requires_grad for all model parameters."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _load_offline_captions(self, task_name: str) -> None:
        """Load pre-generated captions for a given task.

        Reads JSONL files from ``offline_caption_dir`` and caches the captions
        associated with ``task_name``.

        Args:
        ----
        task_name:
            Name of the task whose offline captions should be loaded.

        """
        import json
        import os

        if task_name in self._offline_captions:
            return

        self._offline_captions[task_name] = {}
        if not self.offline_caption_dir:
            return

        for f in os.listdir(self.offline_caption_dir):
            if f.endswith(f"_{task_name}_captions.jsonl"):
                path = os.path.join(self.offline_caption_dir, f)
                with open(path) as json_f:
                    for line_num, line in enumerate(json_f, start=1):
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            # A truncated trailing line can happen on abrupt interruption.
                            log.warning(
                                "Skipping malformed JSONL line %d in %s "
                                "while loading offline captions.",
                                line_num,
                                path,
                            )
                            continue
                        doc_id = data.get("doc_id")
                        if doc_id is None:
                            log.warning(
                                "Skipping JSONL line %d in %s without doc_id.",
                                line_num,
                                path,
                            )
                            continue
                        self._offline_captions[task_name][doc_id] = [
                            (c["prompt"], c["caption"]) for c in data["captions"]
                        ]
                log.info(
                    "Loaded %d offline captions for task %s",
                    len(self._offline_captions[task_name]),
                    task_name,
                )
                return

        log.warning(
            "Could not find offline captions for task %s in %s",
            task_name,
            self.offline_caption_dir,
        )

    def _get_warmup_captions(
        self, image: Image.Image, task_name: str | None = None, doc_id: int | None = None
    ) -> list[tuple[str, str]]:
        """Fetch offline warmup captions or generate them online with CLIP filtering."""
        warmup_captions = []
        if self.offline_caption_dir and task_name and doc_id is not None:
            if task_name not in self._offline_captions:
                self._load_offline_captions(task_name)

            if doc_id in self._offline_captions.get(task_name, {}):
                warmup_captions = self._offline_captions[task_name][doc_id]
                log.info("Using loaded offline captions for doc_id %s", doc_id)
            else:
                log.warning(
                    "No offline captions found for doc_id %s, falling back to generation...",
                    doc_id,
                )

        # online generation fallback if no offline warmup captions available
        if not warmup_captions:
            self._init_clip()
            log.debug("Generating caption candidates (%d per prompt)...", self.ttw_num_candidates)
            caption_results = self._base.ttw_generate_captions(
                image,
                TTW_AUXILIARY_PROMPTS,
                self.ttw_num_candidates,
                self.ttw_caption_temperature,
                self.ttw_max_new_tokens,
                batch_size=1,  # Default to 1 for online evaluation fallback
            )

            # CLIP-filter: pick best candidate per prompt
            log.debug("CLIP-filtering generated captions to find best match per prompt...")
            for prompt, candidates in caption_results:
                clip_inputs = self._clip_processor(
                    text=candidates,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to("cuda")
                with torch.no_grad():
                    clip_out = self._clip_model(**clip_inputs)
                scores = clip_out.logits_per_image[0]
                best = candidates[scores.argmax().item()]
                warmup_captions.append((prompt, best))
                log.debug("Prompt: %s\n  Best Caption: %s", prompt, best)
        return warmup_captions

    def _configure_trainable_params(
        self, model: torch.nn.Module, doc_id: int | None = None
    ) -> list[torch.nn.Parameter]:
        """Apply the selected TTW finetuning policy and return trainable params."""
        rank = getattr(self._base, "rank", 0)
        return configure_trainable_params(
            model,
            self.ttw_finetune_method,
            svf_layers=self._svf_layers,
            unfreeze_connector_fn=self._unfreeze_connector,
            set_requires_grad_fn=self._set_requires_grad,
            get_vision_encoder_fn=self._base.ttw_get_vision_encoder,
            lora_backend=self.ttw_lora_backend,
            get_connector_owner_fn=self._get_connector_owner,
            log_gpu_memory_fn=_log_gpu_memory,
            doc_id=doc_id,
            rank=rank,
        )

    # ── Training helpers – delegated to src.models.ttw._training ─────

    def _enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        """Enable checkpointing in the FSDP-safe non-reentrant mode."""
        enable_gradient_checkpointing(model)

    def _disable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        """Disable gradient checkpointing after TTW warmup."""
        disable_gradient_checkpointing(model)

    def _build_warmup_training_dataset(
        self,
        image: Image.Image,
        warmup_captions: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Convert TTW caption pairs into trainer-ready chat examples."""
        return build_warmup_training_dataset(
            image,
            warmup_captions,
            self._base.ttw_format_chat,
        )

    def _build_trainer_config(self, output_dir: str, model_dtype: torch.dtype) -> SFTConfig:
        """Build TRL SFTConfig for TTW warmup."""
        return build_trainer_config(
            output_dir,
            model_dtype,
            epochs=self.ttw_epochs,
            lr=self.ttw_lr,
            batch_size=self.ttw_batch_size,
            lora_backend=self.ttw_lora_backend,
        )

    def _build_trainer_kwargs(
        self,
        trainer_class: type,
        model: torch.nn.Module,
        processor: ProcessorMixin,
        collator: _TTWVisionDataCollator,
        train_dataset: object,
        config: SFTConfig,
    ) -> dict[str, object]:
        """Support TRL versions that renamed tokenizer -> processing_class."""
        return build_trainer_kwargs(
            trainer_class,
            model,
            processor,
            collator,
            train_dataset,
            config,
        )

    def _get_unsloth_response_templates(self) -> tuple[str, str] | None:
        """Return assistant boundary markers for Unsloth response-only masking."""
        model_name = str(getattr(self._base, "_model_name_or_path", ""))
        return get_unsloth_response_templates(model_name)

    def _run_trainer_warmup_optimization(
        self,
        model: torch.nn.Module,
        image: Image.Image,
        warmup_captions: list[tuple[str, str]],
        doc_id: int | None = None,
    ) -> None:
        """Run TTW warmup through TRL instead of a hand-written loop."""
        try:
            from trl import SFTTrainer
        except ImportError as exc:
            raise ImportError(
                "TTW trainer warmup requires TRL. Install it with `pip install trl`."
            ) from exc

        processor = self._base.processor
        rank = getattr(self._base, "rank", 0)
        world_size = getattr(self._base, "world_size", 1)
        device = next(model.parameters()).device
        model_dtype = getattr(self._base.model, "dtype", torch.bfloat16)
        train_dataset = self._build_warmup_training_dataset(image, warmup_captions)
        output_dir = os.path.join("logs", "ttw_trainer", f"rank{rank}")
        os.makedirs(output_dir, exist_ok=True)

        if self.ttw_lora_backend == "unsloth":
            try:
                from unsloth import FastVisionModel, unsloth_train
                from unsloth.trainer import UnslothVisionDataCollator
            except ImportError as exc:
                raise ImportError("TTW unsloth warmup requires the `unsloth` package.") from exc

            FastVisionModel.for_training(model)
            response_templates = self._get_unsloth_response_templates()
            collator_kwargs = {"completion_only_loss": True}
            if response_templates is not None:
                instruction_part, response_part = response_templates
                collator_kwargs.update(
                    {
                        "train_on_responses_only": True,
                        "instruction_part": instruction_part,
                        "response_part": response_part,
                    }
                )
            collator = UnslothVisionDataCollator(model, processor, **collator_kwargs)
        else:
            collator = _TTWVisionDataCollator(processor)

        config = self._build_trainer_config(output_dir, model_dtype)
        trainer_kwargs = self._build_trainer_kwargs(
            SFTTrainer,
            model,
            processor,
            collator,
            train_dataset,
            config,
        )

        model.train()
        _log_gpu_memory("TTW warmup: before trainer.train", doc_id=doc_id, rank=rank)

        trainer = SFTTrainer(**trainer_kwargs)
        try:
            with torch.enable_grad():
                if self.ttw_lora_backend == "unsloth":
                    train_result = unsloth_train(trainer)
                else:
                    train_result = trainer.train()
            if hasattr(train_result, "training_loss"):
                global_step = getattr(self, "_ttw_wandb_step", 0)
                _ttw_log_loss_to_wandb(
                    float(train_result.training_loss),
                    rank,
                    world_size,
                    global_step,
                    device=device,
                )
                self._ttw_wandb_step = global_step + 1
            _log_gpu_memory("TTW warmup: after trainer.train", doc_id=doc_id, rank=rank)
        finally:
            del trainer

        # going into inference mode, disable grad and checkpointing to save memory
        self._disable_gradient_checkpointing(model)
        model.eval()  # inference-ready
        torch.cuda.empty_cache()
        self._set_requires_grad(model, False)
        _log_gpu_memory("TTW warmup: done", doc_id=doc_id, rank=rank)
        log.info("TTW warmup finished.")

    def _run_warmup_optimization(
        self,
        model: torch.nn.Module,
        image: Image.Image,
        warmup_captions: list[tuple[str, str]],
        trainable_params: list[torch.nn.Parameter],
        doc_id: int | None = None,
    ) -> None:
        """Run the TTW optimization loop for a single image."""
        # LoRA fine-tuning uses the HF SFTTrainer API
        if self.ttw_finetune_method == "lora" and self.ttw_lora_backend in {"unsloth", "peft"}:
            self._run_trainer_warmup_optimization(
                model,
                image,
                warmup_captions,
                doc_id=doc_id,
            )
            return

        # For "full" and "svf" fine-tuning methods, run a custom optimization loop with AdamW
        optimizer = AdamW(trainable_params, lr=self.ttw_lr, foreach=False)

        log.info(
            "TTW warmup: method=%s epochs=%d lr=%s batch_size=%d",
            self.ttw_finetune_method,
            self.ttw_epochs,
            self.ttw_lr,
            self.ttw_batch_size,
        )
        model_dtype = getattr(self._base.model, "dtype", torch.bfloat16)

        self._enable_gradient_checkpointing(model)
        model.train()
        rank = getattr(self._base, "rank", 0)
        _log_gpu_memory("TTW warmup: before training loop", doc_id=doc_id, rank=rank)

        device = next(model.parameters()).device
        world_size = getattr(self._base, "world_size", 1)
        use_grad_accum = self.ttw_grad_accum
        with torch.enable_grad(), torch.autocast("cuda", dtype=model_dtype):
            for epoch in range(self.ttw_epochs):
                for start in range(0, len(warmup_captions), self.ttw_batch_size):
                    batch_captions = warmup_captions[start : start + self.ttw_batch_size]
                    step_loss = 0.0
                    if use_grad_accum:
                        # Build full batch once to compute total tokens for normalization
                        batch_full = self._ttw_build_training_batch(image, batch_captions)
                        total_tokens = (batch_full["labels"] != -100).sum().item()
                        if total_tokens == 0:
                            raise ValueError(
                                f"total_tokens is 0 for image {doc_id}. "
                                "This indicates an issue with the caption data or batch building. "
                                "Check that labels contain valid tokens (not all -100)."
                            )

                        for caption_pair in batch_captions:
                            micro_batch = self._ttw_build_training_batch(image, [caption_pair])
                            outputs = model(**micro_batch)
                            _log_gpu_memory(
                                "TTW warmup: after forward (grad_accum)",
                                doc_id=doc_id,
                                rank=rank,
                                level=logging.DEBUG,
                            )
                            loss = outputs.loss / total_tokens
                            loss.backward()
                            _log_gpu_memory(
                                "TTW warmup: after backward (grad_accum)",
                                doc_id=doc_id,
                                rank=rank,
                                level=logging.DEBUG,
                            )
                            step_loss += loss.detach().item()
                    else:
                        batch_inputs = self._ttw_build_training_batch(image, batch_captions)
                        _log_gpu_memory(
                            "TTW warmup: after build_batch (before forward)",
                            doc_id=doc_id,
                            rank=rank,
                            level=logging.DEBUG,
                        )
                        outputs = model(**batch_inputs)
                        _log_gpu_memory(
                            "TTW warmup: after forward",
                            doc_id=doc_id,
                            rank=rank,
                            level=logging.DEBUG,
                        )
                        loss = outputs.loss
                        loss.backward()
                        _log_gpu_memory(
                            "TTW warmup: after backward",
                            doc_id=doc_id,
                            rank=rank,
                            level=logging.DEBUG,
                        )
                        step_loss = loss.detach().item()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # tensor deallocated entirely
                    _log_gpu_memory(
                        "TTW warmup: after optimizer.step",
                        doc_id=doc_id,
                        rank=rank,
                        level=logging.DEBUG,
                    )
                    log.debug(
                        "TTW epoch %d/%d step %d loss=%.4f",
                        epoch + 1,
                        self.ttw_epochs,
                        start // self.ttw_batch_size + 1,
                        step_loss,
                    )
                    global_step = getattr(self, "_ttw_wandb_step", 0)
                    _ttw_log_loss_to_wandb(
                        step_loss,
                        self._base.rank,
                        world_size,
                        global_step,
                        device=device,
                    )
                    self._ttw_wandb_step = global_step + 1
                    torch.cuda.empty_cache()
                if epoch == 0:
                    _log_gpu_memory(
                        "TTW warmup: after epoch 1",
                        doc_id=doc_id,
                        rank=rank,
                        level=logging.DEBUG,
                    )

        self._disable_gradient_checkpointing(model)
        model.eval()
        del optimizer, trainable_params
        torch.cuda.empty_cache()
        self._set_requires_grad(model, False)
        _log_gpu_memory("TTW warmup: done", doc_id=doc_id, rank=rank)
        log.info("TTW warmup finished.")

    def _ttw_warmup(
        self,
        image: Image.Image,
        task_name: str | None = None,
        doc_id: int | None = None,
    ) -> None:
        """Run TTW warmup: fetch captions, configure trainables, and optimize."""
        model = self._get_model_after_ddp_wrapping()

        # Log model identity for debugging (TODO #4)
        log.debug(
            "TTW warmup: prepared model = %s (type: %s), is DDP: %s, is FSDP: %s",
            id(model),
            type(model).__name__,
            isinstance(model, DDP),
            "FSDP" in type(model).__name__,
        )

        log.info("Starting TTW Warmup for new image...")

        warmup_captions = self._get_warmup_captions(image, task_name=task_name, doc_id=doc_id)
        rank = getattr(self._base, "rank", 0)
        _log_gpu_memory(
            "TTW warmup: after get_warmup_captions (CLIP done)",
            doc_id=doc_id,
            rank=rank,
            level=logging.DEBUG,
        )
        trainable_params = self._configure_trainable_params(model, doc_id=doc_id)

        # Wrap in no_sync for DDP (svf/lora) to suppress gradient all-reduce.
        # Each rank adapts independently to its own image, so cross-rank gradient
        # averaging would mix gradients from different images (incorrect).
        # FSDP (full) is excluded because it requires gradient sync to complete
        # the sharded forward/backward pass.
        if isinstance(model, DDP):
            with model.no_sync():
                self._run_warmup_optimization(
                    model,
                    image,
                    warmup_captions,
                    trainable_params,
                    doc_id=doc_id,
                )
        else:  # FSDP
            self._run_warmup_optimization(
                model,
                image,
                warmup_captions,
                trainable_params,
                doc_id=doc_id,
            )

    def _get_first_request_image(self, req: TaskInstance) -> tuple[Image.Image | None, str, int]:
        """Extract the first PIL image and TTW metadata from a request."""
        args = req.args
        doc_to_visual_fn = args[2]
        doc_id = args[3]
        task = args[4]
        split = args[5]

        doc = self._base.task_dict[task][split][doc_id]
        visuals = doc_to_visual_fn(doc)
        image = next((vis for vis in visuals if isinstance(vis, Image.Image)), None)
        return image, task, doc_id

    def _generate_single_request(self, req: TaskInstance) -> list[Any]:
        """Delegate single-request generation to the base model."""
        return self._base.generate_until([req])

    def _ttw_build_training_batch(
        self,
        image: Image.Image,
        warmup_captions: list[tuple[str, str]],
    ) -> dict[str, torch.Tensor]:
        """Build a training batch from (prompt, caption) pairs with proper label masking.

        Delegates to :func:`src.models.ttw._batch.build_training_batch`.
        """
        log.debug("Building TTW training batch...")
        return build_training_batch(
            processor=self._base.processor,
            image=image,
            warmup_captions=warmup_captions,
            device="cuda",
            format_chat_fn=self._base.ttw_format_chat,
        )

    # ── Concurrent warmup pool helpers ─────────────────────────────────────
    #
    # Execution mode: ``resolve_warmup_execution_profile`` in ``_strategy.py``.
    # Batched pool infer/restore: module-level ``_execute_concurrent_ttw_batches``.
    #
    # Invariants (``ttw_concurrent_warmups > 1`` and method ``lora`` or ``svf``):
    # - LoRA: workers return CPU LoRA tensors; main merges via ``_merge_adapter``.
    # - SVF: workers return CPU ``trainable_svf_S`` + ``visual.merger`` tensors; main merges
    #   the same way (keys may need ``module.`` prefix under DDP).
    # - CUDA MPS time-slices the GPU; each worker holds a full model copy (MPS does not share
    #   weights).

    @staticmethod
    def _merge_worker_state_dict(
        model: torch.nn.Module, worker_state: dict[str, torch.Tensor]
    ) -> int:
        """Copy worker tensors into ``model.state_dict()`` (handles optional DDP ``module.``)."""
        current_state = model.state_dict()
        loaded = 0
        for k, v in worker_state.items():
            dest_key = None
            if k in current_state:
                dest_key = k
            elif f"module.{k}" in current_state:
                dest_key = f"module.{k}"
            if dest_key is not None:
                current_state[dest_key].copy_(v.to(current_state[dest_key].device))
                loaded += 1
        return loaded

    def _get_warmup_pool(self) -> ProcessPoolExecutor:
        """Lazily create the ProcessPool with persistent workers."""
        if self._warmup_pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from src.models.ttw._worker import init_worker as _init_worker

            ctx = mp.get_context("spawn")
            model_path = self._base._model_name_or_path
            lora_config = get_lora_config_dict()
            gpu_id = getattr(
                getattr(self._base, "accelerator", None),
                "local_process_index",
                0,
            )
            self._warmup_pool = ProcessPoolExecutor(
                max_workers=self.ttw_concurrent_warmups,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(
                    model_path,
                    "lora",
                    {"lora_config": lora_config},
                    "bfloat16",
                    gpu_id,
                ),
            )
            log.info(
                "Created warmup ProcessPool with %d workers for model %s",
                self.ttw_concurrent_warmups,
                model_path,
            )
        return self._warmup_pool

    def _get_svf_warmup_pool(self) -> ProcessPoolExecutor:
        """Lazily create the ProcessPool for SVF concurrent warmup."""
        if self._svf_warmup_pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from src.models.ttw._worker import init_worker as _init_svf_worker

            ctx = mp.get_context("spawn")
            model_path = self._base._model_name_or_path
            gpu_id = getattr(
                getattr(self._base, "accelerator", None),
                "local_process_index",
                0,
            )
            self._svf_warmup_pool = ProcessPoolExecutor(
                max_workers=self.ttw_concurrent_warmups,
                mp_context=ctx,
                initializer=_init_svf_worker,
                initargs=(
                    model_path,
                    "svf",
                    {"svf_rank": int(self.ttw_svf_rank)},
                    "bfloat16",
                    gpu_id,
                ),
            )
            log.info(
                "Created SVF warmup ProcessPool with %d workers for model %s (svf_rank=%s)",
                self.ttw_concurrent_warmups,
                model_path,
                self.ttw_svf_rank,
            )
        return self._svf_warmup_pool

    def _merge_adapter(
        self, model: torch.nn.Module, adapter_state: dict[str, torch.Tensor]
    ) -> None:
        """Load warmup worker's LoRA weights into the main process model."""
        loaded = self._merge_worker_state_dict(model, adapter_state)
        log.debug("Merged %d LoRA keys from worker into main model", loaded)

    def _merge_svf_worker_state(
        self, model: torch.nn.Module, svf_state: dict[str, torch.Tensor]
    ) -> None:
        """Load SVF worker S + connector tensors into the main process model."""
        loaded = self._merge_worker_state_dict(model, svf_state)
        log.debug("Merged %d SVF/connector keys from worker into main model", loaded)

    # ── generate_until ───────────────────────────────────────────────────

    def generate_until(self, requests: Sequence[TaskInstance]) -> list[Any]:
        """Process requests with TTW per-image adaptation.

        TTW adapts per-image, so we process one request at a time:
        save weights → warmup on image → infer → restore weights.

        When ``ttw_concurrent_warmups > 1`` and the finetune method is ``lora`` or ``svf``,
        warmup is offloaded to a ``ProcessPoolExecutor`` (persistent workers, one model each)
        so multiple images can warm up concurrently on the same GPU via CUDA MPS.

        Images are extracted via the doc_to_visual mechanism in TaskInstance.args:
        args = (context, gen_kwargs, doc_to_visual_fn, doc_id, task, split).
        """
        prepared_model = self._get_model_after_ddp_wrapping()
        rank = getattr(self._base, "rank", 0)
        N = self.ttw_concurrent_warmups

        profile = resolve_warmup_execution_profile(self.ttw_finetune_method, N)

        _log_gpu_memory(
            "generate_until: before restore-state snapshot", rank=rank, level=logging.DEBUG
        )
        restore_ctx = self._snapshot_restore_state(
            prepared_model,
            use_lora_zero_restore=profile.snapshot_uses_lora_zero_restore,
        )
        _log_gpu_memory(
            "generate_until: after restore-state snapshot", rank=rank, level=logging.DEBUG
        )
        res = []

        if not profile.uses_process_pool:
            # Sequential: N==1, or ``full`` / other modes (no worker pool)
            for req in requests:
                image, task, doc_id = self._get_first_request_image(req)
                if image is not None:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._ttw_warmup(image, task_name=task, doc_id=doc_id)
                _log_gpu_memory(
                    "generate_until: after warmup (before inference)", doc_id=doc_id, rank=rank
                )

                single_result = self._generate_single_request(req)
                res.extend(single_result)
                _log_gpu_memory("generate_until: after inference", doc_id=doc_id, rank=rank)

                self._restore_model_state(prepared_model, restore_ctx)
                _log_gpu_memory("generate_until: after restore", doc_id=doc_id, rank=rank)
                torch.cuda.empty_cache()

            return res

        # Concurrent LoRA or SVF — batching + infer via _execute_concurrent_ttw_batches
        from src.models.ttw._worker import run_warmup as _run_pool_warmup
        from src.models.ttw._worker import serialize_image

        if self.ttw_finetune_method == "lora":
            pool = self._get_warmup_pool()
            merge_fn = self._merge_adapter
        else:
            pool = self._get_svf_warmup_pool()
            merge_fn = self._merge_svf_worker_state

        log.info(
            "Concurrent TTW (%s, profile=%s): processing %d requests in batches of %d",
            self.ttw_finetune_method,
            profile.name,
            len(requests),
            N,
        )

        return _execute_concurrent_ttw_batches(
            requests,
            N,
            prepared_model,
            restore_ctx,
            pool=pool,
            run_pool_warmup=_run_pool_warmup,
            merge_worker_state=merge_fn,
            get_first_request_image=self._get_first_request_image,
            get_warmup_captions=self._get_warmup_captions,
            serialize_image=serialize_image,
            generate_single_request=self._generate_single_request,
            restore_model_state_fn=self._restore_model_state,
            ttw_epochs=self.ttw_epochs,
            ttw_lr=self.ttw_lr,
            ttw_batch_size=self.ttw_batch_size,
            ttw_grad_accum=self.ttw_grad_accum,
            log_gpu_memory=_log_gpu_memory,
            get_total_gpu_memory_gib=_get_total_gpu_memory_gib,
            rank=rank,
        )
