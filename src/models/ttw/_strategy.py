"""Restore strategy, trainable configuration, and warmup execution profile for TTW.

Groups snapshot/restore, ``configure_trainable_params``, and
``WarmupExecutionProfile`` (how concurrent vs sequential warmup is chosen).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.utils import get_logger

log = get_logger(__name__, rank_zero_only=True)


def _tensor_bytes(t: torch.Tensor) -> int:
    """Return total bytes for both regular and ShardedTensors.

    ShardedTensor does not support .numel() or .element_size() directly;
    both are available via metadata instead.
    """
    if isinstance(t, ShardedTensor):
        meta = t.metadata()
        numel = math.prod(meta.size)
        element_size = torch.finfo(meta.tensor_properties.dtype).bits // 8
        return numel * element_size
    return t.numel() * t.element_size()


# ── FSDP helpers ─────────────────────────────────────────────────────


def _fsdp_ensure_initialized(model: torch.nn.Module) -> None:
    """Trigger FSDP lazy init before state-dict APIs, when needed."""
    if not isinstance(model, FSDP):
        return
    # FSDP1 sets _is_root on first forward; before that, some APIs can assert.
    if getattr(model, "_is_root", None) is not None:
        return
    device = next(model.parameters()).device
    with torch.no_grad():
        try:
            model(input_ids=torch.zeros(1, 1, dtype=torch.long, device=device))
        except Exception:
            # Expected for many VLM forwards; goal is only to trigger FSDP init path.
            log.debug("FSDP lazy-init dummy forward raised (expected)")


def _fsdp_get_state_dict(model: torch.nn.Module) -> dict:
    """Capture a CPU-offloaded, parallelism-agnostic sharded state dict.

    Uses PyTorch's unified API, which works across FSDP1/FSDP2/DDP/non-sharded
    models and returns canonical parameter FQNs.

    Note: with ``full_state_dict=False`` this stores local shards. Restore is
    intended for the same run topology (same world size / sharding config).
    """
    _fsdp_ensure_initialized(model)  # Avoid FSDP1 lazy init issues with state dict APIs.
    # Only get model state; CPU offload avoids GPU memory issues.
    model_state, _ = get_state_dict(
        model,
        (),
        options=StateDictOptions(
            full_state_dict=False, cpu_offload=True, ignore_frozen_params=False
        ),
    )
    return model_state


def _fsdp_set_state_dict(model: torch.nn.Module, state: dict) -> None:
    """Restore a state dict captured via ``_fsdp_get_state_dict``.

    This expects a sharded state dict from the same topology.
    """
    set_state_dict(
        model,
        (),
        model_state_dict=state,
        optim_state_dict=None,
        options=StateDictOptions(full_state_dict=False, strict=True),
    )


@dataclass
class RestoreContext:
    """Encapsulates the state needed to restore a model after TTW warmup."""

    mode: str  # "full" | "partial" | "lora_zero"
    state: dict[str, torch.Tensor] = field(default_factory=dict)


# ── Snapshot helpers ─────────────────────────────────────────────────────


# def log_snapshot_size(state: dict[str, torch.Tensor], label: str) -> None:
#     """Log the CPU snapshot size for TTW restore state."""
#     total_bytes = sum(_tensor_numel(t) * t.element_size() for t in state.values())
#     log.info(
#         "TTW %s snapshot captured: %d tensors (%.2f GiB on this rank)",
#         label,
#         len(state),
#         total_bytes / (1024**3),
#     )
def log_snapshot_size(state: dict[str, torch.Tensor], label: str) -> None:
    """Log the CPU snapshot size for TTW restore state."""
    total_bytes = sum(_tensor_bytes(t) for t in state.values())
    log.info(
        "TTW %s snapshot captured: %d tensors (%.2f GiB on this rank)",
        label,
        len(state),
        total_bytes / (1024**3),
    )


def snapshot_selected_state(
    model: torch.nn.Module, predicate: Callable[[str], bool], label: str
) -> dict[str, torch.Tensor]:
    """Snapshot only the tensors whose names satisfy *predicate*."""
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if predicate(name)
    }
    log_snapshot_size(state, label)
    return state


def snapshot_restore_state(
    model: torch.nn.Module,
    finetune_method: str,
    *,
    use_lora_zero_restore: bool,
) -> RestoreContext:
    """Capture the minimal state needed to restore the model after TTW.

    Args:
    ----
        model: The prepared model (possibly FSDP/DDP wrapped).
        finetune_method: One of ``"full"``, ``"svf"``, ``"lora"``.
        use_lora_zero_restore: If True and method is ``"lora"``, skip the CPU
            snapshot of LoRA weights and restore by **zeroing LoRA params**
            instead (used only for **concurrent LoRA** pool warmup). For SVF
            concurrent, this must be False.

    Returns:
    -------
        RestoreContext: Context containing mode and state for later restoration.

    """
    if finetune_method == "full":
        state = _fsdp_get_state_dict(model)
        log_snapshot_size(state, "full-model")
        return RestoreContext(mode="full", state=state)

    if finetune_method == "svf":
        return RestoreContext(
            mode="partial",
            state=snapshot_selected_state(
                model,
                lambda name: "trainable_svf_S" in name or "visual.merger." in name,
                "svf",
            ),
        )

    if finetune_method == "lora":
        if use_lora_zero_restore:
            log.info("TTW concurrent LoRA restore: no CPU snapshot needed")
            return RestoreContext(mode="lora_zero")
        return RestoreContext(
            mode="partial",
            state=snapshot_selected_state(
                model,
                lambda name: "lora_" in name or "visual.merger." in name,
                "lora",
            ),
        )

    raise ValueError(
        f"Unknown finetune_method: {finetune_method!r}. Expected one of: 'full', 'svf', 'lora'"
    )


# ── Restore helpers ──────────────────────────────────────────────────────


def _restore_selected_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore a subset of model tensors from CPU snapshots."""
    if not state:
        return
    current_state = model.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            if name in current_state:
                current_state[name].copy_(value.to(current_state[name].device))


def zero_lora_parameters(model: torch.nn.Module) -> None:
    """Reset LoRA weights to a zero delta between concurrent warmup inferences."""
    zeroed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.zero_()
                zeroed += 1
    log.debug("Zeroed %d LoRA tensors after concurrent warmup inference", zeroed)


def restore_model_state(
    model: torch.nn.Module,
    ctx: RestoreContext,
) -> None:
    """Restore the model state using the strategy captured in *ctx*.

    Args:
    ----
        model: The model to restore.
        ctx: The restore context from :func:`snapshot_restore_state`.

    """
    if ctx.mode == "full":
        _fsdp_set_state_dict(model, ctx.state)
        return
    if ctx.mode == "lora_zero":
        zero_lora_parameters(model)
        return
    _restore_selected_state(model, ctx.state)


# ── Trainable parameter configuration ───────────────────────────────────


def configure_trainable_params(
    model: torch.nn.Module,
    finetune_method: str,
    *,
    svf_layers: list | None = None,
    unfreeze_connector_fn: Callable | None = None,
    set_requires_grad_fn: Callable | None = None,
    get_vision_encoder_fn: Callable | None = None,
    lora_backend: str = "peft",
    get_connector_owner_fn: Callable | None = None,
    log_gpu_memory_fn: Callable | None = None,
    doc_id: int | None = None,
    rank: int = 0,
) -> list[torch.nn.Parameter]:
    """Apply the selected TTW finetuning policy and return trainable params.

    This is the single function that replaces the ``if/elif`` chain
    previously inside ``TTWModel._configure_trainable_params``.

    Args:
    ----
        model: The model to configure.
        finetune_method: One of ``"full"``, ``"svf"``, ``"lora"``.
        svf_layers: SVF layer references (required for ``"svf"`` mode).
        unfreeze_connector_fn: ``(model) -> None``; unfreezes the connector.
        set_requires_grad_fn: ``(model, bool) -> None``; bulk grad toggle.
        get_vision_encoder_fn: Returns the vision encoder modules (``"full"``).
        lora_backend: ``"peft"`` or ``"unsloth"`` (logging only).
        get_connector_owner_fn: ``(model) -> owner``; for unsloth diagnostics.
        log_gpu_memory_fn: Optional GPU memory logging callable.
        doc_id: Document ID for logging.
        rank: Process rank for logging.

    Returns:
    -------
        list[torch.nn.Parameter]: List of trainable parameters.

    """
    if log_gpu_memory_fn:
        log_gpu_memory_fn("TTW warmup: before freeze/unfreeze", doc_id=doc_id, rank=rank)

    if finetune_method == "svf":
        set_requires_grad_fn(model, False)
        for svf in svf_layers:
            svf.S.requires_grad_(True)
        if unfreeze_connector_fn:
            unfreeze_connector_fn(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        log.debug("SVF: %s trainable params", f"{sum(p.numel() for p in trainable_params):,}")

    elif finetune_method == "lora":
        set_requires_grad_fn(model, False)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        if unfreeze_connector_fn:
            unfreeze_connector_fn(model)
        if lora_backend == "unsloth" and get_connector_owner_fn is not None:
            connector_owner = get_connector_owner_fn(model)
            merger_params = list(connector_owner.visual.merger.parameters())
            connector_trainable = any(p.requires_grad for p in merger_params)
            log.info(
                "LoRA Unsloth: connector (visual.merger) trainable=%s (%d params)",
                connector_trainable,
                len(merger_params),
            )
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        log.debug("LoRA: %s trainable params", f"{sum(p.numel() for p in trainable_params):,}")

    elif finetune_method == "full":
        log.debug("Freezing vision encoders and unfreezing LLM and connector...")
        set_requires_grad_fn(model, True)
        if get_vision_encoder_fn:
            for module in get_vision_encoder_fn():
                for param in module.parameters():
                    param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    else:
        raise ValueError(f"Unknown finetune_method '{finetune_method}'")

    trainable_numel = sum(p.numel() for p in trainable_params)
    trainable_gib = sum(p.numel() * p.element_size() for p in trainable_params) / (1024**3)
    log.info(
        "TTW trainable: %s params (%.2f GiB weights, ~%.2f GiB opt+grad)",
        f"{trainable_numel:,}",
        trainable_gib,
        3 * trainable_gib,
    )
    if log_gpu_memory_fn:
        log_gpu_memory_fn("TTW warmup: after freeze/unfreeze", doc_id=doc_id, rank=rank)

    return trainable_params


# ── Warmup execution profile (pool vs sequential, LoRA zero-restore) ─────


class WarmupExecutionProfile(Enum):
    """How TTW warmup runs for a given finetune method and ``ttw_concurrent_warmups``."""

    SEQUENTIAL_MAIN = "sequential_main"
    """Main process only: ``N <= 1`` or unsupported concurrent combo."""

    CONCURRENT_LORA_POOL = "concurrent_lora_pool"
    """``ProcessPoolExecutor`` + LoRA workers; LoRA zero-restore."""

    CONCURRENT_SVF_POOL = "concurrent_svf_pool"
    """``ProcessPoolExecutor`` + SVF workers; snapshot uses partial SVF+merger CPU tensors."""

    @property
    def uses_process_pool(self) -> bool:
        """If True, warmup runs in a separate process pool instead of the main process."""
        return self is not WarmupExecutionProfile.SEQUENTIAL_MAIN

    @property
    def snapshot_uses_lora_zero_restore(self) -> bool:
        """If True, restore uses LoRA zero-restore (no CPU snapshot)."""
        return self is WarmupExecutionProfile.CONCURRENT_LORA_POOL


def resolve_warmup_execution_profile(
    finetune_method: str,
    concurrent_warmups: int,
) -> WarmupExecutionProfile:
    """Map ``(ttw_finetune_method, ttw_concurrent_warmups)`` to a single execution profile.

    - ``N <= 1`` → :attr:`SEQUENTIAL_MAIN`.
    - ``N > 1`` and ``lora`` → :attr:`CONCURRENT_LORA_POOL`.
    - ``N > 1`` and ``svf`` → :attr:`CONCURRENT_SVF_POOL`.
    - ``N > 1`` and ``full`` (or unknown) → :attr:`SEQUENTIAL_MAIN``.
    """
    n = int(concurrent_warmups)
    if n <= 1:
        return WarmupExecutionProfile.SEQUENTIAL_MAIN

    m = finetune_method.lower() if isinstance(finetune_method, str) else ""
    if m == "lora":
        return WarmupExecutionProfile.CONCURRENT_LORA_POOL
    if m == "svf":
        return WarmupExecutionProfile.CONCURRENT_SVF_POOL
    return WarmupExecutionProfile.SEQUENTIAL_MAIN
