# src/models/_ttw_wrapper.py
"""Test-Time Warmup (TTW) wrapper

Wraps any Model to add per-image TTW adaptation before inference.
Intercepts `generate_until` to:
  1. Save original model weights
  2. For each image:
    2.1 generate captions
    2.2 CLIP-filter
    2.3 warmup train
    2.4 infer
  3. Restore original weights

"""
import csv
import gc
import os
from datetime import datetime

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.models._base import Model
from src.utils import get_logger

log = get_logger(__name__, rank_zero_only=True)

# Per-process CSV file and writer for GPU memory logging (lazily opened)
_gpu_csv_file = None
_gpu_csv_writer = None


def _fsdp_ensure_initialized(model):
    """Force FSDP lazy init by running a tiny dummy forward pass.

    FSDP defers internal setup (root detection, handle sharing) until the first
    ``forward()`` call.  Calling ``state_dict()`` before that corrupts
    ``_is_root`` flags and raises
    ``AssertionError: Non-root FSDP instance's `_is_root` should not have been
    set yet or should have been set to `False```.

    This is a no-op when the model is not FSDP-wrapped or has already been
    initialised.
    """
    if not isinstance(model, FSDP):
        return
    # Check if FSDP has already run its lazy init
    if getattr(model, "_is_root", None) is not None:
        return
    log.info("FSDP: triggering lazy init with a dummy forward pass...")
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
    with torch.no_grad():
        try:
            model(input_ids=dummy)
        except Exception as e:
            log.debug("FSDP dummy forward failed (expected): %s", e)
    log.info("FSDP: lazy init complete.")


def _fsdp_get_state_dict(model):
    """Save each rank's local shard to CPU via direct FlatParameter access.

    Avoids FULL_STATE_DICT's expensive All-Gather (each rank gathers the full
    15 GiB model, fragmenting GPU cache across images) and LOCAL_STATE_DICT's
    ShardedTensor machinery (which triggers cross-rank requires_grad consistency
    checks that fail when frozen and trainable params coexist).

    Directly accessing ``_flat_param.data`` gives the raw shard tensor on each
    rank — plain GPU→CPU copy, zero inter-rank communication.
    """
    _fsdp_ensure_initialized(model)
    if isinstance(model, FSDP):
        saved = {}
        for name, module in model.named_modules():
            fp = getattr(module, "_flat_param", None)
            if fp is not None:
                saved[name] = fp.data.detach().cpu().clone()
        return saved
    # DDP or single-GPU: plain clone to CPU
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _fsdp_set_state_dict(model, state):
    """Restore each rank's local shard from CPU via direct FlatParameter write.

    Mirrors _fsdp_get_state_dict: writes back to the FlatParameter's data
    storage directly, bypassing FSDP's state-dict machinery entirely.
    """
    if isinstance(model, FSDP):
        with torch.no_grad():
            for name, module in model.named_modules():
                fp = getattr(module, "_flat_param", None)
                if fp is not None and name in state:
                    fp.data.copy_(state[name].to(fp.device))
    else:
        model.load_state_dict(state)


# 10 auxiliary prompts (from the original TTW repo: utils.get_baseline_prompts())
TTW_AUXILIARY_PROMPTS = [
    "What is happening in this image?",
    "Describe the main subject of this image in detail.",
    "What objects or people are visible in this image?",
    "What actions are the subjects performing in this image?",
    "What does the background reveal about this image?",
    "What is unusual or unique about this image?",
    "What details in this image might someone easily overlook?",
    "Are there any signs, symbols, or text in this image? If so, what do they say?",
    "Explain the possible relationships or roles of the people, animals, or objects in  this scene. What hints or clues suggest these relationships?",
    "Based on visual cues, infer what might have happened just before and what  might happen right after this image was captured.",
]


def _ensure_gpu_csv_open(rank: int) -> None:
    """Open the GPU memory CSV file on first use."""
    global _gpu_csv_file, _gpu_csv_writer
    if _gpu_csv_file is not None:
        return
    log_dir = "logs/gpu"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"ttw_gpu_memory_{timestamp}_rank{rank}.csv")
    _gpu_csv_file = open(path, "w", newline="", encoding="utf-8")
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
        ]
    )
    _gpu_csv_file.flush()
    log.info("GPU memory CSV: %s", path)


def _log_gpu_memory(
    label: str,
    doc_id: int | None = None,
    rank: int | None = None,
) -> None:
    """Log per-GPU memory for debugging OOM issues. Optionally append to CSV in logs/gpu/."""
    if not torch.cuda.is_available():
        return
    r = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
    ts = datetime.now().isoformat()
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        log.info(
            "[Rank %d GPU %d] %s: %.2f/%.2f GiB allocated (%.2f GiB reserved)",
            r,
            i,
            label,
            alloc,
            total,
            reserved,
        )
        # Append to CSV for later analysis
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
                ]
            )
            _gpu_csv_file.flush()
        except Exception as e:
            log.debug("Failed to write GPU CSV: %s", e)


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


class TTWModel:
    """Wraps any Model to add Test-Time Warmup"""

    # Original TTW paper default hyperparameters:
    # - Learning rate: 1e-6 (AdamW)
    # - Batch size: 5
    # - Epochs: 2 per test image
    # - Candidates per prompt: 10
    # - Generation temperature: 0.75
    def __init__(
        self,
        base_model: Model,
        ttw_lr=1e-6,  # 1e-4 if using LoRA
        ttw_epochs=2,  # 5 if using LoRA
        ttw_batch_size=5,  # half the 10 captions → 2 gradient steps per epoch
        ttw_num_candidates=10,  # N candidates per prompt, CLIP picks best
        ttw_caption_temperature=0.75,
        ttw_max_new_tokens=128,
        clip_model_name=None,
        offline_caption_dir=None,
        ttw_finetune_method="full",  # "full" or "svf" or "lora"
        ttw_lora_backend="peft",  # "peft" or "unsloth"
        ttw_svf_rank=-1,  # SVD truncation rank (-1 = full)
        ttw_grad_accum=False,  # micro-batch gradient accumulation (fallback for OOM)
    ):
        self._base = base_model
        self.ttw_lr = ttw_lr
        self.ttw_epochs = ttw_epochs
        self.ttw_batch_size = ttw_batch_size
        self.ttw_num_candidates = ttw_num_candidates
        self.ttw_caption_temperature = ttw_caption_temperature
        self.ttw_max_new_tokens = ttw_max_new_tokens
        self.offline_caption_dir = offline_caption_dir
        self._offline_captions = {}  # {task_name: {doc_id: [(prompt, caption)]}}
        self._clip_model_name = clip_model_name or "openai/clip-vit-large-patch14-336"
        self._clip_model = None
        self._clip_processor = None
        self.ttw_finetune_method = ttw_finetune_method
        self.ttw_lora_backend = ttw_lora_backend
        self.ttw_svf_rank = ttw_svf_rank
        self.ttw_grad_accum = ttw_grad_accum

        # SVF and LoRA are applied in Qwen2VL._transform_model_before_prepare (before FSDP).
        # For SVF, the base model stores _svf_layers for warmup.
        self._svf_layers = getattr(self._base, "_svf_layers", None)

    def _unfreeze_connector(self, model) -> None:
        """Unfreeze the vision-language connector (guaranteed in our multimodal setups)."""
        if hasattr(model, "module"):
            model = model.module
        connector_owner = model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            # PEFT-wrapped path
            connector_owner = model.base_model.model
        elif hasattr(model, "model"):
            # Base Qwen2-VL path
            connector_owner = model.model

        # TODO: change for other models, this is qwen2-vl specific
        for param in connector_owner.visual.merger.parameters():
            param.requires_grad = True

    def _init_clip(self):
        """Lazy load CLIP model only when needed to save memory."""
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(self._clip_model_name).to("cuda").eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)

    def _get_prepared_model(self):
        """Return the model object used for warmup, snapshotting, and inference."""
        # unsloth returns the model as _model, otherwise self._base.model is the model
        return getattr(self._base, "_model", self._base.model)

    def _snapshot_model_state(self, model):
        """Snapshot the current model state, preserving local FSDP shards."""
        state = _fsdp_get_state_dict(model)
        total_bytes = sum(t.numel() * t.element_size() for t in state.values())
        log.info(
            "TTW snapshot captured: %d tensors (%.2f GiB on this rank)",
            len(state),
            total_bytes / (1024**3),
        )
        return state

    def _restore_model_state(self, model, state) -> None:
        """Restore the model state from a TTW snapshot."""
        _fsdp_set_state_dict(model, state)

    def __getattr__(self, name):
        """Delegate everything not defined on TTWModel to the base model."""
        return getattr(self._base, name)

    def __delattr__(self, name):
        """Delegate attribute deletion to the base model if not on TTWModel."""
        if name in self.__dict__:
            super().__delattr__(name)
        else:
            delattr(self._base, name)

    def _set_requires_grad(self, model, requires_grad: bool) -> None:
        """Set requires_grad for all model parameters."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _load_offline_captions(self, task_name: str):
        """Load pre-generated captions from JSONL files in offline_caption_dir for the given task."""
        import os, json

        if task_name in self._offline_captions:
            return

        self._offline_captions[task_name] = {}
        if not self.offline_caption_dir:
            return

        for f in os.listdir(self.offline_caption_dir):
            if f.endswith(f"_{task_name}_captions.jsonl"):
                path = os.path.join(self.offline_caption_dir, f)
                with open(path, "r") as json_f:
                    for line_num, line in enumerate(json_f, start=1):
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            # A truncated trailing line can happen on abrupt interruption.
                            log.warning(
                                f"Skipping malformed JSONL line {line_num} in {path} "
                                "while loading offline captions."
                            )
                            continue
                        doc_id = data.get("doc_id")
                        if doc_id is None:
                            log.warning(
                                f"Skipping JSONL line {line_num} in {path} without doc_id."
                            )
                            continue
                        self._offline_captions[task_name][doc_id] = [
                            (c["prompt"], c["caption"]) for c in data["captions"]
                        ]
                log.info(
                    f"Loaded {len(self._offline_captions[task_name])} offline captions for task {task_name}"
                )
                return

        log.warning(
            f"Could not find offline captions for task {task_name} in {self.offline_caption_dir}"
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
                log.info(f"Using loaded offline captions for doc_id {doc_id}")
            else:
                log.warning(
                    f"No offline captions found for doc_id {doc_id}, falling back to generation..."
                )

        # online generation fallback if no offline warmup captions available
        if not warmup_captions:
            self._init_clip()
            log.debug(f"Generating caption candidates ({self.ttw_num_candidates} per prompt)...")
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
                log.debug(f"Prompt: {prompt}\n  Best Caption: {best}")
        return warmup_captions

    def _configure_trainable_params(self, model, doc_id: int | None = None):
        """Apply the selected TTW finetuning policy and return trainable params."""
        rank = getattr(self._base, "rank", 0)
        _log_gpu_memory("TTW warmup: before freeze/unfreeze", doc_id=doc_id, rank=rank)
        if self.ttw_finetune_method == "svf":
            # SVF layers were replaced at init. Freeze everything, then unfreeze
            # only the S vectors and the connector.
            self._set_requires_grad(model, False)
            for svf in self._svf_layers:
                svf.S.requires_grad_(True)
            self._unfreeze_connector(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            log.debug(f"SVF: {sum(p.numel() for p in trainable_params):,} trainable params")
        elif self.ttw_finetune_method == "lora":
            self._set_requires_grad(model, False)
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            self._unfreeze_connector(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            log.debug(f"LoRA: {sum(p.numel() for p in trainable_params):,} trainable params")
        elif self.ttw_finetune_method == "full":
            log.debug("Freezing vision encoders and unfreezing LLM and connector...")
            self._set_requires_grad(model, True)
            for module in self._base.ttw_get_vision_encoder():
                for param in module.parameters():
                    param.requires_grad = False
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        trainable_numel = sum(p.numel() for p in trainable_params)
        trainable_gib = sum(p.numel() * p.element_size() for p in trainable_params) / (1024**3)
        log.info(
            "TTW trainable: %s params (%.2f GiB weights, ~%.2f GiB opt+grad)",
            f"{trainable_numel:,}",
            trainable_gib,
            3 * trainable_gib,
        )
        _log_gpu_memory("TTW warmup: after freeze/unfreeze", doc_id=doc_id, rank=rank)
        return trainable_params

    def _enable_gradient_checkpointing(self, model) -> None:
        """Enable checkpointing in the FSDP-safe non-reentrant mode."""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }  # otherwise sharding doesn't work
            )
            log.info("TTW: gradient checkpointing enabled (use_reentrant=False)")

    def _disable_gradient_checkpointing(self, model) -> None:
        """Disable gradient checkpointing after TTW warmup."""
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    def _run_warmup_optimization(
        self,
        model,
        image: Image.Image,
        warmup_captions,
        trainable_params,
        task_name: str | None = None,
        doc_id: int | None = None,
    ):
        """Run the TTW optimization loop for a single image."""
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
                        for caption_pair in batch_captions:
                            micro_batch = self._ttw_build_training_batch(image, [caption_pair])
                            outputs = model(**micro_batch)
                            _log_gpu_memory(
                                "TTW warmup: after forward (grad_accum)", doc_id=doc_id, rank=rank
                            )
                            loss = outputs.loss
                            loss.backward()
                            _log_gpu_memory(
                                "TTW warmup: after backward (grad_accum)", doc_id=doc_id, rank=rank
                            )
                            step_loss += loss.detach().item()
                    else:
                        batch_inputs = self._ttw_build_training_batch(image, batch_captions)
                        _log_gpu_memory(
                            "TTW warmup: after build_batch (before forward)",
                            doc_id=doc_id,
                            rank=rank,
                        )
                        outputs = model(**batch_inputs)
                        _log_gpu_memory("TTW warmup: after forward", doc_id=doc_id, rank=rank)
                        loss = outputs.loss
                        loss.backward()
                        _log_gpu_memory("TTW warmup: after backward", doc_id=doc_id, rank=rank)
                        step_loss = loss.detach().item()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # tensor deallocated entirely
                    _log_gpu_memory("TTW warmup: after optimizer.step", doc_id=doc_id, rank=rank)
                    log.debug(
                        "TTW epoch %d/%d step %d loss=%.4f",
                        epoch + 1,
                        self.ttw_epochs,
                        start // self.ttw_batch_size + 1,
                        step_loss,
                    )
                    # Log TTW warmup loss to WandB when enabled (e.g. --wandb_args project=...)
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
                    _log_gpu_memory("TTW warmup: after epoch 1", doc_id=doc_id, rank=rank)

        self._disable_gradient_checkpointing(model)
        model.eval()
        del optimizer, trainable_params
        torch.cuda.empty_cache()
        self._set_requires_grad(model, False)
        _log_gpu_memory("TTW warmup: done", doc_id=doc_id, rank=rank)
        log.info("TTW warmup finished.")

    def _ttw_warmup(self, image: Image.Image, task_name: str = None, doc_id: int = None):
        """Run TTW warmup: fetch captions, configure trainables, and optimize."""
        model = self._get_prepared_model()
        log.info("Starting TTW Warmup for new image...")

        warmup_captions = self._get_warmup_captions(image, task_name=task_name, doc_id=doc_id)
        rank = getattr(self._base, "rank", 0)
        _log_gpu_memory(
            "TTW warmup: after get_warmup_captions (CLIP done)", doc_id=doc_id, rank=rank
        )
        trainable_params = self._configure_trainable_params(model, doc_id=doc_id)
        self._run_warmup_optimization(
            model,
            image,
            warmup_captions,
            trainable_params,
            task_name=task_name,
            doc_id=doc_id,
        )

    def _get_first_request_image(self, req):
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

    def _generate_single_request_with_fsdp_support(self, prepared_model, req):
        """Delegate single-request generation to the base model."""
        return self._base.generate_until([req])

    def _ttw_build_training_batch(self, image, warmup_captions):
        """Build a training batch from (prompt, caption) pairs with proper label masking.

        Model-agnostic: uses the base model's `ttw_format_chat` to handle
        prompt formatting, and `apply_chat_template` to find the boundary between
        prompt tokens (masked to -100) and caption tokens (loss computed on these).

        Example with 2 warmup captions of different lengths:
        ─────────────────────────────────────────────────────
        Caption A: prompt="Describe the image" → caption="A cat on a mat"
        Caption B: prompt="What objects are visible?" → caption="A cat"

        Step 1 — Tokenize prompt-only (with generation marker):
          prompt_A tokens: [SYS][IMG]...[USR] Describe the image [ASST]  → prompt_len = 5
          prompt_B tokens: [SYS][IMG]...[USR] What objects are visible? [ASST]  → prompt_len = 7

        Step 2 — Tokenize full conversation (prompt + response):
          full_A tokens: [SYS][IMG]...[USR] Describe the image [ASST] A cat on a mat [END]  → len = 10
          full_B tokens: [SYS][IMG]...[USR] What objects are visible? [ASST] A cat [END]  → len = 9

        Step 3 — Build labels (mask prompt tokens to -100):
          labels_A: [-100, -100, -100, -100, -100,  tok,  tok,  tok,  tok,  tok]
                     |______ prompt (5) ______|  |____ caption (5) → loss _____|
          labels_B: [-100, -100, -100, -100, -100, -100, -100,  tok,  tok]
                     |_________ prompt (7) _________|  |_ caption (2) _|

        Padding — Pad all sequences to max_len = 10:
          input_ids_A: [ t0,  t1,  t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9]     (no padding)
          input_ids_B: [ t0,  t1,  t2,  t3,  t4,  t5,  t6,  t7,  t8, PAD]     (1 pad token)

          attn_mask_A: [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
          attn_mask_B: [  1,   1,   1,   1,   1,   1,   1,   1,   1,   0]

          labels_A:    [-100,-100,-100,-100,-100, tok, tok, tok, tok, tok]
          labels_B:    [-100,-100,-100,-100,-100,-100,-100, tok, tok,-100]     (pad → -100)

        Final batch (stacked along dim 0):
          input_ids:      [2, 10]
          attention_mask:  [2, 10]
          labels:          [2, 10]
          pixel_values:    repeated for batch_size=2
        """
        log.debug("Building TTW training batch...")
        processor = self._base.processor

        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        image_tensors = {}

        for count, (prompt_text, caption_text) in enumerate(warmup_captions):
            # Step 1: Tokenize prompt-only (with assistant marker appended)
            # This tells us how many tokens are "prompt" (should be masked).
            prompt_msg = self._base.ttw_format_chat(image, prompt_text)
            prompt_only_text = processor.apply_chat_template(
                prompt_msg, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = processor(text=[prompt_only_text], images=[image], return_tensors="pt")
            prompt_len = prompt_tokens.input_ids.shape[1]

            # Step 2: Tokenize full conversation (prompt + assistant response)
            full_msg = self._base.ttw_format_chat(image, prompt_text, caption_text)
            full_text = processor.apply_chat_template(
                full_msg, tokenize=False, add_generation_prompt=False
            )
            full_text_tokens = processor(text=[full_text], images=[image], return_tensors="pt")

            # Extract non-text batch keys (like pixel_values, image_grid_thw) from the first iteration
            # They are identical for all candidates because it's the exact same image.
            if count == 0:
                for key, tensor in full_text_tokens.items():
                    if key not in ["input_ids", "attention_mask", "labels"]:
                        image_tensors[key] = tensor

            # Step 3: Build labels — mask prompt tokens to -100
            labels = full_text_tokens.input_ids.clone()
            labels[:, :prompt_len] = -100

            all_input_ids.append(full_text_tokens.input_ids)
            all_attention_masks.append(full_text_tokens.attention_mask)
            all_labels.append(labels)

        # Pad and stack into a batch
        max_len = max(ids.shape[1] for ids in all_input_ids)
        pad_id = processor.tokenizer.pad_token_id or 0

        padded_ids, padded_masks, padded_labels = [], [], []
        for ids, mask, lab in zip(all_input_ids, all_attention_masks, all_labels):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((1, pad_len), pad_id, dtype=ids.dtype)], dim=1)
                mask = torch.cat([mask, torch.zeros(1, pad_len, dtype=mask.dtype)], dim=1)
                lab = torch.cat([lab, torch.full((1, pad_len), -100, dtype=lab.dtype)], dim=1)
            padded_ids.append(ids)
            padded_masks.append(mask)
            padded_labels.append(lab)

        batch = {
            "input_ids": torch.cat(padded_ids, dim=0).to("cuda"),
            "attention_mask": torch.cat(padded_masks, dim=0).to("cuda"),
            "labels": torch.cat(padded_labels, dim=0).to("cuda"),
        }

        # collect non-text tensors like pixel_values, image_grid_thw
        # expanding them across the batch. We use .repeat() along dim 0 because
        # Qwen2VL expects flat tensors (e.g. image_grid_thw as [num_images, 3],
        # pixel_values as [total_patches, channels]). The old unsqueeze+expand
        batch_size = len(all_input_ids)
        for key, tensor in image_tensors.items():
            repeat_dims = [1] * tensor.ndim
            repeat_dims[0] = batch_size
            batch[key] = tensor.repeat(*repeat_dims).to("cuda")
        # batch[key] = tensor.repeat(batch_size, *([1] * (tensor.ndim - 1))).to("cuda")

        return batch

    def generate_until(self, requests):
        """Intercept: warmup → delegate → restore.

        TTW adapts per-image, so we process one request at a time:
        save weights → warmup on image → infer → restore weights.

        Images are extracted via the doc_to_visual mechanism in TaskInstance.args:
        args = (context, gen_kwargs, doc_to_visual_fn, doc_id, task, split)
        """
        prepared_model = self._get_prepared_model()
        rank = getattr(self._base, "rank", 0)

        _log_gpu_memory("generate_until: before state_dict copy", rank=rank)
        original_state = self._snapshot_model_state(prepared_model)
        _log_gpu_memory("generate_until: after state_dict copy to CPU", rank=rank)
        res = []

        for req in requests:
            image, task, doc_id = self._get_first_request_image(req)
            if image is not None:
                gc.collect()
                torch.cuda.empty_cache()
                self._ttw_warmup(image, task_name=task, doc_id=doc_id)
            _log_gpu_memory(
                "generate_until: after warmup (before inference)", doc_id=doc_id, rank=rank
            )

            single_result = self._generate_single_request_with_fsdp_support(prepared_model, req)
            res.extend(single_result)
            _log_gpu_memory("generate_until: after inference", doc_id=doc_id, rank=rank)

            self._restore_model_state(prepared_model, original_state)
            _log_gpu_memory("generate_until: after restore", doc_id=doc_id, rank=rank)
            torch.cuda.empty_cache()

        return res
