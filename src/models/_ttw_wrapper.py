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
import torch
from torch.optim import AdamW
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.models._base import Model
from src.models.apply_svf_to_llm import apply_svf_to_llm
import logging

log = logging.getLogger(__name__)


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


def _log_gpu_memory(label: str) -> None:
    """Log per-GPU memory for debugging OOM issues."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        log.info(
            "[GPU %d] %s: %.2f/%.2f GiB allocated (%.2f GiB reserved)",
            i,
            label,
            alloc,
            total,
            reserved,
        )


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
        ttw_svf_rank=-1,
    ):  # SVD truncation rank (-1 = full)
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

        # Prepare finetune mode once at init.
        if self.ttw_finetune_method == "svf":
            # Apply SVF at init time so we pay the SVD cost once.
            # The state_dict save/restore in generate_until handles resetting
            # the S vectors between images.
            # Use raw _model (not base.model) so we get the HF model with model.model.layers
            # intact; accelerator.unwrap_model can alter structure in some setups.
            model_to_svf = getattr(self._base, "_model", self._base.model)
            self._svf_layers = apply_svf_to_llm(model_to_svf, rank=self.ttw_svf_rank)
        elif self.ttw_finetune_method == "lora":
            self._inject_lora_adapters()
        elif self.ttw_finetune_method != "full":
            raise ValueError(
                f"Unknown ttw_finetune_method '{self.ttw_finetune_method}'. "
                "Expected one of: full, svf, lora."
            )

    def _inject_lora_adapters(self) -> None:
        """Attach LoRA adapters using the selected backend."""
        backend = str(self.ttw_lora_backend).lower()
        if backend == "unsloth":
            self._inject_lora_unsloth()
            return
        if backend != "peft":
            log.warning(f"Unknown LoRA backend '{self.ttw_lora_backend}', defaulting to PEFT.")
        self._inject_lora_peft()

    def _inject_lora_peft(self) -> None:
        """Attach LoRA adapters with PEFT (default backend)."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA finetuning requested but PEFT is not installed. "
                "Install with `pip install peft`."
            ) from exc

        model = self._base.model

        # Match the reference TTW repository hyperparameters.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA on the full conditional-generation model. For Qwen2-VL,
        # target modules (q/k/v/o + gate/up/down) live under `model.layers.*`.
        # Wrapping the top-level model avoids assuming a specific submodule name
        # like `language_model` across architectures.
        peft_model = get_peft_model(model, lora_config)
        self._base._model = peft_model
        log.info("Injected LoRA adapters for TTW (backend=peft).")

    def _inject_lora_unsloth(self) -> None:
        """Attach LoRA adapters with Unsloth; fallback to PEFT on failure."""
        try:
            from unsloth import FastVisionModel
        except ImportError:
            log.warning(
                "Unsloth requested for TTW LoRA, but package is not available. "
                "Falling back to PEFT backend."
            )
            self._inject_lora_peft()
            return

        model = self._base.model
        try:
            unsloth_model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self._base._model = unsloth_model
            log.info("Injected LoRA adapters for TTW (backend=unsloth).")
        except Exception as exc:
            log.warning(
                "Failed to apply Unsloth LoRA backend; falling back to PEFT. " f"Reason: {exc}"
            )
            self._inject_lora_peft()

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

        for param in connector_owner.visual.merger.parameters():
            param.requires_grad = True

    def _init_clip(self):
        """Lazy load CLIP model only when needed to save memory."""
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(self._clip_model_name).to("cuda").eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)

    def __getattr__(self, name):
        """Delegate everything not defined on TTWModel to the base model."""
        return getattr(self._base, name)

    def __delattr__(self, name):
        """Delegate attribute deletion to the base model if not on TTWModel."""
        if name in self.__dict__:
            super().__delattr__(name)
        else:
            delattr(self._base, name)

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

    def _ttw_warmup(self, image: Image.Image, task_name: str = None, doc_id: int = None):
        """Run TTW warmup: fetch or generate captions, train on best."""
        model = getattr(self._base, "_model", self._base.model)  # use prepared model for DDP

        log.info("Starting TTW Warmup for new image...")

        # 1. Fetch or Generate Captions
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

        # 2. Freeze/unfreeze based on finetuning method
        _log_gpu_memory("TTW warmup: before freeze/unfreeze")
        if self.ttw_finetune_method == "svf":
            # SVF layers were replaced at init. Freeze everything, then unfreeze
            # only the S vectors and the connector (state_dict restore in generate_until resets requires_grad).
            for param in model.parameters():
                param.requires_grad = False
            for svf in self._svf_layers:
                svf.S.requires_grad_(True)
                # if svf.bias is not None:
                #     svf.bias.requires_grad_(True)
            # Also unfreeze the vision-language connector (matches original TTW repo) and don't apply svf to it
            self._unfreeze_connector(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            log.debug(f"SVF: {sum(p.numel() for p in trainable_params):,} trainable params")
        elif self.ttw_finetune_method == "lora":
            # LoRA TTW mode: train only LoRA params + connector.
            for param in model.parameters():
                param.requires_grad = False

            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

            self._unfreeze_connector(model)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            log.debug(f"LoRA: {sum(p.numel() for p in trainable_params):,} trainable params")
        else:
            # Full FT: unfreeze everything, then freeze vision
            log.debug("Freezing vision encoders and unfreezing LLM and connector...")
            for param in model.parameters():
                param.requires_grad = True

            vision_modules = self._base.ttw_get_vision_encoder()
            for module in vision_modules:
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

        # 3. Warmup training – simple loop matching the original TTW repo.
        # With DDP each replica trains on the same captions (correct for
        # per-image adaptation). No Accelerate accumulate() needed.
        if self.ttw_finetune_method == "full":
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(trainable_params, lr=self.ttw_lr)
                log.info("TTW: using 8-bit AdamW (bitsandbytes)")
            except ImportError:
                log.warning("bitsandbytes not installed, falling back to regular AdamW")
                optimizer = AdamW(trainable_params, lr=self.ttw_lr, foreach=False)
        else:
            optimizer = AdamW(trainable_params, lr=self.ttw_lr, foreach=False)

        log.info(
            "TTW warmup: method=%s epochs=%d lr=%s batch_size=%d",
            self.ttw_finetune_method,
            self.ttw_epochs,
            self.ttw_lr,
            self.ttw_batch_size,
        )
        model_dtype = getattr(self._base.model, "dtype", torch.bfloat16)

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            log.info("TTW: gradient checkpointing enabled")

        model.train()
        _log_gpu_memory("TTW warmup: before training loop")

        # Full FT uses gradient accumulation (micro-batch size 1) to avoid OOM
        # with ~7.6B trainable params. LoRA/SVF have small trainable param counts
        # so they can forward the full batch at once for speed.
        use_grad_accum = self.ttw_finetune_method == "full"

        with torch.enable_grad(), torch.autocast("cuda", dtype=model_dtype):
            for epoch in range(self.ttw_epochs):
                for start in range(0, len(warmup_captions), self.ttw_batch_size):
                    batch_captions = warmup_captions[start : start + self.ttw_batch_size]
                    step_loss = 0.0
                    if use_grad_accum:
                        for caption_pair in batch_captions:
                            micro_batch = self._ttw_build_training_batch(image, [caption_pair])
                            outputs = model(**micro_batch)
                            loss = outputs.loss
                            loss.backward()
                            step_loss += loss.detach().item()
                    else:
                        batch_inputs = self._ttw_build_training_batch(image, batch_captions)
                        outputs = model(**batch_inputs)
                        loss = outputs.loss
                        loss.backward()
                        step_loss = loss.detach().item()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    log.debug(
                        "TTW epoch %d/%d step %d loss=%.4f",
                        epoch + 1,
                        self.ttw_epochs,
                        start // self.ttw_batch_size + 1,
                        step_loss,
                    )
                    torch.cuda.empty_cache()
                if epoch == 0:
                    _log_gpu_memory("TTW warmup: after epoch 1")

        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        model.eval()
        del optimizer, trainable_params
        torch.cuda.empty_cache()
        for param in model.parameters():
            param.requires_grad = False
        _log_gpu_memory("TTW warmup: done")
        log.info("TTW warmup finished.")

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
        _log_gpu_memory("generate_until: before state_dict copy")
        original_state = {k: v.cpu().clone() for k, v in self._base.model.state_dict().items()}
        _log_gpu_memory("generate_until: after state_dict copy to CPU")
        res = []

        for req in requests:
            args = req.args
            doc_to_visual_fn = args[2]
            doc_id = args[3]
            task = args[4]
            split = args[5]

            doc = self._base.task_dict[task][split][doc_id]
            visuals = doc_to_visual_fn(doc)

            # TTW warmup on the first image
            for vis in visuals:
                if isinstance(vis, Image.Image):
                    self._ttw_warmup(vis, task_name=task, doc_id=doc_id)
                    break

            # Infer on this single request with warmed-up weights
            single_result = self._base.generate_until([req])
            res.extend(single_result)

            # Restore original weights for the next request
            self._base.model.load_state_dict(original_state)

        return res
