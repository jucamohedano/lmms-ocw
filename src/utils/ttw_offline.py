"""TTW Offline Dataset Generation.

Supports two independent output paths:

**Path 1 — CLIP baseline (default)**
    Generates CLIP-filtered captions for TTW warmup and saves them as JSONL.
    Entry point: ``generate_offline_captions``
    Backends:
      - HuggingFace (default): uses the evaluation model's ``ttw_generate_captions``
      - vLLM (``--ttw_use_vllm``): uses vLLM's optimised inference engine

**Path 2 — verl GRPO**
    Generates verl-compatible parquet datasets for GRPO training (reward v3:
    nested ``<HasProperty>`` / ``<HasA>`` / ``<AtLocation>`` tags inside a
    ``<think>...</think>`` scratchpad; see ``docs/grpo/reward_design_v3.md``).
    No model inference at dataset-prep time — verl's own vLLM handles rollouts
    during training. No external metadata is required by the reward.
    Images use a top-level ``images`` column (``data.image_key=images``).
    Entry point: ``generate_grpo_dataset``

    Path 1 (CLIP JSONL warmup) is unchanged and still uses free-form caption
    prompts from ``TTW_AUXILIARY_PROMPTS`` — independent of this GRPO schema.
    Use ``--ttw_offline_vanilla_chat`` for a short default system message instead
    of the GRPO scratchpad when generating those captions.

Both paths share task-loading helpers but are otherwise independent.
"""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from PIL import Image

from src import utils

log = utils.get_logger(__name__, rank_zero_only=True)

# ---------------------------------------------------------------------------
# GRPO prompt constants (defaults when no --ttw_grpo_prompt_config is given)
# ---------------------------------------------------------------------------

_GRPO_SYSTEM_PROMPT = """You are an expert visual reasoner. For every image,
first think through what you see inside <think>...</think>, using three sub-tags:
<HasProperty> for visible properties, <HasA> for visible parts, and
<AtLocation> for visible setting or context.

Use this exact structure:

<think>
<HasProperty>2-5 short, comma-separated visible properties</HasProperty>
<HasA>2-5 short, comma-separated visible parts</HasA>
<AtLocation>2-5 short, comma-separated visible settings or places</AtLocation>
Reason on the attributes extracted to answer the user's question after thinking.

</think>
answer the user's question directly

Guidelines:
- Use lowercase common terms (e.g. "striped", "tail", "jungle").
- Keep each entry short and concrete.
- After </think>, answer the user's question directly.
- If they ask you to classify, emit a single label only.
- If they ask you to describe or explain, emit a natural-language answer."""

_GRPO_USER_PROMPT = "Classify the main object in this image."

# Short instruct-system text for offline caption runs that should *not* use the GRPO scratchpad prompt.
_TTW_VANILLA_SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# GRPO prompt config loader
# ---------------------------------------------------------------------------


def _load_grpo_prompt_config(
    config_path: str | None,
) -> tuple[str, str]:
    """Load system/user prompt pair from a YAML file.

    Falls back to the built-in ``_GRPO_SYSTEM_PROMPT`` / ``_GRPO_USER_PROMPT``
    when *config_path* is ``None``.

    Args:
    ----
        config_path: Path to a YAML file with ``system_prompt`` and ``user_prompt`` keys.
            See ``configs/grpo_prompts/`` for examples.

    Returns:
    -------
        (system_prompt, user_prompt) tuple.

    """
    if not config_path:
        return _GRPO_SYSTEM_PROMPT, _GRPO_USER_PROMPT

    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    system_prompt = cfg.get("system_prompt")
    user_prompt = cfg.get("user_prompt")

    if system_prompt is None or user_prompt is None:
        raise ValueError(
            f"GRPO prompt config '{config_path}' must contain both "
            f"'system_prompt' and 'user_prompt' keys. Got: {list(cfg.keys())}"
        )

    log.info("Loaded GRPO prompt config from %s", config_path)
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Shared task-loading helpers
# ---------------------------------------------------------------------------


def _load_processed_doc_ids(out_file: str) -> set[int]:
    """Read an existing JSONL file and return the set of already-processed doc IDs."""
    processed = set()
    if not os.path.exists(out_file):
        return processed
    with open(out_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                log.warning(
                    f"Skipping malformed JSONL line {line_num} in {out_file}; "
                    "this instance will be regenerated."
                )
                continue
            doc_id = data.get("doc_id")
            if doc_id is not None:
                processed.add(doc_id)
    return processed


def _get_docs(task_obj) -> list:
    """Return documents from the first available split (test > val > train)."""
    if task_obj.has_test_docs():
        return list(task_obj.test_docs())
    if task_obj.has_validation_docs():
        return list(task_obj.validation_docs())
    if task_obj.has_training_docs():
        return list(task_obj.training_docs())
    # Fallback: access underlying dataset directly if split exists but not declared in config
    if hasattr(task_obj, "dataset") and "test" in task_obj.dataset:
        return list(task_obj.dataset["test"])
    return []


def _get_train_docs(task_obj) -> list:
    """Return training documents, falling back to validation if unavailable."""
    if task_obj.has_training_docs():
        return list(task_obj.training_docs())
    if task_obj.has_validation_docs():
        return list(task_obj.validation_docs())
    # Fallback: access underlying dataset directly if split exists but not declared in config
    if hasattr(task_obj, "dataset") and "train" in task_obj.dataset:
        return list(task_obj.dataset["train"])
    return []


def _extract_gt_label(doc, task_obj) -> str:
    """Extract the ground-truth label string from a task document.

    Handles both string labels (caltech101, dtd, oxford_pets) and integer
    choice indices (some lm-eval tasks).
    """
    gt = task_obj.doc_to_target(doc)
    if isinstance(gt, int):
        choices = doc.get("choices", doc.get("options", []))
        gt = choices[gt] if choices else str(gt)
    return str(gt).strip()


# ---------------------------------------------------------------------------
# Path 1 — CLIP baseline: offline caption generation -> JSONL
# (all functions below are unchanged from the original)
# ---------------------------------------------------------------------------


def _run_offline_loop(args, task_manager, task_names, backend):
    """Shared dataset iteration loop for offline caption generation.

    ``backend`` must be a callable with signature::

        backend(image: PIL.Image.Image) -> list[dict]

    where each dict has keys ``{"prompt": str, "caption": str}``.
    """
    from src.data.tasks import get_tasks_as_dict

    os.makedirs(args.output_path, exist_ok=True)

    for task_name in task_names:
        task_obj_dict = get_tasks_as_dict([task_name], task_manager)
        if not task_obj_dict or task_name not in task_obj_dict:
            log.error(f"Failed to load task {task_name}")
            continue

        task_obj = task_obj_dict[task_name]
        docs = _get_docs(task_obj)
        if not docs:
            log.error(f"Task {task_name} has no documents to process.")
            continue

        # Build output filename (safe for model ids containing '/')
        safe_model_id = args.model.replace("/", "__")
        out_file = os.path.join(args.output_path, f"{safe_model_id}_{task_name}_captions.jsonl")

        processed_doc_indices = _load_processed_doc_ids(out_file)
        limit = args.ttw_offline_limit if args.ttw_offline_limit else len(docs)

        log.info("-" * 60)
        log.info(
            f"Task: {task_name} | Docs: {len(docs)} | "
            f"Limit: {limit} | Done: {len(processed_doc_indices)}"
        )
        log.info("-" * 60)

        with open(out_file, "a") as f_out:
            generated_so_far = len(processed_doc_indices)

            for i, doc in enumerate(docs):
                if generated_so_far >= limit:
                    log.info(f"Reached limit of {limit}. Stopping.")
                    break

                if i in processed_doc_indices:
                    continue

                # Extract image
                visuals = task_obj.doc_to_visual(doc)
                if not visuals or not isinstance(visuals[0], Image.Image):
                    log.warning(f"No valid image for doc_id {i}, skipping.")
                    continue
                image = visuals[0]

                log.info(f"[{generated_so_far + 1}/{limit}] doc_id={i}")

                # Backend call
                captions = backend(image)

                # Write to disk immediately (fault-tolerant against SLURM timeout)
                out_record = {"doc_id": i, "captions": captions}
                f_out.write(json.dumps(out_record) + "\n")
                f_out.flush()
                generated_so_far += 1

        log.info(f"Finished {task_name}! Processed {generated_so_far} items.")

    log.info("Offline generation complete.")
    return None, None


def _make_hf_backend(args):
    """Initialise the HuggingFace model + CLIP and return a generate callable."""
    from src.models._api import get_model
    from src.models.ttw import TTW_AUXILIARY_PROMPTS

    log.info(f"Loading Base MLLM '{args.model}' (Args: {args.model_args})...")
    model_kwargs = utils.parse_string_args(args.model_args)
    model = get_model(args.model, **model_kwargs)
    log.info("Base MLLM loaded successfully.")

    log.info("Initializing auxiliary CLIP model...")
    model._init_clip()
    log.info("CLIP model initialized.")

    def generate(image: Image.Image) -> list[dict]:
        caption_results = model._base.ttw_generate_captions(
            image,
            TTW_AUXILIARY_PROMPTS,
            args.ttw_offline_num_candidates,
            temperature=args.ttw_offline_temperature,
            max_new_tokens=args.ttw_offline_max_new_tokens,
            batch_size=args.ttw_offline_batch_size,
            vanilla_chat=getattr(args, "ttw_offline_vanilla_chat", False),
        )

        filtered = []
        for prompt, candidates in caption_results:
            clip_inputs = model._clip_processor(
                text=candidates,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")
            with torch.no_grad():
                scores = model._clip_model(**clip_inputs).logits_per_image[0]
            filtered.append({"prompt": prompt, "caption": candidates[scores.argmax().item()]})
        return filtered

    return generate


def _make_vllm_backend(args):
    """Initialise the vLLM engine + CLIP and return a generate callable."""
    import random

    import numpy as np
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor

    from src.models.ttw import TTW_AUXILIARY_PROMPTS

    model_kwargs = utils.parse_string_args(args.model_args)
    max_model_len = model_kwargs.get("ttw_vllm_max_model_len", 8192)
    gpu_memory_utilization = model_kwargs.get("ttw_vllm_gpu_util", 0.85)
    mm_encoder_attn_backend = model_kwargs.get("ttw_vllm_mm_encoder_attn_backend", "TORCH_SDPA")
    pretrained_path = (
        model_kwargs.get("pretrained") or model_kwargs.get("model_name_or_path") or args.model
    )

    # vLLM must see this before import/engine construction. ``spawn`` avoids
    # CUDA re-init failures if another library has already touched torch.cuda.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    # Set seeds
    seed = args.seed[0] if isinstance(args.seed, list) else args.seed
    random.seed(seed)
    np.random.seed(seed)

    log.info(f"Loading vLLM engine from '{pretrained_path}'...")
    log.info(
        "vLLM config: VLLM_WORKER_MULTIPROC_METHOD=%s, mm_encoder_attn_backend=%s",
        os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
        mm_encoder_attn_backend,
    )
    llm = LLM(
        model=pretrained_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        mm_encoder_attn_backend=mm_encoder_attn_backend,
    )
    log.info("vLLM engine loaded.")

    sampling_params = SamplingParams(
        temperature=args.ttw_offline_temperature,
        max_tokens=args.ttw_offline_max_new_tokens,
        n=args.ttw_offline_num_candidates,
        seed=seed,
    )

    processor = AutoProcessor.from_pretrained(pretrained_path, trust_remote_code=True)

    use_vanilla = getattr(args, "ttw_offline_vanilla_chat", False)
    system_text = _TTW_VANILLA_SYSTEM_PROMPT if use_vanilla else utils._GRPO_SYSTEM_PROMPT

    def _format_prompt(prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    formatted_prompts = {p: _format_prompt(p) for p in TTW_AUXILIARY_PROMPTS}

    # Load CLIP for filtering
    clip_model_name = model_kwargs.get("clip_model_name", "openai/clip-vit-large-patch14-336")
    log.info(f"Loading auxiliary CLIP '{clip_model_name}'...")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to("cuda").eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    log.info("CLIP loaded.")

    def generate(image: Image.Image) -> list[dict]:
        vllm_requests = [
            {
                "prompt": formatted_prompts[prompt],
                "multi_modal_data": {"image": image},
            }
            for prompt in TTW_AUXILIARY_PROMPTS
        ]

        outputs = llm.generate(vllm_requests, sampling_params)

        filtered = []
        for prompt, output in zip(TTW_AUXILIARY_PROMPTS, outputs):
            candidates = [gen.text for gen in output.outputs]
            inputs = clip_processor(
                text=candidates,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(clip_model.device)
            with torch.no_grad():
                scores = clip_model(**inputs).logits_per_image[0]
            scores_list = scores.tolist()
            filtered.append(
                {
                    "prompt": prompt,
                    "caption": candidates[scores.argmax().item()],
                    "candidates": [
                        {"text": c, "clip_score": s} for c, s in zip(candidates, scores_list)
                    ],
                }
            )
        return filtered

    return generate


def generate_offline_captions(args, task_manager, task_names):
    """Generate TTW caption datasets offline and save them to disk as JSONL.

    Picks the backend (HF or vLLM) based on ``args.ttw_use_vllm``, then
    delegates to the shared ``_run_offline_loop``.
    """
    if not args.output_path:
        raise ValueError("You must provide --output_path when using --ttw_offline_generate")

    use_vllm = getattr(args, "ttw_use_vllm", False)
    backend_name = "vLLM" if use_vllm else "HuggingFace"

    log.info("=" * 60)
    log.info(f"TTW Offline Caption Generation ({backend_name})")
    log.info("=" * 60)
    log.info(f"  Model:            {args.model}")
    log.info(f"  Model args:       {args.model_args}")
    log.info(f"  Tasks:            {', '.join(task_names)}")
    log.info(f"  Num candidates:   {args.ttw_offline_num_candidates}")
    log.info(f"  Temperature:      {args.ttw_offline_temperature}")
    log.info(f"  Max new tokens:   {args.ttw_offline_max_new_tokens}")
    log.info(f"  Output path:      {args.output_path}")
    log.info(f"  Vanilla chat:    {getattr(args, 'ttw_offline_vanilla_chat', False)}")
    log.info("=" * 60)

    if use_vllm:
        backend = _make_vllm_backend(args)
    else:
        backend = _make_hf_backend(args)

    result = _run_offline_loop(args, task_manager, task_names, backend)

    if getattr(args, "ttw_upload_to_hf", False):
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "You must provide --hf_token or set HF_TOKEN environment variable "
                "when using --ttw_upload_to_hf"
            )
        upload_to_hf(
            hf_token=hf_token,
            output_path=args.output_path,
            file_pattern="*.jsonl",
            repo_name=args.hf_repo_name,
            model_name=args.model,
            commit_message=args.hf_commit_message,
            private=args.hf_private_repo,
        )

    return result


# ---------------------------------------------------------------------------
# Path 2 — verl GRPO: image + label -> parquet
# No model inference. verl's own vLLM handles rollouts during training.
# ---------------------------------------------------------------------------


def _build_grpo_parquet(
    task_obj,
    docs,
    split_name,
    task_name,
    limit,
    *,
    system_prompt: str,
    user_prompt: str,
):
    """Build a GRPO parquet dataset from a list of documents.

    Schema matches verl expectations.

    Args:
        task_obj: The task object.
        docs: The list of documents.
        split_name: The name of the split.
        task_name: The name of the task.
        limit: The limit on the number of documents to process.
        system_prompt: System message text injected into every sample.
        user_prompt: User message text (``<image>`` tag prepended automatically).

    """
    import datasets as hf_datasets

    rows_list = []
    generated = 0

    for i, doc in enumerate(docs):
        if generated >= limit:
            break

        visuals = task_obj.doc_to_visual(doc)
        if not visuals or not isinstance(visuals[0], Image.Image):
            log.warning(f"No valid image for doc_id={i}, skipping.")
            continue

        image: Image.Image = visuals[0].convert("RGB")
        gt_label = _extract_gt_label(doc, task_obj)

        log.info(f"[{generated + 1}/{limit}] doc_id={i} | label='{gt_label}'")

        rows_list.append(
            {
                "data_source": task_name,
                # Stored as plain list of dicts — matches GSM8K exactly.
                # Dataset.from_list() auto-infers this as List<Struct<role, content>>
                # which pandas reads back correctly as a list of dicts.
                # DO NOT wrap in hf_datasets.Sequence() — that creates Struct<List>
                # which pandas returns as {"role": [...], "content": [...]} breaking verl.
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        # verl splits content on <image> tags to inject images.
                        # Content must be a plain string — NOT a list of dicts.
                        "content": f"<image>\n{user_prompt}",
                    },
                ],
                "ability": "classification",
                "reward_model": {"style": "rule", "ground_truth": gt_label},
                "images": [image],
                "extra_info": {"split": split_name, "index": i, "gt_label": gt_label},
            }
        )
        generated += 1

    if not rows_list:
        return None, 0

    # Step 1: Create dataset without explicit features.
    # from_list() infers prompt as List<Struct> (row-oriented) — same as
    # what GSM8K's dataset.map() produces. This is what verl expects.
    dataset = hf_datasets.Dataset.from_list(rows_list)

    # Step 2: Cast only the images column to Image() so PIL Images get
    # serialised to PNG bytes. All other columns keep their inferred types.
    dataset = dataset.cast_column("images", hf_datasets.Sequence(hf_datasets.Image()))

    return dataset, generated


def generate_grpo_dataset(args, task_manager, task_names):
    """Generate verl-compatible parquet datasets for GRPO classification training.

    No model inference is performed. Images and ground-truth labels are read
    directly from the lm-eval task objects and written to parquet files that
    verl can load with ``data.image_key=images``. Reward v3 uses only the
    sample ground-truth label, so no external metadata build step is required.

    Output layout::

        <output_path>/
          <task_name>/
            train.parquet
            test.parquet

    Required verl launch script flags
    ----------------------------------
    ``data.train_files=<output_path>/<task>/train.parquet``
    ``data.val_files=<output_path>/<task>/test.parquet``
    ``data.image_key=images``
    ``data.return_raw_chat=True``
    """
    from src.data.tasks import get_tasks_as_dict

    if not args.output_path:
        raise ValueError("You must provide --output_path when using --ttw_grpo_generate")

    log.info("=" * 60)
    log.info("TTW GRPO Dataset Generation -> parquet")
    log.info("=" * 60)
    log.info(f"  Tasks:       {', '.join(task_names)}")
    log.info(f"  Output path: {args.output_path}")
    log.info("=" * 60)

    system_prompt, user_prompt = _load_grpo_prompt_config(
        getattr(args, "ttw_grpo_prompt_config", None)
    )
    log.info(f"  System prompt: {system_prompt[:60]}...")
    log.info(f"  User prompt:   {user_prompt[:60]}...")
    log.info("=" * 60)

    os.makedirs(args.output_path, exist_ok=True)

    for task_name in task_names:
        task_obj_dict = get_tasks_as_dict([task_name], task_manager)
        if not task_obj_dict or task_name not in task_obj_dict:
            log.error(f"Failed to load task {task_name}")
            continue

        task_obj = task_obj_dict[task_name]
        task_out_dir = Path(args.output_path) / task_name
        task_out_dir.mkdir(parents=True, exist_ok=True)

        limit = args.ttw_offline_limit if args.ttw_offline_limit else None

        # Process both splits; (docs, split_name, output_filename)
        splits = [
            (_get_train_docs(task_obj), "train", "train"),
            (_get_docs(task_obj), "test", "test"),
        ]

        for docs, split_name, out_split in splits:
            if not docs:
                log.warning(f"Task {task_name}: no docs for split '{split_name}', skipping.")
                continue

            effective_limit = limit if limit else len(docs)
            out_file = task_out_dir / f"{out_split}.parquet"

            log.info("-" * 60)
            log.info(
                f"Task: {task_name} | Split: {split_name} -> {out_split} | "
                f"Docs: {len(docs)} | Limit: {effective_limit}"
            )
            log.info("-" * 60)

            dataset, n_written = _build_grpo_parquet(
                task_obj,
                docs,
                split_name,
                task_name,
                effective_limit,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if dataset is None:
                log.warning(f"No rows generated for {task_name}/{out_split}, skipping.")
                continue

            dataset.to_parquet(str(out_file))
            log.info(f"Saved {n_written} rows -> {out_file}")

        log.info(f"Finished task: {task_name}")

    log.info("GRPO dataset generation complete.")

    if getattr(args, "ttw_upload_to_hf", False):
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "You must provide --hf_token or set HF_TOKEN environment variable "
                "when using --ttw_upload_to_hf"
            )
        upload_to_hf(
            hf_token=hf_token,
            output_path=args.output_path,
            file_pattern="**/*.parquet",
            repo_name=args.hf_repo_name,
            model_name=None,
            commit_message=args.hf_commit_message,
            private=args.hf_private_repo,
        )

    return None, None


# ---------------------------------------------------------------------------
# Shared HuggingFace upload (handles both JSONL and parquet)
# ---------------------------------------------------------------------------


def upload_to_hf(
    hf_token: str,
    output_path: str,
    file_pattern: str = "*.jsonl",
    repo_name: str | None = None,
    model_name: str | None = None,
    commit_message: str | None = None,
    private: bool = True,
    repo_type: str = "dataset",
    namespace: str = "ttw-captions",
) -> str:
    """Upload generated datasets to a HuggingFace dataset repository.

    Works for both JSONL (CLIP path) and parquet (GRPO path) files.
    Preserves sub-directory structure (e.g. oxford_pets/train.parquet).

    Args:
        hf_token:       HuggingFace auth token with write permissions.
        output_path:    Local directory containing the generated files.
        file_pattern:   Glob pattern relative to output_path, e.g. "*.jsonl"
                        or "**/*.parquet".
        repo_name:      HuggingFace repo id (e.g. "username/my-dataset").
                        Auto-generated from model_name if None.
        model_name:     Used to auto-generate repo_name when repo_name is None.
        commit_message: Custom commit message.
        private:        Create the repo as private. Default: True.
        repo_type:      "dataset" or "model". Default: "dataset".
        namespace:      HF namespace used when auto-generating repo_name.

    Returns:
        URL of the created/updated HuggingFace repository.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        raise ValueError(f"Output path does not exist: {output_path}")

    files = list(output_path.glob(file_pattern))
    if not files:
        raise ValueError(f"No files matching '{file_pattern}' found in {output_path}")

    if repo_name is None:
        safe_name = (model_name or "ttw-dataset").replace("/", "--")
        repo_name = f"{namespace}/{safe_name}"

    log.info(f"Uploading {len(files)} file(s) -> {repo_name}")

    api = HfApi(token=hf_token)
    create_repo(
        repo_id=repo_name,
        token=hf_token,
        private=private,
        repo_type=repo_type,
        exist_ok=True,
    )

    for file in files:
        # Preserve relative path so task/split structure is kept in the repo.
        relative = file.relative_to(output_path)
        log.info(f"  Uploading {relative}")
        api.upload_file(
            repo_id=repo_name,
            path_or_fileobj=str(file),
            path_in_repo=str(relative),
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {relative}",
        )

    repo_url = f"https://huggingface.co/{repo_name}"
    log.info(f"Upload complete -> {repo_url}")
    return repo_url


# ---------------------------------------------------------------------------
# Backward-compatible alias for the original upload function name
# ---------------------------------------------------------------------------


def upload_offline_captions_to_hf(
    hf_token: str,
    output_path: str,
    repo_name: str | None = None,
    model_name: str | None = None,
    commit_message: str | None = None,
    private: bool = True,
    repo_type: str = "dataset",
    namespace: str = "ttw-captions",
) -> str:
    """Backward-compatible wrapper around ``upload_to_hf`` for JSONL files."""
    return upload_to_hf(
        hf_token=hf_token,
        output_path=output_path,
        file_pattern="*.jsonl",
        repo_name=repo_name,
        model_name=model_name,
        commit_message=commit_message,
        private=private,
        repo_type=repo_type,
        namespace=namespace,
    )
