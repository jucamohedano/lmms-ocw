"""TTW Offline Caption Generation.

Generates CLIP-filtered captions for TTW warmup and saves them as JSONL.
Supports two backends:
  - **HuggingFace** (default): uses the evaluation model's ``ttw_generate_captions``
  - **vLLM** (``--ttw_use_vllm``): uses vLLM's optimised inference engine

Both backends share the same dataset loop, resume logic, and output format
so the resulting files are interchangeable.
"""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from PIL import Image

from src import utils

log = utils.get_logger(__name__, rank_zero_only=True)


# Helper methods
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
    return []


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


# HuggingFace backend
def _make_hf_backend(args):
    """Initialise the HuggingFace model + CLIP and return a generate callable."""
    from src.models._api import get_model
    from src.models._ttw_wrapper import TTW_AUXILIARY_PROMPTS

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


# vLLM backend
def _make_vllm_backend(args):
    """Initialise the vLLM engine + CLIP and return a generate callable."""
    import random

    import numpy as np
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor
    from vllm import LLM, SamplingParams

    from src.models._ttw_wrapper import TTW_AUXILIARY_PROMPTS

    model_kwargs = utils.parse_string_args(args.model_args)
    max_model_len = model_kwargs.get("ttw_vllm_max_model_len", 4096)
    gpu_memory_utilization = model_kwargs.get("ttw_vllm_gpu_util", 0.85)
    pretrained_path = model_kwargs.get("pretrained", args.model)

    # Set seeds
    seed = args.seed[0] if isinstance(args.seed, list) else args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log.info("Loading vLLM engine...")
    llm = LLM(
        model=pretrained_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    log.info("vLLM engine loaded.")

    sampling_params = SamplingParams(
        temperature=args.ttw_offline_temperature,
        max_tokens=args.ttw_offline_max_new_tokens,
        n=args.ttw_offline_num_candidates,
        seed=seed,
    )

    processor = AutoProcessor.from_pretrained(pretrained_path, trust_remote_code=True)

    def _format_prompt(prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": prompt},
                ],
            }
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
            filtered.append({"prompt": prompt, "caption": candidates[scores.argmax().item()]})
        return filtered

    return generate


# Public entry point
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
    log.info("=" * 60)

    if use_vllm:
        backend = _make_vllm_backend(args)
    else:
        backend = _make_hf_backend(args)

    result = _run_offline_loop(args, task_manager, task_names, backend)

    # Upload to HuggingFace if requested
    if getattr(args, "ttw_upload_to_hf", False):
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "You must provide --hf_token or set HF_TOKEN environment variable "
                "when using --ttw_upload_to_hf"
            )
        upload_offline_captions_to_hf(
            hf_token=hf_token,
            output_path=args.output_path,
            repo_name=args.hf_repo_name,
            model_name=args.model,
            commit_message=args.hf_commit_message,
            private=args.hf_private_repo,
        )

    return result


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
    """Upload TTW offline caption datasets to HuggingFace as a dataset repository.

    Args:
        hf_token: HuggingFace authentication token with write permissions.
        output_path: Local directory containing the generated JSONL caption files.
        repo_name: HuggingFace repository name (e.g., "username/dataset-name").
                  If None, generates a name based on model_name or infers from filenames.
        model_name: Model identifier (used to generate repo_name if not provided).
        commit_message: Custom commit message for the upload.
        private: Whether to create the repository as private. Default: True.
        repo_type: Repository type, either "dataset" or "model". Default: "dataset".
        namespace: HuggingFace namespace/username to use when repo_name is auto-generated.
                  Default: "ttw-captions".

    Returns:
        The URL of the created/updated HuggingFace repository.

    Raises:
        ValueError: If output_path doesn't exist or repo_name cannot be determined.
    """
    output_path = Path(output_path)

    if not output_path.exists():
        raise ValueError(f"Output path does not exist: {output_path}")

    # Find all JSONL files in the output directory
    jsonl_files = list(output_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {output_path}")

    # Determine repository name
    if repo_name is None:
        if model_name is None:
            # Try to infer from first JSONL filename
            first_file = jsonl_files[0]
            # Extract model part from filename (format: modelid_task_captions.jsonl)
            stem = first_file.stem
            parts = stem.split("_")
            if len(parts) >= 2:
                model_part = parts[0]
            else:
                model_part = "ttw-captions"
            repo_name = f"{namespace}/{model_part}"
        else:
            safe_model_name = model_name.replace("/", "--")
            repo_name = f"{namespace}/{safe_model_name}"

    log.info(f"Uploading {len(jsonl_files)} JSONL files to HuggingFace repo: {repo_name}")
    log.info(f"Repository type: {repo_type}, Private: {private}")

    # Initialize HuggingFace API
    api = HfApi(token=hf_token)

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            repo_type=repo_type,
            exist_ok=True,
        )
        log.info(f"Repository created/verified: {repo_name}")
    except Exception as e:
        log.error(f"Failed to create repository: {e}")
        raise

    # Upload each JSONL file
    for jsonl_file in jsonl_files:
        path_in_repo = jsonl_file.name
        log.info(f"Uploading {jsonl_file.name} to {path_in_repo}")

        try:
            api.upload_file(
                repo_id=repo_name,
                path_or_fileobj=str(jsonl_file),
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                commit_message=commit_message or f"Upload {jsonl_file.name}",
            )
            log.info(f"Successfully uploaded {jsonl_file.name}")
        except Exception as e:
            log.error(f"Failed to upload {jsonl_file.name}: {e}")
            raise

    repo_url = f"https://huggingface.co/{repo_name}"
    log.info(f"All files uploaded successfully!")
    log.info(f"Repository URL: {repo_url}")

    return repo_url
