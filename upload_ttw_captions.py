#!/usr/bin/env python3
"""Standalone script to upload TTW offline caption datasets to HuggingFace.

This script demonstrates how to use the upload_offline_captions_to_hf function
directly without running the full generation pipeline.

Usage:
    # Upload all JSONL files from the default offline_captions directory:
    python upload_ttw_captions.py --hf_token YOUR_HF_TOKEN

    # Upload from a specific directory:
    python upload_ttw_captions.py \
        --output_path /path/to/captions \
        --hf_token YOUR_HF_TOKEN \
        --repo_name username/dataset-name \
        --model_name "llama-3.2-11b-vision-instruct"
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _setup_environment() -> None:
    """Set up environment variables required for HuggingFace upload.

    Sets HF_XET_CACHE to a secure temporary directory if not already set.
    """
    if "HF_XET_CACHE" not in os.environ:
        os.environ["HF_XET_CACHE"] = tempfile.mkdtemp(prefix="hf_xet_")


def _add_project_root_to_path() -> None:
    """Add project root to sys.path for local imports."""
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> int:
    """Upload TTW offline caption datasets to HuggingFace.

    Returns:
        Exit code (0 for success, 1 for failure).

    """  # noqa: D406,D407
    # Setup environment and paths before importing project modules
    # Setup environment and paths before importing project modules
    _setup_environment()
    _add_project_root_to_path()

    from src import utils
    from src.utils.ttw_offline import upload_offline_captions_to_hf

    log = utils.get_logger(__name__, rank_zero_only=True)
    parser = argparse.ArgumentParser(
        description="Upload TTW offline caption datasets to HuggingFace"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="offline_captions",
        help=(
            "Directory containing the generated JSONL caption files " "(default: offline_captions)"
        ),  # noqa: E501
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (can also set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="ttw-captions",
        help="HuggingFace namespace/username (default: ttw-captions)",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help=(
            "HuggingFace repository name (e.g., 'username/dataset-name'). "
            "Overrides --namespace and --model_name if provided."
        ),  # noqa: E501
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Model identifier (used to generate repo name). "
            "If not provided, will be inferred from JSONL filenames."
        ),
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Create public repository (default: private)",
    )

    args = parser.parse_args()

    if not args.hf_token:
        parser.error(
            "HF token is required. Provide --hf_token or set HF_TOKEN environment variable"
        )

    # Construct repo_name if not provided
    if args.repo_name is None:
        if args.model_name:
            # Use provided model_name with namespace
            safe_model_name = args.model_name.replace("/", "--")
            args.repo_name = f"{args.namespace}/{safe_model_name}"
        else:
            # Will be inferred from JSONL filenames inside upload_offline_captions_to_hf
            # but we need to pass the namespace separately
            pass

    try:
        repo_url = upload_offline_captions_to_hf(
            hf_token=args.hf_token,
            output_path=args.output_path,
            repo_name=args.repo_name,
            model_name=args.model_name,
            commit_message=args.commit_message,
            private=not args.public,
            namespace=args.namespace,
        )
        log.info("✅ Upload complete! Repository: %s", repo_url)
        return 0
    except Exception as e:
        log.error("❌ Upload failed: %s", e)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
