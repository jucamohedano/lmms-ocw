from __future__ import annotations

import os
from typing import Any

from transformers import AutoConfig, Qwen2VLForConditionalGeneration


def is_qwen2_5_vl_checkpoint(model_name_or_path: str | os.PathLike[str]) -> bool:
    """Return True when a model path/repo points to a Qwen2.5-VL checkpoint."""
    model_ref = str(model_name_or_path)
    if "qwen2.5" in model_ref.lower() or "qwen2_5" in model_ref.lower():
        return True

    try:
        config = AutoConfig.from_pretrained(model_ref)
    except Exception:
        return False

    model_type = getattr(config, "model_type", "")
    architectures = getattr(config, "architectures", None) or []
    return model_type == "qwen2_5_vl" or any("Qwen2_5" in str(arch) for arch in architectures)


def get_qwen_vl_model_class(model_name_or_path: str | os.PathLike[str]) -> type[Any]:
    """Select the correct Transformers class for Qwen2-VL vs Qwen2.5-VL."""
    if not is_qwen2_5_vl_checkpoint(model_name_or_path):
        return Qwen2VLForConditionalGeneration

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ValueError(
            "Failed to import Qwen2_5_VLForConditionalGeneration. "
            "Please upgrade transformers to a version with Qwen2.5-VL support."
        ) from exc

    return Qwen2_5_VLForConditionalGeneration
