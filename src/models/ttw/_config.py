"""Shared TTW constants and configuration.

Single source of truth for LoRA hyperparameters and auxiliary prompts.
"""

from __future__ import annotations

from typing import Any

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
    (
        "Explain the possible relationships or roles of the people, animals, "
        "or objects in this scene. What hints or clues suggest these relationships?"
    ),
    (
        "Based on visual cues, infer what might have happened just before and "
        "what might happen right after this image was captured."
    ),
]


def get_lora_config_dict() -> dict[str, Any]:
    """Single source of truth for LoRA hyperparameters.

    Used by:
        - ``_qwen2_vl.py`` (main-process LoRA init before FSDP prepare)
        - ``_wrapper.py`` (passed to worker pool initializer)
        - ``_worker.py`` (received as ``{"lora_config": ...}`` config dict)
    """
    from peft import TaskType

    return {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
