"""Apply SVF (Singular Value Finetuning) to a language model's linear layers.

Walks the model's decoder blocks and replaces target nn.Linear modules with
SVFLinear wrappers. Only the singular values (S vectors) remain trainable;
U and V are frozen. This drastically reduces trainable parameter count.

Usage:
    from src.models.apply_svf_to_llm import apply_svf_to_llm

    apply_svf_to_llm(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        rank=-1,  # full rank (-1) or truncated
    )
"""

import logging

import torch.nn as nn

from src import utils
from src.models.svf import SVFLinear

log = utils.get_logger(__name__, rank_zero_only=True)

TARGET_MODULES_DEFAULT = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def apply_svf_to_llm(
    model: nn.Module,
    target_modules: list[str] | None = None,
    rank: int = -1,
) -> list[SVFLinear]:
    """Replace target nn.Linear layers inside the LM decoder with SVFLinear.

    Targets are matched by attribute name within each decoder block.
    All model parameters are frozen first, then only the SVFLinear
    trainable params (S vectors) are unfrozen.

    Works with any HF causal LM that has a `.model.layers` attribute
    (Qwen2-VL, LLaMA, Mistral, etc.).

    Args:
    ----
        model: The full model (e.g. Qwen2VLForConditionalGeneration).
        target_modules: List of linear layer names to replace. Defaults to
            q/k/v/o_proj + gate/up/down_proj.
        rank: SVD truncation rank. -1 means full rank (no truncation).

    Returns:
    -------
        List of created SVFLinear modules (useful for optimizer param groups).

    """
    if target_modules is None:
        target_modules = TARGET_MODULES_DEFAULT

    # Find decoder layers by walking the module tree. This is robust to
    # accelerate hooks, PEFT wrapping, and other model transformations that
    # break direct attribute access like model.model.layers.
    decoder_layers = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if "DecoderLayer" in type(first).__name__ or "Block" in type(first).__name__:
                decoder_layers = module
                log.info("Found decoder layers at '%s' (%d blocks)", name, len(module))
                break

    if decoder_layers is None:
        raise AttributeError(
            f"Cannot find decoder layers in {type(model).__name__}. "
            "No ModuleList with DecoderLayer/Block children found."
        )

    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Walk decoder blocks and replace targets with SVFLinear
    svf_layers: list[SVFLinear] = []
    replaced = 0

    for layer_idx, block in enumerate(decoder_layers):
        for name, module in block.named_modules():
            # Match leaf module name (e.g. "q_proj" from "self_attn.q_proj")
            leaf_name = name.split(".")[-1] if name else ""
            if leaf_name not in target_modules or not isinstance(module, nn.Linear):
                continue

            # Navigate to parent to perform the replacement
            # walks down the module tree to find the parent module
            parts = name.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)

            # Replace with SVFLinear (manual implementation correctly freezes U/Vh
            orig_dtype = module.weight.dtype
            svf = SVFLinear(
                module, rank=rank, implementation="manual"
            )  # set to manual so that only the S vectors are trainable (requires_grad=True)

            # TODO: at the moment keep it like this to debug on 1 gpu
            # Cast all SVF components back to model's dtype (e.g. bf16) to save memory.
            # and avoid dtype mismatches in forward pass (SVFLinear does x = x.to(S.dtype),
            # so all components must match). SVD requires float32 but we can train in bf16
            # for the short TTW warmup.
            svf.U.data = svf.U.data.to(orig_dtype)
            svf.S.data = svf.S.data.to(orig_dtype)
            svf.Vh.data = svf.Vh.data.to(orig_dtype)

            # Replace the original linear layer with the SVFLinear wrapper
            setattr(parent, parts[-1], svf)
            svf_layers.append(svf)
            replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        f"SVF applied: replaced {replaced} linear layers across "
        f"{len(decoder_layers)} decoder blocks (rank={rank}). "
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return svf_layers


if __name__ == "__main__":
    import torch
    from transformers import Qwen2VLForConditionalGeneration

    logging.basicConfig(level=logging.INFO)

    print("Loading Qwen2-VL-2B...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32,  # float32 needed for SVD
    )

    print(f"\nBefore SVF:")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")

    print("\nApplying SVF...")
    svf_layers = apply_svf_to_llm(model, rank=-1)

    print(f"\nAfter SVF:")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  SVF layers:       {len(svf_layers)}")

    # Show a few replaced layers
    print(f"\nFirst 5 SVF layers:")
    for i, svf in enumerate(svf_layers[:5]):
        print(f"  [{i}] {svf}")

    # Quick forward check: verify S vectors have gradients
    print(f"\nGradient check:")
    for i, svf in enumerate(svf_layers[:3]):
        print(
            f"  [{i}] S.requires_grad={svf.trainable_svf_S.requires_grad}, "
            f"S.shape={svf.trainable_svf_S.shape}"
        )
