import logging
from collections.abc import Callable

from src.models._api import (
    MODELS,
    get_model,
    get_model_builder,
    get_model_info,
    get_models_info,
    register_model,
)
from src.models._base import Model
from src.models._embedder_clip import EmbedderCLIP
from src.models._embedder_gme import EmbedderGME
from src.models._embedder_qwen2_vl import EmbedderQwen2VL
from src.models._idefics2 import Idefics2
from src.models._instructblip import InstructBLIP
from src.models._internvl2 import InternVL2
from src.models._llava_hf import LLaVA

try:
    from src.models._llava_onevision import LLaVAOnevision
except ImportError:
    LLaVAOnevision = None
    logging.getLogger(__name__).warning(
        "Could not import LLaVAOnevision (likely a transformers version mismatch). "
        "LLaVA-OneVision models will not be available."
    )

from src.models._openai import OpenAIAPI
from src.models._phi3v import Phi3v
from src.models._qwen2_vl import Qwen2VL
from src.models._qwen2_vl_cluster import Qwen2VLCluster
from src.models._rag_majority_voting import RAGMajorityVoting
from src.models.ttw import TTWModel

__all__ = [
    "MODELS",
    "Model",
    "TTWModel",
    "Idefics2",
    "InstructBLIP",
    "InternVL2",
    "LLaVA",
    "LLaVAOnevision",
    "Phi3v",
    "Qwen2VL",
    "register_model",
    "get_model",
    "get_model_builder",
    "get_model_info",
    "get_models_info",
]

MODEL_TYPES: dict[str, Callable] = {
    "idefics2": Idefics2,
    "instructblip": InstructBLIP,
    "internvl2": InternVL2,
    "llava": LLaVA,
    "phi3v": Phi3v,
    "qwen2-vl": Qwen2VL,
    "qwen2-vl-cluster": Qwen2VLCluster,
    "embedder-clip": EmbedderCLIP,
    "embedder-gme": EmbedderGME,
    "embedder-qwen2vl": EmbedderQwen2VL,
    "rag-majority-voting": RAGMajorityVoting,
    "openai": OpenAIAPI,
}

if LLaVAOnevision is not None:
    MODEL_TYPES["llava-onevision"] = LLaVAOnevision


@register_model("custom-model")
def custom_model(model_type: str, model_name_or_path: str, **model_kwargs) -> Callable:
    model_cls = MODEL_TYPES.get(model_type)
    if model_cls is None:
        raise ValueError(f"Model type '{model_type}' not found.")

    model_instance = model_cls(model_name_or_path, **model_kwargs)
    return model_instance
