import base64
import gc
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2 as T
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from src import utils
from src.data.tasks import TaskInstance
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model

__all__ = ["EmbedderQwen2VL"]

log = utils.get_logger(__name__, rank_zero_only=True)


def _flatten_list(input: list[list[Any]]) -> list[Any]:
    """Flatten a nested list into a single list.

    Args:
    ----
        input (list): A nested list containing elements to be flattened.

    """
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


class EmbedderQwen2VL(Model):
    """Qwen2VL Model, used to embed stuff.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "Qwen/Qwen2-VL-7B-Instruct".
        use_cache (bool): Whether to use KV cache during generation. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether to use flash attention 2. Default to False.
        max_pixels (int): The max number of pixels in an image. Defaults to 12'845'056.
        min_pixels (int): The min number of pixels in an image. Defaults to 3'316.
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        use_cache: bool = True,
        use_flash_attention_2: bool | None = utils.package_available("flash_attn"),
        max_pixels: int = 1024 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        spatial_merge_factor: int = 2,
        pool_method: str = "mean",
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "float16",
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._use_cache = use_cache
        self.batch_size_per_gpu = batch_size
        self._use_flash_attention_2 = use_flash_attention_2
        self._max_pixels = max_pixels
        self._min_pixels = min_pixels
        self._spatial_merge_factor = spatial_merge_factor
        self._pool_method = pool_method

        db_root = os.getenv("RAG_DATABASE_ROOT")
        if not db_root:
            raise ValueError("RAG_DATABASE_ROOT is not set")
        self._db_root = Path(db_root)

        self._transform: T.Compose | None = None

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            distributed_types=["FSDP", "MULTI_GPU"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        """Load the model in memory."""
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self._device_map,
        }
        processor_kwargs = {
            "max_pixels": self._max_pixels,
            "min_pixels": self._min_pixels,
        }

        if self._use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        PretrainedModel = Qwen2VLForConditionalGeneration
        if "Qwen2.5" in self._model_name_or_path:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
            except ImportError as e:
                raise ValueError(
                    "Failed to import Qwen2_5_VLForConditionalGeneration."
                    " Please upgrade transformers to a later version."
                ) from e

            PretrainedModel = Qwen2_5_VLForConditionalGeneration

        self._model = PretrainedModel.from_pretrained(self._model_name_or_path, **model_kwargs)
        del self._model.model
        del self._model.lm_head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model = torch.compile(self._model, mode="max-autotune", fullgraph=True)
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        if "Qwen2.5" in self._model_name_or_path:
            self._processor.tokenizer.padding_side = "left"
            self._tokenizer.padding_side = "left"

    def loglikelihood(self, requests: list[TaskInstance]) -> list[tuple[float, bool]]:
        """Compute the log-likelihood of the given requests.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, target). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError

    def generate_until(self, requests: list[TaskInstance]) -> list[str]:
        """Generate greedily until a stopping sequence.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, until). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        res = []

        def _collate(x: tuple[str, ...]) -> tuple[int, str]:
            """Group and sort requests by context length for efficient batching.

            The negative sign on len(tokens) sorts in descending order, which provides several
                advantages:
                - Time estimates will be overestimates rather than underestimates, which is more
                    useful for planning;
                - The first item in a batch determines the padded context length, simplifying
                    batching logic;
                - Makes automatic adaptive batches much easier to implement;
                - Any out-of-memory errors occur immediately rather than near the end.

            Args:
            ----
                x: A tuple containing the context string and other arguments

            """
            return 1, x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__
        dataset_name = configurable_task.task_name.split("_embed")[0]

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        doc_ids = []
        visuals = []
        targets = []

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk, strict=True)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [
                batched_doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in batched_doc_id
            ]
            batched_visuals = _flatten_list(batched_visuals)

            doc_ids.extend(batched_doc_id)
            visuals.extend([self.task_dict[task][split][ids]["visual"] for ids in batched_doc_id])
            targets.extend([self.task_dict[task][split][ids]["target"] for ids in batched_doc_id])

            messages = []
            for visual in batched_visuals:
                base64_image = visual.convert("RGB")
                buffer = BytesIO()
                base64_image.save(buffer, format="JPEG")
                base64_bytes = base64.b64encode(buffer.getvalue())
                base64_string = base64_bytes.decode("utf-8")

                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this image.",
                                },
                            ],
                        }
                    ]
                )

            # Prepare inputs
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda") if self.device_map == "auto" else inputs.to(self.device)

            with torch.no_grad():
                embeddings = self._model.visual(
                    inputs["pixel_values"],
                    grid_thw=inputs["image_grid_thw"],
                )

            pooled_embeddings = []
            start_idx = 0
            for thw in inputs["image_grid_thw"]:
                t, h, w = thw.tolist()

                compressed_h = (h + 1) // self._spatial_merge_factor
                compressed_w = (w + 1) // self._spatial_merge_factor
                num_tokens = t * compressed_h * compressed_w

                # Extract this image's embeddings
                image_emb = embeddings[start_idx : start_idx + num_tokens]

                if self._pool_method == "mean":
                    pooled = image_emb.mean(dim=0)
                elif self._pool_method == "max":
                    pooled = image_emb.max(dim=0)[0]
                elif self._pool_method == "first":
                    pooled = image_emb[0]
                else:
                    raise ValueError(f"Unknown pool_method: {self._pool_method}")

                pooled_embeddings.append(pooled)
                start_idx += num_tokens

            embeddings = torch.stack(pooled_embeddings)  # [batch_size, hidden_dim]

            res.append(embeddings.to("cpu"))
            pbar.update(len(embeddings))

        res = torch.cat(res, dim=0)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)
        res = torch.stack(res, dim=0)
        doc_ids = reordered.get_original(doc_ids)
        visuals = reordered.get_original(visuals)
        targets = reordered.get_original(targets)

        metadata = {}
        for i, (ids, vis, tgt) in enumerate(zip(doc_ids, visuals, targets, strict=False)):
            metadata[i] = {
                "doc_id": ids,
                "visual": vis,
                "target": tgt,
            }

        model_nicename = self._model_name_or_path.replace("/", "-")
        save_path = self._db_root / dataset_name / split
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(res, save_path / f"{model_nicename}.pt")

        with open(save_path / f"{model_nicename}_metadata.json", "w") as f:
            json.dump(metadata, f)

        pbar.close()

        out = f"Embeddings saved to {save_path}. Exiting"
        log.info(out)
        exit()

        return res

    def generate_until_multi_round(self, requests: list[TaskInstance]) -> list[str]:
        """Generate greedily until a stopping sequence with multi-round generation.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, until). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError


@register_model("qwen2-vl-2b-ve")
def qwen2_vl_2b_ve(**model_kwargs) -> Model:
    """Load the Qwen2 VL 2B model, using its Vision Encoder (`ve`)."""
    model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
    model = EmbedderQwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2-vl-7b-ve")
def qwen2_vl_7b_ve(**model_kwargs) -> Model:
    """Load the Qwen2 VL 7B model, using its Vision Encoder (`ve`)."""
    model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
    model = EmbedderQwen2VL(model_name_or_path, **model_kwargs)
    return model
