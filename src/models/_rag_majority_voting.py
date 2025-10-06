import os
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src import utils
from src.data.tasks import TaskInstance, TaskSingleOutput
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model
from src.retrieval import Retriever

__all__ = ["RAGMajorityVoting"]

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


class RAGMajorityVoting(Model):
    """RAG Majority Voting Model.

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

    References:
    ----------
        - https://github.com/QwenLM/Qwen2-VL

    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self.batch_size_per_gpu = batch_size

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            distributed_types=["FSDP", "MULTI_GPU"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path, device_map=self.device_map
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self._model_name_or_path)

    def loglikelihood(self, requests: list[TaskInstance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.

        Downstream tasks should attempt to use loglikelihood instead of other
        LMM calls whenever possible.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, continuation). The arguments are as follows:
                - context (str): Context string. Implementations of LMM must be able to handle an
                    empty context string.
                - continuation (str):  The continuation over which log likelihood will be
                    calculated. If there is a word boundary, the space should be in the
                    continuation, e.g., context="hello" continuation=" world" is correct.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError

    def _is_rag_enabled(self, gen_kwargs: dict) -> bool:
        """Check if RAG is enabled in generation kwargs.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.

        """
        rag = (gen_kwargs or {}).get("rag") or {}
        return rag.get("enabled", False)

    def _setup_rag(self, gen_kwargs: dict) -> None:
        """Set up RAG retriever if not already set up.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.

        """
        if hasattr(self, "_retriever"):
            return

        db_root = os.getenv("RAG_DATABASE_ROOT")
        if not db_root:
            log.warning(
                "RAG is enabled but RAG_DATABASE_ROOT is not set; skipping retriever init."
            )
            return

        rag = (gen_kwargs or {}).get("rag") or {}
        self._retriever = Retriever(
            Path(db_root) / rag.get("database_path"),
            format=rag.get("database_format", "faiss"),
            model_name=rag.get("model_name"),
        )
        log.info("Retrieval database loaded!")

        self._retriever.set_vocab_transform(rag)

    def _retrieve(
        self, gen_kwargs: dict, images: list[Image.Image]
    ) -> None | list[dict[str, Any]]:
        """Retrieve relevant data for each image using the retriever.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.
            images (list[Image.Image]): List of images to retrieve data for.

        """
        if not hasattr(self, "_retriever"):
            log.error("RAG is enabled but retriever is not set up.")
            exit()

        def prepare(image: Image.Image) -> dict[str, Any]:
            payload = {"type": "image", "image": image}

            return payload

        def result_callback(
            rag_data: list[dict[str, Any]], images: list | None
        ) -> dict[str, list]:
            # Keep only the text elements and add a placeholder for images if any
            _rag_data = []
            for elem in rag_data:
                if elem.get("type") == "text":
                    _rag_data.append(elem.get("text"))
                elif elem.get("type") == "image" and images is not None:
                    _rag_data.append("<images>")

            return {
                "rag_data": _rag_data,
                "images": [x.get("image") for x in images] if images is not None else [],
            }

        result = self._retriever.retrieve_and_prepare(
            gen_kwargs, images, prepare=prepare, result_callback=result_callback
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        result = cast(list[dict[str, list]], result)

        return result

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

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

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

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # RAG setup (guard-style, avoids deep nesting later)
            rag_enabled = self._is_rag_enabled(gen_kwargs)
            if rag_enabled:
                rag = (gen_kwargs or {}).get("rag") or {}
                rag["doc_to_target"] = configurable_task.doc_to_target
                rag["doc_to_visual"] = configurable_task.doc_to_visual
                rag["test_docs"] = configurable_task.test_docs
                rag_position = rag.get("position", "pre-sample")  # noqa: F841
                self._setup_rag(gen_kwargs)
            else:
                raise ValueError("RAG is not enabled in generation kwargs.")

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        "Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            if isinstance(batched_contexts, tuple):
                batched_contexts = list(batched_contexts)

            for i in range(len(batched_contexts)):
                if "<image>" in batched_contexts[i]:
                    batched_contexts[i] = batched_contexts[i].replace("<image>", "")

            # RAG optimization: pre-load all images and retrieve in batches
            rag_messages = {}
            if rag_enabled:
                batched_visuals_for_rag = {}
                for i, _ in enumerate(batched_contexts):
                    visual = batched_visuals[i] if i < len(batched_visuals) else None
                    if isinstance(visual, Image.Image):
                        visual = visual.convert("RGB")
                        batched_visuals_for_rag[i] = visual

                # Batch retrieve data
                retrieved_data = self._retrieve(gen_kwargs, list(batched_visuals_for_rag.values()))
                for k, v in zip(batched_visuals_for_rag.keys(), retrieved_data, strict=True):
                    rag_messages[k] = v

            # For each sample, just keep the most frequent label
            answers = []
            for i in range(len(batched_contexts)):
                rag_message = rag_messages.get(i)

                if rag_message is None:
                    raise ValueError(f"No retrieved data for sample {i}.")

                rag_message = cast(dict[str, list], rag_message)

                # Get the actual message and the images
                rag_message, _ = rag_message["rag_data"], rag_message["images"]
                rag_message = rag_message[0] if len(rag_message) > 0 else ""
                candidates = Counter(rag_message.split("\n"))
                prediction = candidates.most_common(1)[0][0]
                answers.append(prediction)

            for ans, context in zip(answers, batched_contexts, strict=True):
                _ans = TaskSingleOutput(
                    answer=ans,
                )
                res.append(_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: list[TaskInstance]) -> list[str]:
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
        raise NotImplementedError


# rm = RAG majority voting
@register_model("rm-clip-vit-b32-openai")
def rm_clip_vit_b32_openai(**model_kwargs) -> Model:
    """Load the CLIP ViT B/32 model from OpenAI."""
    model_name_or_path = "openai/clip-vit-base-patch32"
    model = RAGMajorityVoting(model_name_or_path, **model_kwargs)
    return model
