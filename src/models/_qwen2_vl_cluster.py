import base64
import copy
import os
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from transformers.modeling_outputs import GenerateOutput

from src import utils
from src.data.tasks import TaskInstance, TaskSingleOutput
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model
from src.retrieval import Retriever

__all__ = ["Qwen2VLCluster"]

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


class Qwen2VLCluster(Model):
    """Qwen2VL Model, with clustering.

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
        model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        use_cache: bool = True,
        use_flash_attention_2: bool | None = utils.package_available("flash_attn"),
        max_pixels: int = 1024 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._use_cache = use_cache
        self._use_flash_attention_2 = use_flash_attention_2
        self._max_pixels = max_pixels
        self._min_pixels = min_pixels
        self.batch_size_per_gpu = batch_size

        if device_map == "None":
            device_map = None

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device_map,
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
        self._model = torch.compile(self._model, mode="max-autotune", fullgraph=True)
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        if "Qwen2.5" in self._model_name_or_path:
            self._processor.tokenizer.padding_side = "left"
            self._tokenizer.padding_side = "left"

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

    def _setup_rag(self, gen_kwargs: dict, multi_step: bool = False) -> None:
        """Set up RAG retriever if not already set up.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.
            multi_step (bool): Whether to set up multi-step RAG retrievers. Defaults to False.

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

        if not multi_step:
            self._retriever = Retriever(
                Path(db_root) / rag.get("database_path"),
                format=rag.get("database_format", "faiss"),
                model_name=rag.get("model_name"),
                few_shot=rag.get("num_shots_per_class"),
            )
            self._retriever.set_vocab_transform(rag)

        else:
            self._retriever = {}
            configs = {}
            for step_idx, rag_config in rag.items():
                # Not actual RAG, just piping from previous step
                if rag_config.get("pipe_from") is not None:
                    continue

                db_path = rag_config.get("database_path")
                # Reuse retriever if already loaded
                if db_path in configs:
                    self._retriever[step_idx] = self._retriever[configs[db_path]]
                else:
                    self._retriever[step_idx] = Retriever(
                        Path(db_root) / rag_config.get("database_path"),
                        format=rag_config.get("database_format", "faiss"),
                        model_name=rag_config.get("model_name"),
                        few_shot=rag_config.get("num_shots_per_class"),
                    )
                    configs[db_path] = step_idx
                self._retriever[step_idx].set_vocab_transform(rag_config)

        log.info("Retrieval database(s) loaded!")

    def _retrieve(
        self,
        gen_kwargs: dict,
        images: list[Image.Image],
        doc_ids: dict | None = None,
        step_idx: int | None = None,
    ) -> None | list[list[dict[str, Any]]]:
        """Retrieve relevant data for each image using the retriever.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments.
            images (list[Image.Image]): List of images to retrieve data for.
            doc_ids (dict | None): Optional dictionary mapping from sample index to document IDs
                to retrieve.
            step_idx (int | None): If using multi-step RAG, the index of the current step.

        """
        if not hasattr(self, "_retriever"):
            log.error("RAG is enabled but retriever is not set up.")
            exit()

        def _prepare(image: Image.Image, resize_image_size: None | int = None) -> dict[str, Any]:
            base64_image = image.convert("RGB")

            if resize_image_size is not None and max(base64_image.size) > resize_image_size:
                base64_image.thumbnail((resize_image_size, resize_image_size), Image.LANCZOS)

            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")

            payload = {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}

            return payload

        prepare = partial(
            _prepare,
            resize_image_size=(gen_kwargs or {}).get("rag", {}).get("resize_images", None),
        )

        use_custom_prepare = not (gen_kwargs or {}).get("rag", {}).get("separate_queries", False)

        retriever = self._retriever
        if step_idx is not None:
            retriever = retriever[step_idx]
        result = retriever.retrieve_and_prepare(
            gen_kwargs,
            images,
            prepare=prepare if use_custom_prepare else lambda x: x,
            doc_ids=doc_ids,
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        result = cast(list[list[dict[str, Any]]], result)

        return result

    def _batch_rag(
        self,
        batched_doc_id: Iterable,
        batched_contexts: Iterable,
        batched_visuals: Iterable,
        gen_kwargs: dict,
        task: str,
        split: str,
        step_idx: int | None = None,
    ) -> dict:
        """Retrieve data for a batch of samples using RAG.

        Args:
        ----
            batched_doc_id (Iterable): An iterable of document IDs for the batch.
            batched_contexts (Iterable): An iterable of context strings for the batch.
            batched_visuals (Iterable): An iterable of visual inputs for the batch.
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            task (str): The task name.
            split (str): The data split name.
            step_idx (int | None): Optional index for multi-step RAG retrieval.

        """
        # RAG optimization: pre-load all images and retrieve in batches
        rag_messages = {}

        cache_and_reuse = (gen_kwargs or {}).get("rag", {}).get("cache_and_reuse", False)
        reuse = (gen_kwargs or {}).get("rag", {}).get("reuse", False)

        batched_visuals_for_rag = {}
        for i, _ in enumerate(batched_contexts):
            visual = batched_visuals[i] if i < len(batched_visuals) else None
            if isinstance(visual, Image.Image):
                visual = visual.convert("RGB")
                batched_visuals_for_rag[i] = visual

                # When caching and reusing, only retrieve for the first image
                # Typically, this is used with random sampling, so we there's no
                # relationship between the images and the retrieved content,
                # meaning we can just retrieve once and reuse the same content for all
                if cache_and_reuse or reuse:
                    break

            elif isinstance(visual, list) and visual[0] is None:
                batched_visuals_for_rag[i] = None

                if cache_and_reuse or reuse:
                    break

        # Batch retrieve data
        payload = [{"doc_id": ids, **self.task_dict[task][split][ids]} for ids in batched_doc_id]
        retrieved_data = self._retrieve(
            gen_kwargs, list(batched_visuals_for_rag.values()), payload, step_idx=step_idx
        )
        for k, v in zip(batched_visuals_for_rag.keys(), retrieved_data, strict=True):
            rag_messages[k] = v

        return rag_messages

    def _prepare_visual_for_context(self, visual: Image.Image) -> str:
        """Prepare a visual input for context by converting it to a base64-encoded string.

        Args:
        ----
            visual (Image.Image): The visual input to prepare.

        """
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        return base64_image, base64_string

    def _log_conversation(self, conversation: list) -> None:
        """Log the given conversation in a readable format.

        Args:
        ----
            conversation (list): The conversation to log, typically a list of messages

        """
        msg = ""
        msg += "--- Conversation ---"
        for count, part in enumerate(conversation):
            msg += f"> Message {count} | Role: {part.get('role')}\n"

            if isinstance(part.get("content"), list):
                for msg_part_count, message in enumerate(part.get("content")):
                    msg += f">> Part {msg_part_count}\n"
                    if message.get("type") == "image":
                        msg += f"<image> ({len(message.get('image', ''))})\n\n"
                    else:
                        msg += f"{message.get('text')}\n\n"
            else:
                msg += f"{part.get('content')}\n\n"

        log.debug(msg)

    def _make_history(
        self,
        gen_kwargs: dict,
        batched_contexts: Iterable,
        batched_visuals: Iterable,
        rag: dict,
        rag_messages: dict,
        history: list | None = None,
        append_history: bool = False,
        step_idx: int | None = None,
    ) -> list | None:
        """Construct the conversation history for each sample in the batch.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            batched_contexts (Iterable): An iterable of context strings for the batch.
            batched_visuals (Iterable): An iterable of visual inputs for the batch.
            rag (dict): RAG configuration dictionary.
            rag_messages (dict): A dictionary mapping from sample index to retrieved RAG messages.
            history (list | None): Optional list of conversation histories for each sample.
                If None, a default history will be created for each sample. Defaults to None.
            append_history (bool): Whether to append the provided history to the default history
                (True) or to replace it entirely (False). Defaults to False.
            step_idx (int | None): Optional index for multi-step RAG retrieval, used to determine
                which retrieved messages to include in the history.

        """
        if history is not None:
            assert len(history) == len(
                batched_contexts
            ), "There must be a conversation for each sample (context)"

        rag_enabled = rag["rag_enabled"]
        rag_position = rag.get("position", "pre-sample")
        include_target_classes = rag.get("include_target_classes") is not None

        if include_target_classes:
            include_target_classes_position = rag.get("include_target_classes", "pre-query")
            target_classes_format = rag.get("target_classes_format", "csv")
            target_classes = sorted(set([rag["doc_to_target"](x) for x in rag["test_docs"]()]))
            target_classes_prompt = rag.get("target_classes_prompt")
            if target_classes_format == "csv":
                target_classes_str = ",".join(target_classes)
            elif target_classes_format == "csv-spaces":
                target_classes_str = ", ".join(target_classes)
            elif target_classes_format == "newline":
                target_classes_str = "\n".join(target_classes)
            elif target_classes_format == "bullet-list":
                target_classes_str = "\n".join([f"- {cls}" for cls in target_classes])
            elif target_classes_format == "numbered-list":
                target_classes_str = "\n".join(
                    [f"{i+1}. {cls}" for i, cls in enumerate(target_classes)]
                )
            else:
                raise ValueError(f"Unknown target_classes_format: {target_classes_format}")

        messages = []
        images_per_request = []

        for i, context in enumerate(batched_contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            # Use a default message as initial context
            # If history is provided, replace the message with the history for the i-th sample
            # `history` should contain the system prompt, as it entirely replaces the default
            # message -- it is not appended to it!
            message = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                    if rag.get("system_prompt") is None
                    else rag.get("system_prompt"),
                }
            ]
            if history is not None and history[i] is not None:
                if append_history:
                    message.extend(history[i])
                else:
                    message = history[i]

            if len(batched_visuals) > 0:
                visual = batched_visuals[i] if i < len(batched_visuals) else None
                _images_counter = len(visual) if isinstance(visual, list | tuple) else 1
                if isinstance(visual, Image.Image):  # Single image
                    base64_image, base64_string = self._prepare_visual_for_context(visual)

                    rag_message = None
                    if rag_enabled:
                        if i in rag_messages:
                            rag_message = rag_messages[i]
                        else:
                            rag_message = self._retrieve(
                                gen_kwargs, [base64_image], step_idx=step_idx
                            )[0]

                        # Update the image counter, used for stats
                        _images_counter += sum(
                            1
                            for _msg in rag_message
                            for k, v in _msg.items()
                            if k == "type" and v == "image"
                        )

                    # Construct the message
                    content = []

                    if (
                        include_target_classes
                        and include_target_classes_position == "begin-of-ctx"
                    ):
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    # Retrieved data goes at the beginning of the context
                    if rag_enabled and rag_position == "pre-sample" and rag_message is not None:
                        content.extend(rag_message)

                    if rag.get("pre_image_prompt") is not None:
                        content.append({"type": "text", "text": rag["pre_image_prompt"]})

                    # When RAG is disabled, always include the image
                    # When it is enabled, check the "include_image" flag (True by default)
                    if not rag_enabled or rag.get("include_image", True):
                        content.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )

                    if include_target_classes and include_target_classes_position == "after-image":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    # Retrieved data goes after the image, before the query
                    if rag_enabled and rag_position == "post-sample" and rag_message is not None:
                        content.extend(rag_message)

                    if include_target_classes and include_target_classes_position == "pre-query":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    content.append({"type": "text", "text": context})

                    # Retrieved data goes at the end of the context
                    if (
                        rag_enabled
                        and rag_position == "post-sample-and-query"
                        and rag_message is not None
                    ):
                        content.extend(rag_message)

                    if include_target_classes and include_target_classes_position == "end-of-ctx":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    message.append(
                        {
                            "role": "user",
                            "content": content,
                        }
                    )
                    images_per_request.append(_images_counter)

                elif isinstance(visual, list | tuple) and all(
                    isinstance(v, Image.Image) for v in visual
                ):  # Multiple images
                    image_content = []
                    for v in visual:
                        base64_image, base64_string = self._prepare_visual_for_context(v)
                        image_content.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )
                    message.append(
                        {
                            "role": "user",
                            "content": image_content + [{"type": "text", "text": context}],
                        }
                    )
                else:
                    content = []

                    if include_target_classes and include_target_classes_position == "pre-query":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    content.append({"type": "text", "text": context})

                    if include_target_classes and include_target_classes_position == "end-of-ctx":
                        content.append(
                            {
                                "type": "text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    message.append({"role": "user", "content": content})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            messages.append(message)

        return messages, images_per_request

    def _generate(self, messages: list, gen_kwargs: dict, rag: dict) -> tuple[dict, Any]:
        """Generate model outputs for the given messages.

        Args:
        ----
            messages (list): A list of messages.
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            rag (dict): RAG configuration dictionary.

        """
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        if video_inputs is not None:
            raise ValueError("Video inputs should be empty for current implementation of Qwen2VL.")

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda") if self.device_map == "auto" else inputs.to(self.device)
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 128
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        if "resize_input_image" in gen_kwargs:
            gen_kwargs.pop("resize_input_image")

        pad_token_id = self.tokenizer.pad_token_id

        generation_output = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=gen_kwargs["temperature"] > 0,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self._use_cache,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,  # In our case, with temp == 0, they should
            # be identical to scores
        )

        return inputs, generation_output

    def _decode(self, inputs: dict, generation_output: GenerateOutput) -> list[str]:
        """Decode the generated output into a list of strings.

        Args:
        ----
            inputs (dict): The input dictionary containing input IDs.
            generation_output (GenerateOutput): The output from the model generation.

        """
        cont = generation_output.sequences

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont, strict=True)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return answers

    def _downscale_images(self, images: list[Image.Image], rag: dict) -> list[Image.Image]:
        """Downscale images based on RAG configuration.

        Args:
        ----
            images (list[Image.Image]): List of images to potentially downscale.
            rag (dict): RAG configuration dictionary containing resizing parameters.

        """
        resize_image_size = rag.get("resize_images")
        if resize_image_size is not None:
            for image in images:
                if not isinstance(image, Image.Image):
                    continue

                # Scale down, keeping aspect ratio, if the image is larger
                if max(image.size) > resize_image_size:
                    image.thumbnail((resize_image_size, resize_image_size), Image.LANCZOS)

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
            tokens = self.tokenizer.encode(x[0])
            return -len(tokens), x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        # Create the initial context via clustering
        gen_kwargs = requests[0].args[1]
        task = requests[0].args[4]
        split = requests[0].args[5]

        rag_enabled = self._is_rag_enabled(gen_kwargs)
        rag = (gen_kwargs or {}).get("rag") or {}
        rag["rag_enabled"] = rag_enabled
        if rag_enabled:
            rag["doc_to_target"] = configurable_task.doc_to_target
            rag["doc_to_visual"] = configurable_task.doc_to_visual
            rag["test_docs"] = configurable_task.test_docs
            self._setup_rag(gen_kwargs)

        rag_messages = {}
        if rag_enabled:
            rag_messages = self._batch_rag([0], [None], [[None]], gen_kwargs, task, split)[0]
        else:
            raise ValueError("RAG must be enabled")

        # Now that we have the images, we can cluster them
        # 1. Load CLIP to get image embeddings
        model_name = "openai/clip-vit-base-patch32"
        clip_model = AutoModel.from_pretrained(model_name, device_map=self.device)
        clip_model.eval()
        clip_processor = AutoProcessor.from_pretrained(model_name)
        inputs = clip_processor(images=rag_messages, return_tensors="pt").to(self._device)

        embeddings = clip_model.get_image_features(**inputs)
        embeddings = F.normalize(embeddings)

        del clip_model, clip_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        embeddings = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=rag.get("kmeans_clusters", 4), random_state=42, n_init="auto")
        kmeans.fit(embeddings)
        labels = kmeans.labels_

        # Now group by cluster assignment
        clustered_images = defaultdict(list)
        for idx, label in enumerate(labels):
            clustered_images[label].append(rag_messages[idx])

        # Generate a label for each cluster
        generated_labels = []
        encoded_images_per_cluster = []
        for cluster in clustered_images.values():
            images = [
                {
                    "type": "image",
                    "image": (
                        f"data:image/jpeg;base64," f"{self._prepare_visual_for_context(image)[1]}"
                    ),
                }
                for image in cluster
            ]
            encoded_images_per_cluster.append(images)

            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        *images,
                        {"type": "text", "text": rag.get("classification_prompt")},
                    ],
                },
            ]

            inputs, generation_output = self._generate(
                [conversation], copy.deepcopy(gen_kwargs), copy.deepcopy(rag)
            )
            answers = self._decode(inputs, generation_output)
            generated_labels.append(answers * len(images))

        # Now we have a label per cluster. We can construct the actual context
        # for subsequent classifications
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        conv_to_append = [
            {
                "type": "text",
                "text": (
                    "You are an open-world visual classifier.\n"
                    "You will see a small set of image-label pairs that "
                    "belong to the same unknown domain.\nUse them to "
                    "infer what kinds of distinctions matter in this domain "
                    "(e.g., texture patterns, object subtypes, material "
                    "differences, color or shape features).\n\nYour task is "
                    "to classify a new image from the same domain.\nMatch the "
                    "level of specificity shown in the context labels.\nAvoid "
                    'generic words such as "object", "thing", "vehicle", "animal", '
                    '"texture" unless they are the most specific category possible\n\n'
                    "Answer concisely with one or two words.\n"
                ),
            }
        ]
        for idx, _ in enumerate(clustered_images.values()):
            for img, gen_label in zip(
                encoded_images_per_cluster[idx], generated_labels[idx], strict=False
            ):
                conv_to_append.append(img)
                conv_to_append.append({"type": "text", "text": f"Label: {gen_label}"})

        conv_to_append.append({"type": "text", "text": "===== EXAMPLES END =====\nTARGET:"})
        conversation.append(
            {
                "role": "user",
                "content": conv_to_append,
            }
        )

        # Now `conversation` contains the entire conversation. We just need to append the
        # input image and the `prompt`

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
            rag = (gen_kwargs or {}).get("rag") or {}
            rag["rag_enabled"] = rag_enabled
            if rag_enabled:
                rag["doc_to_target"] = configurable_task.doc_to_target
                rag["doc_to_visual"] = configurable_task.doc_to_visual
                rag["test_docs"] = configurable_task.test_docs
                self._setup_rag(gen_kwargs)

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

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

            rag_messages = {}
            if rag_enabled:
                rag_messages = self._batch_rag(
                    batched_doc_id, batched_contexts, batched_visuals, gen_kwargs, task, split
                )

            if gen_kwargs.get("resize_input_image", None) is not None:
                for img_idx in range(len(batched_visuals)):
                    if not isinstance(batched_visuals[img_idx], Image.Image):
                        continue

                    # Scale down, keeping aspect ratio, if the image is larger
                    if max(batched_visuals[img_idx].size) > gen_kwargs["resize_input_image"]:
                        batched_visuals[img_idx].thumbnail(
                            (gen_kwargs["resize_input_image"], gen_kwargs["resize_input_image"]),
                            Image.LANCZOS,
                        )

            # Construct history
            messages = []
            for ctx, input_image in zip(batched_contexts, batched_visuals, strict=False):
                conv = copy.deepcopy(conversation)
                conv[-1]["content"].append(
                    {
                        "type": "image",
                        "image": (
                            f"data:image/jpeg;base64,"
                            f"{self._prepare_visual_for_context(input_image)[1]}"
                        ),
                    }
                )
                conv[-1]["content"].append(
                    {
                        "type": "text",
                        "text": ctx,
                    }
                )
                messages.append(conv)

            inputs, generation_output = self._generate(
                messages, copy.deepcopy(gen_kwargs), copy.deepcopy(rag)
            )
            answers = self._decode(inputs, generation_output)

            for ans_idx, answer in enumerate(answers):
                if "Answer:" in answer:
                    answers[ans_idx] = (
                        answer.split("Answer:")[-1]
                        .replace("[", "")
                        .replace("]", "")
                        .replace('"', "")
                        .strip()
                    )

            rag.pop("doc_to_target", None)
            rag.pop("doc_to_visual", None)
            rag.pop("test_docs", None)

            if rag.get("store_memory", False):
                self._retriever.store_memory(batched_visuals, answers)

            for idx, (ans, context) in enumerate(zip(answers, batched_contexts, strict=True)):
                _ans = TaskSingleOutput(
                    answer=ans,
                    context=context,
                    context_tokens_count=inputs["attention_mask"][idx].sum(),
                    num_images=None,
                    loglikelihood=None,
                    perplexity=None,
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


@register_model("qwen2-vl-7b-cluster")
def qwen2_vl_7b_cluster(**model_kwargs) -> Model:
    """Load the Qwen2VL model with 7B params."""
    model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLCluster(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2-vl-2b-cluster")
def qwen2_vl_2b_cluster(**model_kwargs) -> Model:
    """Load the Qwen2VL model with 2B params."""
    model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLCluster(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2.5-vl-7b-cluster")
def qwen25_vl_7b_cluster(**model_kwargs) -> Model:
    """Load the Qwen2.5VL model with 7B params."""
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2VLCluster(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2.5-vl-3b-cluster")
def qwen25_vl_3b_cluster(**model_kwargs) -> Model:
    """Load the Qwen2.5VL model with 3B params."""
    model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2VLCluster(model_name_or_path, **model_kwargs)
    return model
