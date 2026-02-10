import base64
import concurrent.futures
import copy
import datetime
import json
import os
from collections.abc import Iterable
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from PIL import Image

from src import utils
from src.data.tasks import TaskInstance, TaskSingleOutput
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model
from src.retrieval import Retriever

__all__ = ["OpenAI"]

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


class OpenAIAPI(Model):
    """OpenAI Model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "Qwen/Qwen2-VL-7B-Instruct".
        batch_size (int): Batch size for model inference. Defaults to 1.
        kwargs: Additional keyword arguments.

    References:
    ----------
        - https://platform.openai.com/docs/models

    """

    def __init__(
        self,
        model_name_or_path: str = "gpt-4o-mini-2024-07-18",
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self.batch_size_per_gpu = batch_size

        super().__init__(
            batch_size=batch_size,
            distributed_types=["FSDP", "MULTI_GPU"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        self._model = OpenAI(max_retries=5)
        self._processor = lambda x: x

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
            doc_ids (dict | None): Optional dictionary mapping document IDs to their metadata.
            step_idx (int | None): Optional index for multi-step RAG retrieval.

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

            payload = {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_string}",
            }

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
    ) -> tuple[list, list]:
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

                        # Replace "text" with "input_text" in "type"
                        sanitized_messages = []

                        for msg in rag_message:
                            # FIX: Convert 'image' to 'input_image'
                            if msg.get("type") == "image":
                                # Assuming your input_image has a 'data' field with base64
                                sanitized_messages.append(
                                    {
                                        "type": "input_image",
                                        "image_url": msg["image"],
                                    }
                                )
                            elif msg.get("type") == "text":
                                sanitized_messages.append(
                                    {"type": "input_text", "text": msg["text"]}
                                )
                            else:
                                sanitized_messages.append(msg)

                        rag_message = sanitized_messages

                    # Construct the message
                    content = []

                    if (
                        include_target_classes
                        and include_target_classes_position == "begin-of-ctx"
                    ):
                        content.append(
                            {
                                "type": "input_text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    # Retrieved data goes at the beginning of the context
                    if rag_enabled and rag_position == "pre-sample" and rag_message is not None:
                        content.extend(rag_message)

                    if rag.get("pre_image_prompt") is not None:
                        content.append({"type": "input_text", "text": rag["pre_image_prompt"]})

                    # When RAG is disabled, always include the image
                    # When it is enabled, check the "include_image" flag (True by default)
                    if not rag_enabled or rag.get("include_image", True):
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )

                    if include_target_classes and include_target_classes_position == "after-image":
                        content.append(
                            {
                                "type": "input_text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    # Retrieved data goes after the image, before the query
                    if rag_enabled and rag_position == "post-sample" and rag_message is not None:
                        content.extend(rag_message)

                    if include_target_classes and include_target_classes_position == "pre-query":
                        content.append(
                            {
                                "type": "input_text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    content.append({"type": "input_text", "text": context})

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
                                "type": "input_text",
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
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )
                    message.append(
                        {
                            "role": "user",
                            "content": image_content + [{"type": "input_text", "text": context}],
                        }
                    )
                else:
                    content = []

                    if include_target_classes and include_target_classes_position == "pre-query":
                        content.append(
                            {
                                "type": "input_text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    content.append({"type": "input_text", "text": context})

                    if include_target_classes and include_target_classes_position == "end-of-ctx":
                        content.append(
                            {
                                "type": "input_text",
                                "text": target_classes_prompt.format(target_classes_str),
                            }
                        )

                    message.append({"role": "user", "content": content})
            else:
                message.append(
                    {"role": "user", "content": [{"type": "input_text", "text": context}]}
                )

            messages.append(message)

        return messages, images_per_request

    def _log_conversation(self, conversation: list) -> None:
        """Log the given conversation in a readable format.

        Args:
        ----
            conversation (list): The conversation to log, typically a list of messages

        """
        res = ""
        res += "--- Conversation ---\n"
        for count, part in enumerate(conversation):
            res += f"> Message {count} | Role: {part.get('role')}\n"

            if isinstance(part.get("content"), list):
                for msg_part_count, message in enumerate(part.get("content")):
                    res += f">> Part {msg_part_count}\n"
                    if message.get("type") == "input_image":
                        res += f"<image> ({len(message.get('image_url', ''))})\n\n"
                    else:
                        res += f"{message.get('text')}\n\n"
            else:
                res += f"{part.get('content')}\n\n"

        log.debug(res)

    def _generate_single_response(self, conversation: list, gen_kwargs: dict) -> str:
        """Make an API call for a single request (conversation).

        Args:
        ----
            conversation (list): The conversation for which to generate a response.
            gen_kwargs (dict): Generation keyword arguments containing parameters
                like temperature, top_p, and max_new_tokens.

        """
        resp = self._model.responses.create(
            model=self._model_name_or_path,
            input=conversation,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            max_output_tokens=gen_kwargs["max_new_tokens"],
        )

        return resp.output_text

    def _generate(
        self, messages: list, gen_kwargs: dict, rag: dict, do_online: bool = True
    ) -> list[Any]:
        """Generate model outputs for the given messages.

        Args:
        ----
            messages (list): A list of messages.
            gen_kwargs (dict): Generation keyword arguments containing RAG configuration.
            rag (dict): RAG configuration dictionary.
            do_online (bool): Whether to generate responses using the streaming API (True)
                or to prepare a batch job for offline generation (False). Defaults to True.

        """
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 128
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None

        if do_online:
            generation_output = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [
                    executor.submit(self._generate_single_response, msg, gen_kwargs)
                    for msg in messages
                ]

                generation_output = [f.result() for f in futures]

            return generation_output

        # Offline (batch) case
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%Y%m%d_%H%M%S")
        jsonl_filename = (
            Path("openai_batched")
            / self._model_name_or_path.replace("/", "_")
            / f"{self._task_name}_{current_time}.jsonl"
        )

        jsonl_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_filename, "w") as f:
            for i, conversation in enumerate(messages):
                sanitized_messages = []

                for msg in conversation:
                    new_content = []

                    # Check if content is a list (multimodal) or just a string
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            # FIX: Convert 'input_image' to 'image_url'
                            if item.get("type") == "input_image":
                                # Assuming your input_image has a 'data' field with base64
                                base64_data = item["image_url"]
                                new_content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": base64_data},
                                    }
                                )
                            else:
                                new_content.append(
                                    {
                                        "type": "text",
                                        "text": item["text"],
                                    }
                                )

                        sanitized_messages.append({"role": msg["role"], "content": new_content})
                    else:
                        # Text-only messages are fine as-is
                        sanitized_messages.append(msg)

                request_obj = {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self._model_name_or_path,
                        "messages": sanitized_messages,
                        "temperature": gen_kwargs["temperature"],
                        "top_p": gen_kwargs["top_p"],
                        "max_tokens": gen_kwargs["max_new_tokens"],
                    },
                }

                f.write(json.dumps(request_obj) + "\n")

        with open(jsonl_filename, "rb") as f:
            batch_file = self._model.files.create(file=f, purpose="batch")
        log.info("Uploaded file ID: %s", batch_file.id)

        batch_job = self._model.batches.create(
            input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
        )
        log.info("Batch Job Created! Save this ID to retrieve results later: %s", batch_job.id)

        with open(jsonl_filename.parent / f"{self._task_name}_{current_time}.txt", "w") as f:
            f.writelines([f"{batch_job.id}"])

        log.info("Batch Job ID saved. Exiting now.")
        exit()

    def _downscale_images(self, images: list[Image.Image], rag: dict) -> list[Image.Image]:
        """Downscale images based on RAG configuration.

        Args:
        ----
            images (list[Image.Image]): List of images to potentially downscale.
            rag (dict): RAG configuration dictionary containing resizing parameters.

        """
        resize_image_size = rag.get("resize_images", 512)
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
            return -1, x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__
        self._task_name = configurable_task.task_name

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        if self.batch_size == -1:
            self.batch_size_per_gpu = len(requests)
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
            rag = (gen_kwargs or {}).get("rag") or {}
            rag["rag_enabled"] = rag_enabled
            if rag_enabled:
                rag["doc_to_target"] = configurable_task.doc_to_target
                rag["doc_to_visual"] = configurable_task.doc_to_visual
                rag["test_docs"] = configurable_task.test_docs
                self._setup_rag(gen_kwargs)

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

            if gen_kwargs.get("resize_input_image", None) is None:
                gen_kwargs["resize_input_image"] = 512

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

            messages, images_per_request = self._make_history(
                gen_kwargs,
                batched_contexts,
                batched_visuals,
                rag,
                rag_messages,
            )

            answers = self._generate(
                messages, copy.deepcopy(gen_kwargs), copy.deepcopy(rag), do_online=False
            )

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
                    context_tokens_count=None,
                    num_images=images_per_request[idx] if len(images_per_request) > idx else None,
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

    def _pipe_from(
        self,
        rag: dict,
        batched_contexts: list,
        round_idx: int,
        step_idx: int,
        round_idx_to_actual_step: dict,
        answers_by_round: list,
        data_by_round: list,
        prompts_by_round: list,
    ) -> list:
        """Get data from a previous step.

        Args:
        ----
            rag (dict): RAG configuration dictionary containing piping configuration.
            batched_contexts (list): A list of context strings for the batch.
            round_idx (int): The current round index.
            step_idx (int): The current step index.
            round_idx_to_actual_step (dict): Determine the actual step to pipe data from.
            answers_by_round (list): Answers from previous rounds.
            data_by_round (list): Retrieved data from previous rounds.
            prompts_by_round (list): Prompts used in previous rounds.

        """

        def _get_conv(text: str, prompt: str) -> dict[str, str]:
            """Construct a conversation message from text and a prompt.

            Args:
            ----
                text (str): The text to include in the message.
                prompt (str): The prompt template, with a `{}` placeholder for the text.

            """
            return [
                {
                    "type": "input_text",
                    "text": prompt.format(text),
                }
            ]

        format = rag.get("format", "list-captions")
        prompt = rag.get("prompt", "{}")
        prompts = prompt.split("<images>")

        # When caching and reusing the retrieved data, we only need to
        # process the first item in the batch, and then reuse the same
        # data for all items in the batch
        _iterator = (
            range(len(batched_contexts))
            if (not rag.get("cache_and_reuse", False) and not rag.get("reuse", False))
            else range(1)
        )
        retrieved_data = []
        for ctx_id in _iterator:
            remove_ids = []
            if rag.get("keep_full_history", False):
                if rag.get("circular", False):
                    indices_of_circular = [
                        k
                        for k, v in round_idx_to_actual_step.items()
                        if v == round_idx_to_actual_step.get(step_idx) and k < round_idx
                    ]
                    indices_of_circular = [rag.get("pipe_from__first_step")] + indices_of_circular

                else:
                    indices_of_circular = [
                        k
                        for k, v in round_idx_to_actual_step.items()
                        if v == round_idx_to_actual_step.get(rag.get("pipe_from"))
                        and k < round_idx
                    ]

                    if rag.get("remove_when_labels_change", False):
                        indices_of_circular = [indices_of_circular[0] - 1] + indices_of_circular

                indices_of_circular = sorted(list(set(indices_of_circular)))
                text = []
                for idx in indices_of_circular:
                    _t = answers_by_round[idx]
                    if isinstance(_t[0], list):
                        _t = _t[ctx_id]
                    text.append(_t)
                text = list(zip(*text, strict=True))

                if len(rag.get("concatenate_labels", "")) > 0:
                    text = [
                        rag.get("concatenate_labels", "").join([t_item.strip() for t_item in t])
                        for t in text
                    ]

                elif rag.get("remove_when_labels_change", False):
                    for t_idx, t in enumerate(text):
                        # Check if all elements in `t` are the same
                        first_elem = t[0].strip()
                        all_same = all(first_elem == t_item.strip() for t_item in t)
                        if not all_same:
                            remove_ids.append(t_idx)

                    # And keep only the last label
                    text = [t[-1] for t in text]

            else:
                text = answers_by_round[rag.get("pipe_from")]

                # In case text is a list of lists (e.g., multiple answers for each item
                # in the batch), select the right one
                if isinstance(text[0], list):
                    text = text[ctx_id]

            if rag.get("circular", False):
                _from_step = rag.get("pipe_from__first_step", rag.get("pipe_from"))
                _data = data_by_round[_from_step]
            elif rag.get("include_image", False):
                _data = data_by_round[
                    0
                ]  # from first step, i.e., the raw images to re-build the history
            else:
                _data = data_by_round[rag.get("pipe_from")]

            if isinstance(_data[0], list):
                _data = _data[ctx_id]
                self._downscale_images(_data, rag)

            if rag.get("exclude_rag_message") is not None:
                text.pop(rag.get("exclude_rag_message"))
                _data.pop(rag.get("exclude_rag_message"))

            # Remove items where labels changed
            _data = copy.deepcopy(_data)
            for remove_id in sorted(remove_ids, reverse=True):
                text.pop(remove_id)
                _data.pop(remove_id)

            if format == "list-captions":
                text = [f"{idx + 1}. {t.strip()}" for idx, t in enumerate(text)]
                text = "\n".join(text)
                _rag_data = _get_conv(text, prompt)

            elif format == "bullet-list-captions":
                text = [f"- {t.strip()}" for t in text]
                text = "\n".join(text)
                _rag_data = _get_conv(text, prompt)

            elif format == "newlines":
                text = "\n".join([t.strip() for t in text])
                _rag_data = _get_conv(text, prompt)

            elif format == "csv":
                text = ", ".join([t.strip() for t in text])
                _rag_data = _get_conv(text, prompt)

            elif format == "interleaved":
                _rag_data = [
                    {"type": "input_text", "text": prompts[0]},
                ]
                for _image, generated_label in zip(_data, text, strict=True):
                    if isinstance(_image, str):
                        # Already a base64 string
                        base64_string = _image
                    else:
                        _, base64_string = self._prepare_visual_for_context(_image)
                    _rag_data.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_string}",
                        }
                    )
                    _rag_data.append(
                        {"type": "input_text", "text": prompts[-1].format(generated_label)}
                    )

                if rag.get("end_context_prompt") is not None:
                    _rag_data.append({"type": "input_text", "text": rag["end_context_prompt"]})

            elif format == "image-answer-pairs":
                _rag_data = []
                for _image, generated_label in zip(_data, text, strict=True):
                    _, base64_string = self._prepare_visual_for_context(_image)
                    _res = []

                    _res.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompts_by_round[
                                        rag.get("pipe_from__first_step", rag.get("pipe_from"))
                                    ][ctx_id],
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_string}",
                                },
                            ],
                        }
                    )

                    # This happens when we keep the full history for circular
                    if isinstance(generated_label, list | tuple):
                        for gl in generated_label:
                            _res.append({"role": "assistant", "content": _get_conv(gl, prompt)})
                            _res.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": batched_contexts[ctx_id]}
                                    ],
                                }
                            )
                        _res = _res[:-1]  # remove last user prompt

                    else:
                        _res.append(
                            {"role": "assistant", "content": _get_conv(generated_label, prompt)}
                        )

                    _rag_data.append(_res)

            elif format == "all-image-answer-pairs":
                _rag_data = []

                for relabel_idx in range(len(text)):
                    _res = []
                    content = []

                    # Add pre-context prompt if specified
                    # If there's only one element, i.e., the one to be re-labelled, use another
                    # prompt that does not mention multiple images
                    if len(_data) - 1 == 0 and rag.get("context_prompt_no_samples") is not None:
                        content.append(
                            {"type": "input_text", "text": rag["context_prompt_no_samples"]}
                        )

                    # Standard case
                    elif rag.get("context_prompt") is not None:
                        content.append({"type": "input_text", "text": rag["context_prompt"]})

                    for in_context_idx, (_image, generated_label) in enumerate(
                        zip(_data, text, strict=True)
                    ):
                        # The image being relabeled will go last
                        if in_context_idx == relabel_idx:
                            continue

                        _, base64_string = self._prepare_visual_for_context(_image)

                        # Add the image
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )

                        # This happens when we keep the full history for circular
                        if isinstance(generated_label, list | tuple):
                            for gl in generated_label:
                                content.extend(_get_conv(gl, prompt))
                                content.append(
                                    {
                                        "type": "input_text",
                                        "text": batched_contexts[ctx_id],
                                    }
                                )

                            content = content[:-1]

                        else:
                            content.extend(_get_conv(generated_label, prompt))

                    # Add a post-context prompt if specified
                    if (
                        len(_data) - 1 == 0
                        and rag.get("end_context_prompt_no_samples") is not None
                    ):
                        content.append(
                            {"type": "input_text", "text": rag["end_context_prompt_no_samples"]}
                        )
                    elif rag.get("end_context_prompt") is not None:
                        content.append({"type": "input_text", "text": rag["end_context_prompt"]})

                    # Now add the image to be re-labelled
                    _image = _data[relabel_idx]
                    generated_label = text[relabel_idx]
                    _, base64_string = self._prepare_visual_for_context(_image)
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_string}",
                        }
                    )

                    # Put the model's output from previous step
                    if rag.get("keep_label_for_hot_image", False):
                        if isinstance(generated_label, list | tuple):
                            for gl in generated_label:
                                content.extend(_get_conv(gl, rag.get("prompt_for_sample", prompt)))
                                content.append(
                                    {"type": "input_text", "text": batched_contexts[ctx_id]}
                                )

                            content = content[:-1]

                        else:
                            content.extend(
                                _get_conv(generated_label, rag.get("prompt_for_sample", prompt))
                            )

                    # Add the conversation to the results
                    _res.append({"role": "user", "content": content})

                    _rag_data.append(_res)

            else:
                raise NotImplementedError(f"Format {format} not implemented for RAG `pipe_from`")

            retrieved_data.append(_rag_data)

        return retrieved_data

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
        res = []

        gen_kwargs = requests[0].args[1]
        start_req = gen_kwargs.get("start_req", 0)
        end_req = gen_kwargs.get("end_req", len(requests))
        original_len = len(requests)
        requests = copy.deepcopy(requests[start_req:end_req])

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
            return -1, x[0]

        configurable_task: ConfigurableTask = requests[0].args[2].__self__
        self._task_name = configurable_task.task_name

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        global_rag_messages_by_round = {}
        global_answers_by_round = {}
        global_data_by_round = {}

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        for chunk_idx, chunk in enumerate(chunks):
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_to_text,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk, strict=True)
            if chunk_idx == 2:
                exit()
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

            if gen_kwargs.get("resize_input_image", None) is None:
                gen_kwargs["resize_input_image"] = 512

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

            # Set default values for until and max_new_tokens
            until = ["\n"]  # or `[]` to disable its effects

            # Setup RAG for multi-round
            rag_all_steps = (gen_kwargs or {}).get("rag") or {}
            self._setup_rag(gen_kwargs, multi_step=True)

            # Setup mapping from round idx to actual step idx
            round_idx_to_actual_step = {}
            round_idx = 0
            for step_idx, rag_config in rag_all_steps.items():
                if rag_config.get("circular", False):
                    for _ in range(rag_config.get("max_iters", 1)):
                        round_idx_to_actual_step[round_idx] = step_idx
                        round_idx += 1
                else:
                    round_idx_to_actual_step[round_idx] = step_idx
                    round_idx += 1

            # This happens when no RAG is used, so we simply map rounds to steps 1:1
            if rag_all_steps == {}:
                # Get the prompts for each step
                round_idx = len(
                    batched_doc_to_text[0]
                    .keywords.get("model_specific_kwargs", {})
                    .get("default", {})
                    .get("prompts", [])
                )
                round_idx_to_actual_step = {k: k for k in range(round_idx)}

            _max_round = round_idx

            # Store answers (generated text), conversation histories
            prompts_by_round = {}
            batched_round_results, batched_round_info = [], []

            # Store answers by round
            answers_by_round = {}
            data_by_round = {}

            round_idx = 0

            # Iterate rounds
            while True:
                last_round_info = None

                # Get RAG config and set it up for this round
                rag = rag_all_steps.get(round_idx_to_actual_step.get(round_idx, _max_round), {})
                rag_enabled = rag.get("enabled", False)
                rag["rag_enabled"] = rag_enabled
                rag["original_pipe_from"] = rag.get("pipe_from")
                if rag_enabled:
                    rag["doc_to_target"] = configurable_task.doc_to_target
                    rag["doc_to_visual"] = configurable_task.doc_to_visual
                    rag["test_docs"] = configurable_task.test_docs

                if rag.get("circular", False):
                    if rag.get("pipe_from") == -1:
                        rag["pipe_from__first_step"] = (
                            round_idx_to_actual_step.get(round_idx, _max_round) - 1
                        )
                    else:
                        rag["pipe_from__first_step"] = rag.get("original_pipe_from")

                if rag.get("pipe_from") == -1:
                    rag["pipe_from"] = round_idx - 1  # previous round

                # Rounds after the first one
                if round_idx != 0:
                    results = []
                    for ids_idx, doc_id in enumerate(batched_doc_id):
                        previous_round_results = [
                            round_results[ids_idx] for round_results in batched_round_results
                        ]
                        if len(batched_round_info) > 0:
                            last_round_info = batched_round_info[-1][ids_idx]

                        result = batched_doc_to_text[0](
                            self.task_dict[task][split][doc_id],
                            round_idx=round_idx_to_actual_step.get(round_idx, _max_round),
                            previous_round_results=previous_round_results,
                            last_round_info=last_round_info,
                        )
                        results.append(result)

                    (
                        _,  # batched_visuals, but we keep them for RAG. Just make sure
                        # to not put them in the prompt again
                        batched_contexts,
                        batched_terminal_signal,
                        batched_round_results,
                        last_round_info,
                    ) = list(zip(*results, strict=True))

                    batched_round_results = list(zip(*batched_round_results, strict=True))
                    if batched_terminal_signal[0]:  # terminal signal from doc_to_text function
                        break

                if isinstance(batched_contexts, tuple):
                    batched_contexts = list(batched_contexts)

                prompts_by_round[round_idx] = batched_contexts

                for i in range(len(batched_contexts)):
                    if "<image>" in batched_contexts[i]:
                        batched_contexts[i] = batched_contexts[i].replace("<image>", "")

                # ================= #
                # Data preparation  #
                # ================= #

                rag_messages = {}
                if rag_enabled:
                    passthrough_answers = None

                    # Get results from a previous step in the multi-round process
                    if rag.get("pipe_from") is not None:
                        # Using cached results from previous rounds
                        if (
                            rag.get("cache_and_reuse", False)
                            and round_idx in global_rag_messages_by_round
                        ):
                            rag_messages = global_rag_messages_by_round[round_idx]

                        # Passing through previous answers directly
                        elif rag.get("passthrough", False):
                            passthrough_answers = []
                            for ctx_id in range(len(batched_contexts)):
                                indices_of_circular = [
                                    k
                                    for k, v in round_idx_to_actual_step.items()
                                    if v == round_idx_to_actual_step.get(rag.get("pipe_from"))
                                    and k < round_idx
                                ]

                                indices_of_circular = sorted(list(set(indices_of_circular)))
                                text = []
                                for idx in indices_of_circular:
                                    _t = answers_by_round[idx]
                                    if isinstance(_t[0], list):
                                        _t = _t[ctx_id]
                                    text.append(_t)
                                text = list(zip(*text, strict=True))

                                if len(rag.get("concatenate_labels", "")) > 0:
                                    text = [
                                        rag.get("concatenate_labels", "").join(
                                            [t_item.strip() for t_item in t]
                                        )
                                        for t in text
                                    ]
                                else:
                                    text = [t[-1] for t in text]

                                # Get the label corresponding to the input image
                                text = text[rag.get("grab_outputs_from", -1)]
                                passthrough_answers.append(text)

                        # Actually load the data from a previous round
                        else:
                            retrieved_data = self._pipe_from(
                                rag,
                                batched_contexts,
                                round_idx,
                                step_idx,
                                round_idx_to_actual_step,
                                answers_by_round,
                                data_by_round,
                                prompts_by_round,
                            )

                            rag_messages = {}
                            for k, v in enumerate(retrieved_data):
                                rag_messages[k] = v

                            # Caching data for future rounds
                            if rag.get("cache_and_reuse", False):
                                global_rag_messages_by_round[round_idx] = rag_messages

                    # Standard RAG retrieval
                    else:
                        # Loading results from cache
                        if (
                            rag.get("cache_and_reuse", False)
                            and round_idx in global_rag_messages_by_round
                        ):
                            rag_messages = copy.deepcopy(global_rag_messages_by_round[round_idx])

                        # Performing actual retrieval
                        else:
                            rag_messages = self._batch_rag(
                                batched_doc_id,
                                batched_contexts,
                                batched_visuals,
                                {"rag": rag},
                                task,
                                split,
                                round_idx,
                            )

                            if rag.get("cache_and_reuse", False):
                                global_rag_messages_by_round[round_idx] = copy.deepcopy(
                                    rag_messages
                                )

                        # Put the input images in the messages (which are a list of
                        # `PIL.Image`s in this case)
                        if rag.get("include_input", False):
                            rag_messages[0].extend(batched_visuals)

                        if rag.get("include_input_separately", False):
                            for k in rag_messages:
                                rag_messages[k].append(batched_visuals[k])

                # ================= #
                # Answer generation #
                # ================= #

                # When there is RAG with separate queries for each retrieved document
                if rag_enabled and rag.get("separate_queries", False):
                    # Load answers from cache directly
                    # If the input is included, we cannot use cached answers directly -- it's
                    # handled in the `else` branch below
                    if (
                        rag.get("cache_and_reuse", False)
                        and round_idx in global_answers_by_round
                        and not rag.get("include_input", False)
                    ):
                        answers_by_round[round_idx] = global_answers_by_round[round_idx]
                        data_by_round[round_idx] = global_data_by_round[round_idx]
                        batched_round_results.append(answers_by_round[round_idx])

                    # In this case, each retrieved document is processed separately
                    # and we do not process the main query
                    else:
                        rag["rag_enabled"] = False  # Disable RAG for the side-query processing
                        answers_by_round[round_idx] = []
                        data_by_round[round_idx] = []

                        # If we are here, it means that we might find the answers in the cache,
                        # but we still have to process the new inputs (the "hot" images)
                        cached_answers, cached_data = None, None
                        if (
                            rag.get("cache_and_reuse", False)
                            and round_idx in global_answers_by_round
                        ):
                            cached_answers = copy.deepcopy(global_answers_by_round[round_idx])

                            # Since `cached_answers` contains the same answers repeated for
                            # each item in the batch, we can just keep the first one (`[0]`)
                            cached_answers = cached_answers[0][: rag.get("num_samples")]

                            # Same for `cached_data`
                            cached_data = global_data_by_round[round_idx][0]

                            # Remove the cached items from the current rag messages, so that
                            # we only process the new ones
                            rag_messages[0] = rag_messages[0][rag.get("num_samples") :]

                        # Process each item in the batch separately
                        for i, _visuals in enumerate(rag_messages.values()):
                            _history = None
                            if rag.get("format", "") in (
                                "image-answer-pairs",
                                "all-image-answer-pairs",
                            ):
                                _history = _visuals

                            # Construct the actual conversation for the model
                            _contexts = [batched_contexts[i]] * len(_visuals)
                            messages, images_per_request = self._make_history(
                                gen_kwargs,
                                _contexts,
                                _visuals,
                                rag,
                                rag_messages,
                                history=_history,
                                append_history=True,
                                step_idx=round_idx,
                            )

                            # Using sub-batches to handle requests for this batch item
                            if rag.get("gen_batch_size", None) is not None:
                                # Process in smaller batches to avoid OOM
                                answers = []
                                for j in range(0, len(messages), rag["gen_batch_size"]):
                                    generation_output = self._generate(
                                        messages[j : j + rag["gen_batch_size"]],
                                        gen_kwargs,
                                        rag,
                                    )
                                    answers.append(generation_output)

                                answers = _flatten_list(answers)

                            # Process all at once
                            else:
                                answers = self._generate(messages, gen_kwargs, rag, do_online=True)

                            # Post-process the answers and store them
                            for i, answer in enumerate(answers):
                                for term in until:
                                    if len(term) > 0:
                                        answer = answer.split(term)[0]

                                if "," in answer and rag.get("clean_output_labels", False):
                                    answer = answer.split(",")

                                    # Keep only the last non-empty label
                                    for test_idx in reversed(range(len(answer))):
                                        temp_answer = answer[test_idx].strip().replace('"', "")
                                        if len(temp_answer) > 0:
                                            answer = temp_answer
                                            break

                                    # Fallback in case all labels were empty after cleaning
                                    if isinstance(answer, list):
                                        answer = answer[0]

                                answers[i] = answer

                            # If we are using cached answers for the fixed items, we have
                            # to store them as if we generated them now. Since fixed images
                            # are the beginning of the batch, we prepend them to the ones
                            # we've just generated
                            if cached_answers is not None:
                                answers = cached_answers + answers
                            if cached_data is not None:
                                _visuals = cached_data + _visuals

                            answers_by_round[round_idx].append(answers)
                            data_by_round[round_idx].append(_visuals)

                        # If caching is enabled and we include input, prepare the answers
                        # and data accordingly
                        if rag.get("cache_and_reuse", False) and rag.get("include_input", False):
                            # Here, `len(rag_messages)` should be 1
                            answers_by_round[round_idx] = answers_by_round[round_idx] * len(
                                batched_contexts
                            )
                            data_by_round[round_idx] = data_by_round[round_idx] * len(
                                batched_contexts
                            )

                            # Now, for each list in the two lists, retain only the first
                            # N items and then the i-th one
                            retain_first = rag.get("num_samples")
                            for i in range(len(answers_by_round[round_idx])):
                                item_to_retain = retain_first + i
                                answers_by_round[round_idx][i] = (
                                    answers_by_round[round_idx][i][:retain_first]
                                    + answers_by_round[round_idx][i][
                                        item_to_retain : item_to_retain + 1
                                    ]
                                )
                                data_by_round[round_idx][i] = (
                                    data_by_round[round_idx][i][:retain_first]
                                    + data_by_round[round_idx][i][
                                        item_to_retain : item_to_retain + 1
                                    ]
                                )

                        # When caching and reusing results, reuse the first result for
                        # all items in the batch
                        if rag.get("cache_and_reuse", False):
                            # When the input is not included, i.e., when all items are
                            # *actually* the same, we can just reuse the cached answers
                            # faking answers for all items in the batch
                            if not rag.get("include_input", False):
                                for _ in range(1, len(batched_contexts)):
                                    answers_by_round[round_idx].append(
                                        answers_by_round[round_idx][0]
                                    )
                                    data_by_round[round_idx].append(data_by_round[round_idx][0])

                            if round_idx not in global_answers_by_round:
                                global_answers_by_round[round_idx] = answers_by_round[round_idx]
                                global_data_by_round[round_idx] = data_by_round[round_idx]

                        batched_round_results.append(answers_by_round[round_idx])

                        rag["rag_enabled"] = True  # Re-enable RAG for compatibility

                # Standard generation process
                else:
                    if rag_enabled and rag.get("passthrough", False):
                        # Special case where circular has been used to generate labels: we just
                        # pass the result through here
                        if passthrough_answers is None:
                            raise ValueError(
                                "`passthrough_answers` cannot be None when "
                                "`rag['passthrough']` is `True`"
                            )
                        answers = passthrough_answers
                        answers_by_round[round_idx] = answers
                        generation_output = None
                        messages = [[]] * len(answers)

                    else:
                        # Standard case: we process the main query (i.e., the context and
                        # provided visuals) with optional retrieved documents if RAG is enabled
                        messages, images_per_request = self._make_history(
                            gen_kwargs,
                            batched_contexts,
                            batched_visuals,
                            copy.deepcopy(rag),
                            copy.deepcopy(rag_messages),
                            history=last_round_info,
                            step_idx=round_idx,
                        )

                        answers = self._generate(messages, gen_kwargs, rag, do_online=False)

                    answers_by_round[round_idx] = answers

                    # Get clean answers and store them in the conversation history
                    for i, answer in enumerate(answers):
                        for term in until:
                            if len(term) > 0:
                                answer = answer.split(term)[0]

                        # Remove duplicates
                        if rag.get("clean_answer", False):
                            seen = set()
                            seen_add = seen.add

                            answer = answer.strip()
                            answer = answer.split(",")
                            answer = ", ".join(
                                [
                                    a.strip().replace('"', "")
                                    for a in answer
                                    if len(a.strip()) > 0
                                    if not (a in seen or seen_add(a))
                                ]
                            )

                        answers[i] = answer
                        messages[i].append(
                            {
                                "role": "assistant",
                                "content": [{"type": "input_text", "text": answer}],
                            }
                        )

                    batched_round_results.append(answers)
                    batched_round_info.append(messages)

                    if rag.get("store_memory", False):
                        self._retriever[0].store_memory(batched_visuals, answers)

                reset_cache = rag.get("reset_cache", False)
                # Check if it's a bool or an int
                # If it's an int, we reset the cache every `reset_cache` batches
                # This is like having a larger batch size
                if not isinstance(reset_cache, bool):
                    reset_cache = (chunk_idx + 1) % reset_cache == 0
                if reset_cache:
                    global_rag_messages_by_round = {}
                    global_answers_by_round = {}
                    global_data_by_round = {}

                round_idx += 1

                # Restore original `pipe_from` to correctly support circular contexts
                rag["pipe_from"] = rag.get("original_pipe_from")

                # Cleanup RAG keys
                rag.pop("doc_to_target", None)
                rag.pop("doc_to_visual", None)
                rag.pop("test_docs", None)

            answers = list(zip(*batched_round_results, strict=True))
            for idx, (ans, context) in enumerate(zip(answers, batched_contexts, strict=True)):
                _ans = TaskSingleOutput(
                    answer=ans,
                    context=context,
                    context_tokens_count=None,
                    num_images=images_per_request[idx] if len(images_per_request) > idx else None,
                    loglikelihood=None,
                    perplexity=None,
                )
                res.append(_ans)
                self.cache_hook.add_partial(
                    "generate_until_multi_round", (context, gen_kwargs), ans
                )
                pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        # Adding back dummy results for skipped requests
        # Doing this after reordering because the `reordered` object only
        # knows about the actual requests being processed
        if start_req > 0:
            res = [TaskSingleOutput(answer=[""], is_dummy=True) for _ in range(start_req)] + res
        if end_req < original_len:
            res = res + [
                TaskSingleOutput(answer=[""], is_dummy=True) for _ in range(original_len - end_req)
            ]

        pbar.close()
        return res


@register_model("gpt-4o-mini")
def gpt4o_mini(**model_kwargs) -> Model:
    """Load the GPT 4o-mini model."""
    model_name_or_path = "gpt-4o-mini-2024-07-18"
    model = OpenAIAPI(model_name_or_path, **model_kwargs)
    return model


@register_model("gpt-4o")
def gpt4o(**model_kwargs) -> Model:
    """Load the GPT 4o model."""
    model_name_or_path = "gpt-4o-2024-08-06"
    model = OpenAIAPI(model_name_or_path, **model_kwargs)
    return model
