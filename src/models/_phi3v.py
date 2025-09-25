import os
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from src import utils
from src.data.tasks import TaskInstance, TaskSingleOutput
from src.data.tasks._manager import ConfigurableTask
from src.models._api import register_model
from src.models._base import Model
from src.retrieval import Retriever

from ._phi3v_processor import Phi3VProcessor

__all__ = ["Phi3v"]

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


class Phi3v(Model):
    """Phi3v model.

    Args:
    ----
        model_name_or_path (str): The name or path of the pre-trained model to use. Defaults to
            "microsoft/Phi-3-vision-128k-instruct".
        attn_implementation (str, optional): The attention implementation to use. Defaults to
            "flash_attention_2" if `flash_attn` is installed, "eager" otherwise.
        use_cache (bool): Whether to use KV cache during generation. Defaults to True.
        batch_size (int): The batch size to use for inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    References:
    ----------
        - https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
        - https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/
        - https://github.com/microsoft/Phi-3CookBook

    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/Phi-3-vision-128k-instruct",
        attn_implementation: str | None = (
            "flash_attention_2" if utils.package_available("flash_attn") else "eager"
        ),
        use_cache: bool = True,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._attn_implementation = attn_implementation
        self._use_cache = use_cache

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU", "DEEPSPEED"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
            "torch_dtype": self.dtype,
            "_attn_implementation": self._attn_implementation,
        }
        processor_kwargs = {
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
        }

        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name_or_path, **model_kwargs
        )
        # Using fixed version from pull request rather than AutoProcessor
        self._processor = Phi3VProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )
        self._processor.tokenizer.padding_side = "left"
        self._tokenizer = self._processor.tokenizer

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
    ) -> None | list[dict[str, list]]:
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
            payload = {
                "type": "image",
                "image": image,
            }

            return payload

        def result_callback(
            rag_data: list[dict[str, Any]], images: list | None
        ) -> dict[str, list]:
            # Keep only the text elements and add a placeholder for images if any
            _rag_data = []
            for elem in rag_data:
                if elem.get("type") == "text":
                    _rag_data.append(elem.get("text", ""))
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
        result = list(cast(list[dict[str, list]], result))

        return result

    def _prepare_image_tokens(
        self, batched_visuals: list[Image.Image], start_counter: int, return_as_list: bool = False
    ) -> tuple[str | list[str], int, int]:
        """Prepare image tokens for the model.

        Args:
        ----
            batched_visuals (list[Image.Image]): List of images to prepare tokens for.
            start_counter (int): Starting counter for image tokens.
            return_as_list (bool): Whether to return image tokens as a list. Defaults to False.

        """
        counter = start_counter

        if return_as_list:
            image_tokens = []
            for _ in range(len(batched_visuals)):
                image_tokens.append(f"<|image_{counter+1}|>")
                counter += 1
        else:
            image_tokens = ""
            for _ in range(len(batched_visuals)):
                image_tokens += f"<|image_{counter+1}|>\n"
                counter += 1

        return image_tokens, counter, start_counter

    def _compose_rag_message(self, rag_message: list[str], rag_image_tokens: list[str]) -> str:
        """Compose the RAG message by replacing <images> placeholders with actual image tokens.

        Args:
        ----
            rag_message (list[str]): The RAG message with placeholders.
            rag_image_tokens (list[str]): The image tokens to replace the placeholders.

        """
        composed_message = []

        image_idx = 0
        for x in rag_message:
            if x != "<images>":
                composed_message.append(x)
            else:
                if image_idx < len(rag_image_tokens):
                    composed_message.append(rag_image_tokens[image_idx])
                    image_idx += 1
                else:
                    log.warning(
                        "Not enough images to replace <images> placeholder in RAG message."
                    )

        return "\n".join(composed_message)

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

        # num_iters = (len(requests) + self.batch_size - 1) // self.batch_size
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

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            if isinstance(batched_contexts, tuple):
                batched_contexts = list(batched_contexts)

            # RAG setup (guard-style, avoids deep nesting later)
            rag_enabled = self._is_rag_enabled(gen_kwargs)
            if rag_enabled:
                rag = (gen_kwargs or {}).get("rag") or {}
                rag["doc_to_target"] = configurable_task.doc_to_target
                rag["doc_to_visual"] = configurable_task.doc_to_visual
                rag["test_docs"] = configurable_task.test_docs
                rag_position = rag.get("position", "pre-sample")
                self._setup_rag(gen_kwargs)

            # RAG optimization: pre-load all images and retrieve in batches
            rag_messages = {}
            if rag_enabled:
                batched_visuals_for_rag = {}
                for i, _ in enumerate(batched_contexts):
                    visual = batched_visuals[i] if i < len(batched_visuals) else None
                    if (
                        visual is not None
                        and len(visual) == 1
                        and isinstance(visual[0], Image.Image)
                    ):
                        visual = visual[0].convert("RGB")
                        batched_visuals_for_rag[i] = visual

                # Batch retrieve data
                retrieved_data = self._retrieve(gen_kwargs, list(batched_visuals_for_rag.values()))
                for k, v in zip(batched_visuals_for_rag.keys(), retrieved_data, strict=True):
                    rag_messages[k] = v

            image_counter = 0
            _batched_visuals = []
            for i in range(len(batched_contexts)):
                _batched_visual_to_add = batched_visuals[i]

                rag_message = None
                if rag_enabled:
                    if i in rag_messages:
                        rag_message = rag_messages[i]
                    else:
                        rag_message = self._retrieve(gen_kwargs, visual)[0]
                    # Get the actual message and the images
                    rag_message, rag_images = rag_message["rag_data"], rag_message["images"]

                # Build the prompt with <image> tokens and RAG content, if any
                if "<image>" in batched_contexts[i]:
                    query = "" + batched_contexts[i]
                    img_placeholder_count = 1
                    while "<image>" in query:
                        query = query.replace("<image>", f"<|image_{img_placeholder_count}|>", 1)
                        img_placeholder_count += 1
                else:
                    # No RAG: just use the vanilla model
                    if not rag_enabled or rag_message is None:
                        image_tokens, image_counter, _ = self._prepare_image_tokens(
                            batched_visuals[i], image_counter
                        )  # noqa: E501
                        query = image_tokens + batched_contexts[i]

                    # RAG is enabled and there is actual retrieved content
                    elif rag_enabled and rag_message is not None:
                        # Retrieved data goes at the beginning of the context
                        if rag_position == "pre-sample":
                            rag_image_tokens, image_counter, _ = self._prepare_image_tokens(
                                rag_images, image_counter, return_as_list=True
                            )  # noqa: E501
                            image_tokens, image_counter, _ = self._prepare_image_tokens(
                                batched_visuals[i], image_counter
                            )  # noqa: E501
                            rag_message = self._compose_rag_message(rag_message, rag_image_tokens)

                            if len(rag_images) > 0:
                                _batched_visual_to_add = rag_images + _batched_visual_to_add

                            query = "".join(
                                [
                                    rag_message,
                                    "\n\n",
                                    image_tokens,
                                    batched_contexts[i],
                                ]
                            )

                        # Retrieved data goes after the image, before the query
                        if rag_position == "post-sample":
                            image_tokens, image_counter, _ = self._prepare_image_tokens(
                                batched_visuals[i], image_counter
                            )  # noqa: E501
                            rag_image_tokens, image_counter, _ = self._prepare_image_tokens(
                                rag_images, image_counter
                            )  # noqa: E501
                            rag_message = self._compose_rag_message(rag_message, rag_image_tokens)

                            if len(rag_images) > 0:
                                _batched_visual_to_add = _batched_visual_to_add + rag_images

                            query = "".join(
                                [
                                    image_tokens,
                                    "\n",
                                    rag_message,
                                    "\n\n",
                                    batched_contexts[i],
                                ]
                            )

                        # Retrieved data goes at the end of the context
                        if rag_position == "post-sample-and-query":
                            image_tokens, image_counter, _ = self._prepare_image_tokens(
                                batched_visuals[i], image_counter
                            )  # noqa: E501
                            rag_image_tokens, image_counter, _ = self._prepare_image_tokens(
                                rag_images, image_counter
                            )  # noqa: E501
                            rag_message = self._compose_rag_message(rag_message, rag_image_tokens)

                            if len(rag_images) > 0:
                                _batched_visual_to_add = _batched_visual_to_add + rag_images

                            query = "".join(
                                [
                                    image_tokens,
                                    "\n",
                                    batched_contexts[i],
                                    "\n\n",
                                    rag_message,
                                ]
                            )

                _batched_visuals.append(_batched_visual_to_add)

                messages = [{"role": "user", "content": query}]
                batched_contexts[i] = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # Moved here so that the number of visuals corresponds to the number of contexts
            # and the code that inserts <image> tokens correctly sees the number of images
            # associated to each context
            images_per_request = [len(x) for x in _batched_visuals]
            batched_visuals = _flatten_list(_batched_visuals)

            context = batched_contexts
            input_ids = self.processor(
                text=context, images=batched_visuals, return_tensors="pt"
            ).to(self.device, self.model.dtype)

            # Setting default parameters.
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # Remove functions from gen_kwargs["rag"] so that pickle doesn't complain
            if rag_enabled:
                rag.pop("doc_to_target")
                rag.pop("doc_to_visual")
                rag.pop("test_docs")

            # Generate answer
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eod_id
            )
            with torch.inference_mode():
                generate_ids = self.model.generate(
                    **input_ids,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=gen_kwargs["temperature"] > 0,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self._use_cache,
                )
            generate_ids = generate_ids[:, input_ids["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for idx, (ans, context) in enumerate(zip(response, batched_contexts, strict=True)):
                _ans = TaskSingleOutput(
                    answer=ans,
                    context=context,
                    context_tokens_count=input_ids["attention_mask"][idx].sum(),
                    num_images=images_per_request[idx],
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
        res = []

        log.warning("The model do not fully support multi-turn as it does not store history")

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
                batched_doc_to_text,
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

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            # Terminate when receiving signal from the doc_to_text
            round_idx = 0
            batched_round_results, batched_round_info = [], []
            while True:
                last_round_info = None

                # Get current round visual and context from doc_to_text function
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
                            round_idx=round_idx,
                            previous_round_results=previous_round_results,
                            last_round_info=last_round_info,
                        )
                        results.append(result)

                    (
                        batched_visuals,
                        batched_contexts,
                        batched_terminal_signal,
                        batched_round_results,
                        last_round_info,
                    ) = list(zip(*results, strict=True))

                    batched_round_results = list(zip(*batched_round_results, strict=True))
                    last_round_info = last_round_info[-1]
                    if batched_terminal_signal[0]:  # terminal signal from doc_to_text function
                        break

                if isinstance(batched_contexts, tuple):
                    batched_contexts = list(batched_contexts)

                if isinstance(batched_visuals, tuple):
                    batched_visuals = list(*batched_visuals)

                for i in range(len(batched_contexts)):
                    if "<image>" in batched_contexts[i]:
                        query = "" + batched_contexts[i]
                        img_placeholder_count = 1
                        while "<image>" in query:
                            query = query.replace(
                                "<image>", f"<|image_{img_placeholder_count}|>", 1
                            )
                            img_placeholder_count += 1
                    else:
                        query = ""
                        for placeholder_id in range(len(batched_visuals)):
                            query += f"<|image_{placeholder_id+1}|>\n"
                        query += batched_contexts[i]
                    messages = [{"role": "user", "content": query}]
                    batched_contexts[i] = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                context = batched_contexts[0]
                input_ids = self.processor(
                    text=context, images=batched_visuals, return_tensors="pt"
                ).to(self.device, self.model.dtype)

                # Setting default parameters.
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                # Generate answer
                pad_token_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eod_id
                )
                generate_ids = self.model.generate(
                    **input_ids,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=gen_kwargs["temperature"] > 0,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self._use_cache,
                )
                generate_ids = generate_ids[:, input_ids["input_ids"].shape[1] :]
                responses = self.processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                round_idx += 1
                batched_round_results.append(responses)

            res.extend(list(zip(*batched_round_results, strict=True)))

            self.cache_hook.add_partial(
                "generate_until_multi_round", (context, gen_kwargs), batched_round_results
            )
            pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res


@register_model("phi3v")
def phi3v(**model_kwargs) -> Model:
    """Load the Phi3V model with 4B params."""
    model_name_or_path = "microsoft/Phi-3-vision-128k-instruct"
    model = Phi3v(model_name_or_path, **model_kwargs)
    return model
