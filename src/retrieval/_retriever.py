import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src import utils
from src.retrieval import RetrievalDatabase, get_images_by_key

__all__ = ["Retriever"]

log = utils.get_logger(__name__, rank_zero_only=True)


class Retriever:
    """Retriever to retrieve similar images from a retrieval database."""

    def __init__(
        self,
        path: str | Path,
        model_name: str = "openai/clip-vit-base-patch32",
        device_map: str = "auto",
    ) -> None:
        """Retrieve similar images from a retrieval database.

        Args:
        ----
            path (str | Path): Path to the retrieval database.
            model_name (str): Name of the pre-trained model to use.
            device_map (str): Device map to use for the model. Either 'auto' or
                'cpu'. If 'auto', the model will be loaded on GPU if available.

        """
        self._path = path
        self._database = RetrievalDatabase(path)

        self._device = "cuda" if torch.cuda.is_available() and device_map == "auto" else "cpu"

        self._model = AutoModel.from_pretrained(model_name, device_map=device_map)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_name)

        self._vocab_transform = lambda x: x

        self._transform = None
        if model_name in [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ]:
            channel_stats = dict(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

            self._transform = T.Compose(
                [
                    T.Resize(size=(224, 224), interpolation=Image.Resampling.BICUBIC),
                    T.CenterCrop(224),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(**channel_stats),
                ]
            )

    def set_vocab_transform(self, rag: dict) -> None:
        """Set the vocabulary transform function based on the RAG configuration.

        Args:
        ----
            rag (dict): RAG configuration dictionary.

        """
        if rag.get("format", "list-captions") == "cased":
            self._vocab_transform = utils.default_vocabulary_transforms()

    @torch.no_grad()
    def retrieve(
        self,
        queries: str | list[str] | Image.Image | list[Image.Image],
        input_type: str = "image",
        search_modality: str = "text",
        num_samples: int = 10,
    ) -> list[dict]:
        """Retrieve similar images from the database.

        Args:
        ----
            queries (str | list[str] | Image.Image | list[Image.Image]): Input queries
            input_type (str): Type of the input queries. Either 'image' or 'text'.
            search_modality (str): Modality to search in the database. Either 'image'
                or 'text'.
            num_samples (int): Number of samples to retrieve.

        """
        assert input_type in ["image", "text"], "input_type must be either 'image' or 'text'"
        assert search_modality in [
            "image",
            "text",
        ], "search_modality must be either 'image' or 'text'"

        # Ensure the input is a list
        if not isinstance(queries, list):
            queries = [queries]

        if input_type == "image":
            for i in range(len(queries)):
                if isinstance(queries[i], str):
                    queries[i] = Image.open(queries[i]).convert("RGB")

            if self._transform is not None:
                queries = torch.stack(
                    [
                        self._transform(
                            T.functional.to_image(image).to(self._device, non_blocking=True)
                        )
                        for image in queries
                    ],
                    dim=0,
                )
                inputs = {
                    "pixel_values": queries,
                }
            else:
                inputs = self._processor(images=queries, return_tensors="pt").to(self._device)

            embeddings = self._model.get_image_features(**inputs)

        elif input_type == "text":
            inputs = self._processor(
                text=queries, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            embeddings = self._model.get_text_features(**inputs)

        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        embeddings = embeddings.cpu().numpy()

        # Retrieve similar images from the database
        results = self._database.query(
            embeddings, modality=search_modality, num_samples=num_samples
        )

        return results

    def retrieve_and_prepare(
        self, gen_kwargs: dict, images: list[Image.Image], prepare: Callable | None = None
    ) -> list[list[dict[str, Any]]] | None:
        """Retrieve relevant data for each image and prepare it for generation.

        Args:
        ----
            gen_kwargs (dict): Generation keyword arguments. It must contain a 'rag' object
            images (list[Image.Image]): List of input images.
            prepare (Callable, optional): Function to prepare the retrieved images for generation.

        """
        rag = (gen_kwargs or {}).get("rag") or {}

        # Input is always images. Search is conducted by comparing with
        # either text embeddings or image embeddings
        search_modality = rag.get("search_modality", "text")

        # From the retrieved data, determine the type of data to grab:
        # the retrieved data itself or the corresponding image/text
        grab_data_type = rag.get("grab_data_type", search_modality)

        results_set = self.retrieve(
            images,
            input_type="image",
            search_modality=search_modality,
            num_samples=rag.get("num_samples", 10),
        )

        # Shuffle the data?
        if rag.get("shuffle", False):
            for results in results_set:
                random.shuffle(results)

        if results_set is None:
            return None

        # ============================ #
        # Use the retrieved image data #
        # ============================ #
        if grab_data_type == "image":
            prompts = rag.get("prompt", "")
            format = rag.get("format", "list-captions")
            limit = rag.get("limit", None)

            prompts = prompts.split("<images>")
            assert len(prompts) == 2, "Prompt must contain exactly one <images> token."

            # Collect all image keys to retrieve in a single batch
            all_images_keys = []
            batch_idxs = []
            for batch_idx, results in enumerate(results_set):
                image_keys = [x.get("image_path") for x in results]
                # Enforce a limit on the maximum number of images retrieved
                if limit is not None:
                    image_keys = image_keys[:limit]

                all_images_keys.extend(image_keys)
                batch_idxs.extend([batch_idx] * len(image_keys))

            # Extract the images
            retrieved_images = self.get_images(all_images_keys)

            # Format the retrieved data for each sample
            rag_data_set = []
            for _, results in enumerate(results_set):
                if limit is not None:
                    results = results[:limit]

                image_keys = [x.get("image_path") for x in results]

                if prepare is None:
                    raise ValueError(
                        "When retrieving images, a `prepare` function must be provided to convert"
                        " images to the desired format."
                    )

                retrieved_images = [
                    prepare(retrieved_images.get(key))
                    for key in image_keys
                    if key in retrieved_images
                ]

                # Text formatting
                if format == "list-captions":
                    captions = [
                        f"{idx + 1}. {result.get('caption')}" for idx, result in enumerate(results)
                    ]
                    rag_text = "\n".join(captions)

                elif format == "cased":
                    captions = [result.get("caption") for result in results]
                    vocabularies = self._vocab_transform(captions)
                    words = list(set([vocab or ["object"] for vocab in vocabularies]))

                    rag_text = ", ".join(words)

                rag_data = [
                    {"type": "text", "text": prompts[0]},
                    *[
                        {"type": "image", "image": f"data:image/jpeg;base64,{img}"}
                        for img in retrieved_images
                    ],
                ]

                if format != "none":
                    rag_data.append({"type": "text", "text": prompts[1].format(rag_text)})

                rag_data_set.append(rag_data)

            return rag_data_set

        # =========================== #
        # Use the retrieved text data #
        # =========================== #
        elif grab_data_type == "text":
            format = rag.get("format", "list-captions")
            prompt = rag.get("prompt", "")
            caption_key = rag.get("caption_key", "caption")

            if format == "list-captions":
                rag_data_set = []
                for results in results_set:
                    rag_data = [
                        f"{idx + 1}. {result.get(caption_key).strip()}"
                        for idx, result in enumerate(results)
                    ]
                    rag_data = "\n".join(rag_data)
                    rag_data_set.append(rag_data)

            elif format == "cased":
                rag_data_set = []
                for results in results_set:
                    captions = [result.get(caption_key) for result in results]
                    vocabularies = self._vocab_transform(captions)
                    words = list(set([vocab or ["object"] for vocab in vocabularies]))

                    # Shuffle the words
                    if rag.get("shuffle", False):
                        random.shuffle(words)

                    rag_data = ", ".join(words)
                    rag_data_set.append(rag_data)

            else:
                raise NotImplementedError(f"RAG format '{format}' is not supported.")

            # Format the prompt with the retrieved data for each sample
            prompts = [prompt.format(rag_data) for rag_data in rag_data_set]

            # Return a dictionary for each input sample
            # The dictionary is inside a list to be compatible with the image format
            return [
                [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
                for prompt in prompts
            ]

        else:
            raise NotImplementedError(f"Search modality '{search_modality}' is not supported.")

    def __call__(self, *args, **kwds) -> list[dict]:
        """Alias for the `retrieve` method."""
        return self.retrieve(*args, **kwds)

    def get_images(self, keys: list[str], from_tar: bool = False) -> dict[str, Image.Image | None]:
        """Get images by their keys.

        Args:
        ----
            keys (list[str]): List of image keys.
            from_tar (bool): Whether to load images from tar files. Default is False.

        """
        root = Path(self._path).parent.parent / "images"

        return get_images_by_key(keys, root, from_tar=from_tar)
