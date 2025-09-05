from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.retrieval import RetrievalDatabase


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
        self._database = RetrievalDatabase(path)

        self._device = "cuda" if torch.cuda.is_available() and device_map == "auto" else "cpu"

        self._model = AutoModel.from_pretrained(
            model_name, device_map=device_map
        )  # .to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_name)

        self._transform = None
        if model_name in [
            "openai/clip-vit-base-patch32",
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

    def __call__(self, *args, **kwds) -> list[dict]:
        """Alias for the `retrieve` method."""
        return self.retrieve(*args, **kwds)
