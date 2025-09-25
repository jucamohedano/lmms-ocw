import json
from pathlib import Path

import torch


class RetrievalTensorDatabase:
    """Retrieval database for tensors stored in .pt files.

    Args:
    ----
        database_dir (str): Path to the tensor directory.
        device (str): Device to load the tensors on.

    """

    def __init__(
        self,
        database_dir: str,
        device: str = "cpu",
    ) -> None:
        self._database_dir = database_dir
        self._device = device

        path = Path(database_dir).parent
        name = Path(database_dir).name
        embeddings_fp = path / f"{name}.pt"
        metadata = path / f"{name}_metadata.json"

        self._embeddings = torch.load(embeddings_fp).to(self._device)
        self._embeddings = self._embeddings.t()  # Transpose for easier matmul
        with open(metadata) as f:
            self._metadata = json.load(f)

    @property
    def _metadata_provider(self) -> str:
        """Return the metadata provider."""
        return self._metadata

    def _find_closest(
        self, query: torch.Tensor, k: int = 10, batch_size: int = 1024
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the k closest samples in the database to the query tensor.

        Args:
        ----
            query (torch.Tensor): Query tensor of shape (num_queries, embedding_dim).
            k (int): Number of closest samples to return.
            batch_size (int): Batch size for processing the database.

        """
        query = query.to(self._device)
        N = self._embeddings.shape[0]

        similarities = []
        for i in range(0, N, batch_size):
            batch = self._embeddings[i : i + batch_size]
            # (#queries, emb_dim) x (emb_dim, #batch) -> (#queries, #batch)
            sim = torch.matmul(query, batch)
            similarities.append(sim.cpu())

        similarities = torch.cat(similarities, dim=1)  # (#queries, N)
        top_values, top_indices = torch.topk(similarities, k=k, dim=1)  # (#queries, k)

        return top_values, top_indices

    def query(
        self,
        query: torch.Tensor,
        num_samples: int = 10,
        batch_size: int = 1024,
        **kwargs,
    ) -> list[list[dict]]:
        """Query the database with a tensor and return the closest samples.

        Args:
        ----
            query (torch.Tensor): Query tensor of shape (num_queries, embedding_dim).
            num_samples (int): Number of closest samples to return for each query.
            batch_size (int): Batch size for processing the database.
            **kwargs: Additional keyword arguments (not used).

        """
        distances, indices = self._find_closest(query, k=num_samples, batch_size=batch_size)

        results = []
        for dists, inds in zip(distances, indices, strict=True):
            res = []
            for dist, ind in zip(dists, inds, strict=True):
                meta = self._metadata[str(ind.item())]
                meta["distance"] = dist.item()
                res.append(meta)
            results.append(res)

        return results
