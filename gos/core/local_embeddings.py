from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fast_graphrag._llm._base import BaseEmbeddingService


@dataclass
class LocalEmbeddingService(BaseEmbeddingService):
    """Offline embedding backend backed by a local `sentence-transformers` model.

    No network calls and no API key. `model` is a sentence-transformers model
    id (e.g. ``sentence-transformers/all-MiniLM-L6-v2``) or a local path to one.
    The model is loaded once, eagerly, so a `GOS_EMBEDDING_DIM` mismatch is
    reported immediately instead of surfacing later as an opaque vector-store
    error.
    """

    device: str | None = field(default=None)
    _st_model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require the `sentence-transformers` package, "
                "which is not installed. Install it with `uv sync --extra local` "
                "(or `pip install sentence-transformers`), or point "
                "GOS_EMBEDDING_MODEL at a hosted provider (openai/, gemini/, "
                "openrouter/) instead."
            ) from exc

        model = SentenceTransformer(self.model, device=self.device)
        native_dim = model.get_sentence_embedding_dimension()
        if native_dim != self.embedding_dim:
            raise ValueError(
                f"GOS_EMBEDDING_DIM is set to {self.embedding_dim}, but "
                f"`{self.model}` produces {native_dim}-dimensional embeddings. "
                f"Set GOS_EMBEDDING_DIM={native_dim} in .env. If you already "
                "have an indexed workspace built with a different embedding "
                "model or dimension, it must be rebuilt -- embeddings from "
                "different models are not interchangeable."
            )

        self._st_model = model

    async def encode(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if model and model != self.model:
            raise ValueError(
                f"LocalEmbeddingService is bound to `{self.model}`; got an "
                f"override request for `{model}`. A local embedding model "
                "cannot be swapped per call -- set GOS_EMBEDDING_MODEL instead."
            )

        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # sentence-transformers is sync and CPU/GPU-bound; keep it off the event loop.
        vectors = await asyncio.to_thread(
            self._st_model.encode,
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
