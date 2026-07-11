"""Embedding models — Strategy pattern. Swap by changing the class."""
from __future__ import annotations

import logging
from typing import Protocol
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embedding model — no API cost, no latency."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=True)
