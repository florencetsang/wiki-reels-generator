"""Local sentence-transformers embeddings."""

from __future__ import annotations

import json
import logging
from array import array


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, summary: str, categories: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()

            logging.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

        source = (
            f"Summary:\n{summary}\n\nWikipedia categories:\n"
            + "\n".join(f"- {category}" for category in categories)
        )
        vector = self._model.encode(
            source,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vector]


def decode_embedding(value: str | bytes) -> list[float]:
    if isinstance(value, bytes):
        # Local databases created before embeddings were stored as JSON.
        vector = array("f")
        vector.frombytes(value)
        return list(vector)
    return json.loads(value)
