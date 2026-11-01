"""Retrieval contracts shared by Qdrant and the NumPy reference index."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from portfolio_ai.rag.domain.documents import Chunk
from portfolio_ai.rag.embeddings.base import Vector


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is more similar


class VectorIndex(Protocol):
    def search(
        self,
        vector: Vector,
        *,
        top_k: int,
        score_threshold: float | None = None,
        exact: bool = False,
    ) -> list[SearchResult]: ...
