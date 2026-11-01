"""The embedding contract every model adapter must satisfy.

An embedding maps text to a fixed-length vector of floats. Texts with similar
meaning land close together. all-MiniLM-L6-v2 produces 384 numbers per text.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from portfolio_ai.rag.errors import EmbeddingDimensionError, EmbeddingError

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]

_MIN_NORM = 1e-12


@dataclass(frozen=True)
class EmbeddingSpec:
    """What a vector *means*. Vectors with different specs must never be compared."""

    model_id: str
    dimension: int

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")

    @property
    def slug(self) -> str:
        """Filesystem/collection-safe name, e.g. 'all-minilm-l6-v2'."""
        name = self.model_id.rsplit("/", 1)[-1].lower()
        return "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")


class EmbeddingProvider(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """Shape (len(texts), dimension), every row unit-length."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Shape (dimension,). Separate from documents because some models
        (e5, bge) want a different prefix for queries. MiniLM does not."""
        ...


def l2_normalize(vectors: ArrayLike) -> Matrix:
    """Scale each row to length 1, so cosine similarity becomes a dot product."""
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < _MIN_NORM):
        raise EmbeddingError("Model returned a zero vector; cosine similarity is undefined")
    return (matrix / norms).astype(np.float32)


def validate_embeddings(vectors: ArrayLike, *, expected_count: int, dimension: int) -> Matrix:
    """Reject wrong shapes and NaN/inf, then normalize. Run on every model output."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_count:
        raise EmbeddingError(f"Expected {expected_count} vectors, got shape {matrix.shape}")
    if matrix.shape[1] != dimension:
        raise EmbeddingDimensionError(f"Expected dimension {dimension}, got {matrix.shape[1]}")
    if not np.all(np.isfinite(matrix)):
        raise EmbeddingError("Model returned NaN or infinite values")
    return l2_normalize(matrix)


def embed_in_batches(
    texts: Sequence[str],
    *,
    spec: EmbeddingSpec,
    batch_size: int,
    encode_batch: Callable[[list[str]], ArrayLike],
) -> Matrix:
    """Encode `texts` in fixed-size batches: bounded memory and request size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        return np.empty((0, spec.dimension), dtype=np.float32)
    parts = []
    for batch in batched(texts, batch_size):
        items = list(batch)
        parts.append(
            validate_embeddings(
                encode_batch(items), expected_count=len(items), dimension=spec.dimension
            )
        )
    return np.vstack(parts)
