"""Cosine similarity and exact nearest-neighbour search in plain NumPy.

cos(a, b) = (a . b) / (|a| * |b|): the cosine of the angle between two vectors.
1 means same direction, 0 means unrelated (orthogonal), -1 means opposite.
For unit-length vectors the denominator is 1, so cosine equals the dot product.

This module is the reference implementation: Qdrant should return the same top
results for the same vectors.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from portfolio_ai.rag.domain.documents import Chunk, Visibility
from portfolio_ai.rag.embeddings.base import Matrix, Vector, l2_normalize
from portfolio_ai.rag.retrieval.base import SearchResult

_MIN_NORM = 1e-12


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine of two 1-D vectors. Raises for zero vectors (angle undefined)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"Expected two 1-D vectors of equal length, got {x.shape} and {y.shape}")
    norm_product = np.linalg.norm(x) * np.linalg.norm(y)
    if norm_product < _MIN_NORM:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return float(np.clip(x @ y / norm_product, -1.0, 1.0))


def cosine_similarity_matrix(queries: ArrayLike, corpus: ArrayLike) -> NDArray[np.float32]:
    """All pairwise cosines: (m, d) x (n, d) -> (m, n), via one matrix multiply."""
    q = l2_normalize(np.atleast_2d(queries))
    c = l2_normalize(np.atleast_2d(corpus))
    if q.shape[1] != c.shape[1]:
        raise ValueError(f"Dimension mismatch: {q.shape[1]} vs {c.shape[1]}")
    return np.clip(q @ c.T, -1.0, 1.0)


def top_k_indices(scores: ArrayLike, k: int) -> NDArray[np.intp]:
    """Indices of the k highest scores, best first; ties keep corpus order."""
    values = np.asarray(scores)
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    return np.argsort(-values, kind="stable")[:k]


class NumpyExactIndex:
    """Brute-force search: compare the query with every vector. O(n * d) per query.

    Perfect recall and fine for thousands of chunks; Qdrant's HNSW index is what
    keeps millions of vectors fast.
    """

    def __init__(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks = list(chunks)
        self._matrix = l2_normalize(vectors) if len(chunks) else np.empty((0, 0), np.float32)

    def search(
        self,
        vector: Vector,
        *,
        top_k: int,
        score_threshold: float | None = None,
        exact: bool = True,
    ) -> list[SearchResult]:
        if not self._chunks:
            return []
        scores = cosine_similarity_matrix(vector, self._matrix)[0]
        public = np.array([c.visibility is Visibility.PUBLIC for c in self._chunks])
        scores = np.where(public, scores, -np.inf)
        results: list[SearchResult] = []
        for index in top_k_indices(scores, top_k):
            score = float(scores[index])
            if score == -np.inf or (score_threshold is not None and score < score_threshold):
                break
            results.append(SearchResult(chunk=self._chunks[index], score=score))
        return results
