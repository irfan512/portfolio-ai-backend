"""Semantic retrieval: question text -> query vector -> nearest chunks."""

import logging
import time

from portfolio_ai.rag.embeddings.base import EmbeddingProvider
from portfolio_ai.rag.ingestion.cleaning import normalize_whitespace
from portfolio_ai.rag.retrieval.base import SearchResult, VectorIndex

logger = logging.getLogger(__name__)


class SemanticRetriever:
    MAX_QUERY_CHARS = 1_000

    def __init__(
        self,
        embedder: EmbeddingProvider,
        index: VectorIndex,
        *,
        default_top_k: int = 5,
        max_top_k: int = 20,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        exact: bool = False,
    ) -> list[SearchResult]:
        text = normalize_query(query, self.MAX_QUERY_CHARS)
        k = self.default_top_k if top_k is None else top_k
        if not 1 <= k <= self.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.max_top_k}")
        if score_threshold is not None and not -1.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between -1 and 1 (cosine similarity)")

        started = time.perf_counter()
        vector = self._embedder.embed_query(text)
        results = self._index.search(vector, top_k=k, score_threshold=score_threshold, exact=exact)
        logger.info(
            "retrieval_done",
            extra={
                "top_k": k,
                "threshold": score_threshold,
                "results": len(results),
                "best_score": round(results[0].score, 4) if results else None,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return results


def normalize_query(query: str, max_chars: int = SemanticRetriever.MAX_QUERY_CHARS) -> str:
    text = normalize_whitespace(query)
    if not text:
        raise ValueError("Query is empty")
    if len(text) > max_chars:
        raise ValueError(f"Query is longer than {max_chars} characters")
    return text
