"""Deterministic test doubles: no model download, no network."""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from portfolio_ai.rag.embeddings.base import EmbeddingSpec, Matrix, Vector, embed_in_batches

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbeddingProvider:
    """Feature-hashing bag of words: texts sharing words get similar vectors.

    It is not semantic (it doesn't know "payments" ~ "Flutterwave"), but it is
    deterministic, fast, and meaningful enough to test ranking end to end.
    """

    def __init__(self, dimension: int = 384, model_id: str = "test/hashing-bow") -> None:
        self._spec = EmbeddingSpec(model_id=model_id, dimension=dimension)
        self.calls = 0
        self.texts_embedded = 0
        self.fail = False

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _encode(self, batch: list[str]) -> Matrix:
        self.calls += 1
        self.texts_embedded += len(batch)
        if self.fail:
            from portfolio_ai.rag.errors import EmbeddingError

            raise EmbeddingError("simulated embedding outage")
        matrix = np.zeros((len(batch), self._spec.dimension), dtype=np.float32)
        for row, text in enumerate(batch):
            words = _WORD.findall(text.lower()) or ["<empty>"]
            for word in words:
                digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self._spec.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, bucket] += sign
        return matrix

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return embed_in_batches(texts, spec=self._spec, batch_size=16, encode_batch=self._encode)

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


class FakeLLM:
    """Records requests; returns a scripted GroundedAnswer (or raises)."""

    def __init__(self, answer=None, error: Exception | None = None) -> None:
        from portfolio_ai.rag.generation.base import GroundedAnswer

        self.answer = answer or GroundedAnswer(
            answer="Irfan built goCare.", cited_sources=["S1"], insufficient_evidence=False
        )
        self.error = error
        self.requests = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def generate_grounded(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.answer
