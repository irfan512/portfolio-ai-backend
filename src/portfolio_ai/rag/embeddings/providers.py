"""Concrete embedding adapters. Both keep text on this machine.

- `SentenceTransformerProvider`: runs the model in this Python process
  (Linux servers, Apple Silicon). Needs the optional `embeddings-local` extra.
- `LocalHttpEmbeddingProvider`: calls the Docker `embedder` service on
  localhost (Intel Macs, where PyTorch has no current build).
"""

import logging
from collections.abc import Collection, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
from numpy.typing import ArrayLike

from portfolio_ai.rag.embeddings.base import (
    EmbeddingSpec,
    Matrix,
    Vector,
    embed_in_batches,
)
from portfolio_ai.rag.errors import EmbeddingDimensionError, EmbeddingError

logger = logging.getLogger(__name__)

MINILM = EmbeddingSpec("sentence-transformers/all-MiniLM-L6-v2", 384)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class SentenceTransformerProvider:
    def __init__(
        self,
        spec: EmbeddingSpec = MINILM,
        *,
        batch_size: int = 32,
        device: str = "cpu",
        model: Any | None = None,  # inject a stand-in in tests: no download
    ) -> None:
        self._spec = spec
        self._batch_size = batch_size
        self._device = device
        self._model = model

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run: pip install -e '.[embeddings-local]' "
                    "(or set PORTFOLIO_AI_EMBEDDING_BACKEND=local_http on Intel Macs)"
                ) from exc
            logger.info("embedding_model_loading", extra={"model": self._spec.model_id})
            self._model = SentenceTransformer(self._spec.model_id, device=self._device)
        dimension_fn = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        if (actual := dimension_fn()) != self._spec.dimension:
            raise EmbeddingDimensionError(
                f"{self._spec.model_id} produces {actual}-d vectors, expected {self._spec.dimension}"
            )
        return self._model

    def _encode(self, batch: list[str]) -> ArrayLike:
        return self._load().encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return embed_in_batches(
            texts, spec=self._spec, batch_size=self._batch_size, encode_batch=self._encode
        )

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


class LocalHttpEmbeddingProvider:
    """Client for `embedder/app.py`. Refuses non-local hosts so portfolio text
    can never be sent to a third-party embedding service by misconfiguration."""

    def __init__(
        self,
        base_url: str,
        spec: EmbeddingSpec = MINILM,
        *,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        allowed_hosts: Collection[str] = LOOPBACK_HOSTS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        host = urlsplit(base_url).hostname
        if host not in allowed_hosts:
            raise EmbeddingError(
                f"Embedding service host {host!r} is not local; allowed: {sorted(allowed_hosts)}"
            )
        self._spec = spec
        self._batch_size = batch_size
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def health(self) -> dict[str, Any]:
        """Check the service is up and serving the model we expect."""
        data = self._request("GET", "/health")
        self._check_identity(data)
        return data

    def _encode(self, batch: list[str]) -> ArrayLike:
        data = self._request("POST", "/embed", json={"texts": batch})
        self._check_identity(data)
        return data["embeddings"]

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"Embedding service at {self._base_url} failed: {exc}. "
                "Is it running? `docker compose up -d embedder`"
            ) from exc
        except ValueError as exc:
            raise EmbeddingError("Embedding service returned invalid JSON") from exc

    def _check_identity(self, data: dict[str, Any]) -> None:
        # Guards against silently mixing models: the service must say which model it ran.
        if data.get("model") != self._spec.model_id:
            raise EmbeddingError(
                f"Embedding service runs {data.get('model')!r}, expected {self._spec.model_id!r}"
            )
        if data.get("dimension") != self._spec.dimension:
            raise EmbeddingDimensionError(
                f"Embedding service reports dimension {data.get('dimension')}, expected {self._spec.dimension}"
            )

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return embed_in_batches(
            texts, spec=self._spec, batch_size=self._batch_size, encode_batch=self._encode
        )

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]

    def close(self) -> None:
        self._client.close()
