import json

import httpx
import numpy as np
import pytest

from portfolio_ai.rag.embeddings.base import EmbeddingSpec
from portfolio_ai.rag.embeddings.providers import (
    MINILM,
    LocalHttpEmbeddingProvider,
    SentenceTransformerProvider,
)
from portfolio_ai.rag.errors import EmbeddingDimensionError, EmbeddingError


class FakeSentenceTransformer:
    """Stands in for the real model object: same method names, no download."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.batches: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        assert kwargs["normalize_embeddings"] is True
        self.batches.append(list(texts))
        rng = np.random.default_rng(len(texts))
        return rng.normal(size=(len(texts), self.dimension))


def test_sentence_transformer_provider_batches_and_normalizes() -> None:
    model = FakeSentenceTransformer()
    provider = SentenceTransformerProvider(batch_size=2, model=model)

    vectors = provider.embed_documents(["a", "b", "c"])

    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert model.batches == [["a", "b"], ["c"]]
    assert provider.embed_query("q").shape == (384,)


def test_sentence_transformer_provider_rejects_wrong_dimension_model() -> None:
    provider = SentenceTransformerProvider(model=FakeSentenceTransformer(dimension=768))
    with pytest.raises(EmbeddingDimensionError, match="768"):
        provider.embed_documents(["a"])


def service(model: str = MINILM.model_id, dimension: int = 384, status: int = 200):
    """An httpx MockTransport that behaves like embedder/app.py."""
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status)
        if request.url.path == "/health":
            return httpx.Response(
                200, json={"status": "ok", "model": model, "dimension": dimension}
            )
        texts = json.loads(request.content)["texts"]
        requests.append(texts)
        vectors = [[1.0] + [0.0] * (dimension - 1) for _ in texts]
        return httpx.Response(
            200, json={"model": model, "dimension": dimension, "embeddings": vectors}
        )

    return httpx.MockTransport(handler), requests


def test_http_provider_happy_path_batches_requests() -> None:
    transport, requests = service()
    provider = LocalHttpEmbeddingProvider(
        "http://127.0.0.1:8081", batch_size=2, transport=transport
    )

    vectors = provider.embed_documents(["a", "b", "c"])

    assert vectors.shape == (3, 384)
    assert requests == [["a", "b"], ["c"]]
    assert provider.health()["model"] == MINILM.model_id


@pytest.mark.parametrize(
    ("model", "dimension", "error"),
    [
        ("sentence-transformers/all-mpnet-base-v2", 384, EmbeddingError),
        (MINILM.model_id, 768, EmbeddingDimensionError),
    ],
)
def test_http_provider_refuses_a_different_model(model: str, dimension: int, error: type) -> None:
    transport, _ = service(model=model, dimension=dimension)
    provider = LocalHttpEmbeddingProvider("http://127.0.0.1:8081", transport=transport)
    with pytest.raises(error):
        provider.embed_documents(["a"])


def test_http_provider_wraps_service_errors() -> None:
    transport, _ = service(status=503)
    provider = LocalHttpEmbeddingProvider("http://localhost:8081", transport=transport)
    with pytest.raises(EmbeddingError, match="docker compose up"):
        provider.embed_documents(["a"])


@pytest.mark.parametrize(
    "url", ["https://api.example-embeddings.com", "http://192.168.1.20:8081", "http://0.0.0.0:8081"]
)
def test_http_provider_refuses_non_local_hosts(url: str) -> None:
    with pytest.raises(EmbeddingError, match="not local"):
        LocalHttpEmbeddingProvider(url)


def test_http_provider_allows_explicit_extra_host() -> None:
    transport, _ = service()
    provider = LocalHttpEmbeddingProvider(
        "http://embedder:8081", allowed_hosts={"embedder"}, transport=transport
    )
    assert provider.spec == EmbeddingSpec(MINILM.model_id, 384)
