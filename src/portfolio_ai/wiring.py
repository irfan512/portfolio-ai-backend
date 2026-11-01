"""Composition root: build concrete objects from Settings.

CLIs (and the API in M3) call these functions. `rag` itself never reads settings.
"""

from urllib.parse import urlsplit

from qdrant_client import QdrantClient

from portfolio_ai.core.config import Settings
from portfolio_ai.rag.embeddings.base import EmbeddingProvider, EmbeddingSpec
from portfolio_ai.rag.embeddings.providers import (
    LOOPBACK_HOSTS,
    LocalHttpEmbeddingProvider,
    SentenceTransformerProvider,
)
from portfolio_ai.rag.retrieval.qdrant_store import QdrantChunkStore


class ConfigurationError(ValueError):
    """Settings are unsafe or inconsistent."""


def embedding_spec(settings: Settings) -> EmbeddingSpec:
    return EmbeddingSpec(settings.embedding_model, settings.embedding_dimension)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    spec = embedding_spec(settings)
    if settings.embedding_backend == "onnx":
        from portfolio_ai.rag.embeddings.onnx_provider import OnnxEmbeddingProvider

        if settings.embedding_model_dir is None:
            raise ConfigurationError(
                "PORTFOLIO_AI_EMBEDDING_MODEL_DIR is required for the onnx backend"
            )
        return OnnxEmbeddingProvider(
            settings.embedding_model_dir,
            spec,
            batch_size=settings.embedding_batch_size,
            threads=settings.embedding_threads,
        )
    if settings.embedding_backend == "sentence_transformers":
        return SentenceTransformerProvider(spec, batch_size=settings.embedding_batch_size)
    return LocalHttpEmbeddingProvider(
        settings.embedding_service_url, spec, batch_size=settings.embedding_batch_size
    )


def is_remote_qdrant(url: str) -> bool:
    return urlsplit(url).hostname not in LOOPBACK_HOSTS


def validate_qdrant_target(
    url: str, api_key: str | None, private_hosts: frozenset[str] = frozenset()
) -> None:
    """Local Qdrant may be plain http. Anything else must be https + API key."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ConfigurationError(f"Invalid Qdrant URL: {url!r}")
    if parts.hostname in LOOPBACK_HOSTS or parts.hostname in private_hosts:
        return
    if parts.scheme != "https":
        raise ConfigurationError("A remote Qdrant must use https")
    if not api_key:
        raise ConfigurationError("A remote Qdrant requires PORTFOLIO_AI_QDRANT_API_KEY")


def build_qdrant_client(settings: Settings) -> QdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    validate_qdrant_target(settings.qdrant_url, api_key, frozenset(settings.qdrant_private_hosts))
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=settings.qdrant_timeout_s)


def build_qdrant_store(
    settings: Settings, embedder: EmbeddingProvider | None = None
) -> QdrantChunkStore:
    return QdrantChunkStore(
        build_qdrant_client(settings),
        embedder or build_embedder(settings),
        prefix=settings.qdrant_collection_prefix,
    )
