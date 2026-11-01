import pytest

from portfolio_ai.core.config import Settings
from portfolio_ai.rag.embeddings.providers import LocalHttpEmbeddingProvider
from portfolio_ai.wiring import (
    ConfigurationError,
    build_embedder,
    build_qdrant_store,
    validate_qdrant_target,
)


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("http://127.0.0.1:6333", None),
        ("http://localhost:6333", None),
        ("https://abc123.eu-central-1-0.aws.cloud.qdrant.io:6333", "secret"),  # Qdrant Cloud
    ],
)
def test_safe_qdrant_targets(url: str, key: str | None) -> None:
    validate_qdrant_target(url, key)


@pytest.mark.parametrize(
    ("url", "key", "message"),
    [
        ("http://qdrant.example.com:6333", "secret", "https"),
        ("https://qdrant.example.com", None, "API_KEY"),
        ("ftp://127.0.0.1", None, "Invalid"),
    ],
)
def test_unsafe_qdrant_targets_are_refused(url: str, key: str | None, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        validate_qdrant_target(url, key)


def test_defaults_build_local_services() -> None:
    settings = Settings(_env_file=None)
    embedder = build_embedder(settings)
    assert isinstance(embedder, LocalHttpEmbeddingProvider)
    store = build_qdrant_store(settings, embedder)  # no network call at construction
    assert store.collection == "portfolio_chunks__all-minilm-l6-v2__d384__v1"


def test_api_key_is_never_printed() -> None:
    settings = Settings(_env_file=None, qdrant_api_key="super-secret")
    assert "super-secret" not in repr(settings)


def test_build_llm_from_real_settings(monkeypatch) -> None:
    """Every LLM setting the container reads must exist on Settings (caught a startup crash)."""
    from portfolio_ai.container import build_llm
    from portfolio_ai.rag.generation.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key-3333")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("PORTFOLIO_AI_GEMINI_THINKING_LEVEL", "low")
    llm = build_llm(Settings(_env_file=None))  # constructs the SDK client; no network call
    assert isinstance(llm, GeminiProvider)
    assert llm._thinking_level == "low"


def test_build_services_never_raises_unexpected_errors() -> None:
    """Startup must degrade to 'not ready', never crash, even with nothing reachable."""
    from portfolio_ai.container import build_services

    settings = Settings(
        _env_file=None,
        embedding_service_url="http://127.0.0.1:9",  # nothing listens on port 9
        gemini_api_key="test-only-not-a-real-key-4444",
        gemini_model="gemini-test",
    )
    services = build_services(settings)
    assert services.answer_service is None
    assert services.readiness()["embeddings"] is False


def test_empty_env_values_mean_unset(monkeypatch) -> None:
    monkeypatch.setenv("PORTFOLIO_AI_GEMINI_THINKING_LEVEL", "")
    monkeypatch.setenv("GEMINI_MODEL", "  ")
    settings = Settings(_env_file=None)
    assert settings.gemini_thinking_level is None and settings.gemini_model is None
