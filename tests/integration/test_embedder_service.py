"""The real all-MiniLM-L6-v2 model, via the Docker embedder service."""

import numpy as np
import pytest

from portfolio_ai.rag.embeddings.providers import LocalHttpEmbeddingProvider
from portfolio_ai.rag.retrieval.similarity import cosine_similarity
from tests.integration.services import require_service

URL = "http://127.0.0.1:8081"
pytestmark = pytest.mark.embedder


@pytest.fixture(scope="module")
def provider() -> LocalHttpEmbeddingProvider:
    require_service(f"{URL}/health", "embedder")
    return LocalHttpEmbeddingProvider(URL)


def test_service_reports_expected_model(provider: LocalHttpEmbeddingProvider) -> None:
    health = provider.health()
    assert health == {
        "status": "ok",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    }


def test_vectors_are_384d_unit_length_and_deterministic(
    provider: LocalHttpEmbeddingProvider,
) -> None:
    first = provider.embed_documents(["Flutter app with in-app payments"] * 2)
    again = provider.embed_query("Flutter app with in-app payments")
    assert first.shape == (2, 384)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert np.allclose(first[0], again, atol=1e-5)


def test_meaning_beats_shared_words(provider: LocalHttpEmbeddingProvider) -> None:
    """No word overlap with the right answer, yet it ranks first: that's semantics."""
    query, near, far = provider.embed_documents(
        [
            "healthcare application",
            "Telemedicine app that connects patients with doctors for consultations.",
            "Application for e-sports tournament brackets.",  # shares 'application'
        ]
    )
    assert cosine_similarity(query, near) > cosine_similarity(query, far)
