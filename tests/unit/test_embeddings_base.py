import numpy as np
import pytest

from portfolio_ai.rag.embeddings.base import (
    EmbeddingSpec,
    embed_in_batches,
    l2_normalize,
    validate_embeddings,
)
from portfolio_ai.rag.errors import EmbeddingDimensionError, EmbeddingError
from tests.fakes import HashingEmbeddingProvider

SPEC = EmbeddingSpec("sentence-transformers/all-MiniLM-L6-v2", 384)


def test_spec_slug_is_collection_safe() -> None:
    assert SPEC.slug == "all-minilm-l6-v2"


def test_l2_normalize_gives_unit_rows() -> None:
    out = l2_normalize([[3.0, 4.0], [0.0, 2.0]])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert out[0].tolist() == pytest.approx([0.6, 0.8])


def test_zero_vectors_are_rejected() -> None:
    with pytest.raises(EmbeddingError, match="zero vector"):
        l2_normalize([[0.0, 0.0]])


@pytest.mark.parametrize(
    ("vectors", "error"),
    [
        (np.ones((2, 384)), EmbeddingError),  # wrong count
        (np.ones((3, 128)), EmbeddingDimensionError),  # wrong dimension
        (np.full((3, 384), np.nan), EmbeddingError),  # NaN
        (np.ones(384), EmbeddingError),  # 1-D
    ],
)
def test_validate_rejects_bad_model_output(vectors: np.ndarray, error: type[Exception]) -> None:
    with pytest.raises(error):
        validate_embeddings(vectors, expected_count=3, dimension=384)


def test_embed_in_batches_splits_and_preserves_order() -> None:
    seen: list[list[str]] = []

    def encode(batch: list[str]) -> np.ndarray:
        seen.append(batch)
        return np.array([[float(len(t)), 1.0] for t in batch])

    spec = EmbeddingSpec("x", 2)
    out = embed_in_batches(
        ["a", "bb", "ccc", "dddd", "e"], spec=spec, batch_size=2, encode_batch=encode
    )

    assert seen == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    assert out.shape == (5, 2)
    assert out[3] == pytest.approx(l2_normalize([[4.0, 1.0]])[0])


def test_empty_input_returns_empty_matrix_without_calling_model() -> None:
    def explode(_: list[str]) -> np.ndarray:
        raise AssertionError("should not be called")

    out = embed_in_batches([], spec=SPEC, batch_size=8, encode_batch=explode)
    assert out.shape == (0, 384)


def test_hashing_fake_is_deterministic_and_word_sensitive() -> None:
    fake = HashingEmbeddingProvider()
    a, b, c = fake.embed_documents(["flutter payments app", "flutter payments app", "matchmaking"])
    assert np.array_equal(a, b)
    assert float(a @ c) < float(a @ b)
