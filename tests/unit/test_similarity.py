import math

import numpy as np
import pytest

from portfolio_ai.rag.retrieval.similarity import (
    cosine_similarity,
    cosine_similarity_matrix,
    top_k_indices,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 0], [1, 0], 1.0),  # same direction
        ([1, 0], [0, 1], 0.0),  # orthogonal: unrelated
        ([1, 0], [-1, 0], -1.0),  # opposite
        ([1, 1], [1, 0], math.cos(math.pi / 4)),  # 45 degrees ~ 0.7071
        ([3, 4], [6, 8], 1.0),  # length does not matter, only direction
    ],
)
def test_cosine_similarity_known_values(a: list[int], b: list[int], expected: float) -> None:
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_rejects_zero_vector_and_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0, 0], [1, 0])
    with pytest.raises(ValueError, match="equal length"):
        cosine_similarity([1, 0, 0], [1, 0])


def test_matrix_matches_pairwise_computation() -> None:
    rng = np.random.default_rng(seed=42)
    queries, corpus = rng.normal(size=(3, 384)), rng.normal(size=(5, 384))

    matrix = cosine_similarity_matrix(queries, corpus)

    assert matrix.shape == (3, 5)
    for i in range(3):
        for j in range(5):
            assert matrix[i, j] == pytest.approx(cosine_similarity(queries[i], corpus[j]), abs=1e-5)


def test_top_k_orders_best_first_and_breaks_ties_by_position() -> None:
    assert top_k_indices([0.1, 0.9, 0.5, 0.9], 3).tolist() == [1, 3, 2]
    assert top_k_indices([0.1, 0.2], 10).tolist() == [1, 0]
    assert top_k_indices([0.1], 0).tolist() == []
