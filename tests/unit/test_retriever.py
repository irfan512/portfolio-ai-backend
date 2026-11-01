import numpy as np
import pytest
from qdrant_client import QdrantClient

from portfolio_ai.rag.retrieval.qdrant_store import QdrantChunkStore
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever, normalize_query
from portfolio_ai.rag.retrieval.similarity import NumpyExactIndex
from tests.fakes import HashingEmbeddingProvider
from tests.unit.test_qdrant_store import EXPERIENCE, WORK, chunks_of


@pytest.fixture
def setup() -> tuple[HashingEmbeddingProvider, QdrantChunkStore]:
    embedder = HashingEmbeddingProvider()
    store = QdrantChunkStore(QdrantClient(":memory:"), embedder, prefix="t")
    store.ensure_collection()
    store.upsert(chunks_of(WORK, EXPERIENCE))
    return embedder, store


def test_retriever_returns_ranked_results(setup) -> None:
    embedder, store = setup
    results = SemanticRetriever(embedder, store).search("  Flutterwave   payments ", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.section == "goCare"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": 21}, "top_k"),
        ({"score_threshold": 1.5}, "score_threshold"),
    ],
)
def test_retriever_validates_parameters(setup, kwargs: dict, message: str) -> None:
    embedder, store = setup
    with pytest.raises(ValueError, match=message):
        SemanticRetriever(embedder, store).search("payments", **kwargs)


@pytest.mark.parametrize("query", ["", "   \n ", "x" * 1001])
def test_bad_queries_are_rejected(query: str) -> None:
    with pytest.raises(ValueError):
        normalize_query(query)


def test_numpy_exact_index_agrees_with_qdrant(setup) -> None:
    """The educational baseline and the database must rank identically."""
    embedder, store = setup
    chunks, vectors = store.export_vectors()
    numpy_index = NumpyExactIndex(chunks, vectors)

    for query in ["flutterwave payments", "matchmaking messaging", "python llm"]:
        vector = embedder.embed_query(query)
        q = store.search(vector, top_k=3, exact=True)
        n = numpy_index.search(vector, top_k=3)
        assert [r.chunk.chunk_id for r in q] == [r.chunk.chunk_id for r in n]
        assert np.allclose([r.score for r in q], [r.score for r in n], atol=1e-5)
