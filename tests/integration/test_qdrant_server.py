"""The M1 pipeline against the real Qdrant server in Docker.

Uses the deterministic hashing embedder, so no model is needed; each test gets
its own throwaway collection.
"""

import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from portfolio_ai.rag.ingestion.chunking import SectionChunker
from portfolio_ai.rag.ingestion.cleaning import TextCleaner
from portfolio_ai.rag.ingestion.loaders import PortfolioSiteLoader
from portfolio_ai.rag.ingestion.pipeline import DocumentStatus, IngestionPipeline
from portfolio_ai.rag.ingestion.sources import SourceCatalog
from portfolio_ai.rag.retrieval.qdrant_store import QdrantChunkStore, QdrantDocumentRegistry
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever
from portfolio_ai.rag.retrieval.similarity import NumpyExactIndex
from tests.conftest import FakeFetcher
from tests.fakes import HashingEmbeddingProvider
from tests.integration.services import require_service

QDRANT_URL = "http://127.0.0.1:6333"
pytestmark = pytest.mark.qdrant


@pytest.fixture
def store() -> Iterator[QdrantChunkStore]:
    require_service(f"{QDRANT_URL}/readyz", "qdrant")
    client = QdrantClient(url=QDRANT_URL)
    store = QdrantChunkStore(
        client, HashingEmbeddingProvider(), prefix=f"test_{uuid.uuid4().hex[:8]}"
    )
    store.ensure_collection()
    yield store
    client.delete_collection(store.collection)


def run(store: QdrantChunkStore, fetcher: FakeFetcher, catalog: SourceCatalog):
    """Fresh registry per run: state must come from Qdrant, not from memory."""
    loader = PortfolioSiteLoader(fetcher, TextCleaner(), catalog.policy)
    pipeline = IngestionPipeline(
        {"portfolio_site": loader}, SectionChunker(), QdrantDocumentRegistry(store), store
    )
    report = pipeline.run(catalog)
    return report, {o.source_uri.rsplit("#", 1)[1]: o.status for o in report.outcomes}


def test_index_twice_creates_no_duplicates(store, fake_fetcher, catalog) -> None:
    report, statuses = run(store, fake_fetcher, catalog)
    assert report.ok and set(statuses.values()) == {DocumentStatus.CREATED}
    count = store.count()
    embedded = store._embedder.texts_embedded

    report, statuses = run(store, fake_fetcher, catalog)

    assert set(statuses.values()) == {DocumentStatus.UNCHANGED}
    assert store.count() == count
    assert store._embedder.texts_embedded == embedded  # nothing re-embedded


def test_update_replaces_obsolete_vectors(store, fake_fetcher, catalog) -> None:
    run(store, fake_fetcher, catalog)
    old_ids = {c.chunk_id for c in store.all() if c.source_uri.endswith("#work")}
    total = store.count()

    fake_fetcher.html = fake_fetcher.html.replace(
        "Beta summary.", "Beta summary with Stripe payments."
    )
    _, statuses = run(store, fake_fetcher, catalog)

    assert statuses["work"] is DocumentStatus.UPDATED
    work = [c for c in store.all() if c.source_uri.endswith("#work")]
    assert not old_ids & {c.chunk_id for c in work}
    assert store.count() == total


def test_removed_section_is_deleted_from_qdrant(store, fake_fetcher, catalog) -> None:
    run(store, fake_fetcher, catalog)
    html = fake_fetcher.html
    start = html.index('<section id="experience">')
    fake_fetcher.html = html[:start] + html[html.index("</section>", start) + len("</section>") :]

    _, statuses = run(store, fake_fetcher, catalog)

    assert statuses["experience"] is DocumentStatus.DELETED
    assert not any(c.source_uri.endswith("#experience") for c in store.all())


def test_embedding_outage_preserves_existing_index(store, fake_fetcher, catalog) -> None:
    run(store, fake_fetcher, catalog)
    before = sorted(c.chunk_id for c in store.all())
    fake_fetcher.html = fake_fetcher.html.replace("Beta summary.", "Changed.")
    store._embedder.fail = True

    report, _ = run(store, fake_fetcher, catalog)

    assert not report.ok and "simulated embedding outage" in report.failures[0].error
    assert sorted(c.chunk_id for c in store.all()) == before


def same_ranking(a, b) -> bool:
    """Equal scores in order; same ids per score group. The tie group at the
    top_k cut-off is excluded: which tied chunk fills the last slot is arbitrary."""
    if [round(r.score, 5) for r in a] != [round(r.score, 5) for r in b]:
        return False
    groups: dict[float, tuple[set, set]] = {}
    for x, y in zip(a, b, strict=True):
        ids = groups.setdefault(round(x.score, 5), (set(), set()))
        ids[0].add(x.chunk.chunk_id)
        ids[1].add(y.chunk.chunk_id)
    boundary = round(a[-1].score, 5) if a else None
    return all(left == right for score, (left, right) in groups.items() if score != boundary)


def test_search_returns_source_metadata_and_matches_numpy(store, fake_fetcher, catalog) -> None:
    run(store, fake_fetcher, catalog)
    retriever = SemanticRetriever(store._embedder, store)
    query = "second category beta"  # words unique to Project Beta: one clear winner

    hnsw = retriever.search(query, top_k=3)
    exact = retriever.search(query, top_k=3, exact=True)
    chunks, vectors = store.export_vectors()
    reference = SemanticRetriever(store._embedder, NumpyExactIndex(chunks, vectors)).search(
        query, top_k=3
    )

    top = hnsw[0].chunk
    assert hnsw[0].score > hnsw[1].score
    assert top.section == "More projects > Project Beta"
    assert top.source_uri == "https://irfan512.github.io/portfolio/#work"
    assert same_ranking(hnsw, exact)
    assert same_ranking(exact, reference)
