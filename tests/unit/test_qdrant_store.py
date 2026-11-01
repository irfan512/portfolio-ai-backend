"""QdrantChunkStore against qdrant-client's in-process engine (`:memory:`).

Same API and scoring as the server, but no Docker. Server tests live in
tests/integration/test_qdrant_server.py.
"""

from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from portfolio_ai.rag.domain.documents import (
    Chunk,
    SourceDocument,
    SourceType,
    TextBlock,
    Visibility,
)
from portfolio_ai.rag.domain.ids import content_hash_for, document_id_for
from portfolio_ai.rag.errors import CollectionMismatchError, EmbeddingError
from portfolio_ai.rag.ingestion.chunking import SectionChunker
from portfolio_ai.rag.retrieval.qdrant_store import (
    QdrantChunkStore,
    QdrantDocumentRegistry,
    collection_name,
)
from tests.fakes import HashingEmbeddingProvider

SITE = "https://irfan512.github.io/portfolio/"


def document(anchor: str, *texts: tuple[str, str]) -> SourceDocument:
    blocks = tuple(TextBlock(text=t, heading_path=(h,)) for h, t in texts)
    uri = f"{SITE}#{anchor}"
    return SourceDocument(
        document_id=document_id_for(SourceType.WEBSITE, uri),
        source_key=SITE,
        source_type=SourceType.WEBSITE,
        source_uri=uri,
        title=anchor.title(),
        blocks=blocks,
        content_hash=content_hash_for(blocks),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


WORK = document(
    "work",
    ("goCare", "Telemedicine app with Agora video calls and Flutterwave payments."),
    ("Tegal", "Matchmaking app with profiles and messaging."),
)
EXPERIENCE = document("experience", ("Capabilities", "Python, LLM integration, TensorFlow Lite."))


@pytest.fixture
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider()


@pytest.fixture
def store(embedder: HashingEmbeddingProvider) -> QdrantChunkStore:
    store = QdrantChunkStore(QdrantClient(":memory:"), embedder, prefix="test_chunks")
    store.ensure_collection()
    return store


def chunks_of(*documents: SourceDocument) -> list[Chunk]:
    chunker = SectionChunker()
    return [chunk for doc in documents for chunk in chunker.chunk(doc)]


def test_collection_name_encodes_model_dimension_and_schema(
    embedder: HashingEmbeddingProvider,
) -> None:
    assert (
        collection_name("portfolio_chunks", embedder.spec)
        == "portfolio_chunks__hashing-bow__d384__v1"
    )


def test_ensure_collection_is_idempotent(store: QdrantChunkStore) -> None:
    assert store.ensure_collection() is False  # already exists, verified


def test_existing_collection_with_other_dimension_is_refused(
    embedder: HashingEmbeddingProvider,
) -> None:
    client = QdrantClient(":memory:")
    name = collection_name("test_chunks", embedder.spec)
    client.create_collection(name, vectors_config=VectorParams(size=128, distance=Distance.DOT))
    with pytest.raises(CollectionMismatchError, match="size=128"):
        QdrantChunkStore(client, embedder, prefix="test_chunks").ensure_collection()


def test_upsert_preserves_ids_and_metadata(store: QdrantChunkStore) -> None:
    chunks = chunks_of(WORK)
    store.upsert(chunks)

    stored = {c.chunk_id: c for c in store.all()}
    assert stored == {c.chunk_id: c for c in chunks}  # every field round-trips
    payload = store.payloads()[0]
    assert payload["embedding_model"] == "test/hashing-bow"
    assert payload["source_uri"] == f"{SITE}#work"


def test_upserting_twice_does_not_duplicate(store: QdrantChunkStore) -> None:
    chunks = chunks_of(WORK, EXPERIENCE)
    store.upsert(chunks)
    store.upsert(chunks)
    assert store.count() == len(chunks)


def test_search_ranks_matching_chunk_first(
    store: QdrantChunkStore, embedder: HashingEmbeddingProvider
) -> None:
    store.upsert(chunks_of(WORK, EXPERIENCE))

    results = store.search(embedder.embed_query("flutterwave payments"), top_k=3)

    assert results[0].chunk.section == "goCare"
    assert results[0].chunk.source_uri == f"{SITE}#work"
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_score_threshold_and_top_k(
    store: QdrantChunkStore, embedder: HashingEmbeddingProvider
) -> None:
    store.upsert(chunks_of(WORK, EXPERIENCE))
    query = embedder.embed_query("flutterwave payments")

    assert len(store.search(query, top_k=1)) == 1
    strict = store.search(query, top_k=10, score_threshold=0.3)
    assert strict and all(r.score >= 0.3 for r in strict)
    assert store.search(query, top_k=10, score_threshold=0.99) == []


def test_non_public_chunks_are_never_returned(
    store: QdrantChunkStore, embedder: HashingEmbeddingProvider
) -> None:
    internal = chunks_of(WORK)[0].model_copy(
        update={
            "chunk_id": "6f1c1b3e-0000-5000-8000-000000000001",
            "visibility": Visibility.INTERNAL,
        }
    )
    store.upsert([internal])
    results = store.search(embedder.embed_query(internal.text), top_k=10)
    assert all(r.chunk.visibility is Visibility.PUBLIC for r in results)


def test_delete_removes_points(store: QdrantChunkStore) -> None:
    chunks = chunks_of(WORK, EXPERIENCE)
    store.upsert(chunks)
    store.delete([chunks[0].chunk_id])
    assert store.count() == len(chunks) - 1
    store.delete([])  # no-op


def test_embedding_failure_leaves_existing_points_untouched(
    store: QdrantChunkStore, embedder: HashingEmbeddingProvider
) -> None:
    store.upsert(chunks_of(WORK))
    before = store.all()
    embedder.fail = True
    with pytest.raises(EmbeddingError):
        store.upsert(chunks_of(EXPERIENCE))
    assert store.all() == before


def test_registry_is_rebuilt_from_points(store: QdrantChunkStore) -> None:
    chunks = chunks_of(WORK)
    store.upsert(chunks)

    entry = QdrantDocumentRegistry(store).get(WORK.document_id)

    assert entry is not None
    assert entry.content_hash == WORK.content_hash
    assert entry.index_fingerprint == SectionChunker().fingerprint
    assert entry.chunk_ids == tuple(c.chunk_id for c in chunks)
    assert entry.source_uri == f"{SITE}#work"


def test_registry_flags_half_written_documents(store: QdrantChunkStore) -> None:
    old = chunks_of(WORK)
    newer = document("work", ("goCare", "Telemedicine app, updated text."))
    store.upsert(old + chunks_of(newer))  # simulate a crash before stale cleanup

    entry = QdrantDocumentRegistry(store).get(WORK.document_id)

    assert entry is not None
    assert entry.content_hash == QdrantDocumentRegistry.INCONSISTENT
    assert len(entry.chunk_ids) == len(old) + 1  # all of them, so all get cleaned up
