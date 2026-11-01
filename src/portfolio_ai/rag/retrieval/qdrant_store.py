"""Qdrant adapter: the M1 `ChunkStore` + `DocumentRegistry` interfaces over vectors.

Collection versioning: the collection name encodes the embedding model and
dimension, e.g. `portfolio_chunks__all-minilm-l6-v2__d384__v1`. A different
model always gets a *different* collection, so vectors from two models can never
be mixed or compared. Switching models = index into the new collection, then
point the app at it. The old one stays until you delete it.
"""

import logging
from collections import defaultdict
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import batched
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    SearchParams,
    VectorParams,
)

from portfolio_ai.rag.domain.documents import Chunk, Visibility
from portfolio_ai.rag.embeddings.base import EmbeddingProvider, EmbeddingSpec, Matrix, Vector
from portfolio_ai.rag.errors import CollectionMismatchError, VectorStoreError
from portfolio_ai.rag.ingestion.storage import RegistryEntry
from portfolio_ai.rag.retrieval.base import SearchResult

logger = logging.getLogger(__name__)

PAYLOAD_SCHEMA_VERSION = 1
INDEXED_FIELDS = ("document_id", "source_key", "visibility")
_PUBLIC_ONLY = Filter(
    must=[FieldCondition(key="visibility", match=MatchValue(value=Visibility.PUBLIC))]
)


def collection_name(prefix: str, spec: EmbeddingSpec) -> str:
    return f"{prefix}__{spec.slug}__d{spec.dimension}__v{PAYLOAD_SCHEMA_VERSION}"


@contextmanager
def _qdrant_errors(action: str) -> Iterator[None]:
    """Translate client/transport exceptions into our own error type."""
    try:
        yield
    except (UnexpectedResponse, ResponseHandlingException, httpx.HTTPError, ConnectionError) as exc:
        raise VectorStoreError(f"Qdrant {action} failed: {exc}") from exc


class QdrantChunkStore:
    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingProvider,
        *,
        prefix: str = "portfolio_chunks",
        upsert_batch_size: int = 64,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._batch_size = upsert_batch_size
        self.spec = embedder.spec
        self.collection = collection_name(prefix, self.spec)

    # -- collection lifecycle -------------------------------------------------

    def ensure_collection(self) -> bool:
        """Create the collection if missing; verify it if present. Returns True if created."""
        with _qdrant_errors("collection check"):
            if self._client.collection_exists(self.collection):
                self._verify_collection()
                return False
            self._client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.spec.dimension, distance=Distance.COSINE),
            )
            for field in INDEXED_FIELDS:
                self._client.create_payload_index(
                    self.collection, field_name=field, field_schema=PayloadSchemaType.KEYWORD
                )
        logger.info("collection_created", extra={"collection": self.collection})
        return True

    def verify_existing(self) -> None:
        """Read-only check for serving: the collection must exist and match. Never creates.

        Works with a Qdrant Cloud *read-only* API key, which is all the API needs.
        """
        with _qdrant_errors("collection check"):
            if not self._client.collection_exists(self.collection):
                raise VectorStoreError(
                    f"Collection {self.collection} does not exist; run ingestion"
                )
            self._verify_collection()

    def _verify_collection(self) -> None:
        vectors = self._client.get_collection(self.collection).config.params.vectors
        if not isinstance(vectors, VectorParams):
            raise CollectionMismatchError(f"{self.collection} uses named vectors; expected one")
        if vectors.size != self.spec.dimension or vectors.distance != Distance.COSINE:
            raise CollectionMismatchError(
                f"{self.collection} has size={vectors.size} distance={vectors.distance}; "
                f"expected size={self.spec.dimension} distance=Cosine"
            )

    def count(self) -> int:
        with _qdrant_errors("count"):
            return self._client.count(self.collection, exact=True).count

    # -- ChunkStore protocol ---------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Embed first, write second: if the model fails, Qdrant is untouched.

        Upsert = insert or overwrite by ID. Deterministic chunk IDs make re-runs
        overwrite the same points instead of adding duplicates.
        """
        if not chunks:
            return
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        indexed_at = datetime.now(UTC).isoformat()
        points = [
            PointStruct(
                id=chunk.chunk_id, vector=vector.tolist(), payload=self._payload(chunk, indexed_at)
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with _qdrant_errors("upsert"):
            for batch in batched(points, self._batch_size):
                self._client.upsert(self.collection, points=list(batch), wait=True)
        logger.debug(
            "points_upserted", extra={"collection": self.collection, "points": len(points)}
        )

    def delete(self, chunk_ids: Collection[str]) -> None:
        if not chunk_ids:
            return
        with _qdrant_errors("delete"):
            self._client.delete(
                self.collection, points_selector=PointIdsList(points=list(chunk_ids)), wait=True
            )

    def all(self) -> list[Chunk]:
        return [chunk for chunk, _ in self._scroll(with_vectors=False)]

    # -- retrieval -------------------------------------------------------------

    def search(
        self,
        vector: Vector,
        *,
        top_k: int,
        score_threshold: float | None = None,
        exact: bool = False,
    ) -> list[SearchResult]:
        """Nearest neighbours by cosine. Only public chunks are ever returned."""
        with _qdrant_errors("search"):
            response = self._client.query_points(
                self.collection,
                query=vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=_PUBLIC_ONLY,
                search_params=SearchParams(exact=exact),
                with_payload=True,
            )
        return [
            SearchResult(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    def export_vectors(self) -> tuple[list[Chunk], Matrix]:
        """Every stored chunk with its vector, for the NumPy comparison."""
        import numpy as np

        rows = list(self._scroll(with_vectors=True))
        chunks = [chunk for chunk, _ in rows]
        matrix = np.array([vector for _, vector in rows], dtype=np.float32)
        return chunks, matrix.reshape(len(rows), self.spec.dimension)

    # -- internals -------------------------------------------------------------

    def _payload(self, chunk: Chunk, indexed_at: str) -> dict[str, Any]:
        return chunk.model_dump(mode="json") | {
            "embedding_model": self.spec.model_id,
            "payload_schema": PAYLOAD_SCHEMA_VERSION,
            "indexed_at": indexed_at,
        }

    def _scroll(self, *, with_vectors: bool) -> Iterator[tuple[Chunk, Any]]:
        for point in self._points(with_vectors=with_vectors):
            yield Chunk.model_validate(point.payload), point.vector

    def payloads(self) -> list[dict[str, Any]]:
        return [point.payload or {} for point in self._points(with_vectors=False)]

    def _points(self, *, with_vectors: bool) -> Iterator[Any]:
        """Page through the whole collection (Qdrant returns 256 points per call)."""
        offset = None
        with _qdrant_errors("scroll"):
            while True:
                points, offset = self._client.scroll(
                    self.collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                yield from points
                if offset is None:
                    return


class QdrantDocumentRegistry:
    """The registry *derived from the collection itself*.

    No local state file: any machine (laptop, CI, server) that points at the same
    Qdrant collection, local or Qdrant Cloud, sees exactly what is indexed there.
    If a crash left a document with chunks from two versions, its hash is marked
    inconsistent, so the next run reindexes it and cleans up.
    """

    INCONSISTENT = "inconsistent"

    def __init__(self, store: QdrantChunkStore) -> None:
        self._store = store
        self._cache: dict[str, RegistryEntry] | None = None

    def _entries(self) -> dict[str, RegistryEntry]:
        if self._cache is None:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for payload in self._store.payloads():
                grouped[payload["document_id"]].append(payload)
            self._cache = {doc_id: self._entry(rows) for doc_id, rows in grouped.items()}
        return self._cache

    def _entry(self, rows: list[dict[str, Any]]) -> RegistryEntry:
        rows.sort(key=lambda row: row["chunk_index"])
        hashes = {row["content_hash"] for row in rows}
        fingerprints = {row.get("index_fingerprint", "") for row in rows}
        indexes = [row["chunk_index"] for row in rows]
        consistent = (
            len(hashes) == 1 and len(fingerprints) == 1 and indexes == list(range(len(rows)))
        )
        first = rows[0]
        return RegistryEntry(
            document_id=first["document_id"],
            source_key=first["source_key"],
            source_uri=first["source_uri"],
            title=first["document_title"],
            content_hash=first["content_hash"] if consistent else self.INCONSISTENT,
            index_fingerprint=fingerprints.pop() if consistent else "",
            chunk_ids=tuple(row["chunk_id"] for row in rows),
            indexed_at=min(datetime.fromisoformat(row["indexed_at"]) for row in rows),
        )

    def get(self, document_id: str) -> RegistryEntry | None:
        return self._entries().get(document_id)

    def entries(self) -> list[RegistryEntry]:
        return list(self._entries().values())

    def upsert(self, entry: RegistryEntry) -> None:
        self._entries()[entry.document_id] = entry  # truth already written as points

    def remove(self, document_id: str) -> None:
        self._entries().pop(document_id, None)
