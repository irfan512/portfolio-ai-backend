"""Idempotent ingestion: load -> compare hashes -> chunk -> upsert -> remove stale.

Run it twice with unchanged sources and the second run writes nothing.
Write order is upsert new chunks, then delete stale ones, then update the
registry. If a run crashes halfway, the next run sees the old registry entry and
redoes the work. Chunk IDs are deterministic, so redoing it creates no duplicates.
"""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel

from portfolio_ai.rag.domain.documents import SourceDocument
from portfolio_ai.rag.errors import IngestionError, RagError
from portfolio_ai.rag.ingestion.chunking import SectionChunker
from portfolio_ai.rag.ingestion.loaders import SourceLoader
from portfolio_ai.rag.ingestion.sources import SourceCatalog
from portfolio_ai.rag.ingestion.storage import ChunkStore, DocumentRegistry, RegistryEntry

logger = logging.getLogger(__name__)


class DocumentStatus(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


class DocumentOutcome(BaseModel):
    document_id: str
    source_uri: str
    title: str
    status: DocumentStatus
    chunk_count: int


class SourceFailure(BaseModel):
    source: str
    error: str


class IngestionReport(BaseModel):
    dry_run: bool
    outcomes: list[DocumentOutcome] = []
    failures: list[SourceFailure] = []

    @property
    def ok(self) -> bool:
        return not self.failures

    def count(self, status: DocumentStatus) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status is status)


class IngestionPipeline:
    def __init__(
        self,
        loaders: Mapping[str, SourceLoader],
        chunker: SectionChunker,
        registry: DocumentRegistry,
        store: ChunkStore,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        self._loaders = loaders
        self._chunker = chunker
        self._registry = registry
        self._store = store
        self._dry_run = dry_run
        self._force = force

    def run(self, catalog: SourceCatalog) -> IngestionReport:
        report = IngestionReport(dry_run=self._dry_run)
        seen: set[str] = set()

        for source in catalog.sources:
            loader = self._loaders.get(source.loader)
            if loader is None:
                report.failures.append(
                    SourceFailure(source=source.url, error=f"No loader {source.loader!r}")
                )
                continue
            try:
                documents = loader.load(source)
            except IngestionError as exc:
                # Skip deletions for this source: a failed fetch is not an empty site.
                logger.error("source_failed", extra={"source": source.url, "error": str(exc)})
                report.failures.append(SourceFailure(source=source.url, error=str(exc)))
                continue

            current: set[str] = set()
            for document in documents:
                if document.document_id in seen:
                    logger.warning("duplicate_document_skipped", extra={"uri": document.source_uri})
                    continue
                seen.add(document.document_id)
                current.add(document.document_id)  # never delete a doc that failed to index
                try:
                    report.outcomes.append(self._index(document))
                except RagError as exc:
                    # e.g. embedding service down: existing chunks stay untouched
                    logger.error(
                        "document_failed", extra={"uri": document.source_uri, "error": str(exc)}
                    )
                    report.failures.append(
                        SourceFailure(source=document.source_uri, error=str(exc))
                    )

            for entry in self._registry.entries():
                if entry.source_key == source.url and entry.document_id not in current:
                    report.outcomes.append(self._delete(entry))

        # A source removed from the allowlist must disappear from the index too.
        approved = {source.url for source in catalog.sources}
        for entry in self._registry.entries():
            if entry.source_key not in approved:
                report.outcomes.append(self._delete(entry))

        logger.info(
            "ingestion_finished",
            extra={f"docs_{status.value}": report.count(status) for status in DocumentStatus}
            | {"failures": len(report.failures), "dry_run": self._dry_run},
        )
        return report

    def _index(self, document: SourceDocument) -> DocumentOutcome:
        existing = self._registry.get(document.document_id)
        fingerprint = self._chunker.fingerprint
        unchanged = (
            existing is not None
            and existing.content_hash == document.content_hash
            and existing.index_fingerprint == fingerprint
        )
        if unchanged and not self._force:
            assert existing is not None
            logger.debug("document_unchanged", extra={"uri": document.source_uri})
            return self._outcome(document, DocumentStatus.UNCHANGED, len(existing.chunk_ids))

        chunks = self._chunker.chunk(document)
        status = DocumentStatus.CREATED if existing is None else DocumentStatus.UPDATED
        if not self._dry_run:
            self._store.upsert(chunks)
            new_ids = {chunk.chunk_id for chunk in chunks}
            stale = set(existing.chunk_ids) - new_ids if existing else set()
            if stale:
                self._store.delete(stale)
            self._registry.upsert(
                RegistryEntry(
                    document_id=document.document_id,
                    source_key=document.source_key,
                    source_uri=document.source_uri,
                    title=document.title,
                    content_hash=document.content_hash,
                    index_fingerprint=fingerprint,
                    chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                    indexed_at=datetime.now(UTC),
                )
            )
        logger.info(
            "document_indexed",
            extra={"uri": document.source_uri, "status": status.value, "chunks": len(chunks)},
        )
        return self._outcome(document, status, len(chunks))

    def _delete(self, entry: RegistryEntry) -> DocumentOutcome:
        if not self._dry_run:
            self._store.delete(entry.chunk_ids)
            self._registry.remove(entry.document_id)
        logger.info(
            "document_deleted", extra={"uri": entry.source_uri, "chunks": len(entry.chunk_ids)}
        )
        return DocumentOutcome(
            document_id=entry.document_id,
            source_uri=entry.source_uri,
            title=entry.title,
            status=DocumentStatus.DELETED,
            chunk_count=len(entry.chunk_ids),
        )

    @staticmethod
    def _outcome(document: SourceDocument, status: DocumentStatus, chunks: int) -> DocumentOutcome:
        return DocumentOutcome(
            document_id=document.document_id,
            source_uri=document.source_uri,
            title=document.title,
            status=status,
            chunk_count=chunks,
        )
