"""Where ingestion results live.

`DocumentRegistry` remembers what has been indexed (hashes and chunk IDs), and
`ChunkStore` holds the chunks. Both are Protocols: in M2 a Qdrant-backed store
implements the same methods, and the pipeline does not change.
"""

import json
import os
import tempfile
from collections.abc import Collection, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from portfolio_ai.rag.domain.documents import Chunk


class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    source_key: str
    source_uri: str
    title: str
    content_hash: str
    index_fingerprint: str
    chunk_ids: tuple[str, ...]
    indexed_at: datetime


class DocumentRegistry(Protocol):
    def get(self, document_id: str) -> RegistryEntry | None: ...
    def entries(self) -> list[RegistryEntry]: ...
    def upsert(self, entry: RegistryEntry) -> None: ...
    def remove(self, document_id: str) -> None: ...


class ChunkStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...
    def delete(self, chunk_ids: Collection[str]) -> None: ...
    def all(self) -> list[Chunk]: ...


def _atomic_write(path: Path, text: str) -> None:
    """Write to a temp file, then rename: readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class JsonDocumentRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, RegistryEntry] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._entries = {e["document_id"]: RegistryEntry.model_validate(e) for e in raw}

    def get(self, document_id: str) -> RegistryEntry | None:
        return self._entries.get(document_id)

    def entries(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def upsert(self, entry: RegistryEntry) -> None:
        self._entries[entry.document_id] = entry
        self._save()

    def remove(self, document_id: str) -> None:
        if self._entries.pop(document_id, None) is not None:
            self._save()

    def _save(self) -> None:
        payload = [e.model_dump(mode="json") for e in self._entries.values()]
        _atomic_write(self._path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


class JsonlChunkStore:
    """One JSON object per line: easy to inspect with `head` or a text editor."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._chunks: dict[str, Chunk] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunk = Chunk.model_validate_json(line)
                    self._chunks[chunk.chunk_id] = chunk

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        self._chunks.update((chunk.chunk_id, chunk) for chunk in chunks)
        self._save()

    def delete(self, chunk_ids: Collection[str]) -> None:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
        self._save()

    def all(self) -> list[Chunk]:
        return sorted(self._chunks.values(), key=lambda c: (c.source_uri, c.chunk_index))

    def _save(self) -> None:
        lines = (chunk.model_dump_json() for chunk in self.all())
        _atomic_write(self._path, "".join(f"{line}\n" for line in lines))


class InMemoryDocumentRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def get(self, document_id: str) -> RegistryEntry | None:
        return self._entries.get(document_id)

    def entries(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def upsert(self, entry: RegistryEntry) -> None:
        self._entries[entry.document_id] = entry

    def remove(self, document_id: str) -> None:
        self._entries.pop(document_id, None)


class InMemoryChunkStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.upsert_calls = 0

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        self.upsert_calls += 1
        self.chunks.update((chunk.chunk_id, chunk) for chunk in chunks)

    def delete(self, chunk_ids: Collection[str]) -> None:
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)

    def all(self) -> list[Chunk]:
        return list(self.chunks.values())
