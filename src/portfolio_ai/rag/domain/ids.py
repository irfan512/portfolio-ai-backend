"""Deterministic identifiers and content hashes.

UUIDv5 ("name-based") always maps the same input to the same UUID. That gives us
stable IDs across runs, and Qdrant (M2) accepts UUIDs as point IDs.
"""

import hashlib
import json
import uuid
from collections.abc import Iterable

from portfolio_ai.rag.domain.documents import SourceType, TextBlock

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://irfan512.github.io/portfolio-ai")


def document_id_for(source_type: SourceType, canonical_uri: str) -> str:
    """Identity of a document = what kind of source it is + where it lives."""
    return str(uuid.uuid5(_NAMESPACE, f"document|{source_type}|{canonical_uri}"))


def content_hash_for(blocks: Iterable[TextBlock]) -> str:
    """SHA-256 over text *and* heading paths (headings end up inside chunk text)."""
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(json.dumps([list(block.heading_path), block.text]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def chunk_id_for(document_id: str, content_hash: str, fingerprint: str, index: int) -> str:
    """Document + version + chunking settings + position -> one stable chunk ID."""
    return str(uuid.uuid5(_NAMESPACE, f"chunk|{document_id}|{content_hash}|{fingerprint}|{index}"))
