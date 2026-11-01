"""Loaders turn one approved source into cleaned `SourceDocument`s."""

import logging
from typing import Protocol

from portfolio_ai.rag.domain.documents import SourceDocument
from portfolio_ai.rag.domain.ids import content_hash_for, document_id_for
from portfolio_ai.rag.errors import ParseError
from portfolio_ai.rag.ingestion.cleaning import TextCleaner
from portfolio_ai.rag.ingestion.fetchers import PageFetcher
from portfolio_ai.rag.ingestion.html_parser import parse_sections
from portfolio_ai.rag.ingestion.sources import ApprovedSource, SourcePolicy

logger = logging.getLogger(__name__)


class SourceLoader(Protocol):
    def load(self, source: ApprovedSource) -> list[SourceDocument]: ...


class PortfolioSiteLoader:
    """One rendered single-page site -> one document per `<section id>`."""

    def __init__(self, fetcher: PageFetcher, cleaner: TextCleaner, policy: SourcePolicy) -> None:
        self._fetcher = fetcher
        self._cleaner = cleaner
        self._policy = policy

    def load(self, source: ApprovedSource) -> list[SourceDocument]:
        page = self._fetcher.fetch(source.url)
        base_url = page.url.split("#", 1)[0]
        documents: list[SourceDocument] = []
        for section in parse_sections(page.html):
            blocks = self._cleaner.clean_blocks(section.blocks)
            if not blocks:
                continue
            uri = self._policy.canonicalize(f"{base_url}#{section.anchor}")
            documents.append(
                SourceDocument(
                    document_id=document_id_for(source.source_type, uri),
                    source_key=source.url,
                    source_type=source.source_type,
                    source_uri=uri,
                    title=self._cleaner.clean(section.title),
                    blocks=blocks,
                    content_hash=content_hash_for(blocks),
                    visibility=source.visibility,
                    updated_at=page.fetched_at,
                )
            )
        if not documents:
            # Never read "rendered nothing" as "the site is empty": that would
            # delete every indexed document for this source.
            raise ParseError(f"No content sections found at {source.url}")
        logger.info("source_loaded", extra={"source": source.url, "documents": len(documents)})
        return documents
