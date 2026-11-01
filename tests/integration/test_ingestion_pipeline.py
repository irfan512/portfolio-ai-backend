"""End-to-end ingestion over the synthetic fixture, with real file-backed storage."""

from pathlib import Path

import pytest

from portfolio_ai.rag.ingestion.chunking import ChunkingConfig, SectionChunker
from portfolio_ai.rag.ingestion.cleaning import EMAIL_PLACEHOLDER, PHONE_PLACEHOLDER, TextCleaner
from portfolio_ai.rag.ingestion.loaders import PortfolioSiteLoader
from portfolio_ai.rag.ingestion.pipeline import DocumentStatus, IngestionPipeline
from portfolio_ai.rag.ingestion.sources import SourceCatalog
from portfolio_ai.rag.ingestion.storage import JsonDocumentRegistry, JsonlChunkStore
from tests.conftest import SITE, FakeFetcher


class Harness:
    def __init__(self, tmp_path: Path, fetcher: FakeFetcher, catalog: SourceCatalog) -> None:
        self.tmp_path = tmp_path
        self.fetcher = fetcher
        self.catalog = catalog

    def run(
        self,
        *,
        chunker: SectionChunker | None = None,
        catalog: SourceCatalog | None = None,
        **kwargs: bool,
    ):
        catalog = catalog or self.catalog
        # Fresh store/registry objects each run: proves state really persists on disk.
        self.store = JsonlChunkStore(self.tmp_path / "chunks.jsonl")
        self.registry = JsonDocumentRegistry(self.tmp_path / "registry.json")
        loader = PortfolioSiteLoader(self.fetcher, TextCleaner(), catalog.policy)
        pipeline = IngestionPipeline(
            {"portfolio_site": loader},
            chunker or SectionChunker(),
            self.registry,
            self.store,
            **kwargs,
        )
        return pipeline.run(catalog)

    def statuses(self, report) -> dict[str, DocumentStatus]:
        return {o.source_uri.rsplit("#", 1)[1]: o.status for o in report.outcomes}


@pytest.fixture
def harness(tmp_path: Path, fake_fetcher: FakeFetcher, catalog: SourceCatalog) -> Harness:
    return Harness(tmp_path, fake_fetcher, catalog)


def test_first_run_indexes_every_section_with_citable_metadata(harness: Harness) -> None:
    report = harness.run()

    assert report.ok
    assert set(harness.statuses(report).values()) == {DocumentStatus.CREATED}
    chunks = harness.store.all()
    assert {c.source_uri for c in chunks} == {
        f"{SITE}#top",
        f"{SITE}#work",
        f"{SITE}#experience",
        f"{SITE}#contact",
    }
    beta = next(c for c in chunks if c.section == "More projects > Project Beta")
    assert beta.text.startswith("Selected work > More projects > Project Beta\n")
    assert beta.document_title == "Selected work"
    assert all(c.visibility == "public" for c in chunks)


def test_contact_details_never_reach_the_index(harness: Harness) -> None:
    harness.run()
    stored = (harness.tmp_path / "chunks.jsonl").read_text()
    assert "alex@example.com" not in stored
    assert "555 555 0100" not in stored
    assert EMAIL_PLACEHOLDER in stored and PHONE_PLACEHOLDER in stored
    assert "4155550123987" in stored  # digits inside a URL are untouched


def test_second_run_is_a_no_op(harness: Harness) -> None:
    harness.run()
    before = (harness.tmp_path / "chunks.jsonl").read_text()

    report = harness.run()

    assert set(harness.statuses(report).values()) == {DocumentStatus.UNCHANGED}
    assert (harness.tmp_path / "chunks.jsonl").read_text() == before


def test_changed_section_is_reindexed_and_stale_chunks_removed(harness: Harness) -> None:
    harness.run()
    old_ids = {c.chunk_id for c in harness.store.all() if c.source_uri.endswith("#work")}

    harness.fetcher.html = harness.fetcher.html.replace(
        "Beta summary.", "Beta summary, now updated."
    )
    report = harness.run()

    statuses = harness.statuses(report)
    assert statuses["work"] is DocumentStatus.UPDATED
    assert statuses["experience"] is DocumentStatus.UNCHANGED
    work = [c for c in harness.store.all() if c.source_uri.endswith("#work")]
    assert any("now updated" in c.text for c in work)
    assert not old_ids & {c.chunk_id for c in work}
    assert len(work) == len(old_ids)  # replaced, not appended


def test_removed_section_is_deleted(harness: Harness) -> None:
    harness.run()
    start = harness.fetcher.html.index('<section id="experience">')
    end = harness.fetcher.html.index("</section>", start) + len("</section>")
    harness.fetcher.html = harness.fetcher.html[:start] + harness.fetcher.html[end:]

    report = harness.run()

    assert harness.statuses(report)["experience"] is DocumentStatus.DELETED
    assert not any(c.source_uri.endswith("#experience") for c in harness.store.all())
    assert all(not e.source_uri.endswith("#experience") for e in harness.registry.entries())


def test_failed_fetch_keeps_existing_index(harness: Harness) -> None:
    harness.run()
    count = len(harness.store.all())
    harness.fetcher.html = "<html><body><main></main></body></html>"  # e.g. JS failed to render

    report = harness.run()

    assert not report.ok
    assert "No content sections" in report.failures[0].error
    assert len(harness.store.all()) == count  # nothing deleted


def test_source_removed_from_catalog_is_purged(harness: Harness, catalog: SourceCatalog) -> None:
    harness.run()
    empty = catalog.model_copy(update={"sources": ()})

    report = harness.run(catalog=empty)

    assert set(harness.statuses(report).values()) == {DocumentStatus.DELETED}
    assert harness.store.all() == []
    assert harness.registry.entries() == []


def test_chunking_change_triggers_reindex(harness: Harness) -> None:
    harness.run()
    report = harness.run(chunker=SectionChunker(ChunkingConfig(max_tokens=64, overlap_tokens=8)))
    assert set(harness.statuses(report).values()) == {DocumentStatus.UPDATED}


def test_dry_run_writes_nothing(harness: Harness) -> None:
    report = harness.run(dry_run=True)
    assert set(harness.statuses(report).values()) == {DocumentStatus.CREATED}
    assert not (harness.tmp_path / "chunks.jsonl").exists()
    assert not (harness.tmp_path / "registry.json").exists()


def test_force_reindexes_unchanged_documents(harness: Harness) -> None:
    harness.run()
    report = harness.run(force=True)
    assert set(harness.statuses(report).values()) == {DocumentStatus.UPDATED}
