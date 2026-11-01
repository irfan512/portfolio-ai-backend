"""`portfolio-ai-ingest`: run the ingestion pipeline from the command line.

This module wires concrete implementations together (composition root). The
`rag` package only defines behaviour and interfaces.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from portfolio_ai.core.config import Settings, load_settings
from portfolio_ai.core.logging import configure_logging
from portfolio_ai.rag.errors import RagError
from portfolio_ai.rag.ingestion.chunking import SectionChunker
from portfolio_ai.rag.ingestion.cleaning import TextCleaner
from portfolio_ai.rag.ingestion.fetchers import (
    PageFetcher,
    PlaywrightFetcher,
    RecordingFetcher,
    SnapshotFetcher,
)
from portfolio_ai.rag.ingestion.loaders import PortfolioSiteLoader
from portfolio_ai.rag.ingestion.pipeline import IngestionPipeline, IngestionReport
from portfolio_ai.rag.ingestion.sources import load_catalog
from portfolio_ai.rag.ingestion.storage import (
    ChunkStore,
    DocumentRegistry,
    InMemoryDocumentRegistry,
    JsonDocumentRegistry,
    JsonlChunkStore,
)
from portfolio_ai.rag.retrieval.qdrant_store import QdrantDocumentRegistry
from portfolio_ai.wiring import (
    ConfigurationError,
    build_embedder,
    build_qdrant_store,
    is_remote_qdrant,
)

logger = logging.getLogger("portfolio_ai.ingest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest approved portfolio sources.")
    parser.add_argument("--sources", type=Path, default=Path("sources.toml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--offline", action="store_true", help="replay saved snapshots")
    parser.add_argument("--dry-run", action="store_true", help="report changes, write nothing")
    parser.add_argument("--force", action="store_true", help="re-chunk unchanged documents")
    parser.add_argument("--show-chunks", action="store_true", help="print stored chunks")
    parser.add_argument(
        "--store",
        choices=["qdrant", "jsonl"],
        default="qdrant",
        help="qdrant = embed + vector index (M2); jsonl = local file only (M1)",
    )
    parser.add_argument(
        "--env-file", type=Path, default=None, help="settings file, e.g. .env.cloud (default .env)"
    )
    parser.add_argument(
        "--confirm-remote",
        action="store_true",
        help="required to write to a non-local Qdrant (e.g. Qdrant Cloud)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.env_file)
    configure_logging(settings.log_level, settings.log_format, settings.secret_values())

    try:
        catalog = load_catalog(args.sources)
    except (OSError, ValueError, RagError) as exc:
        logger.error("catalog_invalid", extra={"path": str(args.sources), "error": str(exc)})
        return 2

    snapshots = args.data_dir / "raw" / "snapshots"
    fetcher: PageFetcher
    if args.offline:
        fetcher = SnapshotFetcher(snapshots, catalog.policy)
    else:
        fetcher = RecordingFetcher(PlaywrightFetcher(catalog.policy), snapshots)

    try:
        if args.store == "qdrant" and not _remote_write_allowed(settings, args):
            return 3
        store, registry = _build_storage(args.store, args.data_dir, settings, dry_run=args.dry_run)
    except (RagError, ConfigurationError) as exc:
        logger.error("storage_unavailable", extra={"store": args.store, "error": str(exc)})
        print(f"\nCannot reach the {args.store} store: {exc}")
        return 2

    pipeline = IngestionPipeline(
        loaders={"portfolio_site": PortfolioSiteLoader(fetcher, TextCleaner(), catalog.policy)},
        chunker=SectionChunker(),
        registry=registry,
        store=store,
        dry_run=args.dry_run,
        force=args.force,
    )
    report = pipeline.run(catalog)
    _print_report(report)
    if args.show_chunks:
        for chunk in store.all():
            print(f"\n--- {chunk.source_uri}  [{chunk.section}]  {chunk.token_count} tokens")
            print(chunk.text)
    return 0 if report.ok else 1


def _remote_write_allowed(settings: Settings, args: argparse.Namespace) -> bool:
    """Writing to a hosted Qdrant must be a deliberate act."""
    if not is_remote_qdrant(settings.qdrant_url):
        return True
    from urllib.parse import urlsplit

    from portfolio_ai.rag.retrieval.qdrant_store import collection_name
    from portfolio_ai.wiring import embedding_spec

    host = urlsplit(settings.qdrant_url).hostname
    target = collection_name(settings.qdrant_collection_prefix, embedding_spec(settings))
    print(f"\nRemote Qdrant target: {host}\nCollection:           {target}")
    if args.dry_run or args.confirm_remote:
        return True
    print("Refusing to write to a remote Qdrant without --confirm-remote (try --dry-run first).")
    return False


def _build_storage(
    kind: str, data_dir: Path, settings: Settings, *, dry_run: bool
) -> tuple[ChunkStore, DocumentRegistry]:
    if kind == "jsonl":
        processed = data_dir / "processed"
        return JsonlChunkStore(processed / "chunks.jsonl"), JsonDocumentRegistry(
            processed / "registry.json"
        )
    embedder = build_embedder(settings)
    health = getattr(embedder, "health", None)
    if health is not None:
        health()  # fail fast, before any document is touched
    store = build_qdrant_store(settings, embedder)
    if dry_run:
        # Never create anything in a dry run; a missing collection just means "all new".
        try:
            store.verify_existing()
        except RagError:
            logger.info("dry_run_collection_missing", extra={"collection": store.collection})
            return store, InMemoryDocumentRegistry()
    else:
        store.ensure_collection()
    logger.info("qdrant_ready", extra={"collection": store.collection, "points": store.count()})
    return store, QdrantDocumentRegistry(store)


def _print_report(report: IngestionReport) -> None:
    title = "Ingestion report (dry run)" if report.dry_run else "Ingestion report"
    print(f"\n{title}")
    for outcome in report.outcomes:
        print(f"  {outcome.status.value:<10} {outcome.chunk_count:>3} chunks  {outcome.source_uri}")
    for failure in report.failures:
        print(f"  FAILED     {failure.source}: {failure.error}")


if __name__ == "__main__":
    sys.exit(main())
