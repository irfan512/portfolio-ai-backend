"""`portfolio-ai-search`: ask the index a natural-language question.

portfolio-ai-search "Which projects use real-time communication?"
portfolio-ai-search "healthcare apps" --top-k 3 --threshold 0.3
portfolio-ai-search "Firebase experience" --compare   # Qdrant vs NumPy
portfolio-ai-search --info
"""

import argparse
import sys
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from portfolio_ai.core.config import load_settings
from portfolio_ai.core.logging import configure_logging
from portfolio_ai.rag.errors import RagError
from portfolio_ai.rag.retrieval.base import SearchResult
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever
from portfolio_ai.rag.retrieval.similarity import NumpyExactIndex
from portfolio_ai.wiring import ConfigurationError, build_embedder, build_qdrant_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search over the indexed portfolio.")
    parser.add_argument("query", nargs="?", help="natural-language question")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("-t", "--threshold", type=float, default=None, help="min cosine score")
    parser.add_argument("--exact", action="store_true", help="force brute-force search in Qdrant")
    parser.add_argument("--compare", action="store_true", help="also rank with NumPy and diff")
    parser.add_argument("--info", action="store_true", help="show collection details")
    parser.add_argument("--env-file", type=Path, default=None, help="settings file (default .env)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.info and not args.query:
        parser.error("a query is required (or use --info)")

    settings = load_settings(args.env_file)
    configure_logging("WARNING", settings.log_format, settings.secret_values())
    try:
        embedder = build_embedder(settings)
        store = build_qdrant_store(settings, embedder)
        if args.info:
            print(f"collection  {store.collection}")
            print(f"model       {store.spec.model_id} ({store.spec.dimension}-d, cosine)")
            print(f"points      {store.count()}")
            print(f"qdrant      {urlsplit(settings.qdrant_url).hostname}")
            return 0

        retriever = SemanticRetriever(embedder, store)
        started = time.perf_counter()
        results = retriever.search(
            args.query, top_k=args.top_k, score_threshold=args.threshold, exact=args.exact
        )
        elapsed = (time.perf_counter() - started) * 1000
        print(f'\n"{args.query}"  ({len(results)} results, {elapsed:.0f} ms)\n')
        _print_results(results)
        if not results:
            print("  No chunk passed the threshold: the index has no good evidence for this.")

        if args.compare:
            chunks, vectors = store.export_vectors()
            reference = SemanticRetriever(embedder, NumpyExactIndex(chunks, vectors)).search(
                args.query, top_k=args.top_k, score_threshold=args.threshold
            )
            _print_comparison(results, reference)
    except (RagError, ConfigurationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def _print_results(results: list[SearchResult]) -> None:
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        body = " ".join(chunk.text.splitlines()[1:])  # skip the context header line
        label = (
            chunk.document_title
            if chunk.section == chunk.document_title
            else f"{chunk.document_title} > {chunk.section}"
        )
        print(f"{rank:>2}. {result.score:.3f}  {label}")
        print(f"    {chunk.source_uri}")
        print(textwrap.indent(textwrap.shorten(body, 220, placeholder=" …"), "    "))
        print()


def _print_comparison(qdrant: list[SearchResult], numpy: list[SearchResult]) -> None:
    print("Qdrant vs NumPy exact (same vectors, same query)")
    for rank, (q, n) in enumerate(zip(qdrant, numpy, strict=False), start=1):
        same = "same" if q.chunk.chunk_id == n.chunk.chunk_id else "DIFFERENT"
        print(
            f"  {rank:>2}. qdrant {q.score:.5f}  numpy {n.score:.5f}  Δ {abs(q.score - n.score):.1e}  {same}"
        )
    identical = [r.chunk.chunk_id for r in qdrant] == [r.chunk.chunk_id for r in numpy]
    print(f"  ranking identical: {identical}")


if __name__ == "__main__":
    sys.exit(main())
