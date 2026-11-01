from datetime import UTC, datetime

import pytest

from portfolio_ai.rag.domain.documents import SourceDocument, SourceType, TextBlock
from portfolio_ai.rag.domain.ids import content_hash_for, document_id_for
from portfolio_ai.rag.ingestion.chunking import (
    ChunkingConfig,
    SectionChunker,
    approximate_token_count,
)

URI = "https://irfan512.github.io/portfolio/#work"


def make_document(*blocks: TextBlock) -> SourceDocument:
    return SourceDocument(
        document_id=document_id_for(SourceType.WEBSITE, URI),
        source_key="https://irfan512.github.io/portfolio/",
        source_type=SourceType.WEBSITE,
        source_uri=URI,
        title="Work",
        blocks=blocks,
        content_hash=content_hash_for(blocks),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def sentences(count: int, prefix: str = "Sentence") -> str:
    return " ".join(f"{prefix} number {i} has six words." for i in range(count))


def test_approximate_token_count_counts_words_and_punctuation() -> None:
    assert approximate_token_count("Flutter, Dart and Python.") == 6


def test_one_chunk_per_heading_path_with_context_header() -> None:
    document = make_document(
        TextBlock(text="Intro line."),
        TextBlock(text="Alpha detail.", heading_path=("Alpha",)),
        TextBlock(text="Alpha more.", heading_path=("Alpha",)),
        TextBlock(text="Beta detail.", heading_path=("More", "Beta")),
    )
    chunks = SectionChunker().chunk(document)

    assert [c.text for c in chunks] == [
        "Work\nIntro line.",
        "Work > Alpha\nAlpha detail.\nAlpha more.",
        "Work > More > Beta\nBeta detail.",
    ]
    assert [c.section for c in chunks] == ["Work", "Alpha", "More > Beta"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.source_uri == URI for c in chunks)


def test_long_unit_is_split_within_budget_with_overlap() -> None:
    config = ChunkingConfig(max_tokens=60, overlap_tokens=10)
    document = make_document(
        *(
            TextBlock(text=f"Line {i} is a short line of text.", heading_path=("Alpha",))
            for i in range(12)
        )
    )
    chunks = SectionChunker(config).chunk(document)

    assert len(chunks) > 1
    assert all(c.token_count <= config.max_tokens for c in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_line = previous.text.splitlines()[-1]
        assert current.text.splitlines()[1] == last_line  # overlap carried over
    # Every original line survives somewhere.
    joined = "\n".join(c.text for c in chunks)
    assert all(f"Line {i} " in joined for i in range(12))


def test_oversized_block_is_split_by_sentence_then_words() -> None:
    config = ChunkingConfig(max_tokens=40, overlap_tokens=0)
    giant_word_run = " ".join(["word"] * 120)  # no sentence boundaries at all
    document = make_document(
        TextBlock(text=sentences(10)), TextBlock(text=giant_word_run, heading_path=("X",))
    )
    chunks = SectionChunker(config).chunk(document)
    assert all(c.token_count <= config.max_tokens for c in chunks)
    assert sum(c.text.count("word") for c in chunks if c.section == "X") == 120


def test_chunk_ids_are_deterministic_and_version_sensitive() -> None:
    document = make_document(TextBlock(text="Alpha detail.", heading_path=("Alpha",)))
    first = [c.chunk_id for c in SectionChunker().chunk(document)]
    second = [c.chunk_id for c in SectionChunker().chunk(document)]
    assert first == second

    changed = make_document(TextBlock(text="Alpha changed.", heading_path=("Alpha",)))
    assert [c.chunk_id for c in SectionChunker().chunk(changed)] != first

    other_settings = SectionChunker(ChunkingConfig(max_tokens=100, overlap_tokens=10))
    assert [c.chunk_id for c in other_settings.chunk(document)] != first


@pytest.mark.parametrize(("max_tokens", "overlap"), [(10, 0), (100, 50), (100, -1)])
def test_invalid_config_is_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap)
