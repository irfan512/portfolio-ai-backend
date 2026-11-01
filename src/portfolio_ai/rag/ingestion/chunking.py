"""Structure-aware chunking.

Why chunk? Embedding models compress a passage into one vector. Long passages blur
many facts together, and all-MiniLM-L6-v2 truncates input beyond 256 word-pieces.
So we split by structure first (one unit per heading path), then by size.

Every chunk starts with a context header such as
"Experience > Senior Mobile Application Developer" so a chunk still makes sense
when it is retrieved on its own.
"""

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby

from portfolio_ai.rag.domain.documents import Chunk, SourceDocument, TextBlock
from portfolio_ai.rag.domain.ids import chunk_id_for

TokenCounter = Callable[[str], int]

_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])")


def approximate_token_count(text: str) -> int:
    """Words + punctuation marks: a dependency-free stand-in for a tokenizer.

    M2 can pass the embedding model's real tokenizer as `count_tokens`.
    """
    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 200  # below MiniLM's 256 limit, leaving room for special tokens
    overlap_tokens: int = 40

    def __post_init__(self) -> None:
        if self.max_tokens < 32:
            raise ValueError("max_tokens must be at least 32")
        if not 0 <= self.overlap_tokens < self.max_tokens // 2:
            raise ValueError("overlap_tokens must be >= 0 and less than half of max_tokens")


class SectionChunker:
    VERSION = "section-v1"  # bump when the algorithm changes -> forces a reindex

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        count_tokens: TokenCounter = approximate_token_count,
    ) -> None:
        self.config = config or ChunkingConfig()
        self._count = count_tokens

    @property
    def fingerprint(self) -> str:
        """Identifies the chunking settings. A change re-chunks unchanged documents."""
        counter = getattr(self._count, "__name__", type(self._count).__name__)
        return f"{self.VERSION}:max={self.config.max_tokens}:overlap={self.config.overlap_tokens}:{counter}"

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path, group in groupby(document.blocks, key=lambda block: block.heading_path):
            header = " > ".join((document.title, *path))
            budget = max(self.config.max_tokens - self._count(header), self.config.max_tokens // 2)
            for body in self._pack(self._pieces(group, budget), budget):
                text = f"{header}\n{body}"
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id_for(
                            document.document_id, document.content_hash, self.fingerprint, index
                        ),
                        document_id=document.document_id,
                        source_key=document.source_key,
                        document_title=document.title,
                        text=text,
                        chunk_index=index,
                        section=" > ".join(path) or document.title,
                        source_uri=document.source_uri,
                        content_hash=document.content_hash,
                        visibility=document.visibility,
                        token_count=self._count(text),
                        index_fingerprint=self.fingerprint,
                    )
                )
        return chunks

    def _pieces(self, blocks: Iterable[TextBlock], budget: int) -> Iterator[tuple[str, int]]:
        """Yield (text, tokens) pieces that each fit the budget: block > sentence > words."""
        for block in blocks:
            tokens = self._count(block.text)
            if tokens <= budget:
                yield block.text, tokens
                continue
            for sentence in _SENTENCE_BREAK.split(block.text):
                tokens = self._count(sentence)
                if tokens <= budget:
                    yield sentence, tokens
                else:
                    yield from self._word_windows(sentence, budget)

    def _word_windows(self, text: str, budget: int) -> Iterator[tuple[str, int]]:
        words: list[str] = []
        size = 0
        for word in text.split():
            tokens = self._count(word)
            if words and size + tokens > budget:
                yield " ".join(words), size
                words, size = [], 0
            words.append(word)
            size += tokens
        if words:
            yield " ".join(words), size

    def _pack(self, pieces: Iterable[tuple[str, int]], budget: int) -> Iterator[str]:
        """Greedy packing; each new chunk repeats the trailing pieces of the last one."""
        current: list[tuple[str, int]] = []
        size = 0
        for text, tokens in pieces:
            if current and size + tokens > budget:
                yield "\n".join(piece for piece, _ in current)
                current = self._overlap_tail(
                    current, min(self.config.overlap_tokens, budget - tokens)
                )
                size = sum(t for _, t in current)
            current.append((text, tokens))
            size += tokens
        if current:
            yield "\n".join(piece for piece, _ in current)

    @staticmethod
    def _overlap_tail(pieces: list[tuple[str, int]], limit: int) -> list[tuple[str, int]]:
        tail: list[tuple[str, int]] = []
        size = 0
        for text, tokens in reversed(pieces):
            if size + tokens > limit:
                break
            tail.insert(0, (text, tokens))
            size += tokens
        return tail
