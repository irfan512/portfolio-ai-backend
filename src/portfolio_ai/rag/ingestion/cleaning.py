"""Text normalization and privacy redaction, applied before anything is indexed."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from portfolio_ai.rag.domain.documents import TextBlock

_ZERO_WIDTH = re.compile("[​-‍⁠﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)])")
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w+(])[+(]?\d[\d\s().-]{6,}\d(?!\w)")

EMAIL_PLACEHOLDER = "[email redacted]"
PHONE_PLACEHOLDER = "[phone redacted]"


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and remove the gap HTML leaves before punctuation."""
    text = _ZERO_WIDTH.sub("", text)
    text = " ".join(text.split())
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


@dataclass(frozen=True)
class TextCleaner:
    """Normalize text and redact contact details (on by default, see context.md §7)."""

    redact_emails: bool = True
    redact_phone_numbers: bool = True
    min_phone_digits: int = 9  # avoids redacting years ("2019-2023") or counts

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = _CONTROL.sub(" ", text)
        text = normalize_whitespace(text)
        return self._redact_outside_urls(text)

    def clean_blocks(self, blocks: Iterable[TextBlock]) -> tuple[TextBlock, ...]:
        """Clean every block, drop empties and exact duplicates, keep order."""
        seen: set[tuple[tuple[str, ...], str]] = set()
        cleaned: list[TextBlock] = []
        for block in blocks:
            text = self.clean(block.text)
            path = tuple(self.clean(h) for h in block.heading_path)
            key = (path, text)
            if text and key not in seen:
                seen.add(key)
                cleaned.append(TextBlock(text=text, heading_path=path))
        return tuple(cleaned)

    def _redact_outside_urls(self, text: str) -> str:
        # Store links contain long digit runs (app IDs) that look like phone numbers.
        pieces: list[str] = []
        position = 0
        for match in _URL.finditer(text):
            pieces.append(self._redact(text[position : match.start()]))
            pieces.append(match.group(0))
            position = match.end()
        pieces.append(self._redact(text[position:]))
        return "".join(pieces)

    def _redact(self, text: str) -> str:
        if self.redact_emails:
            text = _EMAIL.sub(EMAIL_PLACEHOLDER, text)
        if self.redact_phone_numbers:
            text = _PHONE.sub(self._phone_replacement, text)
        return text

    def _phone_replacement(self, match: re.Match[str]) -> str:
        digits = sum(char.isdigit() for char in match.group(0))
        return PHONE_PLACEHOLDER if digits >= self.min_phone_digits else match.group(0)
