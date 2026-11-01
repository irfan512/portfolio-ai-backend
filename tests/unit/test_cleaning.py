import pytest

from portfolio_ai.rag.domain.documents import TextBlock
from portfolio_ai.rag.ingestion.cleaning import (
    EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    TextCleaner,
    normalize_whitespace,
)


def test_normalize_whitespace_collapses_and_fixes_punctuation_gaps() -> None:
    assert (
        normalize_whitespace("  Built   with\n\tFlutter , Dart .​ ") == "Built with Flutter, Dart."
    )


def test_clean_applies_unicode_normalization_and_strips_control_chars() -> None:
    assert TextCleaner().clean("Ｆｌｕｔｔｅｒ\x07 app") == "Flutter app"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Email: someone@example.com", f"Email: {EMAIL_PLACEHOLDER}"),
        ("Call +44 20 7946 0000 today", f"Call {PHONE_PLACEHOLDER} today"),  # Ofcom fiction range
        ("Phone: (555) 555-0100", f"Phone: {PHONE_PLACEHOLDER}"),
    ],
)
def test_contact_details_are_redacted(text: str, expected: str) -> None:
    assert TextCleaner().clean(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "BS Information Technology, 2019–2023",
        "Worked 2019-2023 on 12 apps",
        "5+ years of experience",
        "Nov 2024 – 2025",
    ],
)
def test_years_and_counts_are_not_mistaken_for_phone_numbers(text: str) -> None:
    assert TextCleaner().clean(text) == text


def test_digits_inside_urls_are_preserved() -> None:
    text = "App Store (https://apps.apple.com/pk/app/x/id6444019910)"
    assert TextCleaner().clean(text) == text


def test_redaction_can_be_disabled() -> None:
    cleaner = TextCleaner(redact_emails=False, redact_phone_numbers=False)
    assert cleaner.clean("a@b.co +1 555 555 0100") == "a@b.co +1 555 555 0100"


def test_clean_blocks_drops_empty_and_duplicate_blocks() -> None:
    blocks = [
        TextBlock(text="Same", heading_path=("A",)),
        TextBlock(text=" Same ", heading_path=("A",)),
        TextBlock(text="Same", heading_path=("B",)),  # different section: kept
        TextBlock(text="​", heading_path=("A",)),
    ]
    cleaned = TextCleaner().clean_blocks(blocks)
    assert [(b.heading_path, b.text) for b in cleaned] == [(("A",), "Same"), (("B",), "Same")]
