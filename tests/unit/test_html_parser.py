import pytest

from portfolio_ai.rag.errors import ParseError
from portfolio_ai.rag.ingestion.html_parser import ParsedSection, parse_sections


def by_anchor(html: str, **kwargs: bool) -> dict[str, ParsedSection]:
    return {section.anchor: section for section in parse_sections(html, **kwargs)}


def texts(section: ParsedSection) -> list[str]:
    return [block.text for block in section.blocks]


def test_sections_are_returned_in_order_and_empty_ones_dropped(synthetic_html: str) -> None:
    anchors = [s.anchor for s in parse_sections(synthetic_html)]
    assert anchors == ["top", "work", "experience", "contact"]


def test_section_title_comes_from_h2_with_fallback(synthetic_html: str) -> None:
    sections = by_anchor(synthetic_html)
    assert sections["work"].title == "Selected work"
    assert sections["top"].title == "Introduction"  # no h2 -> fallback map


def test_unit_blocks_are_filed_under_their_heading(synthetic_html: str) -> None:
    work = by_anchor(synthetic_html)["work"]
    alpha = [b.text for b in work.blocks if b.heading_path == ("Project Alpha",)]
    # The category line appears *before* the <h3> in markup but belongs to the unit.
    assert alpha[0] == "Example category"
    assert "Technologies: Flutter, Firebase" in alpha
    assert "Hidden detail about Alpha." in alpha  # collapsed panels are published content


def test_nested_heading_paths(synthetic_html: str) -> None:
    work = by_anchor(synthetic_html)["work"]
    paths = {b.heading_path for b in work.blocks}
    assert ("More projects",) in paths
    assert ("More projects", "Project Beta") in paths


def test_definition_lists_become_term_value_lines(synthetic_html: str) -> None:
    experience = by_anchor(synthetic_html)["experience"]
    assert "Mobile: Flutter, Dart" in texts(experience)
    role = [b for b in experience.blocks if b.heading_path == ("Example Engineer",)]
    assert [b.text for b in role] == ["2020 – 2022", "Example Co", "Shipped example things."]


def test_external_links_keep_their_url_and_drop_screen_reader_text(synthetic_html: str) -> None:
    work = texts(by_anchor(synthetic_html)["work"])
    assert "Google Play (https://play.example.com/store/apps/details?id=4155550123987)" in work
    assert not any("opens in a new tab" in t for t in work)


def test_ui_chrome_is_not_indexed(synthetic_html: str) -> None:
    everything = " ".join(t for s in parse_sections(synthetic_html) for t in texts(s))
    for unwanted in (
        "Explore my work",  # in-page link
        "Project details",  # button
        "Send message",  # form
        "Form notice.",
        "Footer text",
        "Skip to main content",
        "tracking",  # script
        "·",  # aria-hidden decoration
    ):
        assert unwanted not in everything


def test_responsive_duplicates_are_collapsed(synthetic_html: str) -> None:
    top = texts(by_anchor(synthetic_html)["top"])
    assert top.count("I build example software.") == 1


def test_hidden_content_can_be_excluded(synthetic_html: str) -> None:
    work = texts(by_anchor(synthetic_html, include_hidden=False)["work"])
    assert "Hidden detail about Alpha." not in work


def test_html_without_body_raises() -> None:
    with pytest.raises(ParseError):
        parse_sections("")
