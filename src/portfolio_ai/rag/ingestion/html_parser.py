"""Turn rendered portfolio HTML into sections of heading-aware text blocks.

Each top-level `<section id="...">` inside `<main>` becomes one section, so it can
be cited as `<page>#<id>`. Inside a section, an `<article>` or `<li>` that holds
exactly one heading is a "unit" (a project, a job, a service), and everything in it
is filed under that heading, even text that comes before the heading in the markup.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from portfolio_ai.rag.domain.documents import TextBlock
from portfolio_ai.rag.errors import ParseError
from portfolio_ai.rag.ingestion.cleaning import normalize_whitespace

HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt", "figcaption",
        "figure", "footer", "header", "li", "main", "ol", "p", "section", "table", "tbody",
        "td", "th", "thead", "tr", "ul", *HEADING_LEVELS,
    }
)  # fmt: skip
UNIT_TAGS = frozenset({"article", "li"})
# Interactive or non-text elements: no retrievable facts, only UI.
NON_CONTENT_TAGS = (
    "script", "style", "noscript", "template", "svg", "img", "picture", "video", "audio",
    "iframe", "canvas", "form", "button", "input", "select", "textarea", "nav",
)  # fmt: skip
DEFAULT_FALLBACK_TITLES: Mapping[str, str] = {"top": "Introduction"}


@dataclass(frozen=True)
class ParsedSection:
    anchor: str
    title: str
    blocks: tuple[TextBlock, ...]


def parse_sections(
    html: str,
    *,
    include_hidden: bool = True,
    fallback_titles: Mapping[str, str] = DEFAULT_FALLBACK_TITLES,
) -> list[ParsedSection]:
    """Parse sections in document order. Sections without text are left out.

    `include_hidden=True` keeps collapsed panels (`hidden` attribute): on this site
    they hold published project details behind a "Project details" toggle.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup.body
    if not isinstance(root, Tag):
        raise ParseError("HTML has no <main> or <body> element")
    _strip_non_content(root, include_hidden=include_hidden)

    sections: list[ParsedSection] = []
    seen_anchors: set[str] = set()
    for element in root.find_all("section", id=True):
        if element.find_parent("section") is not None:
            continue  # nested sections belong to their outer section
        anchor = str(element["id"]).strip()
        if not anchor or anchor in seen_anchors:
            continue
        seen_anchors.add(anchor)

        title_tag = element.find("h2")
        title = _text(title_tag) if title_tag else ""
        if title_tag is not None:
            title_tag.decompose()
        title = title or fallback_titles.get(anchor) or anchor.replace("-", " ").title()

        walker = _SectionWalker()
        walker.walk(element, unit_path=None, skip=None)
        if walker.blocks:
            sections.append(ParsedSection(anchor=anchor, title=title, blocks=tuple(walker.blocks)))
    return sections


def _strip_non_content(root: Tag, *, include_hidden: bool) -> None:
    doomed: list[Tag] = list(root.find_all(NON_CONTENT_TAGS))
    doomed += root.find_all(attrs={"aria-hidden": "true"})
    doomed += root.find_all(class_="sr-only")  # e.g. "(opens in a new tab)"
    if not include_hidden:
        doomed += root.find_all(attrs={"hidden": True})
    for tag in doomed:
        if not tag.decomposed:
            tag.decompose()

    # Links are rewritten after removal, so their label no longer has sr-only text.
    for anchor in root.find_all("a", href=True):
        href = str(anchor["href"])
        if href.startswith("#"):
            anchor.decompose()  # in-page navigation ("Explore my work")
        elif href.startswith(("http://", "https://")):
            label = _text(anchor)
            anchor.clear()
            anchor.append(NavigableString(href if label in ("", href) else f"{label} ({href})"))


def _text(tag: Tag) -> str:
    return normalize_whitespace(tag.get_text(" "))


def _is_leaf(tag: Tag) -> bool:
    return tag.find(BLOCK_TAGS) is None


class _SectionWalker:
    """Depth-first walk that tracks the current heading path."""

    def __init__(self) -> None:
        self.blocks: list[TextBlock] = []
        self._headings: list[tuple[int, str]] = []  # headings outside units
        self._seen: set[tuple[tuple[str, ...], str]] = set()

    def walk(self, node: Tag, unit_path: tuple[str, ...] | None, skip: Tag | None) -> None:
        for child in node.children:
            if isinstance(child, Tag):
                self._visit(child, unit_path, skip)
            elif type(child) is NavigableString:  # excludes comments, CDATA, doctype
                self._emit(str(child), unit_path)

    def _visit(self, tag: Tag, unit_path: tuple[str, ...] | None, skip: Tag | None) -> None:
        if tag is skip:
            return
        level = HEADING_LEVELS.get(tag.name)
        if level is not None and unit_path is None:
            if text := _text(tag):
                self._headings = [(lvl, h) for lvl, h in self._headings if lvl < level]
                self._headings.append((level, text))
            return
        if unit_path is None and tag.name in UNIT_TAGS:
            headings = tag.find_all(list(HEADING_LEVELS))
            if len(headings) == 1 and (title := _text(headings[0])):
                unit_level = HEADING_LEVELS[headings[0].name]
                self.walk(tag, self._path_above(unit_level) + (title,), skip=headings[0])
                return
        if tag.name == "dl":
            self._emit_definitions(tag, unit_path)
            return
        if tag.name in {"ul", "ol"} and tag.get("aria-label") and _is_flat_list(tag):
            items = [_text(li) for li in tag.find_all("li")]
            self._emit(f"{tag['aria-label']}: {', '.join(i for i in items if i)}", unit_path)
            return
        if _is_leaf(tag):
            self._emit(_text(tag), unit_path)
            return
        self.walk(tag, unit_path, skip)

    def _emit_definitions(self, dl: Tag, unit_path: tuple[str, ...] | None) -> None:
        """`<dt>Mobile</dt><dd>Flutter, Dart</dd>` -> "Mobile: Flutter, Dart"."""
        term = ""
        for item in dl.find_all(["dt", "dd"]):
            if item.name == "dt":
                term = _text(item)
            elif value := _text(item):
                self._emit(f"{term}: {value}" if term else value, unit_path)

    def _path_above(self, level: int) -> tuple[str, ...]:
        return tuple(text for lvl, text in self._headings if lvl < level)

    def _emit(self, text: str, unit_path: tuple[str, ...] | None) -> None:
        text = normalize_whitespace(text)
        if not text:
            return
        path = unit_path if unit_path is not None else self._path_above(99)
        if (path, text) in self._seen:  # responsive layouts often render text twice
            return
        self._seen.add((path, text))
        self.blocks.append(TextBlock(text=text, heading_path=path))


def _is_flat_list(tag: Tag) -> bool:
    return all(_is_leaf(li) for li in tag.find_all("li"))
