"""Core data model shared by ingestion, retrieval and generation."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field


class SourceType(StrEnum):
    WEBSITE = "website"
    MARKDOWN = "markdown"
    PDF = "pdf"
    GITHUB_README = "github_readme"
    CURATED_PROFILE = "curated_profile"


class Visibility(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"


class TextBlock(BaseModel):
    """One paragraph-like unit of text plus the headings it sits under."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()


class SourceDocument(BaseModel):
    """A cleaned, citable document, e.g. one section of the portfolio page."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source_key: str  # the approved source that produced it; scopes stale-document deletion
    source_type: SourceType
    source_uri: str  # citation URL shown to visitors
    title: str
    blocks: tuple[TextBlock, ...] = Field(min_length=1)
    content_hash: str
    visibility: Visibility = Visibility.PUBLIC
    updated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content(self) -> str:
        return "\n".join(block.text for block in self.blocks)


class Chunk(BaseModel):
    """A retrieval unit: small enough to embed, self-describing enough to cite."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source_key: str
    document_title: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    section: str
    page_number: int | None = None
    source_uri: str
    content_hash: str  # hash of the parent document version
    visibility: Visibility
    token_count: int = Field(ge=1)
    index_fingerprint: str = ""  # chunker settings that produced it (provenance)
