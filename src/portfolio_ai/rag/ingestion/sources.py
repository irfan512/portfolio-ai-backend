"""The approved-source allowlist. Nothing outside it is ever fetched or indexed."""

import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from portfolio_ai.rag.domain.documents import SourceType, Visibility
from portfolio_ai.rag.errors import UnapprovedSourceError


class SourcePolicy(BaseModel):
    """Which hosts and path prefixes may be fetched (HTTPS only)."""

    model_config = ConfigDict(frozen=True)

    allowed_hosts: frozenset[str] = Field(min_length=1)
    allowed_path_prefixes: tuple[str, ...] = ("/",)

    def canonicalize(self, url: str) -> str:
        """Return a normalized URL, or raise `UnapprovedSourceError`.

        Normalizing first means the same page always has the same identity, and
        means tricks like `https://irfan512.github.io@evil.example/` are rejected.
        """
        try:
            parts = urlsplit(url.strip())
            port = parts.port
        except ValueError as exc:
            raise UnapprovedSourceError(f"Malformed URL: {url!r}") from exc

        if parts.scheme.lower() != "https":
            raise UnapprovedSourceError(f"Only https URLs are allowed: {url!r}")
        if parts.username or parts.password:
            raise UnapprovedSourceError(f"URLs with credentials are not allowed: {url!r}")
        host = (parts.hostname or "").lower()
        if host not in self.allowed_hosts:
            raise UnapprovedSourceError(f"Host {host!r} is not approved: {url!r}")
        if port not in (None, 443):
            raise UnapprovedSourceError(f"Non-standard port is not allowed: {url!r}")

        path = parts.path or "/"
        if any(segment in {".", ".."} for segment in path.split("/")):
            raise UnapprovedSourceError(f"Relative path segments are not allowed: {url!r}")
        if not any(_path_matches(path, prefix) for prefix in self.allowed_path_prefixes):
            raise UnapprovedSourceError(f"Path {path!r} is outside the approved prefixes")

        return urlunsplit(("https", host, path, parts.query, parts.fragment))

    def is_allowed(self, url: str) -> bool:
        try:
            self.canonicalize(url)
        except UnapprovedSourceError:
            return False
        return True


def _path_matches(path: str, prefix: str) -> bool:
    return path.startswith(prefix) or path == prefix.rstrip("/")


class ApprovedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    loader: Literal["portfolio_site"]
    source_type: SourceType = SourceType.WEBSITE
    visibility: Visibility = Visibility.PUBLIC


class SourceCatalog(BaseModel):
    """The full ingestion allowlist: a policy plus the exact sources to load."""

    model_config = ConfigDict(frozen=True)

    policy: SourcePolicy
    sources: tuple[ApprovedSource, ...]

    @field_validator("sources")
    @classmethod
    def _validate_sources(
        cls, sources: tuple[ApprovedSource, ...], info: ValidationInfo
    ) -> tuple[ApprovedSource, ...]:
        policy: SourcePolicy | None = info.data.get("policy")
        if policy is None:  # policy itself failed validation; that error is reported
            return sources
        canonical: list[ApprovedSource] = []
        for source in sources:
            if source.visibility is not Visibility.PUBLIC:
                raise ValueError(
                    f"{source.url}: the public assistant may only index public sources"
                )
            try:
                url = policy.canonicalize(source.url)
            except UnapprovedSourceError as exc:
                raise ValueError(str(exc)) from exc
            canonical.append(source.model_copy(update={"url": url}))
        urls = [source.url for source in canonical]
        if len(urls) != len(set(urls)):
            raise ValueError("Duplicate source URLs in catalog")
        return tuple(canonical)


def load_catalog(path: Path) -> SourceCatalog:
    """Read `sources.toml` (stdlib `tomllib`) into a validated catalog."""
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return SourceCatalog.model_validate(data)
