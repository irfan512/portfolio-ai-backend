from pathlib import Path

import pytest
from pydantic import ValidationError

from portfolio_ai.rag.errors import UnapprovedSourceError
from portfolio_ai.rag.ingestion.sources import SourceCatalog, SourcePolicy, load_catalog

POLICY = SourcePolicy(
    allowed_hosts=frozenset({"irfan512.github.io"}), allowed_path_prefixes=("/portfolio/",)
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://irfan512.github.io/portfolio/", "https://irfan512.github.io/portfolio/"),
        ("  HTTPS://IRFAN512.github.io/portfolio/  ", "https://irfan512.github.io/portfolio/"),
        ("https://irfan512.github.io/portfolio", "https://irfan512.github.io/portfolio"),
        (
            "https://irfan512.github.io:443/portfolio/#work",
            "https://irfan512.github.io/portfolio/#work",
        ),
        (
            "https://irfan512.github.io/portfolio/static/js/main.js",
            "https://irfan512.github.io/portfolio/static/js/main.js",
        ),
    ],
)
def test_canonicalize_accepts_approved_urls(url: str, expected: str) -> None:
    assert POLICY.canonicalize(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://irfan512.github.io/portfolio/",  # not https
        "https://evil.example/portfolio/",  # other host
        "https://irfan512.github.io.evil.example/portfolio/",  # look-alike host
        "https://irfan512.github.io@evil.example/portfolio/",  # userinfo trick
        "https://irfan512.github.io/other-repo/",  # outside path prefix
        "https://irfan512.github.io/portfolio/../secret",  # traversal
        "https://irfan512.github.io:8443/portfolio/",  # odd port
        "https://fonts.googleapis.com/css2",  # third-party subresource
        "javascript:alert(1)",
    ],
)
def test_canonicalize_rejects_unapproved_urls(url: str) -> None:
    with pytest.raises(UnapprovedSourceError):
        POLICY.canonicalize(url)
    assert POLICY.is_allowed(url) is False


def test_catalog_canonicalizes_source_urls() -> None:
    catalog = SourceCatalog.model_validate(
        {
            "policy": {"allowed_hosts": ["irfan512.github.io"]},
            "sources": [
                {"url": "HTTPS://irfan512.github.io/portfolio/", "loader": "portfolio_site"}
            ],
        }
    )
    assert catalog.sources[0].url == "https://irfan512.github.io/portfolio/"


@pytest.mark.parametrize(
    "source",
    [
        {"url": "https://evil.example/", "loader": "portfolio_site"},
        {
            "url": "https://irfan512.github.io/",
            "loader": "portfolio_site",
            "visibility": "internal",
        },
        {"url": "https://irfan512.github.io/", "loader": "unknown_loader"},
    ],
)
def test_catalog_rejects_bad_sources(source: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        SourceCatalog.model_validate(
            {"policy": {"allowed_hosts": ["irfan512.github.io"]}, "sources": [source]}
        )


def test_catalog_rejects_duplicate_sources() -> None:
    url = "https://irfan512.github.io/portfolio/"
    with pytest.raises(ValidationError, match="Duplicate"):
        SourceCatalog.model_validate(
            {
                "policy": {"allowed_hosts": ["irfan512.github.io"]},
                "sources": [{"url": url, "loader": "portfolio_site"}] * 2,
            }
        )


def test_repository_sources_toml_is_valid() -> None:
    catalog = load_catalog(Path(__file__).resolve().parents[2] / "sources.toml")
    assert [s.url for s in catalog.sources] == ["https://irfan512.github.io/portfolio/"]
