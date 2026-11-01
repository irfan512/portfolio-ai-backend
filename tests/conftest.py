from datetime import UTC, datetime
from pathlib import Path

import pytest

from portfolio_ai.rag.ingestion.fetchers import FetchedPage
from portfolio_ai.rag.ingestion.sources import SourceCatalog

FIXTURES = Path(__file__).parent / "fixtures"
SITE = "https://irfan512.github.io/portfolio/"


@pytest.fixture
def synthetic_html() -> str:
    return (FIXTURES / "portfolio_synthetic.html").read_text(encoding="utf-8")


@pytest.fixture
def catalog() -> SourceCatalog:
    return SourceCatalog.model_validate(
        {
            "policy": {
                "allowed_hosts": ["irfan512.github.io"],
                "allowed_path_prefixes": ["/portfolio/"],
            },
            "sources": [{"url": SITE, "loader": "portfolio_site"}],
        }
    )


class FakeFetcher:
    """Serves whatever HTML the test sets; counts calls."""

    def __init__(self, html: str) -> None:
        self.html = html
        self.calls = 0

    def fetch(self, url: str) -> FetchedPage:
        self.calls += 1
        return FetchedPage(url=url, html=self.html, fetched_at=datetime(2026, 1, 1, tzinfo=UTC))


@pytest.fixture
def fake_fetcher(synthetic_html: str) -> FakeFetcher:
    return FakeFetcher(synthetic_html)
