"""Optional smoke test against the real site. Run with: RUN_LIVE_TESTS=1 pytest -m live"""

import os

import pytest

from portfolio_ai.rag.ingestion.cleaning import TextCleaner
from portfolio_ai.rag.ingestion.fetchers import PlaywrightFetcher
from portfolio_ai.rag.ingestion.loaders import PortfolioSiteLoader
from portfolio_ai.rag.ingestion.sources import SourceCatalog

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1"),
]


def test_live_portfolio_renders_expected_sections(catalog: SourceCatalog) -> None:
    pytest.importorskip("playwright")
    loader = PortfolioSiteLoader(PlaywrightFetcher(catalog.policy), TextCleaner(), catalog.policy)

    documents = loader.load(catalog.sources[0])

    anchors = {d.source_uri.rsplit("#", 1)[1] for d in documents}
    assert {"work", "services", "experience", "contact"} <= anchors
    contact = next(d for d in documents if d.source_uri.endswith("#contact"))
    assert "@" not in contact.content  # email redacted
