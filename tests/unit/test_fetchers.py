from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from portfolio_ai.rag.errors import FetchError, UnapprovedSourceError
from portfolio_ai.rag.ingestion.fetchers import (
    FetchedPage,
    PlaywrightFetcher,
    RecordingFetcher,
    SnapshotFetcher,
)
from portfolio_ai.rag.ingestion.sources import SourcePolicy

POLICY = SourcePolicy(
    allowed_hosts=frozenset({"irfan512.github.io"}), allowed_path_prefixes=("/portfolio/",)
)
URL = "https://irfan512.github.io/portfolio/"


@dataclass
class FakeRoute:
    url: str
    outcome: list[str] = field(default_factory=list)

    @property
    def request(self) -> "FakeRoute":
        return self

    def continue_(self) -> None:
        self.outcome.append("continue")

    def abort(self) -> None:
        self.outcome.append("abort")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (f"{URL}static/js/main.js", "continue"),
        ("https://fonts.googleapis.com/css2?family=Inter", "abort"),
        ("https://api.emailjs.com/api/v1.0/email/send", "abort"),
    ],
)
def test_browser_requests_are_limited_to_the_policy(url: str, expected: str) -> None:
    route = FakeRoute(url)
    PlaywrightFetcher(POLICY)._guard_request(route)
    assert route.outcome == [expected]


def test_playwright_fetcher_refuses_unapproved_url_before_launching() -> None:
    with pytest.raises(UnapprovedSourceError):
        PlaywrightFetcher(POLICY).fetch("https://evil.example/")


class StaticFetcher:
    def fetch(self, url: str) -> FetchedPage:
        return FetchedPage(
            url=url, html="<main>hi</main>", fetched_at=datetime(2026, 1, 1, tzinfo=UTC)
        )


def test_recorded_snapshot_round_trips(tmp_path: Path) -> None:
    recorded = RecordingFetcher(StaticFetcher(), tmp_path).fetch(URL)
    replayed = SnapshotFetcher(tmp_path, POLICY).fetch(URL)
    assert replayed == recorded


def test_missing_snapshot_raises_fetch_error(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="No snapshot"):
        SnapshotFetcher(tmp_path, POLICY).fetch(URL)
