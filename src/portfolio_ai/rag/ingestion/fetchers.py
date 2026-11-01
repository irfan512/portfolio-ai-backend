"""Ways to obtain a page's HTML: render it live, or replay a saved snapshot.

The portfolio is a client-side React app, so a plain HTTP GET returns an almost
empty shell. `PlaywrightFetcher` runs the page in headless Chromium and reads the
DOM after JavaScript has run.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from portfolio_ai.rag.errors import FetchError
from portfolio_ai.rag.ingestion.sources import SourcePolicy

logger = logging.getLogger(__name__)

# Buttons that reveal more published content ("Show 4 more projects", "Project details").
DEFAULT_EXPAND_SELECTOR = 'main button[aria-expanded="false"][aria-controls]'


@dataclass(frozen=True)
class FetchedPage:
    url: str
    html: str
    fetched_at: datetime


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


class PlaywrightFetcher:
    """Render an approved page in headless Chromium. Off-policy requests are aborted."""

    def __init__(
        self,
        policy: SourcePolicy,
        *,
        timeout_ms: int = 30_000,
        expand_selector: str = DEFAULT_EXPAND_SELECTOR,
        max_expand_clicks: int = 50,
        max_html_bytes: int = 5_000_000,
    ) -> None:
        self._policy = policy
        self._timeout_ms = timeout_ms
        self._expand_selector = expand_selector
        self._max_expand_clicks = max_expand_clicks
        self._max_html_bytes = max_html_bytes

    def fetch(self, url: str) -> FetchedPage:
        canonical = self._policy.canonicalize(url)
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # optional dependency
            raise FetchError(
                "Playwright is not installed. Run: pip install -e '.[ingest]' "
                "&& python -m playwright install chromium"
            ) from exc

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    context = browser.new_context(service_workers="block")
                    context.route("**/*", self._guard_request)
                    page = context.new_page()
                    response = page.goto(
                        canonical, wait_until="networkidle", timeout=self._timeout_ms
                    )
                    if response is None or not response.ok:
                        status = response.status if response else "no response"
                        raise FetchError(f"{canonical} returned {status}")
                    clicks = self._expand_collapsed_content(page)
                    html = page.content()
                finally:
                    browser.close()
        except PlaywrightError as exc:
            raise FetchError(f"Rendering {canonical} failed: {exc}") from exc

        if len(html.encode("utf-8")) > self._max_html_bytes:
            raise FetchError(f"{canonical} exceeded {self._max_html_bytes} bytes")
        logger.info(
            "page_rendered", extra={"url": canonical, "bytes": len(html), "expanded": clicks}
        )
        return FetchedPage(url=canonical, html=html, fetched_at=datetime.now(UTC))

    def _guard_request(self, route: Any) -> None:
        """Only let the browser load approved URLs (scripts, data, the page itself)."""
        request_url = route.request.url
        if self._policy.is_allowed(request_url):
            route.continue_()
        else:
            logger.debug("request_blocked", extra={"url": request_url})
            route.abort()

    def _expand_collapsed_content(self, page: Any) -> int:
        clicks = 0
        while clicks < self._max_expand_clicks:
            button = page.locator(self._expand_selector).first
            if button.count() == 0:
                break
            button.click()
            clicks += 1
        page.wait_for_timeout(300)  # let React commit the last render
        return clicks


def snapshot_path(directory: Path, url: str) -> Path:
    return directory / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.json"


class SnapshotFetcher:
    """Replay pages saved by `RecordingFetcher`: offline, fast, reproducible."""

    def __init__(self, directory: Path, policy: SourcePolicy) -> None:
        self._directory = directory
        self._policy = policy

    def fetch(self, url: str) -> FetchedPage:
        canonical = self._policy.canonicalize(url)
        path = snapshot_path(self._directory, canonical)
        if not path.exists():
            raise FetchError(f"No snapshot for {canonical} at {path}; run once without --offline")
        data = json.loads(path.read_text(encoding="utf-8"))
        return FetchedPage(
            url=data["url"],
            html=data["html"],
            fetched_at=datetime.fromisoformat(data["fetched_at"]),
        )


class RecordingFetcher:
    """Wrap another fetcher and save every page it returns as a snapshot."""

    def __init__(self, inner: PageFetcher, directory: Path) -> None:
        self._inner = inner
        self._directory = directory

    def fetch(self, url: str) -> FetchedPage:
        page = self._inner.fetch(url)
        self._directory.mkdir(parents=True, exist_ok=True)
        payload = {"url": page.url, "fetched_at": page.fetched_at.isoformat(), "html": page.html}
        snapshot_path(self._directory, page.url).write_text(json.dumps(payload), encoding="utf-8")
        return page
