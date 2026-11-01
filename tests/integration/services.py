"""Skip service-backed tests when the service is down; REQUIRE_SERVICES=1 fails instead."""

import os

import httpx
import pytest


def require_service(url: str, name: str) -> None:
    try:
        httpx.get(url, timeout=1.0).raise_for_status()
    except httpx.HTTPError:
        message = f"{name} is not running at {url} (docker compose up -d {name})"
        if os.getenv("REQUIRE_SERVICES") == "1":
            pytest.fail(message)
        pytest.skip(message)
