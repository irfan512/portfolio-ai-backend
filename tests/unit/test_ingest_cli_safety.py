"""The ingest CLI must never write to a hosted Qdrant by accident."""

import argparse

import pytest

from portfolio_ai.core.config import Settings
from portfolio_ai.ingest_cli import _remote_write_allowed

CLOUD = "https://abc.eu-central-1-0.aws.cloud.qdrant.io"


def args(**kw: bool) -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=kw.get("dry_run", False), confirm_remote=kw.get("confirm", False)
    )


def settings(url: str) -> Settings:
    return Settings(_env_file=None, qdrant_url=url, qdrant_api_key="k" * 20)


def test_local_target_needs_no_confirmation() -> None:
    assert _remote_write_allowed(settings("http://127.0.0.1:6333"), args()) is True


def test_remote_target_is_refused_without_confirmation(capsys: pytest.CaptureFixture[str]) -> None:
    assert _remote_write_allowed(settings(CLOUD), args()) is False
    out = capsys.readouterr().out
    assert "abc.eu-central-1-0.aws.cloud.qdrant.io" in out
    assert "portfolio_chunks__all-minilm-l6-v2__d384__v1" in out
    assert "k" * 20 not in out  # the key is never printed


@pytest.mark.parametrize("kw", [{"dry_run": True}, {"confirm": True}])
def test_remote_target_allowed_for_dry_run_or_explicit_confirmation(kw: dict) -> None:
    assert _remote_write_allowed(settings(CLOUD), args(**kw)) is True
