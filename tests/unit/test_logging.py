import json
import logging

from portfolio_ai.core.logging import JsonFormatter


def test_json_formatter_emits_event_and_extra_fields() -> None:
    record = logging.LogRecord(
        "portfolio_ai.test", logging.INFO, __file__, 1, "doc_indexed", None, None
    )
    record.uri = "https://irfan512.github.io/portfolio/#work"
    record.chunks = 3

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "doc_indexed"
    assert payload["level"] == "INFO"
    assert payload["uri"].endswith("#work")
    assert payload["chunks"] == 3
    assert "msg" not in payload and "args" not in payload  # standard attrs filtered out


def test_redacting_formatter_masks_secrets() -> None:
    from portfolio_ai.core.logging import MASK, RedactingFormatter

    secret = "test-only-not-a-real-key-2222"
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, f"failed with {secret}", None, None)
    record.detail = f"header={secret}"
    out = RedactingFormatter(JsonFormatter(), [secret]).format(record)
    assert secret not in out and out.count(MASK) == 2


def test_settings_read_gemini_names_from_spec(monkeypatch) -> None:
    from portfolio_ai.core.config import Settings

    monkeypatch.setenv("GEMINI_API_KEY", "k-123456")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    settings = Settings(_env_file=None)
    assert settings.gemini_model == "gemini-test"
    assert settings.secret_values() == ["k-123456"]
    assert "k-123456" not in repr(settings) and "k-123456" not in settings.model_dump_json()
