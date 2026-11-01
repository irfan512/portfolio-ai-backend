"""Logging setup: one JSON object per line (machine-readable) or plain text.

Library code just calls `logging.getLogger(__name__).info("event_name", extra={...})`.
Only the application entry point decides how logs are formatted.
Configured secret values are masked in every line as a last line of defence.
"""

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal

# Attributes every LogRecord has; anything else came from `extra=`.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
}
MASK = "***"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class RedactingFormatter(logging.Formatter):
    """Wrap another formatter and replace secret values in its output."""

    def __init__(self, inner: logging.Formatter, secrets: Iterable[str]) -> None:
        super().__init__()
        self._inner = inner
        self._secrets = sorted({s for s in secrets if len(s) >= 6}, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        text = self._inner.format(record)
        for secret in self._secrets:
            text = text.replace(secret, MASK)
        return text


def configure_logging(
    level: str = "INFO", fmt: Literal["json", "text"] = "json", secrets: Iterable[str] = ()
) -> None:
    inner: logging.Formatter = (
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter(inner, secrets))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Uvicorn's access log includes client IPs; ours (api/middleware.py) does not.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
