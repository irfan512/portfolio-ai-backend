"""Gemini adapter using the official Google GenAI SDK (`google-genai`).

- The API key goes in the `x-goog-api-key` header (never the URL), and it is
  never put in logs or exception messages.
- The SDK's own retries are disabled. Retries here are explicit: only for
  429/5xx/timeouts, exponential backoff with jitter, bounded attempts.
- Structured output (`response_mime_type=application/json` plus a schema) makes
  the model return `GroundedAnswer` JSON, which is validated with Pydantic.
"""

import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from portfolio_ai.rag.errors import (
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from portfolio_ai.rag.generation.base import GenerationRequest, GroundedAnswer

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class GeminiProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        backoff_s: float = 0.75,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        client: Any | None = None,  # injectable fake for tests
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key or not model:
            raise LLMConfigurationError("GEMINI_API_KEY and GEMINI_MODEL must both be set")
        from google.genai import types

        self._types = types
        self._model = model
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._thinking_budget = thinking_budget
        self._thinking_level = thinking_level
        self._sleep = sleep
        self.usage: list[dict[str, int]] = []  # token usage of each successful call
        if client is None:
            from google import genai

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(timeout_s * 1000),  # SDK expects milliseconds
                    retry_options=types.HttpRetryOptions(attempts=1),  # we retry ourselves
                ),
            )
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model

    def generate_grounded(self, request: GenerationRequest) -> GroundedAnswer:
        config = self._config(request)
        contents = [
            self._types.Content(role=m.role.value, parts=[self._types.Part(text=m.text)])
            for m in request.messages
        ]
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
                answer = self._parse(response)
                self.usage.append(_usage(response))
                logger.info(
                    "llm_call_succeeded",
                    extra={
                        "model": self._model,
                        "attempt": attempt,
                        "ms": round((time.perf_counter() - started) * 1000),
                        **_usage(response),
                    },
                )
                return answer
            except LLMResponseError:
                raise
            except Exception as exc:  # classified below; never re-raised raw
                error = _classify(exc)
                retryable = not isinstance(error, LLMConfigurationError) and _is_retryable(exc)
                logger.warning(
                    "llm_call_failed",
                    extra={
                        "model": self._model,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "status": getattr(exc, "code", None),
                        "retryable": retryable,
                    },
                )
                if not retryable or attempt > self._max_retries:
                    raise error from None  # `from None`: don't chain SDK internals
                delay = self._backoff_s * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                self._sleep(delay)

    def _config(self, request: GenerationRequest) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": request.system_instruction,
            "max_output_tokens": request.max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": GroundedAnswer,
        }
        thinking: dict[str, Any] = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if self._thinking_budget is not None:
            thinking["thinking_budget"] = self._thinking_budget
        if self._thinking_level is not None:
            thinking["thinking_level"] = self._thinking_level
        if thinking:
            kwargs["thinking_config"] = self._types.ThinkingConfig(**thinking)
        return self._types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _parse(response: Any) -> GroundedAnswer:
        text = getattr(response, "text", None)
        if not text:
            reason = _finish_reason(response)
            raise LLMResponseError(f"Model returned no text (finish_reason={reason})")
        try:
            return GroundedAnswer.model_validate_json(text)
        except ValidationError as exc:
            raise LLMResponseError("Model output did not match the answer schema") from exc


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError)):
        return True
    return getattr(exc, "code", None) in RETRYABLE_STATUS


def _classify(exc: Exception) -> LLMError:
    """Map SDK/transport errors to our types, with messages that contain no secrets."""
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return LLMTimeoutError("Gemini request timed out")
    code = getattr(exc, "code", None)
    if code == 429:
        return LLMRateLimitError("Gemini quota or rate limit reached")
    if code in (400, 404):
        return LLMConfigurationError(
            f"Gemini rejected the request (HTTP {code}); check GEMINI_MODEL"
        )
    if code in (401, 403):
        return LLMConfigurationError(f"Gemini rejected the credentials (HTTP {code})")
    if isinstance(code, int):
        return LLMUnavailableError(f"Gemini failed with HTTP {code}")
    return LLMUnavailableError(f"Gemini call failed ({type(exc).__name__})")


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    return (
        str(getattr(candidates[0], "finish_reason", "unknown")) if candidates else "no_candidates"
    )


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    fields = {
        "prompt_tokens": "prompt_token_count",
        "output_tokens": "candidates_token_count",
        "total_tokens": "total_token_count",
    }
    return {
        k: int(v) for k, attr in fields.items() if (v := getattr(usage, attr, None)) is not None
    }
