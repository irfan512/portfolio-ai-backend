"""GeminiProvider with a fake SDK client: no network, no cost."""

from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from portfolio_ai.rag.errors import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from portfolio_ai.rag.generation.base import GenerationRequest, Message, Role
from portfolio_ai.rag.generation.gemini import GeminiProvider

SECRET = "test-only-not-a-real-key-1111"  # fake; tests assert it never leaks
VALID = '{"answer": "Irfan built goCare.", "cited_sources": ["S1"], "insufficient_evidence": false}'


def api_error(code: int) -> errors.APIError:
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": f"HTTP {code}", "status": "X"}})


@dataclass
class FakeModels:
    script: list  # each item: response text, or an exception to raise
    calls: list

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        usage = SimpleNamespace(
            prompt_token_count=120, candidates_token_count=30, total_token_count=150
        )
        return SimpleNamespace(text=item, candidates=[], usage_metadata=usage)


def provider(*script, **kwargs) -> tuple[GeminiProvider, FakeModels, list[float]]:
    models = FakeModels(list(script), [])
    sleeps: list[float] = []
    p = GeminiProvider(
        api_key=SECRET,
        model="gemini-test",
        client=SimpleNamespace(models=models),
        sleep=sleeps.append,
        **kwargs,
    )
    return p, models, sleeps


REQUEST = GenerationRequest(
    system_instruction="rules", messages=[Message(role=Role.USER, text="question")]
)


def test_successful_call_sends_structured_config() -> None:
    p, models, _ = provider(VALID)

    answer = p.generate_grounded(REQUEST)

    assert answer.cited_sources == ["S1"]
    config = models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.system_instruction == "rules"
    assert config.temperature is None  # not sent unless configured (deprecated on Gemini 3.x)
    assert models.calls[0]["model"] == "gemini-test"
    assert config.thinking_config is None  # only sent when configured


def test_thinking_level_is_passed_when_configured() -> None:
    p, models, _ = provider(VALID, thinking_level="low")
    p.generate_grounded(REQUEST)
    assert str(models.calls[0]["config"].thinking_config.thinking_level).lower().endswith("low")


def test_thinking_budget_is_passed_when_configured() -> None:
    p, models, _ = provider(VALID, thinking_budget=0)
    p.generate_grounded(REQUEST)
    assert models.calls[0]["config"].thinking_config.thinking_budget == 0


@pytest.mark.parametrize("failure", [api_error(429), api_error(503), httpx.ReadTimeout("slow")])
def test_transient_failures_are_retried_with_backoff(failure: Exception) -> None:
    p, models, sleeps = provider(failure, VALID, max_retries=2)
    assert p.generate_grounded(REQUEST).answer == "Irfan built goCare."
    assert len(models.calls) == 2
    assert len(sleeps) == 1 and sleeps[0] > 0


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (api_error(429), LLMRateLimitError),
        (api_error(500), LLMUnavailableError),
        (httpx.ReadTimeout("slow"), LLMTimeoutError),
    ],
)
def test_retries_are_bounded(failure: Exception, error: type[Exception]) -> None:
    p, models, sleeps = provider(failure, failure, failure, max_retries=2)
    with pytest.raises(error):
        p.generate_grounded(REQUEST)
    assert len(models.calls) == 3
    assert sleeps[1] > sleeps[0]  # exponential


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried(code: int) -> None:
    p, models, sleeps = provider(api_error(code))
    with pytest.raises(LLMConfigurationError):
        p.generate_grounded(REQUEST)
    assert len(models.calls) == 1 and sleeps == []


@pytest.mark.parametrize("text", ["", "not json", '{"answer": "x"}'])
def test_malformed_output_raises_response_error(text: str) -> None:
    p, _, _ = provider(text)
    with pytest.raises(LLMResponseError):
        p.generate_grounded(REQUEST)


def test_api_key_never_appears_in_errors_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    leaky = api_error(500)
    leaky.args = (f"failed with key {SECRET}",)
    p, _, _ = provider(leaky, max_retries=0)
    with caplog.at_level("DEBUG"), pytest.raises(LLMUnavailableError) as info:
        p.generate_grounded(REQUEST)
    assert SECRET not in str(info.value)
    assert info.value.__cause__ is None  # SDK exception not chained
    assert SECRET not in caplog.text


@pytest.mark.parametrize(("key", "model"), [("", "gemini-test"), (SECRET, "")])
def test_missing_configuration_is_rejected(key: str, model: str) -> None:
    with pytest.raises(LLMConfigurationError):
        GeminiProvider(api_key=key, model=model, client=object())


def test_temperature_is_sent_only_when_configured() -> None:
    p, models, _ = provider(VALID)
    p.generate_grounded(REQUEST.model_copy(update={"temperature": 0.3}))
    assert models.calls[0]["config"].temperature == 0.3
    assert p.usage == [{"prompt_tokens": 120, "output_tokens": 30, "total_tokens": 150}]


def test_default_timeout_leaves_room_for_thinking_models() -> None:
    """The SDK forwards this as X-Server-Timeout; 20 s produced 504s on gemini-3.8-flash."""
    from portfolio_ai.core.config import Settings

    assert Settings(_env_file=None).gemini_timeout_s >= 60
