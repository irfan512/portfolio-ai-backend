"""Real Gemini + local Qdrant + local ONNX embedder. COSTS MONEY (at most GEMINI_TEST_MAX_CALLS).

    RUN_GEMINI_TESTS=1 pytest -m gemini -v -s

Hard limits: automatic retries are forced off (gemini_max_retries=0; SDK retries are
always off), and a budget guard refuses any generation request beyond
GEMINI_TEST_MAX_CALLS (default 4) before it reaches the network.
"""

import logging
import os

import pytest

from portfolio_ai.container import build_services
from portfolio_ai.core.config import Settings
from portfolio_ai.rag.service import Turn

pytestmark = [
    pytest.mark.gemini,
    pytest.mark.skipif(
        os.getenv("RUN_GEMINI_TESTS") != "1", reason="paid API: set RUN_GEMINI_TESTS=1"
    ),
]
PORTFOLIO = "https://irfan512.github.io/portfolio/#"
MAX_CALLS = int(os.getenv("GEMINI_TEST_MAX_CALLS", "4"))


class BudgetedLLM:
    """Counts generation requests and refuses to exceed the approved budget."""

    def __init__(self, inner, max_calls: int) -> None:
        self.inner, self.max_calls, self.calls = inner, max_calls, 0

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    def generate_grounded(self, request):
        if self.calls >= self.max_calls:
            raise AssertionError(
                f"Gemini request budget of {self.max_calls} exhausted; not sending"
            )
        self.calls += 1
        return self.inner.generate_grounded(request)


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.getMessage()} {record.__dict__}")


@pytest.fixture(scope="module")
def live():
    settings = Settings(gemini_max_retries=0)  # one attempt per request, no retries
    if not (settings.gemini_api_key and settings.gemini_model):
        pytest.skip("GEMINI_API_KEY / GEMINI_MODEL not configured")
    services = build_services(settings)
    if services.answer_service is None or not all(services.readiness().values()):
        pytest.skip("local stack not ready (docker compose up -d && portfolio-ai-ingest)")
    service = services.answer_service
    budget = BudgetedLLM(service._llm, MAX_CALLS)
    service._llm = budget
    logs = _Collect()
    logging.getLogger().addHandler(logs)
    yield service, budget, settings, logs
    logging.getLogger().removeHandler(logs)

    usage = budget.inner.usage
    prompt = sum(u.get("prompt_tokens", 0) for u in usage)
    output = sum(u.get("output_tokens", 0) for u in usage)
    total = sum(u.get("total_tokens", 0) for u in usage)
    print(f"\n[gemini usage] model={settings.gemini_model} requests={budget.calls} "
          f"successful={len(usage)} prompt_tokens={prompt} output_tokens={output} total_tokens={total}")  # fmt: skip


def check_citations(result) -> None:
    assert result.citations, "expected at least one citation"
    for citation in result.citations:
        assert citation.url.startswith(PORTFOLIO), citation.url


def no_secrets(text: str, settings: Settings) -> None:
    for secret in settings.secret_values():
        assert secret not in text


def test_grounded_answer_with_citation(live) -> None:
    service, _, settings, _ = live
    result = service.answer("Has Irfan built a telemedicine or healthcare app?")
    print(
        f"\n[grounded] {result.answer}\n  citations={[c.url + ' / ' + c.section for c in result.citations]}"
    )
    assert result.insufficient_evidence is False
    check_citations(result)
    assert any(c.section == "goCare" for c in result.citations)
    assert "goCare" in result.answer
    no_secrets(result.model_dump_json(), settings)


def test_follow_up_question(live) -> None:
    service, _, settings, _ = live
    first = service.answer("Tell me about the goCare app.")
    follow_up = service.answer(
        "Which payment provider did it use?",
        [Turn(question="Tell me about the goCare app.", answer=first.answer)],
    )
    print(f"\n[follow-up] {follow_up.answer}\n  citations={[c.url for c in follow_up.citations]}")
    assert "Flutterwave" in follow_up.answer
    check_citations(follow_up)
    no_secrets(first.model_dump_json() + follow_up.model_dump_json(), settings)


def test_unanswerable_question_abstains(live) -> None:
    service, budget, settings, logs = live
    result = service.answer("What is Irfan's hourly rate for a 3-month contract?")
    print(f"\n[unanswerable] insufficient_evidence={result.insufficient_evidence} {result.answer}")
    assert result.insufficient_evidence is True
    assert "redacted" not in result.answer
    no_secrets(result.model_dump_json() + "\n".join(logs.lines), settings)
    assert budget.calls <= MAX_CALLS
