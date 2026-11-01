"""Real Gemini + local Qdrant + local embedder. COSTS MONEY (a few small requests).

    RUN_GEMINI_TESTS=1 pytest -m gemini -v

Needs GEMINI_API_KEY / GEMINI_MODEL in the environment or .env, and the
portfolio indexed locally (`docker compose up -d && portfolio-ai-ingest`).
"""

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


@pytest.fixture(scope="module")
def answer_service():
    settings = Settings()
    if not (settings.gemini_api_key and settings.gemini_model):
        pytest.skip("GEMINI_API_KEY / GEMINI_MODEL not configured")
    services = build_services(settings)
    if services.answer_service is None or not all(services.readiness().values()):
        pytest.skip("local stack not ready (docker compose up -d && portfolio-ai-ingest)")
    return services.answer_service


def test_grounded_answer_with_citation(answer_service) -> None:
    result = answer_service.answer("Has Irfan built a telemedicine or healthcare app?")
    assert result.insufficient_evidence is False
    assert any(c.url.endswith("#work") for c in result.citations)
    assert "goCare" in result.answer


def test_follow_up_question(answer_service) -> None:
    first = answer_service.answer("Tell me about the goCare app.")
    follow_up = answer_service.answer(
        "Which payment provider did it use?",
        [Turn(question="Tell me about the goCare app.", answer=first.answer)],
    )
    assert "Flutterwave" in follow_up.answer


def test_unanswerable_question_abstains(answer_service) -> None:
    result = answer_service.answer("What is Irfan's hourly rate for a 3-month contract?")
    assert result.insufficient_evidence is True
