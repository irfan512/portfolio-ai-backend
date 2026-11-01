"""The public API with real wiring but fake model/LLM: no network, no cost."""

import logging

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from portfolio_ai.api.rate_limit import RateLimiter
from portfolio_ai.container import Services
from portfolio_ai.conversations.sessions import SessionStore
from portfolio_ai.core.config import Settings
from portfolio_ai.main import create_app
from portfolio_ai.rag.errors import LLMRateLimitError, LLMTimeoutError, VectorStoreError
from portfolio_ai.rag.generation.base import GroundedAnswer
from portfolio_ai.rag.retrieval.qdrant_store import QdrantChunkStore
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever
from portfolio_ai.rag.service import AnswerService, AnswerSettings
from tests.fakes import FakeLLM, HashingEmbeddingProvider
from tests.unit.test_qdrant_store import EXPERIENCE, WORK, chunks_of

ORIGIN = "https://irfan512.github.io"
SECRET = "test-only-not-a-real-key-0000"  # fake; tests assert it never leaks


def make(
    llm: FakeLLM | None = None, *, configured: bool = True, **overrides
) -> tuple[TestClient, Services, FakeLLM]:
    settings = Settings(
        _env_file=None,
        environment=overrides.pop("environment", "test"),
        cors_origins=[ORIGIN],
        gemini_api_key=SECRET,
        gemini_model="fake",
        **overrides,
    )
    embedder = HashingEmbeddingProvider()
    store = QdrantChunkStore(QdrantClient(":memory:"), embedder, prefix="t")
    store.ensure_collection()
    store.upsert(chunks_of(WORK, EXPERIENCE))
    llm = llm or FakeLLM()
    answer = AnswerService(SemanticRetriever(embedder, store), llm, AnswerSettings(min_score=0.1))
    services = Services(
        settings=settings,
        sessions=SessionStore(),
        rate_limiter=RateLimiter(
            per_minute=settings.rate_limit_per_minute,
            per_day=settings.rate_limit_per_day,
            global_per_day=settings.chat_global_daily_limit,
            global_per_minute=settings.chat_global_per_minute_limit,
        ),
        embedder=embedder,
        store=store,
        answer_service=answer if configured else None,
    )
    return TestClient(create_app(settings, services), raise_server_exceptions=False), services, llm


def ask(client: TestClient, message: str = "Which app has Flutterwave payments?", **extra):
    return client.post(
        "/api/v1/chat", json={"message": message, **extra}, headers={"Origin": ORIGIN}
    )


def test_chat_returns_answer_citations_and_session() -> None:
    client, _, _ = make()
    response = ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Irfan built goCare."
    assert body["insufficient_evidence"] is False
    assert body["citations"] == [
        {"title": "Work", "section": "goCare", "url": "https://irfan512.github.io/portfolio/#work"}
    ]
    assert len(body["session_id"]) >= 16
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"]


def test_follow_up_reuses_session_history() -> None:
    client, _, llm = make()
    first = ask(client, "Tell me about the matchmaking app").json()

    ask(client, "Which technologies did it use?", session_id=first["session_id"])

    messages = llm.requests[1].messages
    assert messages[0].text == "Tell me about the matchmaking app"
    assert len(messages) == 3


def test_unknown_session_starts_a_new_one() -> None:
    client, _, _ = make()
    body = ask(client, session_id="x" * 20).json()
    assert body["session_id"] != "x" * 20


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 1001},
        {"message": "hi", "session_id": "bad id!"},
        {"message": "hi", "unexpected": True},
        {},
    ],
)
def test_invalid_requests_get_422_without_echoing_input(payload: dict) -> None:
    client, _, llm = make()
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert all(set(detail) == {"field", "issue"} for detail in error["details"])  # no echoed input
    assert "x" * 50 not in response.text
    assert llm.requests == []


def test_oversized_body_is_rejected_before_parsing() -> None:
    client, _, _ = make()
    response = client.post(
        "/api/v1/chat",
        content=b'{"message": "' + b"x" * 20_000 + b'"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_rate_limit_returns_429_with_retry_after() -> None:
    client, _, llm = make(rate_limit_per_minute=2)
    assert ask(client).status_code == 200
    assert ask(client).status_code == 200
    response = ask(client)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"
    assert int(response.headers["retry-after"]) > 0
    assert len(llm.requests) == 2


def test_global_daily_cap_protects_llm_budget() -> None:
    client, _, llm = make(chat_global_daily_limit=1)
    assert ask(client).status_code == 200
    response = ask(client)
    assert response.status_code == 503  # site-wide budget used up: friendly "busy"
    assert response.json()["error"]["code"] == "busy"
    assert int(response.headers["retry-after"]) > 0
    assert len(llm.requests) == 1  # no further LLM call, no fallback


def test_global_per_minute_cap_returns_busy() -> None:
    client, _, llm = make(chat_global_per_minute_limit=1)
    assert ask(client).status_code == 200
    assert ask(client).json()["error"]["code"] == "busy"
    assert len(llm.requests) == 1


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMTimeoutError("t"), 504, "timeout"),
        (LLMRateLimitError("q"), 503, "busy"),
        (VectorStoreError("qdrant.internal:6333 refused"), 503, "service_unavailable"),
    ],
)
def test_dependency_failures_map_to_safe_errors(error: Exception, status: int, code: str) -> None:
    client, _, _ = make(FakeLLM(error=error))
    response = ask(client)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "qdrant" not in response.text and "6333" not in response.text


def test_unexpected_exceptions_return_generic_500(caplog: pytest.LogCaptureFixture) -> None:
    client, _, _ = make(FakeLLM(error=RuntimeError(f"boom {SECRET}")))
    with caplog.at_level(logging.ERROR):
        response = ask(client)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "boom" not in response.text and SECRET not in response.text


def test_unconfigured_llm_returns_503_and_not_ready() -> None:
    client, _, _ = make(configured=False)
    assert ask(client).status_code == 503
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "checks": {"embeddings": True, "vector_store": True, "llm_configured": False},
    }


def test_ready_when_everything_is_up() -> None:
    client, _, _ = make()
    assert client.get("/ready").json()["status"] == "ready"
    assert client.get("/health").json() == {"status": "ok"}


def test_insufficient_evidence_is_reported() -> None:
    client, _, _ = make(
        FakeLLM(
            GroundedAnswer(
                answer="Not in the portfolio.", cited_sources=[], insufficient_evidence=True
            )
        )
    )
    body = ask(client, "What is Irfan's salary expectation? Flutterwave").json()
    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_docs_are_disabled_in_production() -> None:
    client, _, _ = make(environment="production")
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_available_in_development() -> None:
    client, _, _ = make(environment="development")
    assert client.get("/docs").status_code == 200


def test_unknown_origin_gets_no_cors_grant() -> None:
    client, _, _ = make()
    response = client.options(
        "/api/v1/chat",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_secret_never_in_responses() -> None:
    client, _, _ = make()
    for response in (ask(client), client.get("/ready"), client.get("/health"), client.get("/nope")):
        assert SECRET not in response.text
