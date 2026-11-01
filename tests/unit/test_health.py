from fastapi.testclient import TestClient

from portfolio_ai.core.config import Settings
from portfolio_ai.main import create_app


def make_client() -> TestClient:
    return TestClient(create_app(Settings(environment="test")))


def test_health_returns_ok() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_portfolio_origin() -> None:
    response = make_client().get("/health", headers={"Origin": "https://irfan512.github.io"})

    assert response.headers["access-control-allow-origin"] == "https://irfan512.github.io"


def test_cors_rejects_unknown_origin() -> None:
    response = make_client().get("/health", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in response.headers
