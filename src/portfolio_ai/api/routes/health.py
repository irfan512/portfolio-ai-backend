from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from portfolio_ai.api.deps import get_services
from portfolio_ai.api.schemas.health import HealthResponse, ReadyResponse
from portfolio_ai.container import Services

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up and serving requests. Checks no dependencies."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
def ready(services: Annotated[Services, Depends(get_services)]) -> JSONResponse:
    """Readiness: model loaded, collection reachable and non-empty, LLM configured.

    Reports booleans only; never hostnames, collection names or error text.
    """
    checks = services.readiness()
    ok = all(checks.values())
    body = ReadyResponse(status="ready" if ok else "not_ready", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)
