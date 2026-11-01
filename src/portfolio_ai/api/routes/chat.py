import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from portfolio_ai.api.deps import get_services
from portfolio_ai.api.errors import ApiError
from portfolio_ai.api.rate_limit import client_key
from portfolio_ai.api.schemas.chat import ChatRequest, ChatResponse, CitationOut
from portfolio_ai.container import Services
from portfolio_ai.rag.errors import LLMRateLimitError, LLMTimeoutError, RagError
from portfolio_ai.rag.service import Turn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

UNAVAILABLE = "The assistant is temporarily unavailable. Please try again later."


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={422: {}, 429: {}, 503: {}, 504: {}},
)
def chat(
    body: ChatRequest, request: Request, services: Annotated[Services, Depends(get_services)]
) -> ChatResponse:
    # A plain `def`: FastAPI runs it in a worker thread, so the blocking model
    # and HTTP calls don't stall the event loop.
    settings = services.settings
    decision = services.rate_limiter.check(client_key(request, settings.trusted_proxy_hops))
    if not decision.allowed:
        logger.info("rate_limited", extra={"scope": decision.scope})
        raise ApiError(
            429,
            "rate_limited",
            "Too many questions in a short time. Please try again later.",
            headers={"Retry-After": str(decision.retry_after_s)},
        )
    if services.answer_service is None:
        raise ApiError(503, "service_unavailable", UNAVAILABLE)

    session_id, history = services.sessions.get(body.session_id)
    try:
        result = services.answer_service.answer(body.message, history)
    except LLMTimeoutError:
        raise ApiError(
            504, "timeout", "The assistant took too long to answer. Please try again."
        ) from None
    except LLMRateLimitError:
        raise ApiError(
            503, "busy", "The assistant is busy right now. Please try again shortly."
        ) from None
    except RagError as exc:
        logger.error("chat_failed", extra={"error_type": type(exc).__name__})
        raise ApiError(503, "service_unavailable", UNAVAILABLE) from None

    services.sessions.append(session_id, Turn(question=body.message, answer=result.answer))
    return ChatResponse(
        answer=result.answer,
        citations=[CitationOut(**c.model_dump()) for c in result.citations],
        session_id=session_id,
        insufficient_evidence=result.insufficient_evidence,
    )
