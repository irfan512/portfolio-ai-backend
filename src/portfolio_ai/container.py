"""The API's service container: built once at startup, shared by all requests."""

import logging
import threading
import time
from dataclasses import dataclass, field

from portfolio_ai.api.rate_limit import RateLimiter
from portfolio_ai.conversations.sessions import SessionStore
from portfolio_ai.core.config import Settings
from portfolio_ai.rag.embeddings.base import EmbeddingProvider
from portfolio_ai.rag.errors import RagError
from portfolio_ai.rag.generation.gemini import GeminiProvider
from portfolio_ai.rag.retrieval.qdrant_store import QdrantChunkStore
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever
from portfolio_ai.rag.service import AnswerService, AnswerSettings
from portfolio_ai.wiring import ConfigurationError, build_embedder, build_qdrant_store

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Everything the API needs, built once at startup."""

    settings: Settings
    sessions: SessionStore
    rate_limiter: RateLimiter
    embedder: EmbeddingProvider | None = None
    store: QdrantChunkStore | None = None
    answer_service: AnswerService | None = None
    _ready_cache: tuple[float, dict[str, bool]] | None = field(default=None, repr=False)
    _ready_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def readiness(self, max_age_s: float = 15.0) -> dict[str, bool]:
        """Dependency checks, cached briefly. Never calls Gemini (that would cost money)."""
        with self._ready_lock:
            if self._ready_cache and time.monotonic() - self._ready_cache[0] < max_age_s:
                return self._ready_cache[1]
            checks = {
                "embeddings": self._check_embeddings(),
                "vector_store": self._check_vector_store(),
                "llm_configured": self.answer_service is not None,
            }
            self._ready_cache = (time.monotonic(), checks)
            return checks

    def _check_embeddings(self) -> bool:
        if self.embedder is None:
            return False
        try:
            return self.embedder.embed_query("readiness probe").shape == (
                self.embedder.spec.dimension,
            )
        except RagError as exc:
            logger.warning("readiness_embeddings_failed", extra={"error": str(exc)})
            return False

    def _check_vector_store(self) -> bool:
        if self.store is None:
            return False
        try:
            self.store.verify_existing()
            return self.store.count() > 0
        except RagError as exc:
            logger.warning("readiness_vector_store_failed", extra={"error": str(exc)})
            return False


def build_llm(settings: Settings) -> GeminiProvider:
    api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    return GeminiProvider(
        api_key=api_key,
        model=settings.gemini_model or "",
        timeout_s=settings.gemini_timeout_s,
        max_retries=settings.gemini_max_retries,
        thinking_budget=settings.gemini_thinking_budget,
        thinking_level=settings.gemini_thinking_level,
    )


def build_services(settings: Settings) -> Services:
    """Build the API's dependencies. Partial failures leave the app up but not ready."""
    services = Services(
        settings=settings,
        sessions=SessionStore(
            ttl_s=settings.session_ttl_s,
            max_turns=settings.session_max_turns,
            max_sessions=settings.session_max_count,
        ),
        rate_limiter=RateLimiter(
            per_minute=settings.rate_limit_per_minute,
            per_day=settings.rate_limit_per_day,
            global_per_day=settings.chat_global_daily_limit,
        ),
    )
    try:
        services.embedder = build_embedder(settings)
        services.embedder.embed_query("warm up")  # load the model now, not on the first visitor
        services.store = build_qdrant_store(settings, services.embedder)
        retriever = SemanticRetriever(
            services.embedder, services.store, max_top_k=max(20, settings.retrieval_top_k)
        )
        answer_settings = AnswerSettings(
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
            evidence_token_budget=settings.evidence_token_budget,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
        )
        services.answer_service = AnswerService(retriever, build_llm(settings), answer_settings)
    except (RagError, ConfigurationError) as exc:
        logger.error("service_startup_incomplete", extra={"error": str(exc)})
    return services
