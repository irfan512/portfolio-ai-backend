"""FastAPI entry point. Run with: uvicorn portfolio_ai.main:app --reload"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from portfolio_ai import __version__
from portfolio_ai.api.errors import install_error_handlers
from portfolio_ai.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from portfolio_ai.api.routes import chat, health
from portfolio_ai.container import Services, build_services
from portfolio_ai.core.config import Settings, get_settings
from portfolio_ai.core.logging import configure_logging


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """App factory: tests inject settings and fake services; production builds real ones."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format, settings.secret_values())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Model loading and Qdrant checks block, so run them off the event loop.
        app.state.services = services or await run_in_threadpool(build_services, settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        # No interactive docs or schema in production: smaller public surface.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    if services is not None:
        app.state.services = services  # available even without running the lifespan

    install_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestContextMiddleware)  # outermost: sees every response
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
