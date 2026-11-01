"""Local embedding service (development on Intel Macs, where onnxruntime has no wheel).

Runs inside the production image with the same ONNX provider the API uses, so
locally ingested vectors are produced exactly like production queries:

    docker compose up -d embedder   # uvicorn portfolio_ai.embedding_service:app --port 8081

Contract (consumed by LocalHttpEmbeddingProvider):
    GET  /health -> {"status", "model", "dimension"}
    POST /embed  {"texts": [...]} -> {"model", "dimension", "embeddings"}
"""

from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints

from portfolio_ai.core.config import get_settings
from portfolio_ai.core.logging import configure_logging
from portfolio_ai.wiring import build_embedder

MAX_TEXTS = 256
MAX_CHARS = 8_000

settings = get_settings()
configure_logging(settings.log_level, settings.log_format, settings.secret_values())
if settings.embedding_backend == "local_http":
    raise RuntimeError("The embedding service cannot itself use the local_http backend")
embedder = build_embedder(settings)
embedder.embed_query("warm up")

app = FastAPI(title="PortfolioAI embedder", docs_url=None, redoc_url=None, openapi_url=None)


class EmbedRequest(BaseModel):
    texts: list[Annotated[str, StringConstraints(max_length=MAX_CHARS)]] = Field(
        min_length=1, max_length=MAX_TEXTS
    )


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "model": embedder.spec.model_id, "dimension": embedder.spec.dimension}


@app.post("/embed")
def embed(request: EmbedRequest) -> EmbedResponse:
    vectors = embedder.embed_documents(request.texts)
    return EmbedResponse(
        model=embedder.spec.model_id, dimension=embedder.spec.dimension, embeddings=vectors.tolist()
    )
