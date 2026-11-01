# syntax=docker/dockerfile:1
# Production image for Render: FastAPI + ONNX Runtime embeddings (no PyTorch).
#
#   docker build -t portfolio-ai-api .
#
# - all-MiniLM-L6-v2 (ONNX export) is downloaded at BUILD time at a pinned
#   revision and verified against models/all-MiniLM-L6-v2.sha256.
# - Only pyproject.toml, README.md, src/ and models/ enter the build (.dockerignore).
# - Non-root user; listens on $PORT (Render sets it), default 8000.

ARG PYTHON_VERSION=3.12

# Model files: their own stage, so code changes don't re-download 90 MB.
FROM python:${PYTHON_VERSION}-slim AS model
ARG MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
COPY models/fetch_model.py models/all-MiniLM-L6-v2.sha256 /fetch/
RUN python /fetch/fetch_model.py "${MODEL_REVISION}" /fetch/all-MiniLM-L6-v2.sha256 /opt/models/all-MiniLM-L6-v2 \
    && chmod -R a+rX /opt/models


FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
# Dependencies first, from pyproject.toml only (a stub package satisfies the build),
# so a code change reuses this layer instead of re-downloading every wheel.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/portfolio_ai && touch src/portfolio_ai/__init__.py \
    && pip install ".[embeddings-onnx]" \
    && pip uninstall -y portfolio-ai
# Then the application itself, without touching dependencies.
COPY src ./src
RUN pip install --no-deps .


FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    ORT_DISABLE_TELEMETRY=1 \
    PORTFOLIO_AI_ENVIRONMENT=production \
    PORTFOLIO_AI_EMBEDDING_BACKEND=onnx \
    PORTFOLIO_AI_EMBEDDING_MODEL_DIR=/opt/models/all-MiniLM-L6-v2 \
    PORT=8000

COPY --from=build /opt/venv /opt/venv
COPY --from=model /opt/models /opt/models
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app
WORKDIR /tmp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"

# One worker keeps memory predictable on 512 MB instances; requests run in its threadpool.
CMD ["sh", "-c", "exec uvicorn portfolio_ai.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-server-header --timeout-keep-alive 5"]
