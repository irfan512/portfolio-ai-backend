# PortfolioAI Backend

An AI assistant for [irfan512.github.io/portfolio](https://irfan512.github.io/portfolio/). It answers visitor questions using retrieval-augmented generation (RAG) and grounds every answer in the portfolio's own content. FastAPI serves the HTTP API, and the RAG engine is a separate internal package under `src/portfolio_ai/rag/`.

See [`context.md`](context.md) for the full specification and milestones.

**Status:** M3, grounded chat. `POST /api/v1/chat` answers questions from portfolio evidence only (Gemini), with citations, follow-ups and abstention. Production deployment: Render + Qdrant Cloud ([docs/render-deployment.md](docs/render-deployment.md)).

| Doc | |
|---|---|
| [docs/api.md](docs/api.md) | HTTP API contract |
| [docs/frontend-integration.md](docs/frontend-integration.md) | React widget integration |
| [docs/render-deployment.md](docs/render-deployment.md) | GitHub → Render → Qdrant Cloud, step by step |
| [docs/decisions/](docs/decisions/) | Architecture decisions |

## Requirements

- macOS / Linux, Python 3.12+
- Docker Desktop (Qdrant, and the embedding service on Intel Macs)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,ingest]"
python -m playwright install chromium   # headless browser, ~100 MB, one-time
cp .env.example .env
```

## Run

```bash
uvicorn portfolio_ai.main:app --reload
curl http://127.0.0.1:8000/health   # {"status":"ok"}
```

Interactive API docs are served at http://127.0.0.1:8000/docs.

## Start local services

```bash
docker compose up -d --build   # qdrant :6333 + embedder :8081 (ONNX), bound to 127.0.0.1 only
curl http://127.0.0.1:8081/health   # {"status":"ok","model":"sentence-transformers/all-MiniLM-L6-v2","dimension":384}
docker compose --profile prod up -d api   # optional: the production API on :8000 (512 MB limit)
```

Qdrant data lives in the `qdrant_storage` Docker volume and survives `docker compose down`.
The embedder container is the production image running `portfolio_ai.embedding_service`: the same
ONNX Runtime code as production, so local and hosted vectors are produced identically
([ADR 0003](docs/decisions/0003-onnx-runtime-embeddings.md)). On Linux or Apple Silicon you can run ONNX
in-process instead: `pip install -e ".[embeddings-onnx]"`, `python models/fetch_model.py <revision> models/all-MiniLM-L6-v2.sha256 data/models/all-MiniLM-L6-v2`,
then `PORTFOLIO_AI_EMBEDDING_BACKEND=onnx PORTFOLIO_AI_EMBEDDING_MODEL_DIR=data/models/all-MiniLM-L6-v2`.

## Ingest the portfolio

Only URLs listed in [`sources.toml`](sources.toml) are fetched.

```bash
portfolio-ai-ingest                 # render the live site, embed, index changes in Qdrant
portfolio-ai-ingest --offline       # re-use the saved snapshot (no network)
portfolio-ai-ingest --dry-run       # show what would change
portfolio-ai-ingest --show-chunks   # print the stored chunks
portfolio-ai-ingest --store jsonl   # M1 behaviour: local file, no embeddings
```

Raw snapshots go to `data/raw/snapshots/` (git-ignored). With `--store qdrant` (the default), the index *and* its sync state live in Qdrant.
Re-running with unchanged content writes nothing. Email addresses and phone numbers are redacted before indexing.

## Search

```bash
portfolio-ai-search "Which projects use Firebase?"
portfolio-ai-search "healthcare apps" --top-k 3 --threshold 0.3
portfolio-ai-search "AI experience" --compare   # Qdrant vs NumPy exact ranking
portfolio-ai-search --info                      # collection, model, point count
```

## Chat API (local)

Add `GEMINI_API_KEY` and `GEMINI_MODEL` to `.env` (requests may cost money), then:

```bash
uvicorn portfolio_ai.main:app --reload --port 8000
curl -s localhost:8000/ready
curl -s -X POST localhost:8000/api/v1/chat -H 'content-type: application/json' \
  -d '{"message":"Which projects use Firebase?"}'
```

Without a Gemini key the API still starts: `/ready` reports `llm_configured: false` and chat returns 503.

## Quality checks

```bash
pytest            # tests (offline)
RUN_LIVE_TESTS=1 pytest -m live   # optional check against the real site
REQUIRE_SERVICES=1 pytest -m "qdrant or embedder"   # fail (not skip) if Docker services are down
python scripts/check_gemini.py        # free: verifies key + model, estimates live-test tokens
RUN_GEMINI_TESTS=1 pytest -m gemini   # real Gemini calls: COSTS MONEY
ruff check .      # lint
ruff format .     # format
```

## Layout

```text
src/portfolio_ai/
  main.py          FastAPI app factory (create_app) + ASGI `app`
  core/            settings, logging, security
  api/routes/      HTTP endpoints (thin; call services)
  api/schemas/     Pydantic request/response models
  rag/             RAG engine — must never import FastAPI or api/
    domain/        Document, Chunk, deterministic IDs
    ingestion/     sources -> fetchers -> html_parser -> cleaning -> chunking -> pipeline
    embeddings/    EmbeddingProvider: ONNX (production), local HTTP, sentence-transformers
    retrieval/     Qdrant store, NumPy cosine reference, SemanticRetriever
  wiring.py        builds embedder / Qdrant / Gemini from settings
  container.py     API service container + readiness checks
    generation/    LLMProvider, Gemini adapter, prompts
    service.py     AnswerService: retrieve → evidence → generate → verify citations
  conversations/   in-memory sessions (TTL, never persisted)
  api/             routes, schemas, errors, middleware, rate limiting
Dockerfile         production image (Render)
  search_cli.py    `portfolio-ai-search` command
  embedding_service.py  local embedding HTTP service (Intel Macs)
models/            pinned model checksums + fetch script (used by the Dockerfile)
scripts/           benchmark_embeddings.py, load_test_api.py, check_gemini.py
docs/decisions/    architecture decision records
  ingest_cli.py    `portfolio-ai-ingest` command (wires the pieces together)
  conversations/   session handling (M4)
  leads/           opt-in lead capture (M7)
  persistence/     storage adapters (later)
tests/unit/        fast, isolated tests
```

## Security

- Secrets live only in `.env`, which Git ignores. Never commit API keys.
- CORS is limited to the origins in `PORTFOLIO_AI_CORS_ORIGINS`.
- Portfolio text never leaves the machine for embedding: the HTTP embedder client only talks to loopback hosts.
- A remote Qdrant (e.g. Qdrant Cloud) must use https and an API key; Qdrant telemetry is disabled.
- Writing to a remote Qdrant needs `--confirm-remote`; the API itself only reads (use a read-only key).
- Public errors never include exception text; secrets are masked in logs; logs contain no IPs or message text.
- Rate limits per client (hashed IP) plus a global daily cap on chat requests (bounds Gemini cost).

## License

MIT. See [LICENSE](LICENSE).
