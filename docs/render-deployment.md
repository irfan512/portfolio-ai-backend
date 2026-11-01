# Deploying PortfolioAI to Render

Architecture: GitHub Pages (React portfolio) → Render web service (this repo,
Docker) → Qdrant Cloud (vectors) and Gemini API (answers). The embedding model
runs inside the Render container.

Nothing here happens automatically. Every step below is run by you.

---

## 0. Resource footprint (production image, measured 2026-11-01)

The production image uses **ONNX Runtime** (no PyTorch). Parity with the
PyTorch model is documented in [ADR 0003](decisions/0003-onnx-runtime-embeddings.md):
min cosine 0.99999988, top-5 rankings 15/15 identical.

Measured locally with `--memory 512m --memory-swap 512m` (Render's small-instance limit):

| Metric | ONNX image (current) | Previous PyTorch image |
|---|---|---|
| Image size, compressed / on disk | **223 MB / 765 MB** | 556 MB / 2.36 GB |
| Start → `/health` | **2.8 s** | 8.7 s |
| Memory, idle (cgroup) | **232 MiB** | 450 MiB |
| Memory, peak during 400 chat requests at 16 concurrent (cgroup `memory.peak`) | **249 MiB** | n/a (88% used when idle) |
| Requests OK / out-of-memory kills | 400/400 / 0 | n/a |
| Server overhead per chat request (stub LLM 300 ms) | ~28 ms at p50 | n/a |
| Embedding throughput, 8 threads | 962 queries/s, p95 10 ms | 41 queries/s, p95 218 ms |

Reproduce: `docker compose --profile prod up -d api`, `scripts/load_test_api.py`,
`scripts/benchmark_embeddings.py`.

**Instance choice:** a 512 MB instance is enough, with roughly 50% headroom.
- **Starter**: always on. `render.yaml` uses this.
- **Free**: same memory, but it spins down when idle, so the first visitor
  after a quiet period waits through a cold start.

Check current plans and prices at https://render.com/pricing.

---

## 1. Review the files before committing

```bash
cd portfolio-ai-backend
git status --short                                # what Git sees
git status --ignored --short | grep '^!!'         # must include .env, data/raw/, data/processed/, .venv/
git ls-files -z --others --exclude-standard | xargs -0 grep -nIE \
  'AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{20,}\.|cloud\.qdrant\.io|@gmail\.com' \
  || echo "no secrets found"
```

Expected: the only `cloud.qdrant.io` hits are the fake `abc…` hosts in tests,
and there are no `@gmail.com` hits. **Never** `git add -f` anything ignored.

Optional, recommended: install [gitleaks](https://github.com/gitleaks/gitleaks)
and run `gitleaks detect --no-git -v` before the first commit. CI runs it on
every push too.

## 2. Push to GitHub

Create an **empty** repository named `portfolio-ai-backend` on GitHub (no README,
no .gitignore). Public or private both work with Render.

```bash
git add .
git status --short                  # read the list once more
git commit -m "PortfolioAI backend: ingestion, semantic search, grounded chat API"
git branch -M main
git remote add origin git@github.com:irfan512/portfolio-ai-backend.git
git push -u origin main
```

Then in GitHub → Settings → Code security, enable **Secret scanning** and
**Push protection**. Check that the **CI** workflow passes under Actions.

## 3. Prepare Qdrant Cloud (one-time ingestion)

The hosted collection must have the same model, dimension and collection
version as local. Both are derived from the same settings, so the name is
`portfolio_chunks__all-minilm-l6-v2__d384__v1` in both places.

1. In the Qdrant Cloud console, create two **database API keys** for the cluster:
   - a **read-write** key, kept only on your Mac and used for ingestion;
   - a **read-only** key, if your plan offers it, for Render. The API never writes.
2. Keep cloud settings in the git-ignored `.env.cloud` (permissions 600), and
   your normal `.env` on local Qdrant (`http://127.0.0.1:6333`, empty key):

   ```bash
   # .env.cloud
   PORTFOLIO_AI_QDRANT_URL="https://<cluster-id>.<region>.cloud.qdrant.io"
   PORTFOLIO_AI_QDRANT_API_KEY="<read-write key>"
   PORTFOLIO_AI_EMBEDDING_BACKEND="local_http"   # the ONNX embedder container
   ```
3. Ingest. The `embedder` container runs the production image's ONNX code, so the
   uploaded vectors are made exactly like production queries:

   ```bash
   docker compose up -d embedder
   portfolio-ai-ingest --env-file .env.cloud --dry-run          # prints target, writes nothing
   portfolio-ai-ingest --env-file .env.cloud --confirm-remote   # creates collection, indexes 38 chunks
   portfolio-ai-ingest --env-file .env.cloud --confirm-remote   # again: all "unchanged"
   portfolio-ai-search --env-file .env.cloud --info             # points: 38
   portfolio-ai-search --env-file .env.cloud "Which projects use Firebase?"
   ```

   Without `--confirm-remote` the CLI refuses to write to a non-local Qdrant.

What goes to Qdrant Cloud: the public portfolio text, chunked, with email and
phone redacted, plus its vectors. Nothing else.

Re-run the same command whenever the portfolio content changes.

## 4. Gemini API key

Create a key in Google AI Studio. Then:

- restrict it to the **Generative Language API**;
- set a budget alert or quota in Google Cloud.

Pick a currently supported model id for `GEMINI_MODEL`, then check it for free
with `python scripts/check_gemini.py`. On 2026-11-01 that check showed
`gemini-2.5-flash` returns 404 for new users, while `gemini-3.8-flash` supports
`generateContent`. Thinking tokens count toward the 800-token output limit; if
answers come back cut off, set `PORTFOLIO_AI_GEMINI_THINKING_LEVEL=low`.
Gemini 3.x deprecates `temperature`, so it is not sent unless
`PORTFOLIO_AI_GEMINI_TEMPERATURE` is set. `PORTFOLIO_AI_GEMINI_TIMEOUT_S` (default 60)
is also Google's server-side deadline: 20 s produced HTTP 504s on gemini-3.8-flash.

Note: on Gemini's free tier, Google may use prompts and responses to improve its
products. Prompts contain visitor questions and public portfolio text. The
widget's privacy notice (see docs/frontend-integration.md) tells visitors not to
share personal information.

## 5. Create the Render service

Dashboard → **New → Web Service** → connect GitHub → choose `portfolio-ai-backend`.

| Setting | Value |
|---|---|
| Language / Runtime | **Docker** (uses `./Dockerfile`) |
| Branch | `main` |
| Region | **Frankfurt**, same region as the Qdrant cluster (eu-central-1) |
| Instance type | **Starter** (512 MB). Check current prices on render.com/pricing |
| Health check path | `/ready` |
| Auto-deploy | Off at first; deploy manually |

Or use **New → Blueprint** with `render.yaml`, which sets the same values.

### Environment variables (Render → Environment)

| Key | Value | Secret? |
|---|---|---|
| `GEMINI_API_KEY` | your key | **yes** |
| `GEMINI_MODEL` | model id | no |
| `PORTFOLIO_AI_QDRANT_URL` | `https://<cluster>.cloud.qdrant.io` | treat as private |
| `PORTFOLIO_AI_QDRANT_API_KEY` | **read-only** key | **yes** |
| `PORTFOLIO_AI_CORS_ORIGINS` | `["https://irfan512.github.io"]` | no |
| `PORTFOLIO_AI_TRUSTED_PROXY_HOPS` | `1` | no |
| `PORTFOLIO_AI_CHAT_GLOBAL_DAILY_LIMIT` | `500` (or lower) | no |
| `PORTFOLIO_AI_GEMINI_THINKING_LEVEL` | `low` (set in `render.yaml`; see section 4) | no |

Already set in the image: `PORTFOLIO_AI_ENVIRONMENT=production`,
`PORTFOLIO_AI_EMBEDDING_BACKEND=onnx`, `PORTFOLIO_AI_EMBEDDING_MODEL_DIR`,
`ORT_DISABLE_TELEMETRY=1`.
Render injects `PORT`. Do not put secrets in `render.yaml`, the Dockerfile or
the frontend.

## 6. Deploy

Click **Manual Deploy → Deploy latest commit**. The first build takes a few
minutes (dependencies, then the 90 MB model with checksum verification). The
deploy goes live only when `/ready` returns 200.

Do **not** set `PORTFOLIO_AI_EMBEDDING_BACKEND` to anything except `onnx` on
Render: the image contains no PyTorch or sentence-transformers.

If the deploy fails health checks, open **Logs** and look for these events:
- `service_startup_incomplete`: a config or dependency problem;
- `readiness_vector_store_failed`: Qdrant URL, key or collection missing;
- `readiness_embeddings_failed`: the model did not load (memory?).

## 7. Verify

```bash
API=https://<your-service>.onrender.com
curl -s $API/health      # {"status":"ok"}
curl -s $API/ready       # {"status":"ready","checks":{"embeddings":true,"vector_store":true,"llm_configured":true}}
curl -s -o /dev/null -w "%{http_code}\n" $API/docs    # 404 (docs disabled in production)
curl -s -D - -o /dev/null -X OPTIONS $API/api/v1/chat \
  -H "Origin: https://evil.example" -H "Access-Control-Request-Method: POST" | grep -i access-control || echo "no CORS grant (good)"
```

## 8. A real RAG question (one paid Gemini call)

```bash
curl -s -X POST $API/api/v1/chat \
  -H "Content-Type: application/json" -H "Origin: https://irfan512.github.io" \
  -d '{"message":"Has Irfan built a healthcare app?"}' | python3 -m json.tool
```

Expect an answer about goCare, a citation to `…/portfolio/#work`, and
`"insufficient_evidence": false`. Then ask a follow-up with the returned
`session_id`, and an unanswerable question ("What is his hourly rate?"), which
should give `insufficient_evidence: true`.

Optional: check that rate limiting sees real client IPs. From two different
networks, each gets its own 10/minute budget. If everyone shares one budget,
adjust `PORTFOLIO_AI_TRUSTED_PROXY_HOPS`.

---

## Security and privacy checklist

- [ ] `.env`, `.env.cloud`, `data/raw/` (raw snapshots with unredacted contact details) and `data/processed/` are git-ignored and absent from the image (`.dockerignore` is an allowlist).
- [ ] Secrets exist only in Render's environment and your local ignored files.
- [ ] Render uses a read-only Qdrant key; the read-write key never leaves your Mac.
- [ ] Gemini key restricted to the Generative Language API, with a budget alert.
- [ ] `PORTFOLIO_AI_CORS_ORIGINS` lists only `https://irfan512.github.io`.
- [ ] Logs contain no IPs, message text, API keys (masked) or stack traces in responses.
- [ ] `/docs` and `/openapi.json` return 404 in production.
- [ ] GitHub secret scanning and push protection enabled.
