# PortfolioAI API

Base URL: configured per environment, for example `https://<your-service>.onrender.com`.
All bodies are JSON (UTF-8). Interactive docs (`/docs`) exist only outside production.

## `POST /api/v1/chat`

Ask a question about the portfolio.

### Request

```json
{ "message": "Which projects use Firebase?", "session_id": null }
```

| Field | Type | Rules |
|---|---|---|
| `message` | string | required, 1–1000 characters after trimming |
| `session_id` | string or null | optional; the `session_id` from a previous response. Pattern `^[A-Za-z0-9_-]{16,64}$` |

Unknown fields are rejected. Maximum body size: 8 KB.

### Response `200`

```json
{
  "answer": "The portfolio lists INGAGE GG and goCare as Flutter apps using Firebase ...",
  "citations": [
    { "title": "Selected work", "section": "INGAGE GG", "url": "https://irfan512.github.io/portfolio/#work" }
  ],
  "session_id": "Vb3kK2...opaque",
  "insufficient_evidence": false
}
```

- `answer`: plain text. Render it as text, never as HTML.
- `citations`: sources the answer used. `url` is always a page of the approved
  portfolio (`https://irfan512.github.io/portfolio/#...`), never a URL written by the model.
- `insufficient_evidence: true`: the portfolio does not contain the answer. The
  `answer` says so (it does not guess), and `citations` may be empty.
- `session_id`: send it with the next question for follow-ups. Sessions expire
  after 30 minutes of inactivity. An unknown or expired id silently starts a new one.

### Errors

Every error has the same shape:

```json
{ "error": { "code": "rate_limited", "message": "Too many questions ...", "request_id": "3f9c..." } }
```

| HTTP | `code` | Meaning / what the widget should do |
|---|---|---|
| 422 | `invalid_request` | Bad input (`details`: `[{field, issue}]`). Fix the input. |
| 413 | `payload_too_large` | Body over 8 KB. |
| 429 | `rate_limited` | Too many requests. Honour the `Retry-After` header (seconds). |
| 503 | `busy` | The free-tier model quota, or the site-wide daily/per-minute budget, is used up. Retry later (`Retry-After`). |
| 503 | `service_unavailable` | A dependency is down or the service is starting. Retry later. |
| 504 | `timeout` | The model took too long. Offer "try again". |
| 500 | `internal_error` | Unexpected. Show a generic message. Quote `request_id` when reporting. |

`message` is always safe to show to visitors.

## `GET /health`

Liveness. `200 {"status":"ok"}` while the process runs.

## `GET /ready`

Readiness. Checks the embedding model, the Qdrant collection (exists and is not
empty), and that Gemini is configured. It never calls Gemini.

```json
{ "status": "ready", "checks": { "embeddings": true, "vector_store": true, "llm_configured": true } }
```

`503` with `"status": "not_ready"` if any check fails.

## Limits (defaults)

| Limit | Default | Setting |
|---|---|---|
| Requests per client per minute | 10 | `PORTFOLIO_AI_RATE_LIMIT_PER_MINUTE` |
| Requests per client per day | 100 | `PORTFOLIO_AI_RATE_LIMIT_PER_DAY` |
| Chat requests per day, all clients | 500 | `PORTFOLIO_AI_CHAT_GLOBAL_DAILY_LIMIT` |
| Message length | 1000 chars | |

## CORS

Only origins in `PORTFOLIO_AI_CORS_ORIGINS` may call the API from a browser
(production: `https://irfan512.github.io`). Methods `GET`/`POST`, header `Content-Type`.
No cookies or credentials are used.
