# Integrating the chat widget into the React portfolio

The portfolio (`irfan512/portfolio`) is a Create React App site on GitHub Pages.
This document is the contract for the widget. **Nothing in the portfolio repo has
been changed yet.**

## 1. Backend URL from the environment

CRA inlines `REACT_APP_*` variables at build time.

```bash
# portfolio/.env.production  (the URL is public anyway; never put keys here)
REACT_APP_PORTFOLIO_AI_URL=https://<your-service>.onrender.com
# portfolio/.env.development
REACT_APP_PORTFOLIO_AI_URL=http://127.0.0.1:8000
```

```js
const API_URL = process.env.REACT_APP_PORTFOLIO_AI_URL;
```

The frontend holds **no secrets**. Gemini and Qdrant keys live only on Render.
Anything in a CRA bundle is public.

## 2. Calling the API

```js
export async function askPortfolioAI(message, sessionId, { signal } = {}) {
  const res = await fetch(`${API_URL}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId ?? null }),
    signal, // AbortController: cancel when the widget closes
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(body?.error?.message ?? "The assistant is unavailable.");
    err.code = body?.error?.code ?? "network_error";
    err.retryAfter = Number(res.headers.get("Retry-After")) || null;
    throw err;
  }
  return body; // { answer, citations, session_id, insufficient_evidence }
}
```

Keep `session_id` in `sessionStorage` (it is gone when the tab closes), not `localStorage`.

## 3. Rendering rules (security)

- Render `answer` as **text** (`{answer}` in JSX), never with `dangerouslySetInnerHTML`.
- Build links only from `citations[].url`. Show `section` as the label.
  Same-site links can scroll in place (`#work`, `#experience`). For anything
  else use `rel="noopener noreferrer"`.
- When `insufficient_evidence` is true, show the answer as is, optionally with a
  link to `#contact`.

## 4. States the widget needs

| State | UI |
|---|---|
| Idle | Input with a visible label, and 2–3 example questions. |
| Sending | Disable submit, show "Thinking…". The first request after the service sleeps can take ~60 s on free hosting. |
| Answer | Text, then a "Sources" list from `citations`. |
| `rate_limited` | "Please wait a moment" (use `retryAfter`). |
| `timeout` / `busy` / `service_unavailable` / network | `error.message` plus a retry button. |

Accessibility: announce new answers with `aria-live="polite"`, trap focus while a
dialog-style widget is open and return focus when it closes, and make the widget
usable by keyboard.

## 5. Privacy notice (show near the input)

> Answers come from the content of this portfolio. Questions are processed by
> Google Gemini. Please don't share personal or confidential information.

Chat history is kept in server memory for 30 minutes and is not stored.

## 6. Local end-to-end test

```bash
# backend
docker compose up -d && portfolio-ai-ingest && uvicorn portfolio_ai.main:app --port 8000
# portfolio: REACT_APP_PORTFOLIO_AI_URL=http://127.0.0.1:8000 npm start   (origin http://localhost:3000 is allowed in dev)
```
