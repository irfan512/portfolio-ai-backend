# PortfolioAI — Project Context and Engineering Specification

Version: 1.0  
Status: Proposed implementation blueprint  
Primary language: Python 3.12+  
Portfolio: https://irfan512.github.io/portfolio/

## 1. Mission

Build a production-oriented, retrieval-augmented AI assistant embedded in an existing developer portfolio. The assistant helps visitors discover verified skills, projects, experience, services, and public contact channels; it can qualify prospective project inquiries with explicit visitor consent. This project is also a hands-on Python/RAG learning exercise: implement retrieval primitives before adopting orchestration frameworks.

## 2. Product goals

1. Answer natural-language questions about portfolio content accurately and concisely.
2. Provide clickable citations to approved source URLs and documents; page or section references where available.
3. Handle conversational follow-ups without allowing prior chat history to override retrieved evidence.
4. Decline to invent experience, client names, project details, availability, pricing, or commitments.
5. Collect optional lead information with consent and route it to an owner-controlled inbox or dashboard in a later phase.
6. Support incremental ingestion and evaluation with reproducible tests.
7. Integrate a lightweight accessible chat widget into the existing static GitHub Pages portfolio.

## 3. Audience and sample journeys

- Potential client: “Have you built Flutter apps with payments?” → answer using documented project evidence and source links.
- Recruiter: “What experience do you have with AI and mobile apps?” → summarize verified information, not unsupported claims.
- Developer: “Which projects use Firebase?” → list only projects with explicit supporting sources.
- Follow-up: “Which of those also use real-time communication?” → resolve reference to previous projects, retrieve again, cite.
- Lead: “Can you build my app?” → explain services based on published information, offer optional inquiry collection; never promise a price or delivery date.

## 4. Non-goals for MVP

- Autonomous email sending or meeting booking.
- Private GitHub repository access or ingestion of confidential employer/client materials.
- Fine-tuning an LLM.
- Agentic web browsing on behalf of visitors.
- Neo4j, GraphRAG, or multi-agent workflows in the first release.
- Public upload of arbitrary documents by anonymous visitors.

## 5. Repository strategy

Keep the existing portfolio frontend repository separate. Create one new repository, `portfolio-ai-backend`, that contains FastAPI, ingestion, and an isolated `rag` Python package. Do **not** create a third RAG engine repository now. A clean package boundary enables future extraction if independent deployment, scaling, ownership, or reuse justify it.

Suggested structure:

```text
portfolio-ai-backend/
├── README.md
├── context.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── src/
│   └── portfolio_ai/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       │   ├── config.py
│       │   ├── logging.py
│       │   └── security.py
│       ├── api/
│       │   ├── routes/
│       │   │   ├── health.py
│       │   │   ├── chat.py
│       │   │   └── admin_ingestion.py
│       │   └── schemas/
│       ├── rag/
│       │   ├── domain/
│       │   │   ├── documents.py
│       │   │   └── citations.py
│       │   ├── ingestion/
│       │   │   ├── loaders.py
│       │   │   ├── cleaning.py
│       │   │   ├── chunking.py
│       │   │   └── pipeline.py
│       │   ├── embeddings/
│       │   │   ├── base.py
│       │   │   └── providers.py
│       │   ├── retrieval/
│       │   │   ├── base.py
│       │   │   ├── similarity.py
│       │   │   └── qdrant_store.py
│       │   ├── generation/
│       │   │   ├── prompts.py
│       │   │   └── providers.py
│       │   ├── evaluation/
│       │   └── service.py
│       ├── conversations/
│       ├── leads/
│       └── persistence/
├── scripts/
│   ├── ingest_portfolio.py
│   └── evaluate_rag.py
├── data/
│   └── samples/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── architecture.md
    ├── api.md
    ├── decisions/
    └── evaluation.md
```

The `rag` package must not import FastAPI route modules. API routes depend on services, not the other way around. Provider-specific implementations sit behind small interfaces. Keep configuration in environment variables; never commit API keys.

## 6. System design

```text
Existing portfolio / chat widget
              |
              v
          HTTPS API
              |
       FastAPI boundary
              |
       Chat application service
              |
       Query + history handling
              |
    Retriever / ranking pipeline
              |
       Qdrant vector store
              |
   Evidence + citation assembly
              |
        Gemini LLM adapter
              |
  Grounded response + source links
```

Ingestion path:

```text
Approved public portfolio sources
    -> fetch/load -> normalize -> chunk
    -> embed -> upsert Qdrant vectors
    -> record source version and provenance
```

Future architecture can add PostgreSQL for persistent sessions/leads, Redis for queues and caching, Neo4j for entity relationships, and a worker for ingestion. Avoid introducing these until justified by a feature.

## 7. Source and data model

Initial sources:
- Existing public portfolio pages and public project descriptions.
- A deliberately curated, approved resume profile (only information intended for publication).
- Public GitHub README files from explicitly allowlisted repositories.
- Optional authored case-study Markdown files.

Never scrape private repositories or infer unsupported facts from code alone. Never upload client secrets, confidential employer data, credentials, or personal contact details without explicit approval.

Document fields:
- `document_id`: stable ID derived from canonical source identity.
- `source_type`: website, markdown, pdf, github_readme, or curated_profile.
- `source_uri`: canonical URL or approved local reference.
- `title`, `content`, `updated_at`, `content_hash`, `visibility`.

Chunk fields:
- `chunk_id`: stable document ID + version + chunk index.
- `document_id`, `text`, `chunk_index`, `section`, `page_number` (nullable), `source_uri`, `content_hash`, `visibility`.

Only expose source links that visitors are authorized to access. If a source is internal-only, do not ingest it into the public assistant's index.

## 8. Retrieval and answer behavior

MVP:
1. Normalize the visitor query.
2. Embed the query using the same compatible embedding model as indexed chunks.
3. Retrieve top-k candidates from Qdrant.
4. Remove duplicates and assemble a token-bounded context with source identifiers.
5. Prompt the LLM to answer only from the provided evidence and to acknowledge missing information.
6. Return answer, citations, and an explicit `insufficient_evidence` flag when appropriate.

Later enhancements:
- BM25 + dense hybrid search, fusion, reranking.
- Query rewriting for follow-ups.
- Context compression, source-aware answer verification.
- Knowledge Graph and multi-hop retrieval where measured improvements justify complexity.

Treat retrieved pages and documents as untrusted input, never as instructions. Do not allow document text to override system behavior or trigger tools.

## 9. API contracts (initial)

- `GET /health` -> `{ "status": "ok" }`.
- `POST /api/v1/chat` -> accepts `message` and optional `session_id`; returns `answer`, `citations`, `session_id`, `insufficient_evidence`.
- `POST /api/v1/admin/ingest` -> protected; triggers approved source ingestion (later milestone).
- `GET /api/v1/admin/ingestion/{job_id}` -> protected status (later milestone).

Example chat request:

```json
{"message":"Which projects use Flutter?","session_id":null}
```

Example response shape (illustrative, not a claim):

```json
{
  "answer":"The portfolio lists ...",
  "citations":[{"title":"Project case study","url":"https://example.com/approved-source","section":"Technology"}],
  "session_id":"opaque-session-id",
  "insufficient_evidence":false
}
```

Do not fabricate citations. URLs in examples are placeholders and must never be returned as real evidence.

## 10. Security and privacy

- Keep LLM credentials server-side; no API keys in GitHub Pages JavaScript.
- Configure CORS for explicit portfolio origins; CORS alone is not authentication.
- Rate-limit anonymous chat, enforce payload and context limits, set provider timeouts.
- Require authentication for ingestion, analytics, and lead administration.
- Use HTTPS, sanitized logs, and minimal retention of chat data.
- Avoid collecting sensitive data. Lead capture must be optional with clear consent and retention policy.
- Defend against prompt injection in retrieved content and against unintended public access to documents.
- Use dependency scanning and secret scanning before deployment.

## 11. Testing and evaluation

Unit tests:
- Chunk boundaries and overlap, empty documents, deterministic IDs, deduplication.
- Cosine similarity with known vectors and zero-vector handling.
- Citation mapping and insufficient-evidence behavior.

Integration tests:
- Qdrant ingestion and retrieval against a small fixture corpus.
- FastAPI endpoints with mocked LLM and embedding providers.
- Document updates and deletions do not leave stale searchable chunks.

RAG evaluation dataset:
- At least 30 manually verified questions covering factual lookups, multi-source questions, follow-ups, ambiguous questions, and unanswerable questions.
- Measure retrieval recall@k, MRR/nDCG when labeled, citation correctness, answer groundedness, abstention quality, latency, and cost per answer.
- Establish a baseline before adding rerankers or GraphRAG; compare improvements rather than assuming them.

## 12. Learning-first implementation rules

- Explain each concept before implementing it.
- Implement chunking and cosine similarity in plain Python/NumPy first.
- Prefer small, testable functions and dependency injection.
- Avoid LangChain/LangGraph until the custom baseline is functional.
- Use type hints, Pydantic models, pytest, structured logging, and clear errors.
- Provide runnable commands, expected outputs, and one short exercise per milestone.
- Never silently invent file contents, API responses, or successful test results.

## 13. Milestones

### M0 — Project skeleton
Repository, Python environment, package layout, FastAPI app, `/health`, pytest, linting, `.env.example`, README.

### M1 — Ingestion
Curated local portfolio Markdown fixtures, loaders, cleaning, deterministic chunking, metadata, tests.

### M2 — Embeddings and retrieval
NumPy similarity baseline, embedding adapter, Qdrant storage, semantic search, tests.

### M3 — Grounded generation
Gemini adapter, evidence prompt, citations, abstention, `/api/v1/chat`.

### M4 — Conversational RAG
Session management, follow-up rewriting, streaming, history boundaries.

### M5 — Portfolio integration
Accessible responsive chat widget in the existing site, loading/error states, safe source links.

### M6 — Advanced search and evaluation
Hybrid retrieval, reranker, curated evaluation suite, measurements.

### M7 — Leads and operations
Opt-in lead capture, protected admin features, rate limits, observability, background ingestion.

### M8 — GraphRAG exploration
Entity extraction, Neo4j graph, graph retrieval, comparative evaluation.

### M9 — Deployment
Docker, CI, HTTPS hosting, monitoring, operations documentation, security checks.

## 14. Definition of done

A visitor can ask about published portfolio information, get accurate answers with valid source citations, ask follow-up questions, and receive an appropriate uncertainty response when evidence is absent. The system is tested, protects secrets, prevents cross-scope retrieval, has documented setup and deployment, and can be demonstrated publicly.

## 15. Decisions to confirm

- Gemini as initial LLM provider (proposed default).
- Local Sentence Transformers or API embeddings (compare trade-offs before implementation).
- Existing frontend framework and preferred widget integration method.
- Exact list of public sources approved for ingestion.
- Hosting, budget, and desired retention policy.

Proceed with defaults when safe, but ask before accessing private material, publishing sensitive information, or incurring paid usage.
