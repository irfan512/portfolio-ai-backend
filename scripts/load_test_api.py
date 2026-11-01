"""Concurrent load test of the real production stack with a stub LLM (no Gemini calls).

Runs uvicorn + FastAPI + ONNX embeddings + Qdrant retrieval inside the production
image, sends concurrent chat requests, and reports the container's memory from
cgroup v2 counters (`memory.peak` is the true high-water mark the 512 MB limit
applies to).

    docker run --rm --memory 512m --network portfolio-ai_default \
      -e PORTFOLIO_AI_QDRANT_URL=http://qdrant:6333 -e PORTFOLIO_AI_QDRANT_PRIVATE_HOSTS='["qdrant"]' \
      -v "$PWD/scripts:/scripts:ro" portfolio-ai-api:local python /scripts/load_test_api.py
"""

import json
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import uvicorn

from portfolio_ai.container import build_services
from portfolio_ai.core.config import Settings
from portfolio_ai.main import create_app
from portfolio_ai.rag.generation.base import GroundedAnswer
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever
from portfolio_ai.rag.service import AnswerService, AnswerSettings

QUESTIONS = [
    "Which Flutter projects have payment integrations?",
    "What AI technologies has the developer worked with?",
    "Has the developer built healthcare applications?",
    "Which projects use real-time communication?",
    "What experience does the developer have with Firebase?",
    "What services does Irfan offer?",
    "Where did Irfan study?",
    "Tell me about the FSMS platform and its AI layer",
]
REQUESTS = 400
CONCURRENCY = 16
CGROUP = Path("/sys/fs/cgroup")


class StubLLM:
    """Simulates Gemini latency; never makes a network call."""

    model_name = "stub"

    def generate_grounded(self, request):
        time.sleep(0.3)
        return GroundedAnswer(
            answer="stub answer", cited_sources=["S1"], insufficient_evidence=False
        )


def cgroup_mb(name: str) -> float | None:
    path = CGROUP / name
    return round(int(path.read_text().split()[0]) / 2**20, 1) if path.exists() else None


def oom_kills() -> int:
    path = CGROUP / "memory.events"
    if not path.exists():
        return -1
    events = dict(line.split() for line in path.read_text().splitlines())
    return int(events.get("oom_kill", 0))


def main() -> None:
    settings = Settings(
        rate_limit_per_minute=1_000_000,
        rate_limit_per_day=1_000_000,
        chat_global_daily_limit=1_000_000,
    )
    services = build_services(settings)
    assert services.embedder is not None and services.store is not None, "stack failed to start"
    services.answer_service = AnswerService(
        SemanticRetriever(services.embedder, services.store), StubLLM(), AnswerSettings()
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings, services), host="127.0.0.1", port=8001, log_level="warning"
        )
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    idle = cgroup_mb("memory.current")

    latencies: list[float] = []
    statuses: Counter[int] = Counter()

    def ask(i: int) -> None:
        started = time.perf_counter()
        response = client.post("/api/v1/chat", json={"message": QUESTIONS[i % len(QUESTIONS)]})
        latencies.append((time.perf_counter() - started) * 1000)
        statuses[response.status_code] += 1

    with httpx.Client(base_url="http://127.0.0.1:8001", timeout=60) as client:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(ask, range(REQUESTS)))
        wall = time.perf_counter() - started
    server.should_exit = True
    lat = sorted(latencies)
    print(
        json.dumps(
            {
                "requests": REQUESTS,
                "concurrency": CONCURRENCY,
                "statuses": dict(statuses),
                "throughput_rps": round(REQUESTS / wall, 1),
                "latency_p50_ms": round(statistics.median(lat)),
                "latency_p95_ms": round(lat[int(0.95 * len(lat)) - 1]),
                "cgroup_limit_mb": cgroup_mb("memory.max"),
                "cgroup_idle_mb": idle,
                "cgroup_after_mb": cgroup_mb("memory.current"),
                "cgroup_peak_mb": cgroup_mb("memory.peak"),
                "oom_kills": oom_kills(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
