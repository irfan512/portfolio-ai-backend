"""Compare the ONNX Runtime and PyTorch embedding backends for all-MiniLM-L6-v2.

Run inside a Linux container that has torch + sentence-transformers + onnxruntime:

    python scripts/benchmark_embeddings.py --model-dir /models/all-MiniLM-L6-v2 \
        --corpus corpus.json --out results.json

`corpus.json`: {"chunks": [{"chunk_id", "section", "text"}], "torch_vectors": [[...]]},
i.e. the indexed chunks and the vectors currently stored in Qdrant.
Memory and startup are measured in fresh child processes, one per backend.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

QUERIES = [
    "Which Flutter projects have payment integrations?",
    "What AI technologies has the developer worked with?",
    "Has the developer built healthcare applications?",
    "Which projects use real-time communication?",
    "What experience does the developer have with Firebase?",
    "Tell me about the FSMS facility management platform",
    "Does Irfan know Python?",
    "Which apps are on the App Store?",
    "What services does Irfan offer?",
    "How does Irfan work with clients?",
    "Where did Irfan study?",
    "e-sports tournament app",
    "speech recognition for children",
    "Laravel PHP MySQL",
    "How can I contact him?",
]


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def peak_mb() -> float:
    """High-water RSS of *this program*. (ru_maxrss would include the parent's peak:
    Linux keeps it across fork+exec, so every child would report the parent's maximum.)"""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def build(backend: str, model_dir: str):
    from portfolio_ai.rag.embeddings.providers import MINILM

    if backend == "onnx":
        from portfolio_ai.rag.embeddings.onnx_provider import OnnxEmbeddingProvider

        return OnnxEmbeddingProvider(Path(model_dir), MINILM, threads=1)
    from portfolio_ai.rag.embeddings.providers import SentenceTransformerProvider

    return SentenceTransformerProvider(MINILM)


def child(backend: str, model_dir: str, corpus_path: str) -> None:
    """Fresh process: startup, memory, and 8-thread concurrency for one backend."""
    base = rss_mb()
    t0 = time.perf_counter()
    provider = build(backend, model_dir)
    provider.embed_query("warm up")
    startup = time.perf_counter() - t0
    idle = rss_mb()

    texts = [c["text"] for c in json.loads(Path(corpus_path).read_text())["chunks"]]
    t1 = time.perf_counter()
    provider.embed_documents(texts)
    corpus_s = time.perf_counter() - t1

    latencies: list[float] = []

    def one(q: str) -> None:
        s = time.perf_counter()
        provider.embed_query(q)
        latencies.append((time.perf_counter() - s) * 1000)

    work = QUERIES * 14  # 210 queries
    t2 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, work))
    wall = time.perf_counter() - t2
    lat = sorted(latencies)
    print(
        json.dumps(
            {
                "backend": backend,
                "torch_imported": "torch" in sys.modules,
                "python_baseline_rss_mb": round(base, 1),
                "startup_s": round(startup, 2),
                "idle_rss_mb": round(idle, 1),
                "embed_38_chunks_s": round(corpus_s, 3),
                "concurrent_queries": len(work),
                "threads": 8,
                "throughput_qps": round(len(work) / wall, 1),
                "latency_p50_ms": round(statistics.median(lat), 1),
                "latency_p95_ms": round(lat[int(0.95 * len(lat)) - 1], 1),
                "rss_after_load_test_mb": round(rss_mb(), 1),
                "peak_rss_mb": round(peak_mb(), 1),
            }
        )
    )


def topk(q: np.ndarray, m: np.ndarray, k: int = 5) -> list[int]:
    return np.argsort(-(m @ q), kind="stable")[:k].tolist()


def parity(model_dir: str, corpus_path: str) -> dict:
    from portfolio_ai.rag.embeddings.onnx_provider import OnnxEmbeddingProvider
    from portfolio_ai.rag.embeddings.providers import MINILM, SentenceTransformerProvider

    corpus = json.loads(Path(corpus_path).read_text())
    texts = [c["text"] for c in corpus["chunks"]]
    stored = np.asarray(corpus["torch_vectors"], dtype=np.float32)
    onnx = OnnxEmbeddingProvider(Path(model_dir), MINILM)
    torch = SentenceTransformerProvider(MINILM)

    # 1. Tokenizer parity (incl. truncation beyond 256 tokens)
    onnx._load()
    hf = torch._load().tokenizer
    probe = [*texts, *QUERIES, " ".join(["tournament"] * 400), "naïve café – 5+ years’ ☕"]
    mismatched = [
        t[:40]
        for t in probe
        if onnx._tokenizer.encode(t).ids
        != hf(t, truncation=True, max_length=onnx.max_seq_length)["input_ids"]
    ]

    # 2. Vectors
    d_onnx, d_torch = onnx.embed_documents(texts), torch.embed_documents(texts)
    q_onnx, q_torch = onnx.embed_documents(QUERIES), torch.embed_documents(QUERIES)
    cos_docs = np.sum(d_onnx * d_torch, axis=1)
    cos_stored = np.sum(d_onnx * stored, axis=1)
    cos_queries = np.sum(q_onnx * q_torch, axis=1)

    # 3. Retrieval: same-backend and mixed (ONNX queries against the stored PyTorch index)
    rankings = []
    for i, query in enumerate(QUERIES):
        ref = topk(q_torch[i], stored)
        all_onnx = topk(q_onnx[i], d_onnx)
        mixed = topk(q_onnx[i], stored)
        score_delta = float(np.max(np.abs((stored @ q_torch[i]) - (stored @ q_onnx[i]))))
        rankings.append(
            {
                "query": query,
                "top5_identical_all_onnx": ref == all_onnx,
                "top5_identical_mixed": ref == mixed,
                "overlap_at_5_mixed": len(set(ref) & set(mixed)),
                "max_score_delta_mixed": round(score_delta, 6),
                "top1": corpus["chunks"][ref[0]]["section"],
            }
        )
    return {
        "texts_tokenized": len(probe),
        "tokenizer_mismatches": mismatched,
        "dimension": {"onnx": int(d_onnx.shape[1]), "torch": int(d_torch.shape[1])},
        "max_norm_error": {
            "onnx": float(np.max(np.abs(np.linalg.norm(d_onnx, axis=1) - 1))),
            "torch": float(np.max(np.abs(np.linalg.norm(d_torch, axis=1) - 1))),
        },
        "cosine_onnx_vs_torch_docs": {"min": float(cos_docs.min()), "mean": float(cos_docs.mean())},
        "cosine_onnx_vs_stored_qdrant": {
            "min": float(cos_stored.min()),
            "mean": float(cos_stored.mean()),
        },
        "cosine_onnx_vs_torch_queries": {
            "min": float(cos_queries.min()),
            "mean": float(cos_queries.mean()),
        },
        "max_abs_component_diff": float(np.max(np.abs(d_onnx - d_torch))),
        "rankings": rankings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--child", choices=["onnx", "torch"])
    parser.add_argument("--processes-only", action="store_true", help="skip the parity section")
    args = parser.parse_args()
    if args.child:
        child(args.child, args.model_dir, args.corpus)
        return

    results = {
        "parity": None if args.processes_only else parity(args.model_dir, args.corpus),
        "processes": [],
    }
    for backend in ("onnx", "torch"):
        out = subprocess.run(
            [sys.executable, __file__, "--child", backend, "--model-dir", args.model_dir,
             "--corpus", args.corpus, "--out", args.out],
            capture_output=True, text=True, check=True, env=os.environ | {"OMP_NUM_THREADS": "1"},
        )  # fmt: skip
        results["processes"].append(json.loads(out.stdout.strip().splitlines()[-1]))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
