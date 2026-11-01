# 0001: Run the embedding model in a local Docker service on Intel Macs

Status: superseded by [0003](0003-onnx-runtime-embeddings.md). The `local_http` design stays; the runtime is now ONNX.

## Context
The development machine is an Intel Mac (x86_64). PyTorch, which
sentence-transformers needs, has no Intel-Mac build for current Python. The last
Intel-Mac build is torch 2.2.2 on Python 3.12, from early 2024. ONNX Runtime has
no Intel-Mac build for Python 3.14 either.

## Decision
`EmbeddingProvider` has two real adapters:
- `LocalHttpEmbeddingProvider` calls `embedder/` (sentence-transformers 6.2.0,
  CPU torch 2.15.0) running in Docker, published on 127.0.0.1 only. The model is
  downloaded at image build time, and the container runs with HF offline mode on.
- `SentenceTransformerProvider` runs the same model in-process on Linux or Apple
  Silicon (`pip install -e ".[embeddings-local]"`).

Select with `PORTFOLIO_AI_EMBEDDING_BACKEND`.

## Consequences
- The venv stays on current Python 3.14. No frozen 2024 stack.
- The HTTP client refuses non-loopback hosts, and every response must name the
  expected model and dimension.
- Same model weights and normalization on both adapters, so their vectors are
  interchangeable within one collection.
