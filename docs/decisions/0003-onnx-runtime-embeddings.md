# 0003: ONNX Runtime for embeddings in production and local ingestion

Status: accepted (M3 deployment). Supersedes the runtime part of 0001.

## Context
With PyTorch, the production image idled at 450 MiB inside a 512 MB limit, too
tight for Render's 512 MB instances. The model's official ONNX export lives in
the same Hugging Face repository and revision (`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`).

## Decision
`OnnxEmbeddingProvider` (onnxruntime + tokenizers, no PyTorch) reproduces the
Sentence Transformers pipeline: Transformer (max 256 tokens) → mean pooling →
L2 normalize. It refuses models whose `modules.json` / pooling config differ.
The production image and the local `embedder` service both use it. Model files
are fetched at the pinned revision and verified against `models/all-MiniLM-L6-v2.sha256`.

## Evidence (scripts/benchmark_embeddings.py, 38 indexed chunks + 15 queries)
| Check | Result |
|---|---|
| Token ids vs sentence-transformers (55 texts incl. >256 tokens, unicode) | identical |
| Dimension / max abs(norm − 1) | 384 / 6e-8 (both) |
| Cosine ONNX vs PyTorch, documents | min 0.99999988 |
| Cosine ONNX vs vectors already stored in Qdrant | min 0.99999988 |
| Top-5 rankings, 15 queries (ONNX index, and ONNX queries vs stored PyTorch index) | 15/15 identical |
| Idle RSS after load | 213 MB vs 498 MB |
| Startup (load + first embedding) | 0.49 s vs 5.42 s |
| 210 queries, 8 threads | 962 q/s, p95 10 ms vs 41 q/s, p95 218 ms |

## Consequences
- ONNX and PyTorch vectors are interchangeable within float32 noise, so the
  collection name (`…__all-minilm-l6-v2__d384__v1`) is unchanged and existing
  PyTorch-built vectors stay valid. All new writes use ONNX.
- `SentenceTransformerProvider` remains available (`.[embeddings-local]`) but
  is no longer used by the images.
- `onnxruntime` has no Intel-Mac wheel for Python 3.14, so on the Mac embeddings
  go through the `embedder` container (`local_http`).
