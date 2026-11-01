"""Real ONNX model vs real PyTorch model (runs where both are installed: Linux containers).

PORTFOLIO_AI_EMBEDDING_MODEL_DIR=/models/all-MiniLM-L6-v2 pytest -m onnx
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.onnx

MODEL_DIR = os.getenv("PORTFOLIO_AI_EMBEDDING_MODEL_DIR")
TEXTS = [
    "Telemedicine app that connects patients with doctors for online consultations.",
    "Integrated Agora video calls and Flutterwave payments",
    "Experience > Capabilities\nMobile: Flutter, Dart, Kotlin, Java, SwiftUI",
    "x",
    " ".join(["tournament"] * 400),  # longer than 256 tokens: exercises truncation
]


@pytest.fixture(scope="module")
def providers():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("sentence_transformers")
    if not MODEL_DIR or not Path(MODEL_DIR, "onnx", "model.onnx").exists():
        pytest.skip("PORTFOLIO_AI_EMBEDDING_MODEL_DIR with onnx/model.onnx not available")
    from portfolio_ai.rag.embeddings.onnx_provider import OnnxEmbeddingProvider
    from portfolio_ai.rag.embeddings.providers import MINILM, SentenceTransformerProvider

    return OnnxEmbeddingProvider(Path(MODEL_DIR), MINILM), SentenceTransformerProvider(MINILM)


def test_tokenization_matches_sentence_transformers(providers) -> None:
    onnx, torch = providers
    onnx._load()
    hf = torch._load().tokenizer
    for text in TEXTS:
        ours = onnx._tokenizer.encode(text).ids
        theirs = hf(text, truncation=True, max_length=onnx.max_seq_length)["input_ids"]
        assert ours == theirs


def test_vectors_match_pytorch(providers) -> None:
    onnx, torch = providers
    a, b = onnx.embed_documents(TEXTS), torch.embed_documents(TEXTS)
    assert a.shape == b.shape == (len(TEXTS), 384)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    cosines = np.sum(a * b, axis=1)
    assert cosines.min() > 0.9999, cosines
