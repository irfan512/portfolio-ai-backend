"""ONNX provider logic with fake session/tokenizer (onnxruntime has no Intel-Mac wheel).

Numerical parity with PyTorch is checked by scripts/benchmark_embeddings.py and
tests/integration/test_onnx_parity.py inside Linux containers.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from portfolio_ai.rag.embeddings.onnx_provider import (
    OnnxEmbeddingProvider,
    mean_pool,
    verify_model_layout,
)
from portfolio_ai.rag.embeddings.providers import MINILM
from portfolio_ai.rag.errors import EmbeddingDimensionError, EmbeddingError

MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]
POOLING = {
    "word_embedding_dimension": 384,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
}


def model_dir(tmp_path: Path, modules=MODULES, pooling=POOLING, bert=None) -> Path:
    (tmp_path / "1_Pooling").mkdir(parents=True, exist_ok=True)
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    (tmp_path / "sentence_bert_config.json").write_text(
        json.dumps(bert or {"max_seq_length": 256, "do_lower_case": False})
    )
    (tmp_path / "1_Pooling" / "config.json").write_text(json.dumps(pooling))
    return tmp_path


def test_layout_of_all_minilm_is_accepted(tmp_path: Path) -> None:
    layout = verify_model_layout(model_dir(tmp_path), MINILM)
    assert (layout.max_seq_length, layout.dimension) == (256, 384)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {
                "pooling": POOLING
                | {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True}
            },
            EmbeddingError,
        ),
        ({"pooling": POOLING | {"pooling_mode_max_tokens": True}}, EmbeddingError),
        ({"pooling": POOLING | {"word_embedding_dimension": 768}}, EmbeddingDimensionError),
        ({"modules": MODULES[:2]}, EmbeddingError),  # no Normalize
        ({"bert": {"max_seq_length": 256, "do_lower_case": True}}, EmbeddingError),
    ],
)
def test_incompatible_layouts_are_refused(tmp_path: Path, kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        verify_model_layout(model_dir(tmp_path, **kwargs), MINILM)


def test_missing_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="Incomplete"):
        verify_model_layout(tmp_path, MINILM)


def test_mean_pool_ignores_padding() -> None:
    hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class FakeTokenizer:
    def encode_batch(self, texts):
        width = max(len(t.split()) for t in texts) + 2
        out = []
        for t in texts:
            n = len(t.split()) + 2
            out.append(
                FakeEncoding(
                    [101] + [7] * (n - 2) + [102] + [0] * (width - n),
                    [1] * n + [0] * (width - n),
                    [0] * width,
                )
            )
        return out


class FakeSession:
    def __init__(self, inputs=("input_ids", "attention_mask", "token_type_ids")) -> None:
        self.inputs = inputs
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.inputs]

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        ids = feeds["input_ids"]
        hidden = np.random.default_rng(0).normal(size=(*ids.shape, 384)).astype(np.float32)
        return [hidden]


def test_provider_feeds_model_inputs_and_returns_unit_vectors(tmp_path: Path) -> None:
    session = FakeSession()
    provider = OnnxEmbeddingProvider(
        model_dir(tmp_path), MINILM, batch_size=2, session=session, tokenizer=FakeTokenizer()
    )
    vectors = provider.embed_documents(["a b c", "d", "e f"])
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert len(session.feeds) == 2  # batched
    assert set(session.feeds[0]) == {"input_ids", "attention_mask", "token_type_ids"}
    assert session.feeds[0]["input_ids"].dtype == np.int64


def test_provider_supports_models_without_token_type_ids(tmp_path: Path) -> None:
    session = FakeSession(inputs=("input_ids", "attention_mask"))
    provider = OnnxEmbeddingProvider(
        model_dir(tmp_path), MINILM, session=session, tokenizer=FakeTokenizer()
    )
    provider.embed_query("hello world")
    assert set(session.feeds[0]) == {"input_ids", "attention_mask"}


def test_missing_onnx_file_is_a_clear_error(tmp_path: Path) -> None:
    provider = OnnxEmbeddingProvider(model_dir(tmp_path), MINILM, tokenizer=FakeTokenizer())
    with pytest.raises(EmbeddingError):
        provider.embed_query("x")


def test_onnx_telemetry_is_disabled_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONNX Runtime's Linux wheels ship Microsoft usage telemetry; it must be off before import."""
    import os
    import sys

    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # make the import itself fail
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"x")
    provider = OnnxEmbeddingProvider(model_dir(tmp_path), MINILM, tokenizer=FakeTokenizer())

    with pytest.raises(EmbeddingError):
        provider._load_session()

    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"  # set before the import attempt
