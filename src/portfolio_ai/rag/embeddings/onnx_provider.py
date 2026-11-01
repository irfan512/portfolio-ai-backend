"""all-MiniLM-L6-v2 on ONNX Runtime: no PyTorch, about a quarter of the memory.

It reproduces the Sentence Transformers pipeline stored in the model folder:

    Transformer (BERT, max_seq_length=256)  ->  mean pooling  ->  L2 normalize

`verify_model_layout` reads `modules.json`, `sentence_bert_config.json` and
`1_Pooling/config.json`, and refuses any model whose pipeline differs (CLS
pooling, no normalize, another dimension). That way a model swap can never
silently produce incompatible vectors.

This module must not import torch or sentence-transformers.
"""

import json
import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from portfolio_ai.rag.embeddings.base import EmbeddingSpec, Matrix, Vector, embed_in_batches
from portfolio_ai.rag.errors import EmbeddingDimensionError, EmbeddingError

logger = logging.getLogger(__name__)

EXPECTED_MODULES = (
    "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "sentence_transformers.models.Normalize",
)


@dataclass(frozen=True)
class ModelLayout:
    max_seq_length: int
    dimension: int


def verify_model_layout(model_dir: Path, spec: EmbeddingSpec) -> ModelLayout:
    """Check the folder describes Transformer -> mean pooling -> Normalize."""
    try:
        modules = json.loads((model_dir / "modules.json").read_text())
        bert = json.loads((model_dir / "sentence_bert_config.json").read_text())
        pooling = json.loads((model_dir / "1_Pooling" / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise EmbeddingError(f"Incomplete Sentence Transformers model folder: {model_dir}") from exc

    types = tuple(module["type"] for module in sorted(modules, key=lambda m: m["idx"]))
    if types != EXPECTED_MODULES:
        raise EmbeddingError(f"Unsupported module pipeline {types}; expected {EXPECTED_MODULES}")
    other_modes = [
        k
        for k, v in pooling.items()
        if k.startswith("pooling_mode_") and v and k != "pooling_mode_mean_tokens"
    ]
    if not pooling.get("pooling_mode_mean_tokens") or other_modes:
        raise EmbeddingError(f"Only mean pooling is supported, got {pooling}")
    if pooling.get("word_embedding_dimension") != spec.dimension:
        raise EmbeddingDimensionError(
            f"Model dimension {pooling.get('word_embedding_dimension')} != expected {spec.dimension}"
        )
    if bert.get("do_lower_case"):
        raise EmbeddingError("do_lower_case=true is not supported (handled by tokenizer.json here)")
    return ModelLayout(max_seq_length=int(bert["max_seq_length"]), dimension=spec.dimension)


def mean_pool(hidden: NDArray[np.float32], attention_mask: NDArray[np.int64]) -> Matrix:
    """Average token vectors, ignoring padding (same as sentence-transformers' Pooling)."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return (summed / counts).astype(np.float32)


class OnnxEmbeddingProvider:
    def __init__(
        self,
        model_dir: Path,
        spec: EmbeddingSpec,
        *,
        batch_size: int = 32,
        threads: int = 1,
        onnx_file: str = "onnx/model.onnx",
        session: Any | None = None,  # injectable fakes for tests
        tokenizer: Any | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._spec = spec
        self._batch_size = batch_size
        self._threads = threads
        self._onnx_path = model_dir / onnx_file
        self._layout = verify_model_layout(model_dir, spec)
        self._session = session
        self._tokenizer = tokenizer
        self._input_names: list[str] = []
        self._lock = threading.Lock()

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def max_seq_length(self) -> int:
        return self._layout.max_seq_length

    def _load(self) -> tuple[Any, Any]:
        with self._lock:  # load once even if the first requests arrive concurrently
            if self._tokenizer is None:
                self._tokenizer = self._load_tokenizer()
            if self._session is None:
                self._session = self._load_session()
            if not self._input_names:
                self._input_names = [i.name for i in self._session.get_inputs()]
        return self._session, self._tokenizer

    def _load_tokenizer(self) -> Any:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingError(
                "tokenizers is not installed: pip install -e '.[embeddings-onnx]'"
            ) from exc
        tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise EmbeddingError("tokenizer.json has no [PAD] token")
        tokenizer.enable_truncation(max_length=self._layout.max_seq_length)
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        return tokenizer

    def _load_session(self) -> Any:
        # ONNX Runtime's Linux wheels ship a Microsoft "1DS" usage-telemetry client.
        # The env var must be set before import; disable_telemetry_events() below is a
        # second guard. Nothing about this service may leave the host.
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EmbeddingError(
                "onnxruntime is not installed: pip install -e '.[embeddings-onnx]'"
            ) from exc
        if not self._onnx_path.exists():
            raise EmbeddingError(f"ONNX model not found: {self._onnx_path}")
        disable = getattr(ort, "disable_telemetry_events", None)
        if disable is not None:
            disable()
        options = ort.SessionOptions()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False  # lower, steadier memory on small instances
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(self._onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        logger.info(
            "onnx_model_loaded", extra={"model": self._spec.model_id, "threads": self._threads}
        )
        return session

    def _encode(self, batch: list[str]) -> Matrix:
        session, tokenizer = self._load()
        encodings = tokenizer.encode_batch(batch)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        available = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        try:
            feeds = {name: available[name] for name in self._input_names}
        except KeyError as exc:
            raise EmbeddingError(
                f"Unexpected ONNX input {exc}; inputs are {self._input_names}"
            ) from exc
        hidden = session.run(None, feeds)[0]  # last_hidden_state: (batch, tokens, 384)
        return mean_pool(np.asarray(hidden, dtype=np.float32), mask)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return embed_in_batches(
            texts, spec=self._spec, batch_size=self._batch_size, encode_batch=self._encode
        )

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]
