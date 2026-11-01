"""Application settings loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration. Fields map to PORTFOLIO_AI_* env vars unless an alias says otherwise."""

    model_config = SettingsConfigDict(
        env_prefix="PORTFOLIO_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "PortfolioAI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    cors_origins: list[str] = ["https://irfan512.github.io", "http://localhost:3000"]

    # Embeddings (M2). local_http = Docker `embedder` service (Intel Macs);
    # sentence_transformers = in-process (Linux, Apple Silicon, production).
    # onnx = ONNX Runtime in-process (production image; no PyTorch).
    embedding_backend: Literal["local_http", "sentence_transformers", "onnx"] = "local_http"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_service_url: str = "http://127.0.0.1:8081"
    embedding_batch_size: int = 32
    embedding_model_dir: Path | None = None  # onnx: folder with onnx/model.onnx + tokenizer.json
    embedding_threads: int = 1

    # Vector store (M2). Local Docker by default; Qdrant Cloud = https URL + API key.
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: str = "portfolio_chunks"
    qdrant_timeout_s: int = 10
    # Extra hostnames allowed over plain http, e.g. ["qdrant"] for the docker-compose network.
    qdrant_private_hosts: list[str] = []

    # Gemini (M3). Names fixed by the deployment spec: GEMINI_API_KEY / GEMINI_MODEL.
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key")
    )
    gemini_model: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model")
    )
    # Also sent to Google as X-Server-Timeout (server-side deadline). 20 s caused 504s on
    # gemini-3.8-flash with structured output; 60 s leaves room for thinking.
    gemini_timeout_s: float = 60.0
    gemini_max_retries: int = 2
    # Thinking tokens count toward max_output_tokens on thinking models. Set ONE of these
    # if answers get cut off: level (Gemini 3.x, e.g. "low") or budget (2.5, e.g. 0).
    gemini_thinking_level: str | None = None
    gemini_thinking_budget: int | None = None
    gemini_temperature: float | None = None  # unset = model default; Gemini 3.x deprecates it
    gemini_max_output_tokens: int = 800

    # Retrieval and answering (M3)
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.30
    evidence_token_budget: int = 1_800

    # Conversations: in memory only, never written to disk or logs.
    session_ttl_s: int = 1_800
    session_max_turns: int = 6
    session_max_count: int = 2_000

    # Abuse and cost protection
    max_message_chars: int = 1_000
    max_body_bytes: int = 8_192
    rate_limit_per_minute: int = 10
    rate_limit_per_day: int = 100
    chat_global_daily_limit: int = 500  # caps Gemini spend even if per-IP limits are evaded
    trusted_proxy_hops: int = 0  # 1 on Render: client IP = last X-Forwarded-For entry

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def secret_values(self) -> list[str]:
        """Every configured secret, so logging can mask them."""
        secrets = (self.gemini_api_key, self.qdrant_api_key)
        return [s.get_secret_value() for s in secrets if s and s.get_secret_value()]


def load_settings(env_file: Path | None = None) -> Settings:
    """Settings from a specific env file (e.g. .env.cloud); real env vars still win."""
    if env_file is None:
        return Settings()
    if not env_file.exists():
        raise FileNotFoundError(f"Settings file not found: {env_file}")
    return Settings(_env_file=env_file)


@lru_cache
def get_settings() -> Settings:
    """Return one cached Settings instance (read env once per process)."""
    return Settings()
