"""
Central configuration, resolved from environment variables with sane defaults.

Everything is a plain dataclass read from ``os.environ`` — no settings framework,
because a handful of values does not justify one. Import :data:`settings` and use it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


@dataclass(frozen=True)
class Settings:
    # Storage
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "chroma_db"))
    collection: str = field(default_factory=lambda: os.getenv("COLLECTION", "constitution"))

    # Models
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    gen_model: str = field(
        default_factory=lambda: os.getenv("GEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    )

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    fetch_k: int = field(default_factory=lambda: _env_int("FETCH_K", 20))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", False))

    # Generation
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    max_new_tokens: int = field(default_factory=lambda: _env_int("MAX_NEW_TOKENS", 400))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.2))
    request_timeout: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT", 60))

    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))

    @property
    def has_token(self) -> bool:
        return bool(self.hf_token)


settings = Settings()
