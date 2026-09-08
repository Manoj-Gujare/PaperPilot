"""Centralised, validated application settings.

Every tunable value in PaperPilot — credentials, model names, chunking sizes,
retrieval limits — is declared here and read from the environment (or a local
``.env`` file) exactly once. Modules import :func:`get_settings` rather than
touching ``os.environ`` directly, which keeps configuration testable and makes
misconfiguration fail loudly at start-up instead of deep inside a request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Credentials ──────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(..., description="OpenAI key for chat and embeddings.")
    tavily_api_key: SecretStr = Field(..., description="Tavily key for web search.")
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster endpoint.")
    qdrant_api_key: SecretStr = Field(..., description="Qdrant Cloud API key.")

    # ── Models ───────────────────────────────────────────────────────────────
    chat_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=4, ge=1, le=50)
    max_retrieval_attempts: int = Field(default=3, ge=1)
    max_query_rewrites: int = Field(default=1, ge=0)
    collection_prefix: str = "paperpilot"

    # ── Web search ───────────────────────────────────────────────────────────
    web_search_max_results: int = Field(default=3, ge=1, le=20)
    verification_max_results: int = Field(default=5, ge=1, le=20)
    max_superseding_papers: int = Field(default=3, ge=1, le=10)

    # ── Timeouts (seconds) ───────────────────────────────────────────────────
    qdrant_timeout: int = Field(default=120, gt=0)
    http_timeout: int = Field(default=30, gt=0)
    pdf_download_timeout: int = Field(default=60, gt=0)

    # ── Storage paths ────────────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / "embedding_cache"
    checkpoint_db_path: Path = PROJECT_ROOT / "data" / "checkpoints.db"
    sessions_file: Path = PROJECT_ROOT / "data" / "sessions.json"

    # ── Observability ────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_chunk_size(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalised = value.upper()
        if normalised not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalised

    def ensure_directories(self) -> None:
        """Create the directories PaperPilot writes to, if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so the ``.env`` file is parsed once per process and every module
    observes the same configuration object.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
