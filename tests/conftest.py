"""Shared pytest fixtures.

The whole suite runs without network access or real credentials: environment
variables are stubbed before any PaperPilot module is imported, and external
clients are replaced with fakes in the tests that need them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

_STUB_ENV = {
    "OPENAI_API_KEY": "test-openai-key",
    "TAVILY_API_KEY": "test-tavily-key",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "test-qdrant-key",
}


@pytest.fixture(autouse=True)
def stub_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give every test a clean, fully populated, throwaway configuration."""
    for key, value in _STUB_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path / "data" / "embedding_cache"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "data" / "checkpoints.db"))
    monkeypatch.setenv("SESSIONS_FILE", str(tmp_path / "data" / "sessions.json"))

    from paperpilot.config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    """The cached settings instance built from the stubbed environment."""
    from paperpilot.config.settings import get_settings

    return get_settings()


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real ``.env`` from leaking into test configuration."""
    monkeypatch.setattr(
        "pydantic_settings.sources.DotEnvSettingsSource.__call__",
        lambda self: {},
        raising=False,
    )


@pytest.fixture
def fake_document():
    """Factory building LangChain documents without importing loaders."""
    from langchain_core.documents import Document

    def _build(content: str = "chunk", **metadata: object) -> Document:
        return Document(page_content=content, metadata=dict(metadata))

    return _build


os.environ.setdefault("PAPERPILOT_TESTING", "1")
