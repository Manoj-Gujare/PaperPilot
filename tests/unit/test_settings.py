"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperpilot.config.settings import Settings, get_settings


def test_settings_read_credentials_from_environment(settings):
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"
    assert settings.qdrant_url == "http://localhost:6333"


def test_secrets_are_not_exposed_by_repr(settings):
    assert "test-openai-key" not in repr(settings)


def test_get_settings_is_cached(settings):
    assert get_settings() is settings


def test_defaults_match_documented_values(settings):
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_top_k == 4
    assert settings.max_retrieval_attempts == 3


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        Settings()


def test_unknown_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "CHATTY")
    with pytest.raises(ValidationError, match="log_level must be one of"):
        Settings()


def test_log_level_is_normalised(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert Settings().log_level == "DEBUG"


def test_missing_credentials_fail_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_ensure_directories_creates_storage_paths(settings):
    settings.ensure_directories()
    assert settings.data_dir.is_dir()
    assert settings.embedding_cache_dir.is_dir()
