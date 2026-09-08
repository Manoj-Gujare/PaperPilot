"""Tests for loader dispatch and error translation."""

from __future__ import annotations

import pytest

from paperpilot.core.exceptions import DocumentLoadError, UnsupportedSourceError
from paperpilot.ingestion import loaders


def test_text_file_is_loaded_and_titled(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Retrieval augmented generation.", encoding="utf-8")
    chunks = loaders.load_document(path)
    assert chunks
    assert chunks[0].metadata["title"] == "notes"


def test_markdown_uses_the_markdown_splitter(tmp_path, monkeypatch):
    used = {}

    def fake_markdown_splitter():
        used["markdown"] = True
        return loaders.get_text_splitter()

    monkeypatch.setattr(loaders, "get_markdown_splitter", fake_markdown_splitter)
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    loaders.load_document(path)
    assert used == {"markdown": True}


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedSourceError, match="Unsupported file type"):
        loaders.load_document(path)


def test_missing_file_raises_document_load_error(tmp_path):
    with pytest.raises(DocumentLoadError):
        loaders.load_document(tmp_path / "absent.txt")


@pytest.mark.parametrize("url", ["http://example.com/p", "https://example.com/p"])
def test_urls_are_routed_to_the_web_loader(url, monkeypatch):
    monkeypatch.setattr(loaders, "load_webpage", lambda source: ["web", source])
    assert loaders.load_document(url) == ["web", url]


def test_web_loader_failure_is_translated(monkeypatch):
    class ExplodingLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            raise TimeoutError("connection timed out")

    monkeypatch.setattr(loaders, "WebBaseLoader", ExplodingLoader)
    with pytest.raises(DocumentLoadError, match="Could not fetch"):
        loaders.load_webpage("https://example.com")


def test_empty_web_page_raises(monkeypatch):
    class EmptyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return []

    monkeypatch.setattr(loaders, "WebBaseLoader", EmptyLoader)
    with pytest.raises(DocumentLoadError, match="No readable content"):
        loaders.load_webpage("https://example.com")
