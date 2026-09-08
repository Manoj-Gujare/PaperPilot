"""Tests for ArXiv identifier handling and lookup."""

from __future__ import annotations

import pytest

from paperpilot.core.exceptions import DocumentLoadError
from paperpilot.ingestion import arxiv

ATOM_FEED = """
<feed>
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
  </entry>
</feed>
"""


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("1706.03762", "1706.03762"),
        ("arxiv:2303.08774v2", "2303.08774"),
        ("see 2303.08774 for details", "2303.08774"),
        ("Attention Is All You Need", None),
        ("", None),
    ],
)
def test_identifier_extraction(query, expected):
    assert arxiv.extract_arxiv_id(query) == expected


def test_title_search_returns_the_versionless_identifier(monkeypatch):
    monkeypatch.setattr(arxiv, "_fetch", lambda url, timeout: ATOM_FEED.encode())
    assert arxiv.search_by_title("Attention Is All You Need") == "1706.03762"


def test_title_search_without_a_match_raises(monkeypatch):
    monkeypatch.setattr(arxiv, "_fetch", lambda url, timeout: b"<feed></feed>")
    with pytest.raises(DocumentLoadError, match="No ArXiv paper found"):
        arxiv.search_by_title("a paper that does not exist")


def test_lookup_title_reads_the_entry_not_the_feed(monkeypatch):
    monkeypatch.setattr(arxiv, "_fetch", lambda url, timeout: ATOM_FEED.encode())
    assert arxiv.lookup_title("1706.03762") == "Attention Is All You Need"


def test_lookup_title_falls_back_to_the_identifier(monkeypatch):
    monkeypatch.setattr(
        arxiv, "_fetch", lambda url, timeout: b"<feed><title>Only feed</title></feed>"
    )
    assert arxiv.lookup_title("1706.03762") == "1706.03762"


def test_network_failure_is_translated(monkeypatch):
    def explode(url, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(arxiv.urllib.request, "urlopen", explode)
    with pytest.raises(DocumentLoadError, match="ArXiv request failed"):
        arxiv._fetch("https://export.arxiv.org/api/query", timeout=1)


def test_load_arxiv_prefers_an_identifier_over_a_search(monkeypatch):
    calls = []
    monkeypatch.setattr(arxiv, "search_by_title", lambda q: calls.append(q) or "0000.00000")
    monkeypatch.setattr(arxiv, "load_by_id", lambda arxiv_id: [arxiv_id])
    assert arxiv.load_arxiv("paper 1706.03762") == ["1706.03762"]
    assert calls == []


def test_load_arxiv_falls_back_to_a_title_search(monkeypatch):
    monkeypatch.setattr(arxiv, "search_by_title", lambda q: "1706.03762")
    monkeypatch.setattr(arxiv, "load_by_id", lambda arxiv_id: [arxiv_id])
    assert arxiv.load_arxiv("Attention Is All You Need") == ["1706.03762"]


def test_temporary_pdf_is_removed_even_when_parsing_fails(monkeypatch, tmp_path):
    created: list[str] = []
    real_loader = arxiv.PyMuPDFLoader

    class RecordingLoader:
        def __init__(self, path):
            created.append(path)

        def load(self):
            raise ValueError("corrupt pdf")

    monkeypatch.setattr(arxiv, "_fetch", lambda url, timeout: b"%PDF-1.4")
    monkeypatch.setattr(arxiv, "PyMuPDFLoader", RecordingLoader)
    with pytest.raises(ValueError, match="corrupt pdf"):
        arxiv.load_by_id("1706.03762")
    assert created
    assert not arxiv.Path(created[0]).exists()
    assert real_loader is not RecordingLoader
