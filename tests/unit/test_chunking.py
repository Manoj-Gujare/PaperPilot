"""Tests for splitters and title propagation."""

from __future__ import annotations

from langchain_core.documents import Document

from paperpilot.ingestion.chunking import (
    get_markdown_splitter,
    get_text_splitter,
    reset_splitter_cache,
    stamp_title,
)


def test_splitter_uses_configured_sizes(settings):
    reset_splitter_cache()
    splitter = get_text_splitter()
    assert splitter._chunk_size == settings.chunk_size
    assert splitter._chunk_overlap == settings.chunk_overlap


def test_splitters_are_cached():
    reset_splitter_cache()
    assert get_text_splitter() is get_text_splitter()
    assert get_markdown_splitter() is get_markdown_splitter()


def test_stamp_title_marks_every_chunk():
    documents = [Document(page_content="a"), Document(page_content="b")]
    stamped = stamp_title(documents, "Attention Is All You Need")
    assert [d.metadata["title"] for d in stamped] == ["Attention Is All You Need"] * 2


def test_stamp_title_preserves_existing_metadata():
    document = Document(page_content="a", metadata={"page": 3})
    stamped = stamp_title([document], "Paper")[0]
    assert stamped.metadata == {"page": 3, "title": "Paper"}


def test_long_text_is_split_into_multiple_chunks():
    reset_splitter_cache()
    long_text = " ".join(f"sentence number {i}." for i in range(600))
    chunks = get_text_splitter().split_documents([Document(page_content=long_text)])
    assert len(chunks) > 1
