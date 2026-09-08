"""Text splitters used when ingesting papers.

Chunk size and overlap are the two knobs that decide retrieval quality. Smaller
chunks retrieve more precisely; the overlap keeps a sentence that straddles a
boundary reachable from both sides. Both come from configuration so they can be
tuned per deployment without touching code.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paperpilot.config import get_settings


@lru_cache(maxsize=1)
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return the general-purpose splitter for prose and PDFs."""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )


@lru_cache(maxsize=1)
def get_markdown_splitter() -> RecursiveCharacterTextSplitter:
    """Return a splitter that prefers Markdown structure when breaking text."""
    settings = get_settings()
    return RecursiveCharacterTextSplitter.from_language(
        "markdown",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )


def stamp_title(documents: list[Document], title: str) -> list[Document]:
    """Record the source title on every chunk.

    The title is what the interface lists as a loaded paper and what lets a
    reader trace an answer back to a document, so it must survive splitting.
    """
    for document in documents:
        document.metadata["title"] = title
    return documents


def reset_splitter_cache() -> None:
    """Drop cached splitters so new chunking settings take effect."""
    get_text_splitter.cache_clear()
    get_markdown_splitter.cache_clear()
