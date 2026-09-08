"""Loading papers from files, the web and ArXiv, and splitting them into chunks."""

from paperpilot.ingestion.arxiv import extract_arxiv_id, load_arxiv
from paperpilot.ingestion.chunking import get_markdown_splitter, get_text_splitter
from paperpilot.ingestion.loaders import (
    load_document,
    load_markdown,
    load_pdf,
    load_text,
    load_webpage,
)

__all__ = [
    "extract_arxiv_id",
    "get_markdown_splitter",
    "get_text_splitter",
    "load_arxiv",
    "load_document",
    "load_markdown",
    "load_pdf",
    "load_text",
    "load_webpage",
]
