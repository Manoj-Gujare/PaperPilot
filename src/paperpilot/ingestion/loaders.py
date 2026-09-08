"""Loaders that turn a source into a list of chunked LangChain documents.

Every loader returns chunks stamped with a human-readable title and raises
:class:`DocumentLoadError` on failure, so callers handle one error type
regardless of whether a PDF was corrupt, a URL timed out or an extension was
unrecognised.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document

from paperpilot.config import get_settings
from paperpilot.core.exceptions import DocumentLoadError, UnsupportedSourceError
from paperpilot.core.logging import get_logger
from paperpilot.ingestion.chunking import get_markdown_splitter, get_text_splitter, stamp_title

logger = get_logger(__name__)

WEB_PREFIXES = ("http://", "https://")
PDF_EXTENSIONS = frozenset({".pdf"})
TEXT_EXTENSIONS = frozenset({".txt"})
MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | TEXT_EXTENSIONS | MARKDOWN_EXTENSIONS


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load and chunk a PDF, titling it after the file name."""
    path = Path(file_path)
    try:
        pages = PyMuPDFLoader(str(path)).load()
    except Exception as exc:  # loader raises library-specific errors
        raise DocumentLoadError(f"Could not read PDF '{path.name}': {exc}") from exc
    if not pages:
        raise DocumentLoadError(f"PDF '{path.name}' contains no extractable text.")
    return stamp_title(get_text_splitter().split_documents(pages), path.stem)


def load_text(file_path: str | Path) -> list[Document]:
    """Load and chunk a UTF-8 plain-text file."""
    path = Path(file_path)
    try:
        documents = TextLoader(str(path), encoding="utf-8").load()
    except Exception as exc:  # loader raises library-specific errors
        raise DocumentLoadError(f"Could not read text file '{path.name}': {exc}") from exc
    return stamp_title(get_text_splitter().split_documents(documents), path.stem)


def load_markdown(file_path: str | Path) -> list[Document]:
    """Load a Markdown file, splitting on its heading and block structure."""
    path = Path(file_path)
    try:
        documents = TextLoader(str(path), encoding="utf-8").load()
    except Exception as exc:  # loader raises library-specific errors
        raise DocumentLoadError(f"Could not read Markdown file '{path.name}': {exc}") from exc
    return stamp_title(get_markdown_splitter().split_documents(documents), path.stem)


def load_webpage(url: str) -> list[Document]:
    """Fetch a web page and chunk it, preferring the page's own title."""
    settings = get_settings()
    try:
        documents = WebBaseLoader(url, requests_kwargs={"timeout": settings.http_timeout}).load()
    except Exception as exc:  # network and parser errors vary widely
        raise DocumentLoadError(f"Could not fetch '{url}': {exc}") from exc
    if not documents:
        raise DocumentLoadError(f"No readable content at '{url}'.")
    title = (documents[0].metadata.get("title") or url).strip() or url
    return stamp_title(get_text_splitter().split_documents(documents), title)


def load_document(source: str | Path) -> list[Document]:
    """Dispatch to the right loader based on the source's scheme or extension.

    Raises:
        UnsupportedSourceError: The extension has no loader.
        DocumentLoadError: The source exists but could not be read.
    """
    source_str = str(source)
    if source_str.startswith(WEB_PREFIXES):
        logger.info("Loading web page %s", source_str)
        return load_webpage(source_str)

    extension = Path(source_str).suffix.lower()
    logger.info("Loading %s file %s", extension or "extensionless", Path(source_str).name)
    if extension in PDF_EXTENSIONS:
        return load_pdf(source_str)
    if extension in TEXT_EXTENSIONS:
        return load_text(source_str)
    if extension in MARKDOWN_EXTENSIONS:
        return load_markdown(source_str)
    raise UnsupportedSourceError(
        f"Unsupported file type '{extension or 'none'}'. "
        f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )
