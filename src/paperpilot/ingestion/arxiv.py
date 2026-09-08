"""ArXiv lookup and download.

The ArXiv Atom API is queried directly rather than through the ``arxiv``
package: the dependency proved unreliable for title searches, and the two calls
needed here — resolve an identifier, fetch a PDF — are small enough that owning
them is cheaper than working around a wrapper.
"""

from __future__ import annotations

import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from paperpilot.config import get_settings
from paperpilot.core.exceptions import DocumentLoadError
from paperpilot.core.logging import get_logger
from paperpilot.ingestion.chunking import get_text_splitter, stamp_title

logger = get_logger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

_ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")
_VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$")
_ENTRY_ID_PATTERN = re.compile(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>")
_TITLE_PATTERN = re.compile(r"<title>(.*?)</title>", re.DOTALL)


def _strip_version(arxiv_id: str) -> str:
    """Return the identifier without its ``vN`` suffix."""
    return _VERSION_SUFFIX_PATTERN.sub("", arxiv_id)


def extract_arxiv_id(query: str) -> str | None:
    """Return the bare ArXiv identifier contained in ``query``, if any."""
    match = _ARXIV_ID_PATTERN.search(query)
    return _strip_version(match.group(1)) if match else None


def _fetch(url: str, timeout: int) -> bytes:
    """GET a URL, translating transport failures into DocumentLoadError."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DocumentLoadError(f"ArXiv request failed: {exc}") from exc


def lookup_title(arxiv_id: str) -> str:
    """Return the paper title for an identifier, falling back to the id itself.

    The Atom feed opens with the feed's own title, so the paper's title is the
    second match.
    """
    settings = get_settings()
    query = urllib.parse.urlencode({"id_list": arxiv_id})
    xml = _fetch(f"{ARXIV_API_URL}?{query}", settings.http_timeout).decode()
    titles = _TITLE_PATTERN.findall(xml)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def search_by_title(query: str) -> str:
    """Return the identifier of the closest title match on ArXiv."""
    settings = get_settings()
    phrase = query.strip().strip('"')
    params = urllib.parse.urlencode(
        {"search_query": f'ti:"{phrase}"', "max_results": 1, "sortBy": "relevance"}
    )
    xml = _fetch(f"{ARXIV_API_URL}?{params}", settings.http_timeout).decode()
    match = _ENTRY_ID_PATTERN.search(xml)
    if not match:
        raise DocumentLoadError(f"No ArXiv paper found for '{query}'.")
    return _strip_version(match.group(1))


def load_by_id(arxiv_id: str) -> list[Document]:
    """Download an ArXiv PDF by identifier and return its chunks."""
    settings = get_settings()
    pdf_bytes = _fetch(ARXIV_PDF_URL.format(arxiv_id=arxiv_id), settings.pdf_download_timeout)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            handle.write(pdf_bytes)
            temp_path = handle.name
        pages = PyMuPDFLoader(temp_path).load()
        if not pages:
            raise DocumentLoadError(f"ArXiv paper {arxiv_id} has no extractable text.")
        title = (pages[0].metadata.get("title") or "").strip() or lookup_title(arxiv_id)
        return stamp_title(get_text_splitter().split_documents(pages), title)
    finally:
        # The temporary file is removed whether parsing succeeded or not.
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    """Load an ArXiv paper from an identifier or a title search."""
    arxiv_id = extract_arxiv_id(query) or search_by_title(query)
    logger.info("Loading ArXiv paper %s", arxiv_id)
    return load_by_id(arxiv_id)
