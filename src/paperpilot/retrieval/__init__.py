"""Context retrieval — semantic search over stored papers and over the live web."""

from paperpilot.retrieval.vector_store import (
    PaperRepository,
    collection_name_for,
    get_repository,
)
from paperpilot.retrieval.web_search import WebSearchResult, get_web_search_client

__all__ = [
    "PaperRepository",
    "WebSearchResult",
    "collection_name_for",
    "get_repository",
    "get_web_search_client",
]
