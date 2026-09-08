"""Web search through Tavily.

Wrapping the provider gives the rest of the application a stable, typed result
shape instead of raw response dictionaries, keeps key handling in one place,
and means swapping providers touches this file alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from tavily import TavilyClient

from paperpilot.config import get_settings
from paperpilot.core.exceptions import WebSearchError
from paperpilot.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TITLE = "Web Result"


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """A single web result, normalised across providers."""

    title: str
    url: str
    content: str

    def to_document(self) -> Document:
        """Represent the result as a document so it can join retrieved context."""
        return Document(page_content=self.content, metadata={"title": self.title, "url": self.url})


class WebSearchClient:
    """Thin, typed façade over the Tavily search API."""

    def __init__(self, client: TavilyClient, default_max_results: int) -> None:
        self._client = client
        self._default_max_results = default_max_results

    def search(self, query: str, max_results: int | None = None) -> list[WebSearchResult]:
        """Return web results for ``query``.

        An empty result list is a legitimate outcome and is returned as such;
        only a provider or transport failure raises.
        """
        limit = max_results if max_results is not None else self._default_max_results
        try:
            response = self._client.search(query, max_results=limit)
        except Exception as exc:  # client raises provider-specific errors
            raise WebSearchError(f"Web search failed for {query[:80]!r}: {exc}") from exc

        results = [
            WebSearchResult(
                title=item.get("title") or DEFAULT_TITLE,
                url=item["url"],
                content=item.get("content", ""),
            )
            for item in response.get("results", [])
            if item.get("url")
        ]
        logger.debug("Web search %r returned %d result(s)", query[:80], len(results))
        return results


@lru_cache(maxsize=1)
def get_web_search_client() -> WebSearchClient:
    """Return the shared web search client."""
    settings = get_settings()
    return WebSearchClient(
        client=TavilyClient(api_key=settings.tavily_api_key.get_secret_value()),
        default_max_results=settings.web_search_max_results,
    )


def reset_web_search_cache() -> None:
    """Drop the cached client. Used by tests and after a settings change."""
    get_web_search_client.cache_clear()
