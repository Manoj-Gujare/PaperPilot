"""Tests for the web search façade."""

from __future__ import annotations

import pytest

from paperpilot.core.exceptions import WebSearchError
from paperpilot.retrieval.web_search import WebSearchClient, WebSearchResult


class FakeTavily:
    def __init__(self, response=None, error=None):
        self.response = response or {"results": []}
        self.error = error
        self.calls = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.response


def test_results_are_normalised():
    client = WebSearchClient(
        FakeTavily({"results": [{"title": "T", "url": "https://a", "content": "body"}]}), 3
    )
    result = client.search("q")[0]
    assert result == WebSearchResult(title="T", url="https://a", content="body")


def test_results_without_a_url_are_dropped():
    client = WebSearchClient(FakeTavily({"results": [{"title": "T", "content": "body"}]}), 3)
    assert client.search("q") == []


def test_missing_title_falls_back_to_a_placeholder():
    client = WebSearchClient(FakeTavily({"results": [{"url": "https://a", "content": "b"}]}), 3)
    assert client.search("q")[0].title == "Web Result"


def test_no_results_is_not_an_error():
    assert WebSearchClient(FakeTavily(), 3).search("q") == []


def test_provider_failure_becomes_a_web_search_error():
    client = WebSearchClient(FakeTavily(error=RuntimeError("rate limited")), 3)
    with pytest.raises(WebSearchError, match="Web search failed"):
        client.search("q")


def test_configured_default_is_used_when_no_limit_is_given():
    provider = FakeTavily()
    WebSearchClient(provider, 7).search("q")
    assert provider.calls == [("q", 7)]


def test_explicit_limit_wins():
    provider = FakeTavily()
    WebSearchClient(provider, 7).search("q", max_results=2)
    assert provider.calls == [("q", 2)]


def test_result_converts_to_a_document():
    document = WebSearchResult(title="T", url="https://a", content="body").to_document()
    assert document.page_content == "body"
    assert document.metadata == {"title": "T", "url": "https://a"}
