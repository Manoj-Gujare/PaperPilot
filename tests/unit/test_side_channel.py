"""Tests for the `/btw` side channel."""

from __future__ import annotations

import pytest

from paperpilot.core.exceptions import WebSearchError
from paperpilot.retrieval.web_search import WebSearchResult
from paperpilot.services import side_channel


@pytest.mark.parametrize(
    ("message", "expected"),
    [("/btw what is DPO", True), ("  /BTW hi", True), ("what is DPO", False), ("", False)],
)
def test_side_channel_detection(message, expected):
    assert side_channel.is_side_channel(message) is expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [("/btw what is DPO", "what is DPO"), ("/btw", ""), ("  /btw   spaced  ", "spaced")],
)
def test_command_prefix_is_stripped(message, expected):
    assert side_channel.strip_command(message) == expected


def test_routing_failure_falls_back_to_general_knowledge(monkeypatch):
    def explode(schema):
        raise RuntimeError("provider down")

    monkeypatch.setattr(side_channel, "get_structured_model", explode)
    assert side_channel._needs_web_search("q") is False


def test_web_prompt_carries_results_and_sources(monkeypatch):
    results = [WebSearchResult(title="T", url="https://a", content="body")]
    monkeypatch.setattr(
        side_channel,
        "get_web_search_client",
        lambda: type("C", (), {"search": lambda self, q, max_results: results})(),
    )
    prompt = side_channel._web_system_prompt("q")
    assert "body" in prompt
    assert "https://a" in prompt


def test_search_failure_yields_no_web_prompt(monkeypatch):
    class Failing:
        def search(self, query, max_results):
            raise WebSearchError("down")

    monkeypatch.setattr(side_channel, "get_web_search_client", lambda: Failing())
    assert side_channel._web_system_prompt("q") is None


def test_empty_results_yield_no_web_prompt(monkeypatch):
    monkeypatch.setattr(
        side_channel,
        "get_web_search_client",
        lambda: type("C", (), {"search": lambda self, q, max_results: []})(),
    )
    assert side_channel._web_system_prompt("q") is None


def test_answer_streams_chunks_and_skips_empty_ones(monkeypatch):
    class FakeModel:
        def stream(self, messages):
            for content in ["Hel", "", "lo"]:
                yield type("Chunk", (), {"content": content})()

    monkeypatch.setattr(side_channel, "_needs_web_search", lambda q: False)
    monkeypatch.setattr(side_channel, "get_chat_model", lambda: FakeModel())
    assert "".join(side_channel.answer("q")) == "Hello"


def test_answer_falls_back_to_direct_when_search_is_unavailable(monkeypatch):
    captured = {}

    class FakeModel:
        def stream(self, messages):
            captured["system"] = messages[0]["content"]
            return iter(())

    monkeypatch.setattr(side_channel, "_needs_web_search", lambda q: True)
    monkeypatch.setattr(side_channel, "_web_system_prompt", lambda q: None)
    monkeypatch.setattr(side_channel, "get_chat_model", lambda: FakeModel())
    list(side_channel.answer("q"))
    assert captured["system"] == side_channel.SIDE_CHANNEL_DIRECT_SYSTEM
