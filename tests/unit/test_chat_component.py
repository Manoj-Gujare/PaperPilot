"""Tests for the chat component's streaming and dispatch behaviour."""

from __future__ import annotations

import pytest

from paperpilot.ui.components import chat


class FakePlaceholder:
    def __init__(self):
        self.renders: list[str] = []

    def markdown(self, text):
        self.renders.append(text)


def test_streaming_draws_a_cursor_until_the_end():
    placeholder = FakePlaceholder()
    text = chat._stream_into(placeholder, iter(["Hel", "lo"]))
    assert text == "Hello"
    assert placeholder.renders == ["Hel▌", "Hello▌", "Hello"]


def test_streaming_nothing_renders_an_empty_answer():
    placeholder = FakePlaceholder()
    assert chat._stream_into(placeholder, iter(())) == ""
    assert placeholder.renders == [""]


@pytest.mark.parametrize(
    ("prompt", "goes_to_side_channel"),
    [("/btw what is DPO", True), ("What does the paper claim?", False)],
)
def test_input_is_dispatched_by_prefix(monkeypatch, prompt, goes_to_side_channel):
    calls = []

    class FakeStreamlit:
        def chat_input(self, placeholder):
            return prompt

    monkeypatch.setattr(chat, "st", FakeStreamlit())
    monkeypatch.setattr(chat, "_handle_side_channel", lambda p: calls.append("side"))
    monkeypatch.setattr(chat, "_handle_turn", lambda state, sid, p: calls.append("turn"))

    chat.render_chat_input(state=None, session_id="s1")
    assert calls == (["side"] if goes_to_side_channel else ["turn"])


def test_no_input_does_nothing(monkeypatch):
    calls = []

    class FakeStreamlit:
        def chat_input(self, placeholder):
            return None

    monkeypatch.setattr(chat, "st", FakeStreamlit())
    monkeypatch.setattr(chat, "_handle_turn", lambda state, sid, p: calls.append("turn"))

    chat.render_chat_input(state=None, session_id="s1")
    assert calls == []
