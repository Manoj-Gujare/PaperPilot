"""Tests for automatic session titles."""

from __future__ import annotations

from paperpilot.services import naming


class FakeModel:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": self.content})()


def test_a_title_is_trimmed(monkeypatch):
    monkeypatch.setattr(naming, "get_chat_model", lambda: FakeModel("  Attention Mechanisms  "))
    assert naming.generate_session_name("Explain attention") == "Attention Mechanisms"


def test_an_empty_title_falls_back(monkeypatch):
    monkeypatch.setattr(naming, "get_chat_model", lambda: FakeModel("   "))
    assert naming.generate_session_name("Explain attention") == naming.FALLBACK_NAME


def test_a_failure_falls_back_instead_of_raising(monkeypatch):
    """Naming is cosmetic; it must never block the answer the user is waiting for."""

    def explode():
        raise RuntimeError("provider down")

    monkeypatch.setattr(naming, "get_chat_model", explode)
    assert naming.generate_session_name("Explain attention") == naming.FALLBACK_NAME


def test_only_the_start_of_a_long_message_is_sent(monkeypatch):
    model = FakeModel("Title")
    monkeypatch.setattr(naming, "get_chat_model", lambda: model)
    naming.generate_session_name("x" * 5000)
    assert len(model.messages[1]["content"]) == naming.MAX_MESSAGE_CHARACTERS
