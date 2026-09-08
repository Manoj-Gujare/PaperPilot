"""Tests for the Streamlit state wrapper.

Streamlit is replaced with a stub exposing only ``session_state``: these tests
cover state transitions, not rendering, and a real script run is not needed to
exercise them.
"""

from __future__ import annotations

import pytest

from paperpilot.services.session_store import SessionMeta, SessionStore
from paperpilot.ui import state as ui_state
from paperpilot.ui.state import AppState


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}


class FakeChatService:
    def __init__(self, history=None):
        self._history = history or []

    def history(self, session_id):
        return [dict(message) for message in self._history]


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(ui_state, "st", FakeStreamlit())
    return AppState(
        chat_service=FakeChatService(),
        store=SessionStore(tmp_path / "sessions.json"),
    )


def test_bootstrap_creates_a_first_session(app):
    app.bootstrap()
    assert app.active_id in app.sessions


def test_bootstrap_selects_the_most_recent_session(monkeypatch, tmp_path):
    monkeypatch.setattr(ui_state, "st", FakeStreamlit())
    store = SessionStore(tmp_path / "sessions.json")
    store.save(
        {
            "old": SessionMeta(id="old", created_at="2026-01-01T00:00:00"),
            "new": SessionMeta(id="new", created_at="2026-06-01T00:00:00"),
        }
    )
    app = AppState(chat_service=FakeChatService(), store=store)
    app.bootstrap()
    assert app.active_id == "new"


def test_bootstrap_is_idempotent(app):
    app.bootstrap()
    first = app.active_id
    app.bootstrap()
    assert app.active_id == first
    assert len(app.sessions) == 1


def test_creating_a_session_persists_it(app, tmp_path):
    app.bootstrap()
    session_id = app.create_session()
    assert session_id in SessionStore(tmp_path / "sessions.json").load()


def test_switching_loads_history_from_the_checkpointer(monkeypatch, tmp_path):
    """A session opened after a restart must show its conversation, not a blank page."""
    monkeypatch.setattr(ui_state, "st", FakeStreamlit())
    service = FakeChatService(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
    )
    app = AppState(chat_service=service, store=SessionStore(tmp_path / "sessions.json"))
    app.bootstrap()
    app.switch_to("restored")
    messages = app.messages("restored")
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [m["turn"] for m in messages if m["role"] == "assistant"] == [1, 2]


def test_switching_back_does_not_reload_history(app):
    app.bootstrap()
    session_id = app.active_id
    app.append_user_message(session_id, "typed just now")
    app.switch_to(session_id)
    assert len(app.messages(session_id)) == 1


def test_turn_counter_increments_per_session(app):
    app.bootstrap()
    session_id = app.active_id
    assert [app.next_turn(session_id), app.next_turn(session_id)] == [1, 2]


def test_a_new_session_is_empty(app):
    app.bootstrap()
    assert app.is_empty(app.active_id)
    app.append_user_message(app.active_id, "hello")
    assert not app.is_empty(app.active_id)


def test_renaming_happens_once(app, monkeypatch):
    monkeypatch.setattr(ui_state, "generate_session_name", lambda message: f"title-{message}")
    app.bootstrap()
    session_id = app.active_id
    app.rename_if_unnamed(session_id, "first")
    app.rename_if_unnamed(session_id, "second")
    assert app.sessions[session_id].name == "title-first"


def test_renaming_an_unknown_session_is_ignored(app):
    app.bootstrap()
    app.rename_if_unnamed("does-not-exist", "hello")


def test_uploads_are_tracked_per_session(app):
    """Streamlit resubmits uploads on every rerun; without this each one re-embeds."""
    app.bootstrap()
    session_id = app.active_id
    assert not app.already_loaded(session_id, "paper.pdf")
    app.mark_loaded(session_id, "paper.pdf")
    assert app.already_loaded(session_id, "paper.pdf")
    assert not app.already_loaded("other-session", "paper.pdf")


def test_assistant_messages_carry_their_graph_state(app):
    app.bootstrap()
    session_id = app.active_id
    app.append_assistant_message(session_id, "answer", {"route": "retrieve"}, turn=1)
    message = app.messages(session_id)[-1]
    assert message["graph_state"] == {"route": "retrieve"}
    assert message["turn"] == 1
