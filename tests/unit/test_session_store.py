"""Tests for session metadata persistence."""

from __future__ import annotations

import json

from paperpilot.services.session_store import SessionMeta, SessionStore


def test_new_sessions_get_a_unique_id():
    assert SessionMeta.new().id != SessionMeta.new().id


def test_new_sessions_start_unnamed():
    meta = SessionMeta.new()
    assert meta.name == "New Session"
    assert meta.is_named is False


def test_sessions_round_trip(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    meta = SessionMeta(id="a", name="Attention paper", created_at="2026-01-01T00:00:00")
    store.save({"a": meta})
    assert store.load()["a"] == meta


def test_saving_creates_the_parent_directory(tmp_path):
    store = SessionStore(tmp_path / "nested" / "sessions.json")
    store.save({"a": SessionMeta(id="a")})
    assert (tmp_path / "nested" / "sessions.json").exists()


def test_a_missing_index_loads_as_empty(tmp_path):
    assert SessionStore(tmp_path / "absent.json").load() == {}


def test_a_corrupt_index_loads_as_empty(tmp_path):
    """Losing the list of past conversations must not block starting a new one."""
    path = tmp_path / "sessions.json"
    path.write_text("{ not json", encoding="utf-8")
    assert SessionStore(path).load() == {}


def test_partial_records_are_filled_with_defaults(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"a": {"id": "a"}}), encoding="utf-8")
    meta = SessionStore(path).load()["a"]
    assert meta.name == "New Session"
    assert meta.is_named is False
    assert meta.created_at


def test_most_recent_picks_the_newest():
    sessions = {
        "old": SessionMeta(id="old", created_at="2026-01-01T00:00:00"),
        "new": SessionMeta(id="new", created_at="2026-06-01T00:00:00"),
    }
    assert SessionStore.most_recent(sessions).id == "new"


def test_most_recent_of_nothing_is_none():
    assert SessionStore.most_recent({}) is None


def test_sessions_are_listed_newest_first():
    sessions = {
        "a": SessionMeta(id="a", created_at="2026-01-01T00:00:00"),
        "b": SessionMeta(id="b", created_at="2026-06-01T00:00:00"),
        "c": SessionMeta(id="c", created_at="2026-03-01T00:00:00"),
    }
    assert [meta.id for meta in SessionStore.sorted_by_recency(sessions)] == ["b", "c", "a"]


def test_the_store_defaults_to_the_configured_path(settings):
    store = SessionStore()
    store.save({"a": SessionMeta(id="a")})
    assert settings.sessions_file.exists()
