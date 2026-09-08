"""Streamlit session-state bootstrap and session switching.

Streamlit reruns the whole script on every interaction, so anything that must
survive a rerun lives in ``st.session_state``. Confining that to one module
keeps the components free of bootstrap logic and makes the set of keys the
application relies on explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from paperpilot.core.logging import get_logger
from paperpilot.services.chat_service import ChatService
from paperpilot.services.naming import generate_session_name
from paperpilot.services.session_store import SessionMeta, SessionStore

logger = get_logger(__name__)

SESSIONS_KEY = "sessions_meta"
CHATS_KEY = "chats"
TURNS_KEY = "turns"
ACTIVE_KEY = "active_session_id"
LOADED_FILES_KEY = "loaded_files"


@dataclass
class AppState:
    """A typed view over the Streamlit session state.

    Streamlit's state is an untyped mapping shared by every component. Reading
    and writing it through one object keeps key names in a single place and
    gives the components a small, documented surface instead.
    """

    chat_service: ChatService = field(default_factory=ChatService)
    store: SessionStore = field(default_factory=SessionStore)

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def bootstrap(self) -> None:
        """Populate session state and select a session on first render."""
        st.session_state.setdefault(SESSIONS_KEY, self.store.load())
        st.session_state.setdefault(CHATS_KEY, {})
        st.session_state.setdefault(TURNS_KEY, {})
        st.session_state.setdefault(LOADED_FILES_KEY, {})

        if ACTIVE_KEY in st.session_state:
            return

        latest = self.store.most_recent(self.sessions)
        if latest is None:
            self.create_session()
        else:
            self.switch_to(latest.id)

    # ── Sessions ─────────────────────────────────────────────────────────────

    @property
    def sessions(self) -> dict[str, SessionMeta]:
        """Every known session, keyed by id."""
        return st.session_state[SESSIONS_KEY]

    @property
    def active_id(self) -> str:
        """The session currently being viewed."""
        return st.session_state[ACTIVE_KEY]

    def create_session(self) -> str:
        """Start a new, empty session and make it active."""
        meta = SessionMeta.new()
        self.sessions[meta.id] = meta
        self.store.save(self.sessions)
        st.session_state[CHATS_KEY][meta.id] = []
        st.session_state[TURNS_KEY][meta.id] = 0
        st.session_state[ACTIVE_KEY] = meta.id
        logger.info("Created session %s", meta.id)
        return meta.id

    def switch_to(self, session_id: str) -> None:
        """Make a session active, loading its history the first time.

        History comes from the checkpointer, so a session opened after a
        restart shows the conversation rather than an empty page.
        """
        st.session_state[ACTIVE_KEY] = session_id
        if session_id not in st.session_state[CHATS_KEY]:
            history = [
                {**message, "graph_state": {}} for message in self.chat_service.history(session_id)
            ]
            st.session_state[CHATS_KEY][session_id] = _number_turns(history)
        st.session_state[TURNS_KEY].setdefault(session_id, self._turn_count(session_id))

    def rename_if_unnamed(self, session_id: str, first_message: str) -> None:
        """Give a session a title derived from its first message, once."""
        meta = self.sessions.get(session_id)
        if meta is None or meta.is_named:
            return
        meta.name = generate_session_name(first_message)
        meta.is_named = True
        self.store.save(self.sessions)

    # ── Conversation ─────────────────────────────────────────────────────────

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        """The rendered conversation for a session."""
        return st.session_state[CHATS_KEY].setdefault(session_id, [])

    def is_empty(self, session_id: str) -> bool:
        """True when a session has no messages yet."""
        return not self.messages(session_id)

    def next_turn(self, session_id: str) -> int:
        """Increment and return the assistant turn counter for a session."""
        turns = st.session_state[TURNS_KEY]
        turns[session_id] = turns.get(session_id, 0) + 1
        return turns[session_id]

    def append_user_message(self, session_id: str, content: str) -> None:
        """Record a user message in the rendered conversation."""
        self.messages(session_id).append({"role": "user", "content": content})

    def append_assistant_message(
        self, session_id: str, content: str, graph_state: dict[str, Any], turn: int
    ) -> None:
        """Record an assistant reply along with the state that produced it."""
        self.messages(session_id).append(
            {
                "role": "assistant",
                "content": content,
                "graph_state": graph_state,
                "turn": turn,
            }
        )

    # ── Uploads ──────────────────────────────────────────────────────────────

    def already_loaded(self, session_id: str, filename: str) -> bool:
        """True when a file has already been ingested into this session.

        Streamlit's uploader re-submits its files on every rerun, so without
        this check a single upload would be embedded again on each interaction.
        """
        return filename in st.session_state[LOADED_FILES_KEY].get(session_id, set())

    def mark_loaded(self, session_id: str, filename: str) -> None:
        """Remember that a file has been ingested into this session."""
        st.session_state[LOADED_FILES_KEY].setdefault(session_id, set()).add(filename)

    # ── Internals ────────────────────────────────────────────────────────────

    def _turn_count(self, session_id: str) -> int:
        return sum(1 for message in self.messages(session_id) if message["role"] == "assistant")


def _number_turns(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Number assistant messages so the state inspector can label them."""
    turn = 0
    for message in history:
        if message["role"] == "assistant":
            turn += 1
            message["turn"] = turn
    return history
