"""The chat transcript, the input box, and how a turn is rendered."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import streamlit as st

from paperpilot.core.logging import get_logger
from paperpilot.services import side_channel
from paperpilot.services.chat_service import serialise_state
from paperpilot.ui.state import AppState

logger = get_logger(__name__)

INPUT_PLACEHOLDER = "Ask about your papers, verify a claim, or search the web…"
SIDE_CHANNEL_NOTICE = "Side channel — not saved to session history."
SIDE_CHANNEL_HELP = "Please add a question after `/btw`, e.g. `/btw What is attention?`"
CURSOR = "▌"


def _stream_into(placeholder, chunks: Iterator[str]) -> str:
    """Render chunks as they arrive and return the finished text.

    A trailing cursor is drawn while streaming and removed at the end, so the
    reply reads as text being typed rather than a box that fills in silently.
    """
    text = ""
    for chunk in chunks:
        text += chunk
        placeholder.markdown(text + CURSOR)
    placeholder.markdown(text)
    return text


def _render_state_inspector(graph_state: dict[str, Any], turn: int) -> None:
    """Show the graph state behind an answer, collapsed by default."""
    with st.expander(f"Graph state · turn {turn}", expanded=False):
        st.json(graph_state)


def render_history(state: AppState, session_id: str) -> None:
    """Replay the conversation so far."""
    for message in state.messages(session_id):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_state_inspector(message.get("graph_state", {}), message.get("turn", 0))


def _handle_side_channel(prompt: str) -> None:
    """Answer a `/btw` question without touching session state.

    Nothing here is appended to the conversation or written to the
    checkpointer: the exchange exists only in this render.
    """
    question = side_channel.strip_command(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(SIDE_CHANNEL_NOTICE)

    with st.chat_message("assistant"):
        if not question:
            st.markdown(SIDE_CHANNEL_HELP)
        else:
            _stream_into(st.empty(), side_channel.answer(question))
        st.caption(SIDE_CHANNEL_NOTICE)


def _handle_turn(state: AppState, session_id: str, prompt: str) -> None:
    """Run one graph turn and render the result."""
    is_first_message = state.is_empty(session_id)

    with st.chat_message("user"):
        st.markdown(prompt)
    state.append_user_message(session_id, prompt)
    turn = state.next_turn(session_id)

    if is_first_message:
        state.rename_if_unnamed(session_id, prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = _stream_into(placeholder, state.chat_service.stream_answer(prompt, session_id))
        if not answer:
            # Some routes assemble their reply in Python rather than streaming
            # it from the model, so an empty token stream is expected, not an error.
            answer = state.chat_service.answer_of_record(session_id)
            placeholder.markdown(answer)

        graph_state = serialise_state(state.chat_service.final_state(session_id))
        _render_state_inspector(graph_state, turn)

    state.append_assistant_message(session_id, answer, graph_state, turn)

    if is_first_message:
        # Rerun so the sidebar picks up the generated session title.
        st.rerun()


def render_chat_input(state: AppState, session_id: str) -> None:
    """Read the next message and dispatch it to the right handler."""
    prompt = st.chat_input(INPUT_PLACEHOLDER)
    if not prompt:
        return
    if side_channel.is_side_channel(prompt):
        _handle_side_channel(prompt)
    else:
        _handle_turn(state, session_id, prompt)
