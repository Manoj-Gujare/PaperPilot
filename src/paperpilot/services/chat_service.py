"""Orchestration between the interface and the compiled graph.

The interface should not know about LangGraph configs, stream modes or state
snapshots. It asks this service for a stream of tokens and, afterwards, for the
final state — everything else stays behind the seam.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph

from paperpilot.core.logging import get_logger
from paperpilot.graph import build_graph, initial_state

logger = get_logger(__name__)

ANSWER_NODE = "generate_answer"
SNAPSHOT_TEXT_LIMIT = 300
EMPTY_RESPONSE_FALLBACK = "No response generated."


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """Return the process-wide compiled graph.

    Compiling opens a SQLite connection and builds every node, so it is done
    once and shared rather than repeated on each interaction.
    """
    return build_graph()


def _thread_config(session_id: str) -> dict[str, Any]:
    """Return the LangGraph config that scopes checkpoints to one session."""
    return {"configurable": {"thread_id": session_id}}


@dataclass
class ChatService:
    """Runs turns against the graph on behalf of a session."""

    graph: CompiledStateGraph = field(default_factory=get_graph)

    def stream_answer(self, query: str, session_id: str) -> Iterator[str]:
        """Stream the answer tokens for one turn.

        Only tokens emitted by the answer node are yielded: the router, the
        grader and the rewriter also call the model, and surfacing their output
        would show the user the machinery instead of the answer.
        """
        for chunk, metadata in self.graph.stream(
            initial_state(query, session_id),
            _thread_config(session_id),
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != ANSWER_NODE:
                continue
            content = getattr(chunk, "content", None)
            if content:
                yield content

    def final_state(self, session_id: str) -> dict[str, Any]:
        """Return the persisted state after a turn, or an empty mapping."""
        snapshot = self.graph.get_state(_thread_config(session_id))
        return snapshot.values if snapshot else {}

    def answer_of_record(self, session_id: str) -> str:
        """Return the answer stored in state.

        Some routes assemble their reply in Python rather than streaming it
        from the model, so an empty token stream is normal, not an error.
        """
        return self.final_state(session_id).get("answer") or EMPTY_RESPONSE_FALLBACK

    def history(self, session_id: str) -> list[dict[str, str]]:
        """Return the session's conversation as role/content pairs.

        Read from the checkpointer rather than from interface state, so a
        restart or a second browser tab still shows the full conversation.
        """
        try:
            messages = self.final_state(session_id).get("messages", [])
        except Exception:  # a missing or corrupt thread is not fatal
            logger.exception("Could not read history for session %s", session_id)
            return []

        history: list[dict[str, str]] = []
        for message in messages:
            role = _role_of(message)
            if role is None:
                continue
            content = message.content
            history.append(
                {"role": role, "content": content if isinstance(content, str) else str(content)}
            )
        return history


def _role_of(message: Any) -> str | None:
    """Map a message class to a chat role, ignoring tool traffic."""
    name = type(message).__name__
    if name == "HumanMessage":
        return "user"
    if name in {"AIMessage", "AIMessageChunk"}:
        return "assistant"
    return None


def serialise_state(values: dict[str, Any]) -> dict[str, Any]:
    """Render graph state as JSON-safe data for the state inspector.

    Messages and documents are truncated: the inspector exists to show the path
    a turn took, and embedding whole papers in it would make it unreadable.
    """
    snapshot: dict[str, Any] = {}
    for key, value in values.items():
        if key == "messages":
            snapshot[key] = [
                {
                    "type": type(message).__name__,
                    "content": _truncate(message.content),
                }
                for message in (value or [])
            ]
        elif key == "retrieved_docs":
            snapshot[key] = [
                {"content": _truncate(document.page_content), "metadata": document.metadata}
                for document in (value or [])
                if isinstance(document, Document)
            ]
        else:
            snapshot[key] = value
    return snapshot


def _truncate(content: Any) -> str:
    """Shorten any content value to a display-sized string."""
    text = content if isinstance(content, str) else repr(content)
    return text[:SNAPSHOT_TEXT_LIMIT]
