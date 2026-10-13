"""The state passed between graph nodes."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class RAGState(MessagesState):
    """Everything a turn accumulates as it moves through the graph.

    Extends ``MessagesState``, so ``messages`` is appended to rather than
    replaced, and the conversation survives in the checkpointer across turns.
    """

    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict[str, Any]] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int


def initial_state(query: str, session_id: str) -> dict[str, Any]:
    """Build the starting state for a turn.

    Counters and per-turn findings are reset explicitly. Carrying a previous
    turn's ``retrieved_docs`` or ``retrieval_attempts`` forward would let one
    question answer the next from stale context, or exhaust the retry budget
    before the new question has been tried once.
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "query": query,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
    }
