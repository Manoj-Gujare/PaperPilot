"""Rewrite a query that failed to retrieve anything relevant."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from paperpilot.core.logging import get_logger
from paperpilot.graph.prompts import QUERY_REWRITE_SYSTEM
from paperpilot.graph.state import RAGState
from paperpilot.llm import get_chat_model

logger = get_logger(__name__)


def query_rewrite_node(state: RAGState) -> dict[str, Any]:
    """Produce a sharper query and restart retrieval with a clean slate.

    Documents and the attempt counter are reset so the rewritten query is
    graded on what *it* retrieves, rather than inheriting the chunks that were
    already judged irrelevant.
    """
    original_query = state["query"]
    try:
        response = get_chat_model().invoke(
            [
                {"role": "system", "content": QUERY_REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Original query: {original_query}\n\nWrite an improved query.",
                },
            ]
        )
        rewritten = response.content.strip() or original_query
    except Exception:  # fall back to retrying the original query
        logger.exception("Query rewrite failed; retrying the original query")
        rewritten = original_query

    logger.info("Rewrote query to %r", rewritten[:120])
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "is_relevant": None,
    }
