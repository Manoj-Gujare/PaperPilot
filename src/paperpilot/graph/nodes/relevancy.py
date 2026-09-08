"""Grade retrieved chunks before they are used to answer."""

from __future__ import annotations

from typing import Any

from paperpilot.core.logging import get_logger
from paperpilot.core.schemas import RelevancyDecision
from paperpilot.graph.prompts import RELEVANCY_SYSTEM
from paperpilot.graph.state import RAGState
from paperpilot.llm import get_structured_model

logger = get_logger(__name__)

MAX_GRADED_DOCS = 3
SNIPPET_LENGTH = 300


def relevancy_check_node(state: RAGState) -> dict[str, Any]:
    """Judge whether the retrieved context can answer the question.

    Only the first few chunks are graded, truncated to a snippet each: the
    grader is a cheap gate on whether retrieval worked at all, and feeding it
    the full context would cost as much as answering.
    """
    documents = state.get("retrieved_docs") or []
    if not documents:
        logger.info("Nothing retrieved; marking context irrelevant")
        return {"is_relevant": False}

    snippets = "\n\n---\n\n".join(
        document.page_content[:SNIPPET_LENGTH] for document in documents[:MAX_GRADED_DOCS]
    )
    prompt = (
        f"Question: {state['query']}\n\nRetrieved chunks:\n{snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )

    try:
        decision: RelevancyDecision = get_structured_model(RelevancyDecision).invoke(
            [
                {"role": "system", "content": RELEVANCY_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception:  # a grading failure must not end the turn
        # Treat the retrieved context as usable: answering from real chunks
        # beats discarding them because the grader was unavailable.
        logger.exception("Relevancy grading failed; accepting retrieved context")
        return {"is_relevant": True}

    logger.info("Relevancy verdict: %s (%s)", decision.is_relevant, decision.reason[:120])
    return {"is_relevant": decision.is_relevant}
