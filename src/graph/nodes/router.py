"""Entry node: decide which branch of the graph handles a query."""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.core.schemas import RouterDecision
from src.graph.prompts import ROUTER_SYSTEM
from src.graph.state import RAGState
from src.llm import get_structured_model

logger = get_logger(__name__)

DEFAULT_ROUTE = "retrieve"


def router_node(state: RAGState) -> dict[str, Any]:
    """Classify the latest message as retrieval, verification or direct answer.

    A classification failure falls back to retrieval: answering from paper
    context is the safe default, whereas defaulting to a direct answer would
    invent content the user believes came from their papers.
    """
    query = state["messages"][-1].content
    try:
        decision: RouterDecision = get_structured_model(RouterDecision).invoke(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ]
        )
        route = decision.route
    except Exception:  # any provider error must not end the turn
        logger.exception("Routing failed; falling back to %s", DEFAULT_ROUTE)
        route = DEFAULT_ROUTE

    logger.info("Routed query to %s", route)
    return {"route": route}
