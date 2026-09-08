"""Route names and the conditional-edge functions that read them."""

from __future__ import annotations

from langgraph.prebuilt import tools_condition

from paperpilot.config import get_settings
from paperpilot.core.logging import get_logger
from paperpilot.graph.state import RAGState

logger = get_logger(__name__)

# Route names produced by the router node.
RETRIEVE_ROUTE = "retrieve"
VERIFY_CLAIM_ROUTE = "verify_claim"
DIRECT_ANSWER_ROUTE = "direct_answer"

# Node names, used as conditional-edge targets.
RETRIEVAL_NODE = "retrieval"
RELEVANCY_NODE = "relevancy_check"
QUERY_REWRITE_NODE = "query_rewrite"
GENERATE_ANSWER_NODE = "generate_answer"


def route_query(state: RAGState) -> str:
    """Send the turn down the branch the router chose."""
    return state.get("route") or RETRIEVE_ROUTE


def route_after_agent(state: RAGState) -> str:
    """Decide what follows the retrieval agent.

    Pending tool calls always win. Short-circuiting to the answer here would
    leave an AIMessage carrying tool_call ids that no ToolMessage answers; the
    checkpointer persists that, and every later turn in the session would then
    replay a malformed history to the provider.
    """
    if tools_condition(state) == "tools":
        return RETRIEVAL_NODE
    if state.get("retrieval_attempts", 0) >= get_settings().max_retrieval_attempts:
        logger.info("Retrieval budget exhausted; answering with what was gathered")
        return GENERATE_ANSWER_NODE
    return RELEVANCY_NODE


def route_after_relevancy(state: RAGState) -> str:
    """Answer if the context is good, otherwise rewrite while budget remains."""
    if state.get("is_relevant", False):
        return GENERATE_ANSWER_NODE
    if state.get("rewrite_count", 0) < get_settings().max_query_rewrites:
        return QUERY_REWRITE_NODE
    return GENERATE_ANSWER_NODE
