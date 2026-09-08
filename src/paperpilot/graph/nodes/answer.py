"""Terminal node: turn whatever the graph gathered into the user's answer."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from paperpilot.config import get_settings
from paperpilot.core.logging import get_logger
from paperpilot.graph.prompts import (
    DIRECT_ANSWER_TEMPLATE,
    GROUNDED_ANSWER_TEMPLATE,
    NO_CONTEXT_ANSWER,
    NO_RELEVANT_CONTEXT_ANSWER,
)
from paperpilot.graph.routing import RETRIEVE_ROUTE, VERIFY_CLAIM_ROUTE
from paperpilot.graph.state import RAGState
from paperpilot.llm import get_chat_model

logger = get_logger(__name__)

CONTEXT_SEPARATOR = "\n\n---\n\n"
GENERATION_FAILED_ANSWER = (
    "Something went wrong while generating the answer. Please try asking again."
)


def _answer_from_context(state: RAGState) -> str:
    """Answer grounded in retrieved chunks, or explain why that was not possible."""
    settings = get_settings()
    exhausted_rewrites = state.get("rewrite_count", 0) >= settings.max_query_rewrites
    if state.get("is_relevant") is False and exhausted_rewrites:
        return NO_RELEVANT_CONTEXT_ANSWER

    documents = state.get("retrieved_docs") or []
    if not documents:
        return NO_CONTEXT_ANSWER

    context = CONTEXT_SEPARATOR.join(document.page_content for document in documents)
    prompt = GROUNDED_ANSWER_TEMPLATE.format(context=context, query=state["query"])
    return get_chat_model().invoke([{"role": "user", "content": prompt}]).content


def _format_verification(state: RAGState) -> str:
    """Render the claim verdict and any superseding papers as Markdown."""
    verdict = state.get("claim_verdict", "")
    papers = state.get("superseding_papers") or []
    header = f"**Claim Verification Result**\n\n> {state['query']}\n\n**Verdict:** {verdict}\n\n"

    if not papers:
        return header + "*No papers directly superseding this claim were found.*"

    listing = "\n\n".join(
        f"{index}. **{paper['title']}**\n   {paper['summary']}\n   Link: {paper['url']}"
        for index, paper in enumerate(papers, start=1)
    )
    return (
        f"{header}**Superseding Papers:**\n\n{listing}\n\n"
        "---\n"
        "*You can load any of these papers into your knowledge base to continue "
        "your research with the latest findings.*"
    )


def generate_answer_node(state: RAGState) -> dict[str, Any]:
    """Produce the final answer for whichever route the turn took."""
    route = state.get("route")
    try:
        if route == RETRIEVE_ROUTE:
            answer = _answer_from_context(state)
        elif route == VERIFY_CLAIM_ROUTE:
            answer = _format_verification(state)
        else:
            prompt = DIRECT_ANSWER_TEMPLATE.format(query=state["query"])
            answer = get_chat_model().invoke([{"role": "user", "content": prompt}]).content
    except Exception:  # the user must always receive a reply
        logger.exception("Answer generation failed for route %s", route)
        answer = GENERATION_FAILED_ANSWER

    return {"answer": answer, "messages": [AIMessage(content=answer)]}
