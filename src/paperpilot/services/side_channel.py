"""The `/btw` side channel for questions unrelated to the loaded papers.

Deliberately isolated from the graph: no vector store, no checkpointer, nothing
written to session history. Someone asking a passing question mid-review should
not have that question become context the assistant reasons from for the rest
of the conversation.
"""

from __future__ import annotations

from collections.abc import Iterator

from paperpilot.config import get_settings
from paperpilot.core.exceptions import WebSearchError
from paperpilot.core.logging import get_logger
from paperpilot.core.schemas import SideChannelDecision
from paperpilot.graph.prompts import (
    SIDE_CHANNEL_DIRECT_SYSTEM,
    SIDE_CHANNEL_ROUTER_SYSTEM,
    SIDE_CHANNEL_WEB_SYSTEM,
)
from paperpilot.llm import get_chat_model, get_structured_model
from paperpilot.retrieval.web_search import get_web_search_client

logger = get_logger(__name__)

COMMAND_PREFIX = "/btw"


def is_side_channel(message: str) -> bool:
    """Return True if a message is addressed to the side channel."""
    return message.strip().lower().startswith(COMMAND_PREFIX)


def strip_command(message: str) -> str:
    """Return the question with the `/btw` prefix removed."""
    return message.strip()[len(COMMAND_PREFIX) :].strip()


def _needs_web_search(question: str) -> bool:
    """Ask the model whether general knowledge is enough to answer."""
    try:
        decision: SideChannelDecision = get_structured_model(SideChannelDecision).invoke(
            [
                {"role": "system", "content": SIDE_CHANNEL_ROUTER_SYSTEM},
                {"role": "user", "content": question},
            ]
        )
        return decision.needs_web_search
    except Exception:  # a routing failure should not block the answer
        logger.exception("Side-channel routing failed; answering without search")
        return False


def _web_system_prompt(question: str) -> str | None:
    """Return a prompt carrying web results, or None if search yielded nothing."""
    settings = get_settings()
    try:
        results = get_web_search_client().search(
            question, max_results=settings.web_search_max_results
        )
    except WebSearchError as exc:
        logger.warning("Side-channel search failed: %s", exc)
        return None
    if not results:
        return None
    return SIDE_CHANNEL_WEB_SYSTEM.format(
        context="\n\n".join(result.content for result in results),
        sources="\n".join(f"- {result.url}" for result in results),
    )


def answer(question: str) -> Iterator[str]:
    """Stream an answer to an off-topic question.

    Yields text chunks so the interface can render tokens as they arrive.
    """
    system_prompt: str | None = None
    if _needs_web_search(question):
        system_prompt = _web_system_prompt(question)
    if system_prompt is None:
        system_prompt = SIDE_CHANNEL_DIRECT_SYSTEM

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    for chunk in get_chat_model().stream(messages):
        if chunk.content:
            yield chunk.content
