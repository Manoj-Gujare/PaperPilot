"""Retrieval agent: chooses which tool to call, and with what arguments."""

from __future__ import annotations

from typing import Any

from paperpilot.config import get_settings
from paperpilot.core.logging import get_logger
from paperpilot.graph.prompts import RETRIEVAL_AGENT_SYSTEM
from paperpilot.graph.state import RAGState
from paperpilot.graph.tools import RETRIEVAL_TOOLS
from paperpilot.llm import get_chat_model

logger = get_logger(__name__)


def agent_node(state: RAGState) -> dict[str, Any]:
    """Let the model gather context, until the attempt budget is spent.

    At the cap the model is invoked *without* tools bound. Simply routing away
    instead would let it emit one last set of tool calls that nothing answers,
    leaving an AIMessage with unmatched tool_call ids in the checkpointer and
    corrupting every later turn in the session.
    """
    settings = get_settings()
    attempts = state.get("retrieval_attempts", 0)

    model = get_chat_model()
    if attempts < settings.max_retrieval_attempts:
        model = model.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
    else:
        logger.info("Retrieval budget of %d spent; answering without tools", attempts)

    messages = [{"role": "system", "content": RETRIEVAL_AGENT_SYSTEM}, *state["messages"]]
    response = model.invoke(messages)

    updates: dict[str, Any] = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = attempts + 1
    return updates
