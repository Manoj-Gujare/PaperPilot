"""Tools the retrieval agent can call.

Each tool returns a ToolMessage plus a Command updating ``retrieved_docs``. The
ToolMessage keeps the message history well-formed — every tool call must be
answered — while the Command carries the documents themselves, so full chunks
never enter the conversation transcript and inflate every subsequent prompt.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from paperpilot.core.exceptions import VectorStoreError, WebSearchError
from paperpilot.core.logging import get_logger
from paperpilot.retrieval import get_repository
from paperpilot.retrieval.web_search import get_web_search_client

logger = get_logger(__name__)


class RetrieverInput(BaseModel):
    """Arguments for a search over the uploaded papers."""

    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    """Arguments for a live web search."""

    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of results to return")


@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    try:
        documents = get_repository().search(query=query, session_id=session_id, k=k)
    except VectorStoreError as exc:
        logger.warning("Vector search failed: %s", exc)
        return [ToolMessage(content=f"Vector store unavailable: {exc}", tool_call_id=tool_call_id)]

    if not documents:
        return [
            ToolMessage(
                content="No relevant documents found in the vector store.",
                tool_call_id=tool_call_id,
            )
        ]
    return [
        ToolMessage(
            content=f"Retrieved {len(documents)} chunk(s) from the vector store.",
            tool_call_id=tool_call_id,
        ),
        Command(update={"retrieved_docs": (current_docs or []) + documents}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information."""
    try:
        results = get_web_search_client().search(optimized_query, max_results=max_results)
    except WebSearchError as exc:
        logger.warning("Web search failed: %s", exc)
        return [ToolMessage(content=f"Web search unavailable: {exc}", tool_call_id=tool_call_id)]

    if not results:
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]

    documents = [result.to_document() for result in results]
    return [
        ToolMessage(
            content=f"Found {len(documents)} web result(s) for: {optimized_query}",
            tool_call_id=tool_call_id,
        ),
        Command(update={"retrieved_docs": (current_docs or []) + documents}),
    ]


RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
