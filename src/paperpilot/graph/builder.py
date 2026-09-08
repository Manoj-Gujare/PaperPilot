"""Assembly of the LangGraph workflow.

```
query ─► router ─┬─ direct_answer ────────────────────────► generate_answer
                 │
                 ├─ retrieve ─► agent ─► tools ─► agent ─► relevancy_check
                 │                 ▲                    │        │
                 │                 └── query_rewrite ◄──┘        │
                 │                                               ▼
                 └─ verify_claim ─────────────────────────► generate_answer
```
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from paperpilot.config import get_settings
from paperpilot.core.logging import get_logger
from paperpilot.graph.nodes import (
    agent_node,
    generate_answer_node,
    query_rewrite_node,
    relevancy_check_node,
    router_node,
    verify_claim_node,
)
from paperpilot.graph.routing import (
    DIRECT_ANSWER_ROUTE,
    GENERATE_ANSWER_NODE,
    QUERY_REWRITE_NODE,
    RELEVANCY_NODE,
    RETRIEVAL_NODE,
    RETRIEVE_ROUTE,
    VERIFY_CLAIM_ROUTE,
    route_after_agent,
    route_after_relevancy,
    route_query,
)
from paperpilot.graph.state import RAGState
from paperpilot.graph.tools import RETRIEVAL_TOOLS

logger = get_logger(__name__)

ROUTER_NODE = "router"
AGENT_NODE = "agent"


def build_graph(db_path: str | Path | None = None) -> CompiledStateGraph:
    """Compile the workflow with a SQLite checkpointer.

    ``check_same_thread=False`` is required because Streamlit serves reruns from
    a pool of threads while the compiled graph is cached and shared across them.
    """
    settings = get_settings()
    resolved_path = Path(db_path) if db_path is not None else settings.checkpoint_db_path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(resolved_path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    logger.info("Compiling graph with checkpoints at %s", resolved_path)

    graph = StateGraph(RAGState)
    graph.add_node(ROUTER_NODE, router_node)
    graph.add_node(AGENT_NODE, agent_node)
    graph.add_node(RETRIEVAL_NODE, ToolNode(RETRIEVAL_TOOLS))
    graph.add_node(RELEVANCY_NODE, relevancy_check_node)
    graph.add_node(QUERY_REWRITE_NODE, query_rewrite_node)
    graph.add_node(VERIFY_CLAIM_ROUTE, verify_claim_node)
    graph.add_node(GENERATE_ANSWER_NODE, generate_answer_node)

    graph.set_entry_point(ROUTER_NODE)
    graph.add_conditional_edges(
        ROUTER_NODE,
        route_query,
        {
            RETRIEVE_ROUTE: AGENT_NODE,
            VERIFY_CLAIM_ROUTE: VERIFY_CLAIM_ROUTE,
            DIRECT_ANSWER_ROUTE: GENERATE_ANSWER_NODE,
        },
    )
    graph.add_conditional_edges(
        AGENT_NODE,
        route_after_agent,
        {
            RETRIEVAL_NODE: RETRIEVAL_NODE,
            RELEVANCY_NODE: RELEVANCY_NODE,
            GENERATE_ANSWER_NODE: GENERATE_ANSWER_NODE,
        },
    )
    graph.add_edge(RETRIEVAL_NODE, AGENT_NODE)
    graph.add_conditional_edges(
        RELEVANCY_NODE,
        route_after_relevancy,
        {
            QUERY_REWRITE_NODE: QUERY_REWRITE_NODE,
            GENERATE_ANSWER_NODE: GENERATE_ANSWER_NODE,
        },
    )
    graph.add_edge(QUERY_REWRITE_NODE, AGENT_NODE)
    graph.add_edge(VERIFY_CLAIM_ROUTE, GENERATE_ANSWER_NODE)
    graph.add_edge(GENERATE_ANSWER_NODE, END)

    return graph.compile(checkpointer=checkpointer)
