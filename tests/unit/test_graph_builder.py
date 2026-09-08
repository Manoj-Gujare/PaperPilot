"""Tests that the workflow compiles with the wiring we expect."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from paperpilot.graph import build_graph, initial_state

EXPECTED_NODES = {
    "router",
    "agent",
    "retrieval",
    "relevancy_check",
    "query_rewrite",
    "verify_claim",
    "generate_answer",
}


def test_graph_compiles_with_every_node(tmp_path):
    graph = build_graph(db_path=tmp_path / "checkpoints.db")
    assert set(graph.get_graph().nodes) >= EXPECTED_NODES


def test_checkpoint_database_is_created(tmp_path):
    db_path = tmp_path / "nested" / "checkpoints.db"
    build_graph(db_path=db_path)
    assert db_path.parent.is_dir()


def test_build_defaults_to_the_configured_path(settings):
    build_graph()
    assert settings.checkpoint_db_path.parent.is_dir()


def test_initial_state_seeds_the_query_as_a_human_message():
    state = initial_state("What is attention?", "session-1")
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["query"] == "What is attention?"
    assert state["session_id"] == "session-1"


def test_initial_state_resets_per_turn_counters():
    """Carrying counters across turns would spend the retry budget before the new question."""
    state = initial_state("q", "session-1")
    assert state["retrieval_attempts"] == 0
    assert state["rewrite_count"] == 0
    assert state["retrieved_docs"] == []
    assert state["is_relevant"] is None
    assert state["answer"] is None
