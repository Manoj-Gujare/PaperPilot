"""Tests for the graph's conditional edges."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from paperpilot.graph import routing


def state(**overrides):
    base = {
        "messages": [HumanMessage(content="q")],
        "query": "q",
        "route": "retrieve",
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": 0,
        "is_relevant": None,
    }
    base.update(overrides)
    return base


def with_tool_call():
    message = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {}, "id": "call-1"}],
    )
    return state(messages=[HumanMessage(content="q"), message])


@pytest.mark.parametrize("route", ["retrieve", "verify_claim", "direct_answer"])
def test_route_query_follows_the_router(route):
    assert routing.route_query(state(route=route)) == route


def test_route_query_defaults_to_retrieval_when_unset():
    assert routing.route_query(state(route=None)) == routing.RETRIEVE_ROUTE


def test_pending_tool_calls_always_run_first():
    """Skipping them would persist tool_call ids no ToolMessage answers."""
    exhausted = with_tool_call()
    exhausted["retrieval_attempts"] = 99
    assert routing.route_after_agent(exhausted) == routing.RETRIEVAL_NODE


def test_agent_moves_to_grading_while_budget_remains():
    assert routing.route_after_agent(state()) == routing.RELEVANCY_NODE


def test_agent_answers_once_the_budget_is_spent(settings):
    spent = state(retrieval_attempts=settings.max_retrieval_attempts)
    assert routing.route_after_agent(spent) == routing.GENERATE_ANSWER_NODE


def test_relevant_context_goes_straight_to_the_answer():
    assert routing.route_after_relevancy(state(is_relevant=True)) == routing.GENERATE_ANSWER_NODE


def test_irrelevant_context_triggers_one_rewrite():
    assert routing.route_after_relevancy(state(is_relevant=False)) == routing.QUERY_REWRITE_NODE


def test_rewrite_budget_is_not_exceeded(settings):
    exhausted = state(is_relevant=False, rewrite_count=settings.max_query_rewrites)
    assert routing.route_after_relevancy(exhausted) == routing.GENERATE_ANSWER_NODE
