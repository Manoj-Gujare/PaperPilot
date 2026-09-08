"""Tests for individual graph nodes."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from paperpilot.core.exceptions import WebSearchError
from paperpilot.core.schemas import (
    ClaimVerificationResult,
    RelevancyDecision,
    RouterDecision,
    SupersedingPaper,
)
from paperpilot.graph.nodes import answer as answer_node
from paperpilot.graph.nodes import relevancy, rewrite, router, verification
from paperpilot.retrieval.web_search import WebSearchResult


def structured(value):
    """Return a stand-in for get_structured_model that yields ``value``."""
    return lambda schema: type("Model", (), {"invoke": lambda self, messages: value})()


def failing(exception):
    """Return a factory that raises when the node tries to call a model."""

    def _factory(*args, **kwargs):
        raise exception

    return _factory


def state(**overrides):
    base = {
        "messages": [HumanMessage(content="What does the paper claim?")],
        "query": "What does the paper claim?",
        "route": "retrieve",
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": 0,
        "is_relevant": None,
    }
    base.update(overrides)
    return base


# ── Router ────────────────────────────────────────────────────────────────────


def test_router_returns_the_models_route(monkeypatch):
    monkeypatch.setattr(
        router, "get_structured_model", structured(RouterDecision(route="verify_claim"))
    )
    assert router.router_node(state())["route"] == "verify_claim"


def test_router_falls_back_to_retrieval_on_failure(monkeypatch):
    """A direct answer would invent content the user reads as coming from their papers."""
    monkeypatch.setattr(router, "get_structured_model", failing(RuntimeError("down")))
    assert router.router_node(state())["route"] == "retrieve"


# ── Relevancy ─────────────────────────────────────────────────────────────────


def test_empty_retrieval_is_irrelevant_without_calling_the_model(monkeypatch):
    monkeypatch.setattr(relevancy, "get_structured_model", failing(AssertionError("not called")))
    assert relevancy.relevancy_check_node(state())["is_relevant"] is False


def test_relevancy_uses_the_models_verdict(monkeypatch):
    monkeypatch.setattr(
        relevancy,
        "get_structured_model",
        structured(RelevancyDecision(is_relevant=False, reason="off topic")),
    )
    result = relevancy.relevancy_check_node(state(retrieved_docs=[Document(page_content="x")]))
    assert result["is_relevant"] is False


def test_grading_failure_accepts_the_retrieved_context(monkeypatch):
    monkeypatch.setattr(relevancy, "get_structured_model", failing(RuntimeError("down")))
    result = relevancy.relevancy_check_node(state(retrieved_docs=[Document(page_content="x")]))
    assert result["is_relevant"] is True


def test_only_the_first_chunks_are_graded(monkeypatch):
    captured = {}

    def capture(schema):
        def invoke(self, messages):
            captured["prompt"] = messages[1]["content"]
            return RelevancyDecision(is_relevant=True, reason="ok")

        return type("Model", (), {"invoke": invoke})()

    monkeypatch.setattr(relevancy, "get_structured_model", capture)
    documents = [Document(page_content=f"chunk-{i}") for i in range(10)]
    relevancy.relevancy_check_node(state(retrieved_docs=documents))
    assert "chunk-0" in captured["prompt"]
    assert "chunk-9" not in captured["prompt"]


# ── Rewrite ───────────────────────────────────────────────────────────────────


def test_rewrite_resets_retrieval_progress(monkeypatch):
    monkeypatch.setattr(
        rewrite,
        "get_chat_model",
        lambda: type(
            "M", (), {"invoke": lambda self, m: type("R", (), {"content": " better query "})()}
        )(),
    )
    result = rewrite.query_rewrite_node(
        state(retrieved_docs=[Document(page_content="stale")], retrieval_attempts=2)
    )
    assert result["query"] == "better query"
    assert result["retrieved_docs"] == []
    assert result["retrieval_attempts"] == 0
    assert result["rewrite_count"] == 1
    assert result["is_relevant"] is None


def test_rewrite_failure_retries_the_original_query(monkeypatch):
    monkeypatch.setattr(rewrite, "get_chat_model", failing(RuntimeError("down")))
    result = rewrite.query_rewrite_node(state())
    assert result["query"] == "What does the paper claim?"


# ── Verification ──────────────────────────────────────────────────────────────


def fake_search_client(results):
    return lambda: type("C", (), {"search": lambda self, q, max_results=None: results})()


def test_verification_returns_the_verdict_and_papers(monkeypatch):
    monkeypatch.setattr(
        verification,
        "get_web_search_client",
        fake_search_client([WebSearchResult("T", "https://arxiv.org/abs/1", "body")]),
    )
    monkeypatch.setattr(
        verification,
        "get_structured_model",
        structured(
            ClaimVerificationResult(
                is_superseded=True,
                verdict_summary="Superseded.",
                superseding_papers=[
                    SupersedingPaper(title="T", url="https://arxiv.org/abs/1", summary="S")
                ],
            )
        ),
    )
    result = verification.verify_claim_node(state())
    assert result["claim_verdict"] == "Superseded."
    assert result["claim_source"] == "https://arxiv.org/abs/1"


def test_verification_caps_the_number_of_papers(monkeypatch, settings):
    papers = [
        SupersedingPaper(title=f"T{i}", url=f"https://arxiv.org/abs/{i}", summary="S")
        for i in range(10)
    ]
    monkeypatch.setattr(verification, "get_web_search_client", fake_search_client([]))
    monkeypatch.setattr(
        verification,
        "get_structured_model",
        structured(
            ClaimVerificationResult(
                is_superseded=True, verdict_summary="v", superseding_papers=papers
            )
        ),
    )
    result = verification.verify_claim_node(state())
    assert len(result["superseding_papers"]) == settings.max_superseding_papers


def test_verification_survives_a_search_outage(monkeypatch):
    class Failing:
        def search(self, query, max_results=None):
            raise WebSearchError("down")

    monkeypatch.setattr(verification, "get_web_search_client", lambda: Failing())
    result = verification.verify_claim_node(state())
    assert result["superseding_papers"] == []
    assert "could not be checked" in result["claim_verdict"]


# ── Answer ────────────────────────────────────────────────────────────────────


def test_answer_is_grounded_in_retrieved_context(monkeypatch):
    captured = {}

    class FakeModel:
        def invoke(self, messages):
            captured["prompt"] = messages[0]["content"]
            return type("R", (), {"content": "grounded answer"})()

    monkeypatch.setattr(answer_node, "get_chat_model", lambda: FakeModel())
    result = answer_node.generate_answer_node(
        state(is_relevant=True, retrieved_docs=[Document(page_content="paper says X")])
    )
    assert result["answer"] == "grounded answer"
    assert "paper says X" in captured["prompt"]


def test_answer_explains_when_nothing_relevant_was_found(monkeypatch, settings):
    monkeypatch.setattr(answer_node, "get_chat_model", failing(AssertionError("not called")))
    result = answer_node.generate_answer_node(
        state(is_relevant=False, rewrite_count=settings.max_query_rewrites)
    )
    assert result["answer"] == answer_node.NO_RELEVANT_CONTEXT_ANSWER


def test_answer_without_context_says_so(monkeypatch):
    monkeypatch.setattr(answer_node, "get_chat_model", failing(AssertionError("not called")))
    result = answer_node.generate_answer_node(state(is_relevant=True))
    assert result["answer"] == answer_node.NO_CONTEXT_ANSWER


def test_verification_answer_lists_superseding_papers():
    result = answer_node.generate_answer_node(
        state(
            route="verify_claim",
            claim_verdict="Superseded.",
            superseding_papers=[{"title": "T", "url": "https://a", "summary": "S"}],
        )
    )
    assert "Superseding Papers" in result["answer"]
    assert "https://a" in result["answer"]


def test_verification_answer_without_papers_says_the_claim_holds():
    result = answer_node.generate_answer_node(
        state(route="verify_claim", claim_verdict="Still holds.", superseding_papers=[])
    )
    assert "No papers directly superseding" in result["answer"]


def test_generation_failure_still_returns_a_reply(monkeypatch):
    monkeypatch.setattr(answer_node, "get_chat_model", failing(RuntimeError("down")))
    result = answer_node.generate_answer_node(state(route="direct_answer"))
    assert result["answer"] == answer_node.GENERATION_FAILED_ANSWER
    assert result["messages"]
