"""Check whether a claim from a paper has been superseded by newer work."""

from __future__ import annotations

from typing import Any

from paperpilot.config import get_settings
from paperpilot.core.exceptions import WebSearchError
from paperpilot.core.logging import get_logger
from paperpilot.core.schemas import ClaimVerificationResult
from paperpilot.graph.prompts import CLAIM_VERIFICATION_SYSTEM
from paperpilot.graph.state import RAGState
from paperpilot.llm import get_structured_model
from paperpilot.retrieval.web_search import WebSearchResult, get_web_search_client

logger = get_logger(__name__)

CLAIM_EXCERPT_LENGTH = 200
SNIPPET_LENGTH = 300
UNVERIFIABLE_VERDICT = (
    "Could not reach the search provider, so this claim could not be checked "
    "against recent literature. Please try again."
)


def _format_results(heading: str, results: list[WebSearchResult]) -> list[str]:
    """Render results as a titled block the model can quote verbatim from."""
    lines = [f"=== {heading} ==="]
    lines.extend(
        f"Title: {result.title}\nURL: {result.url}\nSnippet: {result.content[:SNIPPET_LENGTH]}\n"
        for result in results
    )
    return lines


def verify_claim_node(state: RAGState) -> dict[str, Any]:
    """Search for superseding work and summarise the verdict.

    Two searches are issued rather than one. A general search surfaces blog
    posts, retractions and news; an arxiv.org-restricted search surfaces the
    academic work that actually supersedes a finding. Either search alone
    reliably misses one of those two categories.
    """
    settings = get_settings()
    claim = state["messages"][-1].content
    excerpt = claim[:CLAIM_EXCERPT_LENGTH]
    client = get_web_search_client()

    try:
        general = client.search(
            f"recent research superseding: {excerpt}",
            max_results=settings.verification_max_results,
        )
        academic = client.search(
            f"site:arxiv.org {excerpt}",
            max_results=settings.verification_max_results,
        )
    except WebSearchError as exc:
        logger.warning("Claim verification search failed: %s", exc)
        return {
            "claim_verdict": UNVERIFIABLE_VERDICT,
            "claim_source": None,
            "superseding_papers": [],
        }

    context = "\n".join(
        [
            *_format_results("General Web Search Results", general),
            *_format_results("arXiv Paper Search Results", academic),
        ]
    )
    prompt = (
        f"{CLAIM_VERIFICATION_SYSTEM}\n\nClaim to verify:\n{claim}\n\nSearch Results:\n{context}"
    )

    try:
        result: ClaimVerificationResult = get_structured_model(ClaimVerificationResult).invoke(
            [{"role": "user", "content": prompt}]
        )
    except Exception:  # a provider error must not end the turn
        logger.exception("Claim analysis failed")
        return {
            "claim_verdict": UNVERIFIABLE_VERDICT,
            "claim_source": None,
            "superseding_papers": [],
        }

    papers = [
        paper.model_dump() for paper in result.superseding_papers[: settings.max_superseding_papers]
    ]
    logger.info("Claim verified: superseded=%s, %d paper(s)", result.is_superseded, len(papers))
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers[0]["url"] if papers else None,
        "superseding_papers": papers,
    }
