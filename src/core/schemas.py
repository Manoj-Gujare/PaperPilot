"""Structured output schemas shared across the application.

These models are handed to the LLM through ``with_structured_output``, so their
field names and docstrings are part of the prompt: keep them descriptive.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["retrieve", "verify_claim", "direct_answer"]


class RouterDecision(BaseModel):
    """Which branch of the graph should handle a user query."""

    route: Route = Field(description="The handler chosen for this query.")


class RelevancyDecision(BaseModel):
    """Whether retrieved chunks are good enough to answer the question."""

    is_relevant: bool = Field(description="True if the chunks address the question.")
    reason: str = Field(description="Short justification for the verdict.")


class SideChannelDecision(BaseModel):
    """Whether an off-topic `/btw` question needs live web results."""

    needs_web_search: bool = Field(description="True if general knowledge is insufficient.")


class SupersedingPaper(BaseModel):
    """A paper that updates, challenges or replaces a claim."""

    title: str = Field(description="Title exactly as it appears in the search results.")
    url: str = Field(description="Link to the paper, preferring arxiv.org.")
    summary: str = Field(description="One sentence on how it supersedes the claim.")


class ClaimVerificationResult(BaseModel):
    """Verdict on whether a claim from a paper still holds."""

    is_superseded: bool = Field(description="True if newer work supersedes the claim.")
    verdict_summary: str = Field(description="One or two sentences shown to the user.")
    superseding_papers: list[SupersedingPaper] = Field(
        default_factory=list,
        description="Papers that supersede the claim; empty when it still holds.",
    )
