"""Tests for the structured-output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperpilot.core.schemas import (
    ClaimVerificationResult,
    RelevancyDecision,
    RouterDecision,
    SupersedingPaper,
)


@pytest.mark.parametrize("route", ["retrieve", "verify_claim", "direct_answer"])
def test_router_accepts_every_supported_route(route):
    assert RouterDecision(route=route).route == route


def test_router_rejects_unknown_route():
    with pytest.raises(ValidationError):
        RouterDecision(route="summarise")


def test_claim_verification_defaults_to_no_superseding_papers():
    result = ClaimVerificationResult(is_superseded=False, verdict_summary="Still holds.")
    assert result.superseding_papers == []


def test_claim_verification_round_trips_papers():
    paper = SupersedingPaper(title="T", url="https://arxiv.org/abs/1", summary="S")
    result = ClaimVerificationResult(
        is_superseded=True, verdict_summary="Superseded.", superseding_papers=[paper]
    )
    assert result.model_dump()["superseding_papers"][0]["url"] == "https://arxiv.org/abs/1"


def test_relevancy_decision_requires_a_reason():
    with pytest.raises(ValidationError):
        RelevancyDecision(is_relevant=True)
