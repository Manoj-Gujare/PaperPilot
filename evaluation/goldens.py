"""Golden test cases for the evaluation run.

Goldens are synthesised once from a source document and then cached: generation
costs many model calls, and regenerating them each run would make scores move
because the questions changed rather than because the system did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paperpilot.core.logging import get_logger

logger = get_logger(__name__)

MAX_CONTEXTS_PER_DOCUMENT = 5
GOLDENS_PER_CONTEXT = 2


@dataclass(frozen=True, slots=True)
class Golden:
    """One question with the answer a correct system should produce."""

    input: str
    expected_output: str


def load(path: Path) -> list[Golden]:
    """Read cached goldens from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Golden(input=item["input"], expected_output=item["expected_output"]) for item in payload
    ]


def save(goldens: list[Golden], path: Path) -> None:
    """Cache goldens so later runs evaluate the same questions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"input": g.input, "expected_output": g.expected_output} for g in goldens]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def generate(document_path: Path) -> list[Golden]:
    """Synthesise question/answer pairs from a source document."""
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import ContextConstructionConfig

    logger.info("Generating goldens from %s", document_path)
    generated = Synthesizer().generate_goldens_from_docs(
        document_paths=[str(document_path)],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS_PER_DOCUMENT,
        ),
    )
    return [
        Golden(input=item.input, expected_output=item.expected_output)
        for item in generated
        if item.input and item.expected_output
    ]


def load_or_generate(document_path: Path, cache_path: Path) -> list[Golden]:
    """Return cached goldens, synthesising and caching them on first run."""
    if cache_path.exists():
        logger.info("Reusing cached goldens from %s", cache_path)
        return load(cache_path)
    goldens = generate(document_path)
    save(goldens, cache_path)
    return goldens
