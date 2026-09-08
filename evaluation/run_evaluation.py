"""Command-line entry point for the evaluation pipeline.

Run with ``uv run paperpilot-eval`` (or ``uv run python -m
evaluation.run_evaluation``). Requires the ``evaluation`` extra and live API
credentials — this exercises the real pipeline end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from evaluation.goldens import Golden, load_or_generate
from evaluation.metrics import METRIC_THRESHOLD, build_metrics
from paperpilot.config import get_settings
from paperpilot.core.logging import configure_logging, get_logger
from paperpilot.graph import build_graph, initial_state
from paperpilot.ingestion import load_document
from paperpilot.retrieval import get_repository

logger = get_logger(__name__)

DEFAULT_DOCUMENT = Path("documents/Openclaw_Research_Report.pdf")
DEFAULT_GOLDENS = Path("goldens.json")
DEFAULT_RESULTS = Path("eval_results.json")
EVAL_CHECKPOINT_DB = Path("data/eval_checkpoints.db")

# Concurrency is throttled to stay inside provider rate limits; raising it makes
# runs fail with 429s rather than finish faster.
MAX_CONCURRENT = 3
THROTTLE_SECONDS = 5

# Nudges the router towards retrieval, which is what this suite measures.
QUERY_SUFFIX = " as per the report in knowledge base"


def _answer_question(graph, question: str, session_id: str) -> tuple[str, list[str]]:
    """Run one question through the graph and return its answer and context."""
    final_state = graph.invoke(
        initial_state(question, session_id),
        config={"configurable": {"thread_id": session_id}},
    )
    context = [document.page_content for document in (final_state.get("retrieved_docs") or [])]
    return final_state.get("answer") or "", context


def _build_test_cases(graph, goldens: list[Golden], document_path: Path) -> list:
    """Ingest the source document and answer every golden against it."""
    from deepeval.test_case import LLMTestCase

    documents = load_document(document_path)
    repository = get_repository()
    test_cases = []

    for index, golden in enumerate(goldens, start=1):
        # A fresh session per case keeps one answer's retrieval from priming the next.
        session_id = f"evaluation_session_{uuid4()}"
        repository.add_documents(documents, session_id)
        logger.info("Running case %d/%d", index, len(goldens))
        answer, context = _answer_question(graph, golden.input + QUERY_SUFFIX, session_id)
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=answer,
                expected_output=golden.expected_output,
                retrieval_context=context,
            )
        )
    return test_cases


def _summarise(results) -> list[dict]:
    """Flatten DeepEval results into a JSON-serialisable report."""
    return [
        {
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "metrics": [
                {
                    "name": metric.name,
                    "score": metric.score,
                    "passed": metric.success,
                    "reason": metric.reason,
                }
                for metric in test_result.metrics_data
            ],
        }
        for test_result in results.test_results
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PaperPilot's RAG quality.")
    parser.add_argument(
        "--document",
        type=Path,
        default=DEFAULT_DOCUMENT,
        help="Source document goldens are generated from.",
    )
    parser.add_argument(
        "--goldens", type=Path, default=DEFAULT_GOLDENS, help="Where generated goldens are cached."
    )
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS, help="Where the scored report is written."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=METRIC_THRESHOLD,
        help="Pass threshold applied to every metric.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Score the pipeline against the goldens and write a JSON report."""
    from deepeval import evaluate
    from deepeval.evaluate import AsyncConfig

    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    if not args.document.exists():
        logger.error("Source document not found: %s", args.document)
        return 1

    goldens = load_or_generate(args.document, args.goldens)
    logger.info("Evaluating %d golden(s)", len(goldens))

    graph = build_graph(db_path=EVAL_CHECKPOINT_DB)
    test_cases = _build_test_cases(graph, goldens, args.document)

    results = evaluate(
        test_cases,
        build_metrics(args.threshold),
        async_config=AsyncConfig(max_concurrent=MAX_CONCURRENT, throttle_value=THROTTLE_SECONDS),
    )

    summary = _summarise(results)
    args.results.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for item in summary if item["success"])
    logger.info("%d/%d case(s) passed. Report written to %s", passed, len(summary), args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
