"""The metric suite scored on every evaluation run.

Retrieval and generation are measured separately on purpose: a low faithfulness
score with healthy contextual recall points at the answer prompt, while the
reverse points at chunking or the retriever. One blended score would hide which.
"""

from __future__ import annotations

from paperpilot.config import get_settings

METRIC_THRESHOLD = 0.7


def build_metrics(threshold: float = METRIC_THRESHOLD) -> list:
    """Return the metrics used to score the system.

    Retrieval quality:
        contextual precision — are the retrieved chunks relevant?
        contextual recall — do they cover what the answer needs?
        contextual relevancy — do they fit the question and expected answer?

    Generation quality:
        answer relevancy — does the reply address the question?
        faithfulness — is the reply grounded in the retrieved context?
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    model = get_settings().chat_model
    return [
        ContextualPrecisionMetric(threshold=threshold, model=model),
        ContextualRecallMetric(threshold=threshold, model=model),
        ContextualRelevancyMetric(threshold=threshold, model=model),
        AnswerRelevancyMetric(threshold=threshold, model=model),
        FaithfulnessMetric(threshold=threshold, model=model),
    ]
