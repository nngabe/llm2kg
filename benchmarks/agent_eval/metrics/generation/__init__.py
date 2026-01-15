"""
Layer 4: Generation Metrics (RAGAS-only).

Evaluates the quality of the final generated answer using standardized RAGAS metrics.
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_generation_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get all generation metrics with configured thresholds.

    Uses RAGAS-only metrics:
    - Faithfulness: Is answer grounded in context? (no hallucination)
    - Answer Relevancy: Does answer address the question?
    - Answer Correctness: Is answer factually correct? (requires ground truth)
    - Factual Correctness: Claim-level verification (requires ground truth)

    Args:
        config: Optional configuration for thresholds

    Returns:
        List of configured generation metrics
    """
    config = config or EvalConfig()
    thresholds = config.thresholds

    from ..ragas_metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        AnswerCorrectnessMetric,
        FactualCorrectnessMetric,
    )

    return [
        FaithfulnessMetric(threshold=thresholds.faithfulness_min),
        AnswerRelevancyMetric(threshold=thresholds.answer_relevance_min),
        AnswerCorrectnessMetric(threshold=0.60),
        FactualCorrectnessMetric(threshold=0.60),
    ]


__all__ = [
    "get_generation_metrics",
]
