"""
Layer 4: Generation Metrics.

Evaluates the quality of the final generated answer.
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_generation_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get all generation metrics with configured thresholds.

    Args:
        config: Optional configuration for thresholds

    Returns:
        List of configured generation metrics
    """
    config = config or EvalConfig()
    thresholds = config.thresholds

    from .ragas_metrics import FaithfulnessMetric, AnswerRelevanceMetric
    from .citation_recall import CitationRecallMetric

    return [
        FaithfulnessMetric(threshold=thresholds.faithfulness_min),
        AnswerRelevanceMetric(threshold=thresholds.answer_relevance_min),
        CitationRecallMetric(
            threshold=thresholds.citation_recall_min,
            judge_model=config.judge_model,
        ),
    ]


__all__ = [
    "get_generation_metrics",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "CitationRecallMetric",
]
