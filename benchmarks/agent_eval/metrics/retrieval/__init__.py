"""
Layer 1: Retrieval Metrics (RAGAS-only).

Evaluates the quality of context retrieval using standardized RAGAS metrics.
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_retrieval_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get all retrieval metrics with configured thresholds.

    Uses RAGAS-only metrics:
    - Context Precision: Are retrieved chunks relevant?
    - Context Recall: Did we retrieve all needed info? (requires ground truth)

    Args:
        config: Optional configuration for thresholds

    Returns:
        List of configured retrieval metrics
    """
    config = config or EvalConfig()
    thresholds = config.thresholds

    from ..ragas_metrics import ContextPrecisionMetric, ContextRecallMetric

    return [
        ContextPrecisionMetric(threshold=thresholds.contextual_precision_min),
        ContextRecallMetric(threshold=thresholds.contextual_recall_min),
    ]


__all__ = [
    "get_retrieval_metrics",
]
