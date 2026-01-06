"""
Layer 3: Graph Integrity Metrics.

Evaluates the quality and correctness of graph updates.
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_integrity_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get all integrity metrics with configured thresholds.

    Args:
        config: Optional configuration for thresholds

    Returns:
        List of configured integrity metrics
    """
    config = config or EvalConfig()
    thresholds = config.thresholds

    from .schema_adherence import SchemaAdherenceMetric
    from .disambiguation import EntityDisambiguationMetric
    from .citation_accuracy import SourceCitationAccuracyMetric

    return [
        SchemaAdherenceMetric(threshold=thresholds.schema_adherence_min),
        EntityDisambiguationMetric(threshold=thresholds.entity_disambiguation_min),
        SourceCitationAccuracyMetric(
            threshold=thresholds.source_citation_accuracy_min,
            judge_model=config.judge_model,
        ),
    ]


__all__ = [
    "get_integrity_metrics",
    "SchemaAdherenceMetric",
    "EntityDisambiguationMetric",
    "SourceCitationAccuracyMetric",
]
