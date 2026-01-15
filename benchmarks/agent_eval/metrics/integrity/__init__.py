"""
Layer 3: Graph Integrity Metrics.

NOTE: This layer has been DISABLED in RAGAS-only mode.
All integrity metrics used LLM-as-judge which has been removed.

Previously included:
- SchemaAdherenceMetric (formula + LLM)
- EntityDisambiguationMetric (embeddings + LLM-as-judge)
- SourceCitationAccuracyMetric (LLM-as-judge)
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_integrity_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get integrity metrics.

    NOTE: Returns empty list in RAGAS-only mode.
    All integrity metrics used LLM-as-judge which has been removed.

    Args:
        config: Optional configuration for thresholds

    Returns:
        Empty list (integrity layer disabled)
    """
    # All integrity metrics used LLM-as-judge - disabled in RAGAS-only mode
    return []


__all__ = [
    "get_integrity_metrics",
]
