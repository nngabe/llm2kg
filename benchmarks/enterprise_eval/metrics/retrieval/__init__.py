"""
Layer 1: Retrieval Metrics.

Evaluates the quality of context retrieval from graph and vector stores.
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_retrieval_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get all retrieval metrics with configured thresholds.

    Args:
        config: Optional configuration for thresholds

    Returns:
        List of configured retrieval metrics
    """
    config = config or EvalConfig()
    thresholds = config.thresholds

    from .ragas_metrics import ContextualPrecisionMetric, ContextualRecallMetric
    from .graph_traversal import GraphTraversalEfficiencyMetric
    from .connectivity import SubgraphConnectivityMetric

    return [
        ContextualPrecisionMetric(threshold=thresholds.contextual_precision_min),
        ContextualRecallMetric(threshold=thresholds.contextual_recall_min),
        GraphTraversalEfficiencyMetric(threshold=thresholds.graph_traversal_efficiency_min),
        SubgraphConnectivityMetric(threshold=thresholds.subgraph_connectivity_min),
    ]


__all__ = [
    "get_retrieval_metrics",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "GraphTraversalEfficiencyMetric",
    "SubgraphConnectivityMetric",
]
