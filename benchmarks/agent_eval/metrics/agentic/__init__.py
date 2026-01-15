"""
Layer 2: Agentic Brain Metrics (Formula-based only).

Evaluates the agent's decision-making and reasoning capabilities
using formula-based metrics (no LLM-as-judge).
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_agentic_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get all agentic metrics with configured thresholds.

    Uses formula-based metrics only:
    - Loop Efficiency: min(minimum_steps / actual_steps, 1.0)
    - Rejection Sensitivity: Detects appropriate rejection of unanswerable questions

    Args:
        config: Optional configuration for thresholds

    Returns:
        List of configured agentic metrics
    """
    config = config or EvalConfig()
    thresholds = config.thresholds

    from .loop_efficiency import LoopEfficiencyMetric
    from .rejection import RejectionSensitivityMetric

    return [
        LoopEfficiencyMetric(threshold=thresholds.loop_efficiency_min),
        RejectionSensitivityMetric(threshold=thresholds.rejection_sensitivity_min),
    ]


__all__ = [
    "get_agentic_metrics",
    "LoopEfficiencyMetric",
    "RejectionSensitivityMetric",
]
