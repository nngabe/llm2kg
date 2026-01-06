"""
Layer 2: Agentic Brain Metrics.

Evaluates the agent's decision-making and reasoning capabilities.
"""

from typing import List, Optional

from ..base import BaseMetric
from ...config import EvalConfig


def get_agentic_metrics(config: Optional[EvalConfig] = None) -> List[BaseMetric]:
    """Get all agentic metrics with configured thresholds.

    Args:
        config: Optional configuration for thresholds

    Returns:
        List of configured agentic metrics
    """
    config = config or EvalConfig()
    thresholds = config.thresholds

    from .tool_selection import ToolSelectionAccuracyMetric
    from .argument_correctness import ArgumentCorrectnessMetric
    from .loop_efficiency import LoopEfficiencyMetric
    from .rejection import RejectionSensitivityMetric

    return [
        ToolSelectionAccuracyMetric(
            threshold=thresholds.tool_selection_accuracy_min,
            judge_model=config.judge_model,
        ),
        ArgumentCorrectnessMetric(
            threshold=thresholds.argument_correctness_min,
            judge_model=config.judge_model,
        ),
        LoopEfficiencyMetric(threshold=thresholds.loop_efficiency_min),
        RejectionSensitivityMetric(threshold=thresholds.rejection_sensitivity_min),
    ]


__all__ = [
    "get_agentic_metrics",
    "ToolSelectionAccuracyMetric",
    "ArgumentCorrectnessMetric",
    "LoopEfficiencyMetric",
    "RejectionSensitivityMetric",
]
