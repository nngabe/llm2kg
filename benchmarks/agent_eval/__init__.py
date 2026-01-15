"""
Agent Evaluation Suite for GraphRAG + ReAct Agent

A comprehensive 4-layer evaluation framework:
- Layer 1: Retrieval Metrics (Graph + Vector)
- Layer 2: Agentic Brain Metrics (Reasoning)
- Layer 3: Research & Update Metrics (Graph Integrity)
- Layer 4: Generation Metrics (Final Output)

Usage:
    from benchmarks.agent_eval import AgentEvaluationRunner, EvalConfig

    runner = AgentEvaluationRunner()
    result = runner.run_evaluation(test_cases)
"""

from .config import (
    EvalConfig,
    EvaluationLayer,
    LayerThresholds,
    MetricResult,
    LayerResult,
    EvaluationResult,
)
from .runner import AgentEvaluationRunner

__all__ = [
    "EvalConfig",
    "EvaluationLayer",
    "LayerThresholds",
    "MetricResult",
    "LayerResult",
    "EvaluationResult",
    "AgentEvaluationRunner",
]
