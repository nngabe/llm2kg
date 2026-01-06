"""
Configuration for Enterprise Evaluation Suite.

Defines thresholds, model configs, and evaluation layer enums.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class EvaluationLayer(Enum):
    """The 4 evaluation layers."""
    RETRIEVAL = "retrieval"
    AGENTIC = "agentic"
    INTEGRITY = "integrity"
    GENERATION = "generation"


@dataclass
class LayerThresholds:
    """Pass/fail thresholds per metric.

    All thresholds are 0.0-1.0 scores. A metric passes if score >= threshold.
    """
    # Layer 1: Retrieval Metrics
    contextual_precision_min: float = 0.70
    contextual_recall_min: float = 0.60
    graph_traversal_efficiency_min: float = 0.40
    subgraph_connectivity_min: float = 0.50

    # Layer 2: Agentic Brain Metrics
    tool_selection_accuracy_min: float = 0.80
    argument_correctness_min: float = 0.85
    loop_efficiency_min: float = 0.60
    rejection_sensitivity_min: float = 0.70

    # Layer 3: Graph Integrity Metrics
    schema_adherence_min: float = 0.95
    entity_disambiguation_min: float = 0.90
    source_citation_accuracy_min: float = 0.80

    # Layer 4: Generation Metrics
    faithfulness_min: float = 0.75
    answer_relevance_min: float = 0.70
    citation_recall_min: float = 0.60

    def get_threshold(self, metric_name: str) -> float:
        """Get threshold for a metric by name."""
        attr_name = f"{metric_name.lower().replace(' ', '_')}_min"
        return getattr(self, attr_name, 0.5)


@dataclass
class EvalConfig:
    """Configuration for enterprise evaluation suite."""

    # Judge model (GPT-5.2 for LLM-as-judge)
    judge_model: str = "gpt-5.2"
    judge_temperature: float = 0.0
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Layers to evaluate
    enabled_layers: List[EvaluationLayer] = field(
        default_factory=lambda: list(EvaluationLayer)
    )

    # Thresholds
    thresholds: LayerThresholds = field(default_factory=LayerThresholds)

    # RAGAS config
    ragas_llm: str = "gpt-4o"
    ragas_embeddings: str = "text-embedding-3-small"

    # Ollama config (for local models)
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    )

    # Neo4j config
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password")
    )

    # Caching
    use_cache: bool = True
    cache_dir: str = "benchmarks/enterprise_eval/cache"

    # Timeouts
    metric_timeout_ms: int = 60000  # 60s per metric
    total_timeout_ms: int = 3600000  # 1 hour total

    # Parallelism
    parallel_metrics: bool = False  # Run metrics in parallel (requires more resources)


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    metric_name: str
    layer: EvaluationLayer
    score: float  # 0.0 to 1.0
    passed: bool
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "layer": self.layer.value,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "details": self.details,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class LayerResult:
    """Aggregated results for an evaluation layer."""
    layer: EvaluationLayer
    metrics: List[MetricResult] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = True

    def calculate_overall(self):
        """Calculate overall score and pass/fail status."""
        if self.metrics:
            # Only count metrics without errors
            valid_metrics = [m for m in self.metrics if m.error is None]
            if valid_metrics:
                self.overall_score = sum(m.score for m in valid_metrics) / len(valid_metrics)
                self.passed = all(m.passed for m in valid_metrics)
            else:
                self.overall_score = 0.0
                self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer": self.layer.value,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result across all layers."""
    config: EvalConfig
    layers: Dict[EvaluationLayer, LayerResult] = field(default_factory=dict)
    overall_score: float = 0.0
    overall_passed: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: int = 0
    test_case_count: int = 0

    def calculate_overall(self):
        """Calculate overall score and pass/fail status across all layers."""
        if self.layers:
            valid_layers = [l for l in self.layers.values() if l.metrics]
            if valid_layers:
                self.overall_score = sum(l.overall_score for l in valid_layers) / len(valid_layers)
                self.overall_passed = all(l.passed for l in valid_layers)
            else:
                self.overall_score = 0.0
                self.overall_passed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "duration_ms": self.duration_ms,
                "test_case_count": self.test_case_count,
                "judge_model": self.config.judge_model,
                "enabled_layers": [l.value for l in self.config.enabled_layers],
            },
            "summary": {
                "overall_score": self.overall_score,
                "overall_passed": self.overall_passed,
            },
            "layers": {
                layer.value: result.to_dict()
                for layer, result in self.layers.items()
            },
        }

    def get_failed_metrics(self) -> List[MetricResult]:
        """Get list of all failed metrics."""
        failed = []
        for layer_result in self.layers.values():
            for metric in layer_result.metrics:
                if not metric.passed:
                    failed.append(metric)
        return failed
