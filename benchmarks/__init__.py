"""
Benchmarks package for Knowledge Graph extraction model comparison.

This package contains tools to compare multiple LLM models for knowledge
graph extraction quality without writing to Neo4j.
"""

from .model_comparison import (
    ModelBenchmark,
    ModelConfig,
    BENCHMARK_MODELS,
    ExtractionResult,
    ComparisonMetrics,
)

__all__ = [
    "ModelBenchmark",
    "ModelConfig",
    "BENCHMARK_MODELS",
    "ExtractionResult",
    "ComparisonMetrics",
]
