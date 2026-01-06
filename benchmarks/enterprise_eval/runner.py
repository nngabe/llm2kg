"""
Enterprise Evaluation Runner.

Orchestrates evaluation across all 4 layers, collecting metrics
and generating results.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Type
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    EvalConfig,
    EvaluationLayer,
    MetricResult,
    LayerResult,
    EvaluationResult,
)
from .metrics.base import BaseMetric, TestCase, AgentOutput

logger = logging.getLogger(__name__)


class EnterpriseEvaluationRunner:
    """Main evaluation orchestrator.

    Runs the agent on test cases and evaluates using the 4-layer framework.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        """Initialize the evaluation runner.

        Args:
            config: Evaluation configuration. Uses defaults if not provided.
        """
        self.config = config or EvalConfig()
        self._metrics: Dict[EvaluationLayer, List[BaseMetric]] = {
            layer: [] for layer in EvaluationLayer
        }
        self._neo4j_driver = None
        self._ontology = None

    def register_metric(self, metric: BaseMetric) -> None:
        """Register a metric for evaluation.

        Args:
            metric: The metric instance to register
        """
        self._metrics[metric.layer].append(metric)
        logger.debug(f"Registered metric: {metric.name} for layer {metric.layer.value}")

    def register_metrics(self, metrics: List[BaseMetric]) -> None:
        """Register multiple metrics.

        Args:
            metrics: List of metric instances to register
        """
        for metric in metrics:
            self.register_metric(metric)

    def _get_neo4j_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password),
                )
                # Test connection
                with self._neo4j_driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Connected to Neo4j")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
                self._neo4j_driver = None
        return self._neo4j_driver

    def _get_ontology(self):
        """Lazy initialization of ontology."""
        if self._ontology is None:
            try:
                import sys
                sys.path.insert(0, "/app")
                from ontologies import Ontology
                self._ontology = Ontology()
                logger.info("Loaded ontology")
            except Exception as e:
                logger.warning(f"Failed to load ontology: {e}")
                self._ontology = None
        return self._ontology

    def _run_metric(
        self,
        metric: BaseMetric,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Run a single metric with timeout and error handling.

        Args:
            metric: The metric to run
            test_case: The test case
            agent_output: The agent's output
            **kwargs: Additional context

        Returns:
            MetricResult with score or error
        """
        start = time.time()
        try:
            result = metric.measure(test_case, agent_output, **kwargs)
            result.latency_ms = int((time.time() - start) * 1000)
            return result
        except Exception as e:
            logger.error(f"Metric {metric.name} failed: {e}")
            return MetricResult(
                metric_name=metric.name,
                layer=metric.layer,
                score=0.0,
                passed=False,
                threshold=metric.threshold,
                error=str(e),
                latency_ms=int((time.time() - start) * 1000),
            )

    def _run_layer(
        self,
        layer: EvaluationLayer,
        test_cases: List[TestCase],
        agent_outputs: List[AgentOutput],
        **kwargs: Any,
    ) -> LayerResult:
        """Run all metrics for a single layer.

        Args:
            layer: The evaluation layer
            test_cases: Test cases for this layer
            agent_outputs: Corresponding agent outputs
            **kwargs: Additional context

        Returns:
            LayerResult with all metric results
        """
        layer_result = LayerResult(layer=layer)
        metrics = self._metrics.get(layer, [])

        if not metrics:
            logger.warning(f"No metrics registered for layer {layer.value}")
            return layer_result

        # Filter test cases relevant to this layer
        relevant_pairs = [
            (tc, ao)
            for tc, ao in zip(test_cases, agent_outputs)
            if tc.matches_layer(layer)
        ]

        if not relevant_pairs:
            logger.info(f"No relevant test cases for layer {layer.value}")
            return layer_result

        logger.info(
            f"Running {len(metrics)} metrics on {len(relevant_pairs)} "
            f"test cases for layer {layer.value}"
        )

        # Aggregate results across all test cases per metric
        metric_aggregates: Dict[str, List[MetricResult]] = {m.name: [] for m in metrics}

        for test_case, agent_output in relevant_pairs:
            if self.config.parallel_metrics:
                # Run metrics in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(
                            self._run_metric, metric, test_case, agent_output, **kwargs
                        ): metric
                        for metric in metrics
                    }
                    for future in as_completed(futures):
                        metric = futures[future]
                        result = future.result()
                        metric_aggregates[metric.name].append(result)
            else:
                # Run metrics sequentially
                for metric in metrics:
                    result = self._run_metric(metric, test_case, agent_output, **kwargs)
                    metric_aggregates[metric.name].append(result)

        # Aggregate per-metric results into single metric scores
        for metric in metrics:
            results = metric_aggregates[metric.name]
            if results:
                # Average scores across test cases
                valid_results = [r for r in results if r.error is None]
                if valid_results:
                    avg_score = sum(r.score for r in valid_results) / len(valid_results)
                    all_passed = all(r.passed for r in valid_results)
                    total_latency = sum(r.latency_ms for r in results)

                    # Create aggregated result
                    aggregated = MetricResult(
                        metric_name=metric.name,
                        layer=layer,
                        score=avg_score,
                        passed=all_passed,
                        threshold=metric.threshold,
                        details={
                            "test_case_count": len(results),
                            "valid_count": len(valid_results),
                            "individual_scores": [r.score for r in valid_results],
                        },
                        latency_ms=total_latency,
                    )
                else:
                    # All results had errors
                    aggregated = MetricResult(
                        metric_name=metric.name,
                        layer=layer,
                        score=0.0,
                        passed=False,
                        threshold=metric.threshold,
                        error="All test cases failed",
                        details={"errors": [r.error for r in results]},
                        latency_ms=sum(r.latency_ms for r in results),
                    )
                layer_result.metrics.append(aggregated)

        layer_result.calculate_overall()
        return layer_result

    def run_evaluation(
        self,
        test_cases: List[TestCase],
        agent_outputs: List[AgentOutput],
        layers: Optional[List[EvaluationLayer]] = None,
    ) -> EvaluationResult:
        """Run full evaluation across specified layers.

        Args:
            test_cases: Golden dataset test cases
            agent_outputs: Agent outputs for each test case
            layers: Layers to evaluate (defaults to config.enabled_layers)

        Returns:
            Complete EvaluationResult
        """
        start_time = time.time()
        layers = layers or self.config.enabled_layers

        logger.info(
            f"Starting evaluation: {len(test_cases)} test cases, "
            f"{len(layers)} layers"
        )

        # Prepare shared resources
        kwargs = {
            "neo4j_driver": self._get_neo4j_driver(),
            "ontology": self._get_ontology(),
        }

        result = EvaluationResult(
            config=self.config,
            test_case_count=len(test_cases),
        )

        # Run each layer
        for layer in layers:
            if layer not in self.config.enabled_layers:
                logger.info(f"Skipping disabled layer: {layer.value}")
                continue

            layer_result = self._run_layer(
                layer, test_cases, agent_outputs, **kwargs
            )
            result.layers[layer] = layer_result

        # Calculate overall results
        result.calculate_overall()
        result.duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Evaluation complete: score={result.overall_score:.2f}, "
            f"passed={result.overall_passed}, duration={result.duration_ms}ms"
        )

        return result

    def run_agent_evaluation(
        self,
        test_cases: List[TestCase],
        agent_fn: callable,
        layers: Optional[List[EvaluationLayer]] = None,
    ) -> EvaluationResult:
        """Run evaluation by executing the agent on test cases.

        This is a convenience method that handles agent execution.

        Args:
            test_cases: Golden dataset test cases
            agent_fn: Function that takes a question and returns (response, state)
            layers: Layers to evaluate

        Returns:
            Complete EvaluationResult
        """
        logger.info(f"Running agent on {len(test_cases)} test cases...")

        agent_outputs = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Processing test case {i+1}/{len(test_cases)}: {test_case.id}")
            try:
                response, state = agent_fn(test_case.question)
                output = AgentOutput.from_agent(response, state)
                agent_outputs.append(output)
            except Exception as e:
                logger.error(f"Agent failed on test case {test_case.id}: {e}")
                # Create empty output for failed cases
                agent_outputs.append(
                    AgentOutput(
                        question=test_case.question,
                        answer=f"ERROR: {e}",
                        confidence=0.0,
                    )
                )

        return self.run_evaluation(test_cases, agent_outputs, layers)

    def close(self) -> None:
        """Clean up resources."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None


def create_default_runner(config: Optional[EvalConfig] = None) -> EnterpriseEvaluationRunner:
    """Create a runner with all default metrics registered.

    Args:
        config: Optional configuration

    Returns:
        Configured EnterpriseEvaluationRunner
    """
    runner = EnterpriseEvaluationRunner(config)

    # Import and register all metrics
    # These will be implemented in subsequent phases
    try:
        from .metrics.retrieval import get_retrieval_metrics
        runner.register_metrics(get_retrieval_metrics(config))
    except ImportError:
        logger.warning("Retrieval metrics not yet implemented")

    try:
        from .metrics.agentic import get_agentic_metrics
        runner.register_metrics(get_agentic_metrics(config))
    except ImportError:
        logger.warning("Agentic metrics not yet implemented")

    try:
        from .metrics.integrity import get_integrity_metrics
        runner.register_metrics(get_integrity_metrics(config))
    except ImportError:
        logger.warning("Integrity metrics not yet implemented")

    try:
        from .metrics.generation import get_generation_metrics
        runner.register_metrics(get_generation_metrics(config))
    except ImportError:
        logger.warning("Generation metrics not yet implemented")

    return runner
