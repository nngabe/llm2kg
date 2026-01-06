"""
Post-Research Hook.

Validates graph updates after autonomous research operations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import EvalConfig, MetricResult, EvaluationLayer
from ..metrics.base import AgentOutput, TestCase
from ..metrics.integrity.schema_adherence import SchemaAdherenceMetric
from ..metrics.integrity.disambiguation import EntityDisambiguationMetric

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of post-research validation."""
    passed: bool
    score: float
    metrics: List[MetricResult]
    warnings: List[str]
    errors: List[str]
    should_rollback: bool = False


class PostResearchHook:
    """Hook for validating graph updates after research operations.

    This hook runs automatically after the agent completes research
    that modifies the knowledge graph. It validates:
    - Schema adherence of new entities/relationships
    - Entity disambiguation (no duplicates created)
    - Source citation accuracy

    Usage:
        hook = PostResearchHook()

        # After agent completes research
        result = hook.validate(new_nodes, new_edges, citations)

        if not result.passed:
            # Optionally rollback changes
            if result.should_rollback:
                rollback_graph_changes(...)
    """

    def __init__(
        self,
        config: Optional[EvalConfig] = None,
        rollback_threshold: float = 0.5,
        warn_threshold: float = 0.8,
    ):
        """Initialize the post-research hook.

        Args:
            config: Evaluation configuration
            rollback_threshold: Score below which to recommend rollback
            warn_threshold: Score below which to generate warnings
        """
        self.config = config or EvalConfig()
        self.rollback_threshold = rollback_threshold
        self.warn_threshold = warn_threshold

        # Initialize metrics
        self.schema_metric = SchemaAdherenceMetric(
            threshold=self.config.thresholds.schema_adherence_min
        )
        self.disambiguation_metric = EntityDisambiguationMetric(
            threshold=self.config.thresholds.entity_disambiguation_min,
            judge_model=self.config.judge_model,
        )

        self._neo4j_driver = None
        self._ontology = None

    def _get_neo4j_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password),
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
        return self._neo4j_driver

    def _get_ontology(self):
        """Lazy initialization of ontology."""
        if self._ontology is None:
            try:
                import sys
                sys.path.insert(0, "/app")
                from ontologies import Ontology
                self._ontology = Ontology()
            except Exception as e:
                logger.warning(f"Failed to load ontology: {e}")
        return self._ontology

    def validate(
        self,
        new_nodes: List[Dict[str, Any]],
        new_edges: List[Dict[str, Any]],
        citations: Optional[List[Dict[str, Any]]] = None,
        question: str = "",
    ) -> ValidationResult:
        """Validate graph updates.

        Args:
            new_nodes: New nodes to be added to the graph
            new_edges: New edges to be added to the graph
            citations: Citations for the updates
            question: Original question that triggered the research

        Returns:
            ValidationResult with pass/fail status and details
        """
        logger.info(f"Validating {len(new_nodes)} new nodes, {len(new_edges)} new edges")

        # Create minimal test case and agent output for metric evaluation
        test_case = TestCase(
            id="post_research_validation",
            question=question,
        )

        agent_output = AgentOutput(
            question=question,
            answer="",
            confidence=1.0,
            new_nodes=new_nodes,
            new_edges=new_edges,
            citations=citations or [],
        )

        metrics = []
        warnings = []
        errors = []

        # Run schema adherence check
        try:
            schema_result = self.schema_metric.measure(
                test_case,
                agent_output,
                ontology=self._get_ontology(),
            )
            metrics.append(schema_result)

            if not schema_result.passed:
                violations = schema_result.details.get("violations", [])
                for v in violations:
                    errors.append(f"Schema violation: {v}")

            schema_warnings = schema_result.details.get("warnings", [])
            warnings.extend(schema_warnings)

        except Exception as e:
            logger.error(f"Schema adherence check failed: {e}")
            errors.append(f"Schema check error: {e}")

        # Run disambiguation check
        try:
            disambig_result = self.disambiguation_metric.measure(
                test_case,
                agent_output,
                neo4j_driver=self._get_neo4j_driver(),
            )
            metrics.append(disambig_result)

            if not disambig_result.passed:
                duplicates = disambig_result.details.get("duplicate_checks", [])
                for check in duplicates:
                    if check.get("is_duplicate"):
                        warnings.append(
                            f"Potential duplicate: {check['entity_name']} "
                            f"similar to {check.get('similar_entities', [{}])[0].get('similar_to', 'unknown')}"
                        )

        except Exception as e:
            logger.error(f"Disambiguation check failed: {e}")
            errors.append(f"Disambiguation check error: {e}")

        # Calculate overall score
        valid_metrics = [m for m in metrics if m.error is None]
        if valid_metrics:
            overall_score = sum(m.score for m in valid_metrics) / len(valid_metrics)
        else:
            overall_score = 0.0

        # Determine pass/fail
        passed = all(m.passed for m in valid_metrics) if valid_metrics else False

        # Determine if rollback is recommended
        should_rollback = overall_score < self.rollback_threshold

        # Add warnings for borderline scores
        if overall_score < self.warn_threshold and not should_rollback:
            warnings.append(
                f"Overall validation score ({overall_score:.2%}) is below warning threshold"
            )

        result = ValidationResult(
            passed=passed,
            score=overall_score,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
            should_rollback=should_rollback,
        )

        # Log result
        if result.passed:
            logger.info(f"Validation PASSED with score {result.score:.2%}")
        else:
            logger.warning(f"Validation FAILED with score {result.score:.2%}")
            for error in errors:
                logger.error(f"  {error}")

        return result

    def validate_and_commit(
        self,
        new_nodes: List[Dict[str, Any]],
        new_edges: List[Dict[str, Any]],
        citations: Optional[List[Dict[str, Any]]] = None,
        question: str = "",
        commit_fn: Optional[callable] = None,
        rollback_fn: Optional[callable] = None,
    ) -> ValidationResult:
        """Validate and optionally commit or rollback changes.

        Args:
            new_nodes: New nodes
            new_edges: New edges
            citations: Citations
            question: Original question
            commit_fn: Function to call to commit changes
            rollback_fn: Function to call to rollback changes

        Returns:
            ValidationResult
        """
        result = self.validate(new_nodes, new_edges, citations, question)

        if result.passed:
            if commit_fn:
                logger.info("Committing validated changes...")
                try:
                    commit_fn()
                except Exception as e:
                    logger.error(f"Commit failed: {e}")
                    result.errors.append(f"Commit error: {e}")
        elif result.should_rollback:
            if rollback_fn:
                logger.warning("Rolling back changes due to validation failure...")
                try:
                    rollback_fn()
                except Exception as e:
                    logger.error(f"Rollback failed: {e}")
                    result.errors.append(f"Rollback error: {e}")

        return result

    def close(self):
        """Clean up resources."""
        if self._neo4j_driver:
            self._neo4j_driver.close()


def create_hook(config: Optional[EvalConfig] = None) -> PostResearchHook:
    """Factory function to create a configured hook.

    Args:
        config: Optional configuration

    Returns:
        Configured PostResearchHook
    """
    return PostResearchHook(config)
