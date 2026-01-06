"""
RAGAS-based retrieval metrics.

Uses RAGAS library for Contextual Precision and Contextual Recall.
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


class ContextualPrecisionMetric(BaseMetric):
    """Measures precision of retrieved context.

    Contextual Precision = (Relevant items ranked highly) / (Total retrieved items)

    Uses RAGAS context_precision metric which measures whether relevant
    context items appear before irrelevant ones in the ranking.
    """

    layer = EvaluationLayer.RETRIEVAL
    name = "Contextual Precision"

    def __init__(self, threshold: float = 0.70):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_ragas_metric(self):
        """Lazy initialization of RAGAS metric."""
        if self._ragas_metric is None:
            try:
                from ragas.metrics import context_precision
                self._ragas_metric = context_precision
            except ImportError:
                logger.warning("RAGAS not installed, using fallback implementation")
        return self._ragas_metric

    def _fallback_precision(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
    ) -> float:
        """Fallback precision calculation without RAGAS.

        Simple overlap-based precision.
        """
        if not retrieved_contexts:
            return 0.0

        # Normalize for comparison
        retrieved_set = {c.lower().strip() for c in retrieved_contexts}
        ground_truth_set = {c.lower().strip() for c in ground_truth_contexts}

        # Check overlap
        relevant_count = 0
        for retrieved in retrieved_set:
            for gt in ground_truth_set:
                # Check if there's significant overlap
                if gt in retrieved or retrieved in gt:
                    relevant_count += 1
                    break

        return relevant_count / len(retrieved_set)

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure contextual precision.

        Args:
            test_case: Test case with ground truth context
            agent_output: Agent output with retrieved context

        Returns:
            MetricResult with precision score
        """
        # Extract retrieved contexts from agent output
        retrieved_contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        ground_truth_contexts = test_case.ground_truth_context

        if not ground_truth_contexts:
            return self._create_result(
                score=1.0 if not retrieved_contexts else 0.5,
                details={"note": "No ground truth context provided"},
            )

        ragas_metric = self._get_ragas_metric()

        if ragas_metric is not None:
            try:
                from ragas import evaluate
                from datasets import Dataset

                # Prepare data for RAGAS
                data = {
                    "question": [test_case.question],
                    "contexts": [retrieved_contexts],
                    "ground_truth": [test_case.expected_answer or ""],
                }
                dataset = Dataset.from_dict(data)

                # Run RAGAS evaluation
                result = evaluate(dataset, metrics=[ragas_metric])
                score = result["context_precision"]

                return self._create_result(
                    score=score,
                    details={
                        "retrieved_count": len(retrieved_contexts),
                        "ground_truth_count": len(ground_truth_contexts),
                        "method": "ragas",
                    },
                )
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed, using fallback: {e}")

        # Fallback implementation
        score = self._fallback_precision(retrieved_contexts, ground_truth_contexts)

        return self._create_result(
            score=score,
            details={
                "retrieved_count": len(retrieved_contexts),
                "ground_truth_count": len(ground_truth_contexts),
                "method": "fallback",
            },
        )


class ContextualRecallMetric(BaseMetric):
    """Measures recall of retrieved context.

    Contextual Recall = (Ground truth covered by context) / (Total ground truth items)

    Uses RAGAS context_recall metric which measures what fraction of the
    ground truth can be attributed to the retrieved context.
    """

    layer = EvaluationLayer.RETRIEVAL
    name = "Contextual Recall"

    def __init__(self, threshold: float = 0.60):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_ragas_metric(self):
        """Lazy initialization of RAGAS metric."""
        if self._ragas_metric is None:
            try:
                from ragas.metrics import context_recall
                self._ragas_metric = context_recall
            except ImportError:
                logger.warning("RAGAS not installed, using fallback implementation")
        return self._ragas_metric

    def _fallback_recall(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
    ) -> float:
        """Fallback recall calculation without RAGAS.

        Simple overlap-based recall.
        """
        if not ground_truth_contexts:
            return 1.0

        retrieved_text = " ".join(retrieved_contexts).lower()

        # Check how many ground truth items are covered
        covered_count = 0
        for gt in ground_truth_contexts:
            gt_lower = gt.lower().strip()
            if gt_lower in retrieved_text:
                covered_count += 1

        return covered_count / len(ground_truth_contexts)

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure contextual recall.

        Args:
            test_case: Test case with ground truth context
            agent_output: Agent output with retrieved context

        Returns:
            MetricResult with recall score
        """
        # Extract retrieved contexts from agent output
        retrieved_contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        ground_truth_contexts = test_case.ground_truth_context

        if not ground_truth_contexts:
            return self._create_result(
                score=1.0,
                details={"note": "No ground truth context to recall"},
            )

        ragas_metric = self._get_ragas_metric()

        if ragas_metric is not None:
            try:
                from ragas import evaluate
                from datasets import Dataset

                # Prepare data for RAGAS
                data = {
                    "question": [test_case.question],
                    "contexts": [retrieved_contexts],
                    "ground_truth": [test_case.expected_answer or ""],
                }
                dataset = Dataset.from_dict(data)

                # Run RAGAS evaluation
                result = evaluate(dataset, metrics=[ragas_metric])
                score = result["context_recall"]

                return self._create_result(
                    score=score,
                    details={
                        "retrieved_count": len(retrieved_contexts),
                        "ground_truth_count": len(ground_truth_contexts),
                        "method": "ragas",
                    },
                )
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed, using fallback: {e}")

        # Fallback implementation
        score = self._fallback_recall(retrieved_contexts, ground_truth_contexts)

        return self._create_result(
            score=score,
            details={
                "retrieved_count": len(retrieved_contexts),
                "ground_truth_count": len(ground_truth_contexts),
                "method": "fallback",
            },
        )
