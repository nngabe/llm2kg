"""
Rejection Sensitivity Metric.

Custom metric measuring the agent's ability to appropriately reject
unanswerable questions.
"""

import logging
from typing import Any, Dict, List

from ..base import BaseMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


# Keywords that indicate the agent is rejecting/declining to answer
REJECTION_INDICATORS = [
    "cannot answer",
    "unable to answer",
    "don't have enough information",
    "do not have enough information",
    "cannot find",
    "no information available",
    "unable to determine",
    "cannot determine",
    "not enough context",
    "insufficient information",
    "i don't know",
    "i cannot",
    "not available in",
    "no relevant information",
    "outside the scope",
    "beyond my knowledge",
    "unable to provide",
    "cannot provide",
    "no data found",
    "unable to locate",
]

# Keywords that indicate high confidence (opposite of rejection)
CONFIDENCE_INDICATORS = [
    "according to",
    "based on the information",
    "the answer is",
    "clearly shows",
    "evidence suggests",
    "data indicates",
    "confirms that",
]


class RejectionSensitivityMetric(BaseMetric):
    """Measures agent's ability to reject unanswerable questions.

    Rejection Sensitivity = correct_rejections / rejection_test_cases

    This metric evaluates whether the agent appropriately:
    - Rejects questions it cannot answer (true positives)
    - Does NOT reject questions it can answer (true negatives)
    """

    layer = EvaluationLayer.AGENTIC
    name = "Rejection Sensitivity"

    def __init__(self, threshold: float = 0.70):
        super().__init__(threshold)

    def _detect_rejection(
        self,
        answer: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Detect if the agent rejected the question.

        Args:
            answer: The agent's answer
            confidence: The agent's reported confidence

        Returns:
            Dictionary with rejection detection results
        """
        answer_lower = answer.lower()

        # Check for rejection indicators
        rejection_matches = [
            indicator for indicator in REJECTION_INDICATORS
            if indicator in answer_lower
        ]

        # Check for confidence indicators
        confidence_matches = [
            indicator for indicator in CONFIDENCE_INDICATORS
            if indicator in answer_lower
        ]

        # Determine if this is a rejection
        is_rejection = False
        rejection_strength = 0.0

        if rejection_matches:
            rejection_strength = len(rejection_matches) / 5  # Normalize
            rejection_strength = min(rejection_strength, 1.0)

            # High rejection if multiple indicators or low confidence
            if len(rejection_matches) >= 2 or confidence < 0.3:
                is_rejection = True
            elif len(rejection_matches) >= 1 and confidence < 0.5:
                is_rejection = True

        # Confidence indicators counteract rejection
        if confidence_matches and confidence > 0.7:
            is_rejection = False

        # Very low confidence is always a rejection signal
        if confidence < 0.2:
            is_rejection = True
            rejection_strength = max(rejection_strength, 0.8)

        return {
            "is_rejection": is_rejection,
            "rejection_strength": rejection_strength,
            "rejection_indicators": rejection_matches,
            "confidence_indicators": confidence_matches,
            "reported_confidence": confidence,
        }

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure rejection sensitivity.

        For test cases that should be rejected:
        - Score 1.0 if agent correctly rejects
        - Score 0.0 if agent incorrectly provides an answer

        For test cases that should NOT be rejected:
        - Score 1.0 if agent provides an answer
        - Score 0.0 if agent incorrectly rejects

        Args:
            test_case: Test case with should_reject flag
            agent_output: Agent output with answer and confidence

        Returns:
            MetricResult with rejection sensitivity score
        """
        should_reject = test_case.should_reject
        rejection_reason = test_case.rejection_reason

        # Detect if agent rejected
        detection = self._detect_rejection(
            agent_output.answer,
            agent_output.confidence,
        )

        did_reject = detection["is_rejection"]

        # Calculate score based on expected vs actual
        if should_reject:
            # This question should be rejected
            if did_reject:
                # Correct rejection (true positive)
                score = 1.0
                outcome = "correct_rejection"
            else:
                # Missed rejection (false negative)
                score = 0.0
                outcome = "missed_rejection"
        else:
            # This question should NOT be rejected
            if did_reject:
                # Incorrect rejection (false positive)
                score = 0.0
                outcome = "incorrect_rejection"
            else:
                # Correct answer (true negative)
                score = 1.0
                outcome = "correct_answer"

        return self._create_result(
            score=score,
            details={
                "should_reject": should_reject,
                "rejection_reason": rejection_reason,
                "did_reject": did_reject,
                "outcome": outcome,
                "detection": detection,
                "answer_preview": agent_output.answer[:200] if agent_output.answer else "",
            },
        )


class AggregatedRejectionMetric(BaseMetric):
    """Aggregates rejection sensitivity across multiple test cases.

    Calculates:
    - True Positive Rate (sensitivity): correct rejections / should reject
    - True Negative Rate (specificity): correct answers / should answer
    - F1 Score: harmonic mean of precision and recall
    """

    layer = EvaluationLayer.AGENTIC
    name = "Rejection F1 Score"

    def __init__(self, threshold: float = 0.70):
        super().__init__(threshold)
        self._rejection_metric = RejectionSensitivityMetric()

    def measure_batch(
        self,
        test_cases: List[TestCase],
        agent_outputs: List[AgentOutput],
        **kwargs: Any,
    ) -> MetricResult:
        """Measure rejection sensitivity across multiple test cases.

        Args:
            test_cases: List of test cases
            agent_outputs: Corresponding agent outputs

        Returns:
            MetricResult with aggregated rejection metrics
        """
        # Count outcomes
        true_positives = 0  # Correct rejections
        false_positives = 0  # Incorrect rejections
        true_negatives = 0  # Correct answers
        false_negatives = 0  # Missed rejections

        individual_results = []

        for test_case, agent_output in zip(test_cases, agent_outputs):
            result = self._rejection_metric.measure(test_case, agent_output, **kwargs)
            individual_results.append(result)

            outcome = result.details.get("outcome", "")
            if outcome == "correct_rejection":
                true_positives += 1
            elif outcome == "incorrect_rejection":
                false_positives += 1
            elif outcome == "correct_answer":
                true_negatives += 1
            elif outcome == "missed_rejection":
                false_negatives += 1

        # Calculate metrics
        total_should_reject = true_positives + false_negatives
        total_should_answer = true_negatives + false_positives

        sensitivity = true_positives / total_should_reject if total_should_reject > 0 else 1.0
        specificity = true_negatives / total_should_answer if total_should_answer > 0 else 1.0

        # Precision and recall for F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = sensitivity

        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Overall accuracy
        total = len(test_cases)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

        return self._create_result(
            score=f1_score,
            details={
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "total_cases": total,
            },
        )

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure for single test case (delegates to RejectionSensitivityMetric)."""
        return self._rejection_metric.measure(test_case, agent_output, **kwargs)
