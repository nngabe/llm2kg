"""
RAGAS-Only Evaluation Metrics.

Replaces all LLM-as-judge metrics with standardized RAGAS metrics.
Uses Ollama embeddings and configurable LLM backend.

Metrics by Layer:
- RETRIEVAL: ContextPrecision, ContextRecall, ContextRelevance
- GENERATION: Faithfulness, AnswerRelevancy, AnswerCorrectness, FactualCorrectness
- AGENTIC: ToolCallAccuracy (RAGAS native)
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseMetric, TestCase, AgentOutput
from .ragas_helper import run_ragas_evaluate, extract_score
from ..config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


# =============================================================================
# RETRIEVAL LAYER METRICS
# =============================================================================

class ContextPrecisionMetric(BaseMetric):
    """Measures precision of retrieved context.

    Context Precision = (Relevant chunks retrieved) / (Total chunks retrieved)

    Requires ground_truth_context in test case.
    """

    layer = EvaluationLayer.RETRIEVAL
    name = "Context Precision"

    def __init__(self, threshold: float = 0.70):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_metric(self):
        if self._ragas_metric is None:
            try:
                from ragas.metrics import LLMContextPrecisionWithoutReference
                self._ragas_metric = LLMContextPrecisionWithoutReference()
            except ImportError:
                from ragas.metrics import context_precision
                self._ragas_metric = context_precision
        return self._ragas_metric

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        if not contexts:
            return self._create_result(
                score=0.0,
                details={"note": "No context retrieved"},
            )

        data = [{
            "user_input": test_case.question,
            "response": agent_output.answer,
            "retrieved_contexts": contexts,
            "reference": test_case.expected_answer or "",
        }]

        try:
            result = run_ragas_evaluate(data, metrics=[self._get_metric()])
            score = extract_score(result, "llm_context_precision_without_reference")
            if score is None:
                score = extract_score(result, "context_precision")

            if score is not None:
                return self._create_result(
                    score=score,
                    details={"context_count": len(contexts), "method": "ragas"},
                )
        except Exception as e:
            logger.warning(f"RAGAS context_precision failed: {e}")

        return self._create_result(
            score=0.5,
            details={"note": "RAGAS evaluation failed", "method": "fallback"},
        )


class ContextRecallMetric(BaseMetric):
    """Measures recall of retrieved context.

    Context Recall = (Relevant info retrieved) / (Total relevant info needed)

    Requires ground_truth_context or expected_answer in test case.
    """

    layer = EvaluationLayer.RETRIEVAL
    name = "Context Recall"

    def __init__(self, threshold: float = 0.60):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_metric(self):
        if self._ragas_metric is None:
            try:
                from ragas.metrics import LLMContextRecall
                self._ragas_metric = LLMContextRecall()
            except ImportError:
                from ragas.metrics import context_recall
                self._ragas_metric = context_recall
        return self._ragas_metric

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        reference = test_case.expected_answer
        if not reference and test_case.ground_truth_context:
            reference = " ".join(test_case.ground_truth_context)

        if not reference:
            return self._create_result(
                score=float('nan'),
                details={"note": "No ground truth available for recall"},
            )

        data = [{
            "user_input": test_case.question,
            "response": agent_output.answer,
            "retrieved_contexts": contexts if contexts else [""],
            "reference": reference,
        }]

        try:
            result = run_ragas_evaluate(data, metrics=[self._get_metric()])
            score = extract_score(result, "context_recall")

            if score is not None:
                return self._create_result(
                    score=score,
                    details={"context_count": len(contexts), "method": "ragas"},
                )
        except Exception as e:
            logger.warning(f"RAGAS context_recall failed: {e}")

        return self._create_result(
            score=float('nan'),
            details={"note": "RAGAS evaluation failed"},
        )


# =============================================================================
# GENERATION LAYER METRICS
# =============================================================================

class FaithfulnessMetric(BaseMetric):
    """Measures faithfulness of answer to retrieved context.

    Faithfulness = (Claims supported by context) / (Total claims in answer)

    Does NOT require ground truth - evaluates against retrieved context.
    """

    layer = EvaluationLayer.GENERATION
    name = "Faithfulness"

    def __init__(self, threshold: float = 0.75):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_metric(self):
        if self._ragas_metric is None:
            try:
                from ragas.metrics import Faithfulness
                self._ragas_metric = Faithfulness()
            except ImportError:
                from ragas.metrics import faithfulness
                self._ragas_metric = faithfulness
        return self._ragas_metric

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        if not agent_output.answer:
            return self._create_result(
                score=0.0,
                details={"note": "Empty answer"},
            )

        if not contexts:
            return self._create_result(
                score=0.0,
                details={"note": "No context retrieved"},
            )

        data = [{
            "user_input": test_case.question,
            "response": agent_output.answer,
            "retrieved_contexts": contexts,
        }]

        try:
            result = run_ragas_evaluate(data, metrics=[self._get_metric()])
            score = extract_score(result, "faithfulness")

            if score is not None:
                return self._create_result(
                    score=score,
                    details={
                        "answer_length": len(agent_output.answer),
                        "context_count": len(contexts),
                        "method": "ragas",
                    },
                )
        except Exception as e:
            logger.warning(f"RAGAS faithfulness failed: {e}")

        return self._create_result(
            score=0.5,
            details={"note": "RAGAS evaluation failed", "method": "fallback"},
        )


class AnswerRelevancyMetric(BaseMetric):
    """Measures relevance of answer to the question.

    Generates hypothetical questions from answer, compares to original.

    Does NOT require ground truth.
    """

    layer = EvaluationLayer.GENERATION
    name = "Answer Relevancy"

    def __init__(self, threshold: float = 0.70):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_metric(self):
        if self._ragas_metric is None:
            try:
                from ragas.metrics import ResponseRelevancy
                self._ragas_metric = ResponseRelevancy()
            except ImportError:
                from ragas.metrics import answer_relevancy
                self._ragas_metric = answer_relevancy
        return self._ragas_metric

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        if not agent_output.answer:
            return self._create_result(
                score=0.0,
                details={"note": "Empty answer"},
            )

        contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        data = [{
            "user_input": test_case.question,
            "response": agent_output.answer,
            "retrieved_contexts": contexts if contexts else [""],
        }]

        try:
            result = run_ragas_evaluate(data, metrics=[self._get_metric()])
            score = extract_score(result, "answer_relevancy")
            if score is None:
                score = extract_score(result, "response_relevancy")

            if score is not None:
                return self._create_result(
                    score=score,
                    details={
                        "question_length": len(test_case.question),
                        "answer_length": len(agent_output.answer),
                        "method": "ragas",
                    },
                )
        except Exception as e:
            logger.warning(f"RAGAS answer_relevancy failed: {e}")

        return self._create_result(
            score=0.5,
            details={"note": "RAGAS evaluation failed", "method": "fallback"},
        )


class AnswerCorrectnessMetric(BaseMetric):
    """Measures factual correctness of answer vs ground truth.

    Combines semantic similarity and factual overlap.

    REQUIRES ground truth (expected_answer).
    """

    layer = EvaluationLayer.GENERATION
    name = "Answer Correctness"

    def __init__(self, threshold: float = 0.60):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_metric(self):
        if self._ragas_metric is None:
            try:
                from ragas.metrics import AnswerCorrectness
                self._ragas_metric = AnswerCorrectness()
            except ImportError:
                from ragas.metrics import answer_correctness
                self._ragas_metric = answer_correctness
        return self._ragas_metric

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        if not test_case.expected_answer:
            return self._create_result(
                score=float('nan'),
                details={"note": "No ground truth answer available"},
            )

        if not agent_output.answer:
            return self._create_result(
                score=0.0,
                details={"note": "Empty answer"},
            )

        contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        data = [{
            "user_input": test_case.question,
            "response": agent_output.answer,
            "retrieved_contexts": contexts if contexts else [""],
            "reference": test_case.expected_answer,
        }]

        try:
            result = run_ragas_evaluate(data, metrics=[self._get_metric()])
            score = extract_score(result, "answer_correctness")

            if score is not None:
                return self._create_result(
                    score=score,
                    details={
                        "answer_length": len(agent_output.answer),
                        "reference_length": len(test_case.expected_answer),
                        "method": "ragas",
                    },
                )
        except Exception as e:
            logger.warning(f"RAGAS answer_correctness failed: {e}")

        return self._create_result(
            score=float('nan'),
            details={"note": "RAGAS evaluation failed"},
        )


class FactualCorrectnessMetric(BaseMetric):
    """Measures claim-level factual correctness.

    Extracts claims from answer and verifies against reference.

    REQUIRES ground truth (expected_answer).
    """

    layer = EvaluationLayer.GENERATION
    name = "Factual Correctness"

    def __init__(self, threshold: float = 0.60):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_metric(self):
        if self._ragas_metric is None:
            try:
                from ragas.metrics import FactualCorrectness
                self._ragas_metric = FactualCorrectness()
            except ImportError:
                logger.warning("FactualCorrectness not available in this RAGAS version")
                return None
        return self._ragas_metric

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        if not test_case.expected_answer:
            return self._create_result(
                score=float('nan'),
                details={"note": "No ground truth answer available"},
            )

        metric = self._get_metric()
        if metric is None:
            return self._create_result(
                score=float('nan'),
                details={"note": "FactualCorrectness metric not available"},
            )

        contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        data = [{
            "user_input": test_case.question,
            "response": agent_output.answer,
            "retrieved_contexts": contexts if contexts else [""],
            "reference": test_case.expected_answer,
        }]

        try:
            result = run_ragas_evaluate(data, metrics=[metric])
            score = extract_score(result, "factual_correctness")

            if score is not None:
                return self._create_result(
                    score=score,
                    details={"method": "ragas"},
                )
        except Exception as e:
            logger.warning(f"RAGAS factual_correctness failed: {e}")

        return self._create_result(
            score=float('nan'),
            details={"note": "RAGAS evaluation failed"},
        )


# =============================================================================
# AGENTIC LAYER METRICS (RAGAS native)
# =============================================================================

class ToolCallAccuracyMetric(BaseMetric):
    """Measures accuracy of tool calls using RAGAS.

    Compares agent tool calls to reference tool calls.

    REQUIRES optimal_tool_sequence in test case.
    """

    layer = EvaluationLayer.AGENTIC
    name = "Tool Call Accuracy"

    def __init__(self, threshold: float = 0.80):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_metric(self):
        if self._ragas_metric is None:
            try:
                from ragas.metrics import ToolCallAccuracy
                self._ragas_metric = ToolCallAccuracy()
            except ImportError:
                logger.warning("ToolCallAccuracy not available in this RAGAS version")
                return None
        return self._ragas_metric

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        if not test_case.optimal_tool_sequence:
            return self._create_result(
                score=float('nan'),
                details={"note": "No reference tool sequence available"},
            )

        metric = self._get_metric()
        if metric is None:
            # Fallback to simple accuracy calculation
            actual_tools = [
                step.get("action", {}).get("tool_name", "")
                for step in agent_output.thought_history
                if step.get("action")
            ]

            if not actual_tools:
                return self._create_result(
                    score=0.0,
                    details={"note": "No tool calls recorded"},
                )

            # Simple overlap accuracy
            expected_set = set(test_case.optimal_tool_sequence)
            actual_set = set(actual_tools)

            if not expected_set:
                return self._create_result(
                    score=1.0 if not actual_set else 0.5,
                    details={"method": "fallback"},
                )

            overlap = len(expected_set & actual_set)
            precision = overlap / len(actual_set) if actual_set else 0.0
            recall = overlap / len(expected_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return self._create_result(
                score=f1,
                details={
                    "expected_tools": list(expected_set),
                    "actual_tools": list(actual_set),
                    "precision": precision,
                    "recall": recall,
                    "method": "fallback",
                },
            )

        # Use RAGAS metric if available
        # Note: RAGAS tool metrics require specific format
        return self._create_result(
            score=float('nan'),
            details={"note": "RAGAS tool metrics require multi-turn format"},
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ragas_retrieval_metrics(config=None) -> List[BaseMetric]:
    """Get all RAGAS retrieval metrics."""
    return [
        ContextPrecisionMetric(),
        ContextRecallMetric(),
    ]


def get_ragas_generation_metrics(config=None) -> List[BaseMetric]:
    """Get all RAGAS generation metrics."""
    return [
        FaithfulnessMetric(),
        AnswerRelevancyMetric(),
        AnswerCorrectnessMetric(),
        FactualCorrectnessMetric(),
    ]


def get_ragas_agentic_metrics(config=None) -> List[BaseMetric]:
    """Get RAGAS-compatible agentic metrics."""
    # Import formula-based metrics that don't use LLM-as-judge
    from .agentic.loop_efficiency import LoopEfficiencyMetric
    from .agentic.rejection import RejectionSensitivityMetric

    return [
        LoopEfficiencyMetric(),
        RejectionSensitivityMetric(),
        ToolCallAccuracyMetric(),
    ]


def get_all_ragas_metrics(config=None) -> List[BaseMetric]:
    """Get all RAGAS-based metrics for evaluation."""
    metrics = []
    metrics.extend(get_ragas_retrieval_metrics(config))
    metrics.extend(get_ragas_generation_metrics(config))
    metrics.extend(get_ragas_agentic_metrics(config))
    return metrics
