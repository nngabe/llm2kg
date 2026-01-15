"""
RAGAS-based generation metrics.

Uses RAGAS library for Faithfulness and Answer Relevance.
Configured to use Ollama embeddings instead of OpenAI.
"""

import logging
from typing import Any, Dict, List

from ..base import BaseMetric, TestCase, AgentOutput
from ..ragas_helper import run_ragas_evaluate, extract_score
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


class FaithfulnessMetric(BaseMetric):
    """Measures faithfulness of the answer to the context.

    Faithfulness = (Claims supported by context) / (Total claims in answer)

    Uses RAGAS faithfulness metric which decomposes the answer into
    claims and verifies each against the retrieved context.
    """

    layer = EvaluationLayer.GENERATION
    name = "Faithfulness"

    def __init__(self, threshold: float = 0.75):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_ragas_metric(self):
        """Lazy initialization of RAGAS metric."""
        if self._ragas_metric is None:
            try:
                from ragas.metrics import faithfulness
                self._ragas_metric = faithfulness
            except ImportError:
                logger.warning("RAGAS not installed, using fallback implementation")
        return self._ragas_metric

    def _fallback_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Fallback faithfulness calculation.

        Simple heuristic: check if answer contains information
        that appears in the context.
        """
        if not answer or not contexts:
            return 0.0

        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())

        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "shall",
                      "can", "need", "dare", "ought", "used", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "into",
                      "through", "during", "before", "after", "above", "below",
                      "between", "under", "again", "further", "then", "once",
                      "here", "there", "when", "where", "why", "how", "all",
                      "each", "few", "more", "most", "other", "some", "such",
                      "no", "nor", "not", "only", "own", "same", "so", "than",
                      "too", "very", "just", "and", "but", "if", "or", "because",
                      "until", "while", "this", "that", "these", "those", "it"}

        content_words = answer_words - stop_words

        if not content_words:
            return 0.5

        # Check how many content words appear in context
        supported = sum(1 for word in content_words if word in context_text)
        return supported / len(content_words)

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure faithfulness.

        Args:
            test_case: Test case
            agent_output: Agent output with answer and context

        Returns:
            MetricResult with faithfulness score
        """
        answer = agent_output.answer
        contexts = [
            item.get("content", "")
            for item in agent_output.context_items
        ]

        if not answer:
            return self._create_result(
                score=0.0,
                details={"note": "Empty answer"},
            )

        if not contexts:
            return self._create_result(
                score=0.0,
                details={"note": "No context retrieved"},
            )

        ragas_metric = self._get_ragas_metric()

        if ragas_metric is not None:
            try:
                from datasets import Dataset

                data = {
                    "question": [test_case.question],
                    "answer": [answer],
                    "contexts": [contexts],
                }
                dataset = Dataset.from_dict(data)

                # Run RAGAS evaluation with Ollama config
                result = run_ragas_evaluate(dataset, metrics=[ragas_metric])
                score = extract_score(result, "faithfulness")

                if score is not None:
                    return self._create_result(
                        score=score,
                        details={
                            "answer_length": len(answer),
                            "context_count": len(contexts),
                            "method": "ragas",
                        },
                    )
                else:
                    logger.warning("RAGAS returned no score, using fallback")
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed, using fallback: {e}")

        # Fallback implementation
        score = self._fallback_faithfulness(answer, contexts)

        return self._create_result(
            score=score,
            details={
                "answer_length": len(answer),
                "context_count": len(contexts),
                "method": "fallback",
            },
        )


class AnswerRelevanceMetric(BaseMetric):
    """Measures relevance of the answer to the question.

    Answer Relevance = semantic_similarity(answer, question)

    Uses RAGAS answer_relevancy metric which measures how well
    the answer addresses the original question.
    """

    layer = EvaluationLayer.GENERATION
    name = "Answer Relevance"

    def __init__(self, threshold: float = 0.70):
        super().__init__(threshold)
        self._ragas_metric = None

    def _get_ragas_metric(self):
        """Lazy initialization of RAGAS metric."""
        if self._ragas_metric is None:
            try:
                from ragas.metrics import answer_relevancy
                self._ragas_metric = answer_relevancy
            except ImportError:
                logger.warning("RAGAS not installed, using fallback implementation")
        return self._ragas_metric

    def _fallback_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Fallback relevance calculation.

        Simple heuristic: word overlap between question and answer.
        """
        if not answer:
            return 0.0

        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "what", "who", "where",
                      "when", "why", "how", "which", "?", ".", ","}

        q_content = question_words - stop_words
        a_content = answer_words - stop_words

        if not q_content:
            return 0.5

        # Check overlap
        overlap = len(q_content & a_content)
        return min(overlap / len(q_content), 1.0)

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure answer relevance.

        Args:
            test_case: Test case with question
            agent_output: Agent output with answer

        Returns:
            MetricResult with relevance score
        """
        question = test_case.question
        answer = agent_output.answer

        if not answer:
            return self._create_result(
                score=0.0,
                details={"note": "Empty answer"},
            )

        ragas_metric = self._get_ragas_metric()

        if ragas_metric is not None:
            try:
                from datasets import Dataset

                contexts = [
                    item.get("content", "")
                    for item in agent_output.context_items
                ]

                data = {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts if contexts else [""]],
                }
                dataset = Dataset.from_dict(data)

                # Run RAGAS evaluation with Ollama config
                result = run_ragas_evaluate(dataset, metrics=[ragas_metric])
                score = extract_score(result, "answer_relevancy")

                if score is not None:
                    return self._create_result(
                        score=score,
                        details={
                            "question_length": len(question),
                            "answer_length": len(answer),
                            "method": "ragas",
                        },
                    )
                else:
                    logger.warning("RAGAS returned no score, using fallback")
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed, using fallback: {e}")

        # Fallback implementation
        score = self._fallback_relevance(question, answer)

        return self._create_result(
            score=score,
            details={
                "question_length": len(question),
                "answer_length": len(answer),
                "method": "fallback",
            },
        )
