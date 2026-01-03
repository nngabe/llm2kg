"""
Q&A evaluation metrics.

Provides automated metrics for answer quality:
- Relevance: Semantic similarity to gold answer
- Groundedness: Claims traceable to context
- Completeness: Coverage of key points
- Fluency: Language quality
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAMetrics(BaseModel):
    """Metrics for Q&A evaluation."""

    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    groundedness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fluency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)

    def compute_overall(self, weights: Optional[Dict[str, float]] = None):
        """Compute weighted overall score."""
        weights = weights or {
            "relevance": 0.3,
            "groundedness": 0.25,
            "completeness": 0.2,
            "fluency": 0.1,
            "citation_accuracy": 0.15,
        }

        self.overall_score = (
            weights.get("relevance", 0.3) * self.relevance_score +
            weights.get("groundedness", 0.25) * self.groundedness_score +
            weights.get("completeness", 0.2) * self.completeness_score +
            weights.get("fluency", 0.1) * self.fluency_score +
            weights.get("citation_accuracy", 0.15) * self.citation_accuracy
        )


class QAEvaluator:
    """
    Evaluator for Q&A system quality.

    Uses embedding similarity and heuristics to measure answer quality.
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize evaluator.

        Args:
            embedding_model: Model for semantic similarity.
            similarity_threshold: Threshold for relevance.
        """
        self._embedding_model = None
        self._embedding_model_config = embedding_model
        self.similarity_threshold = similarity_threshold

    def _ensure_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            if self._embedding_model_config:
                self._embedding_model = self._embedding_model_config
            else:
                from langchain_openai import OpenAIEmbeddings
                self._embedding_model = OpenAIEmbeddings(
                    model="text-embedding-3-small"
                )

    def evaluate(
        self,
        question: str,
        answer: str,
        gold_answer: Optional[str] = None,
        context: Optional[str] = None,
        citations: Optional[List[Dict]] = None,
    ) -> QAMetrics:
        """
        Evaluate an answer.

        Args:
            question: Original question.
            answer: Generated answer.
            gold_answer: Optional gold standard answer.
            context: Retrieved context.
            citations: Answer citations.

        Returns:
            QAMetrics with scores.
        """
        metrics = QAMetrics()

        # Relevance: similarity to question intent
        metrics.relevance_score = self._compute_relevance(question, answer)

        # Groundedness: claims in context
        if context:
            metrics.groundedness_score = self._compute_groundedness(answer, context)
        else:
            metrics.groundedness_score = 0.5  # Unknown without context

        # Completeness: key point coverage
        if gold_answer:
            metrics.completeness_score = self._compute_completeness(answer, gold_answer)
        else:
            metrics.completeness_score = 0.5  # Unknown without gold

        # Fluency: language quality
        metrics.fluency_score = self._compute_fluency(answer)

        # Citation accuracy
        if citations and context:
            metrics.citation_accuracy = self._compute_citation_accuracy(
                citations, context
            )
        else:
            metrics.citation_accuracy = 0.5

        # Overall
        metrics.compute_overall()

        return metrics

    def _compute_relevance(self, question: str, answer: str) -> float:
        """Compute relevance using embedding similarity."""
        self._ensure_embedding_model()

        try:
            q_embedding = self._embedding_model.embed_query(question)
            a_embedding = self._embedding_model.embed_query(answer)

            # Cosine similarity
            import numpy as np
            q_vec = np.array(q_embedding)
            a_vec = np.array(a_embedding)

            similarity = np.dot(q_vec, a_vec) / (
                np.linalg.norm(q_vec) * np.linalg.norm(a_vec)
            )

            return float(max(0, min(1, similarity)))

        except Exception as e:
            logger.warning(f"Relevance computation failed: {e}")
            return 0.5

    def _compute_groundedness(self, answer: str, context: str) -> float:
        """Compute groundedness by checking claim overlap."""
        # Simple word overlap heuristic
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "that", "it"}
        answer_words -= common_words
        context_words -= common_words

        if not answer_words:
            return 0.5

        overlap = len(answer_words & context_words) / len(answer_words)
        return min(1.0, overlap * 1.5)  # Scale up slightly

    def _compute_completeness(self, answer: str, gold_answer: str) -> float:
        """Compute completeness by comparing to gold answer."""
        self._ensure_embedding_model()

        try:
            ans_embedding = self._embedding_model.embed_query(answer)
            gold_embedding = self._embedding_model.embed_query(gold_answer)

            import numpy as np
            ans_vec = np.array(ans_embedding)
            gold_vec = np.array(gold_embedding)

            similarity = np.dot(ans_vec, gold_vec) / (
                np.linalg.norm(ans_vec) * np.linalg.norm(gold_vec)
            )

            return float(max(0, min(1, similarity)))

        except Exception as e:
            logger.warning(f"Completeness computation failed: {e}")
            return 0.5

    def _compute_fluency(self, answer: str) -> float:
        """Compute fluency using simple heuristics."""
        # Length check
        word_count = len(answer.split())
        if word_count < 5:
            return 0.3
        if word_count > 500:
            return 0.7

        # Sentence structure
        sentences = answer.split('.')
        if len(sentences) < 1:
            return 0.4

        # Check for common issues
        score = 0.8

        # Very short sentences
        avg_sentence_length = word_count / max(len(sentences), 1)
        if avg_sentence_length < 3:
            score -= 0.2

        # Repetition
        words = answer.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.3:
            score -= 0.2

        return max(0.2, min(1.0, score))

    def _compute_citation_accuracy(
        self,
        citations: List[Dict],
        context: str,
    ) -> float:
        """Compute citation accuracy."""
        if not citations:
            return 0.5

        valid_citations = 0
        for citation in citations:
            source_id = citation.get("source_id", "")
            excerpt = citation.get("excerpt", "")

            # Check if source appears in context
            if source_id.lower() in context.lower():
                valid_citations += 1
            elif excerpt and excerpt.lower() in context.lower():
                valid_citations += 1

        return valid_citations / len(citations)

    def batch_evaluate(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[QAMetrics]:
        """
        Evaluate multiple samples.

        Args:
            samples: List of dicts with question, answer, etc.

        Returns:
            List of metrics.
        """
        results = []
        for sample in samples:
            metrics = self.evaluate(
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                gold_answer=sample.get("gold_answer"),
                context=sample.get("context"),
                citations=sample.get("citations"),
            )
            results.append(metrics)

        return results

    def aggregate_metrics(self, metrics_list: List[QAMetrics]) -> QAMetrics:
        """Compute aggregate metrics across samples."""
        if not metrics_list:
            return QAMetrics()

        n = len(metrics_list)
        return QAMetrics(
            relevance_score=sum(m.relevance_score for m in metrics_list) / n,
            groundedness_score=sum(m.groundedness_score for m in metrics_list) / n,
            completeness_score=sum(m.completeness_score for m in metrics_list) / n,
            fluency_score=sum(m.fluency_score for m in metrics_list) / n,
            citation_accuracy=sum(m.citation_accuracy for m in metrics_list) / n,
            overall_score=sum(m.overall_score for m in metrics_list) / n,
        )
