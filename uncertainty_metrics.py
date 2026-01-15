#!/usr/bin/env python3
"""
Uncertainty Metrics Module.

Provides three objective uncertainty quantification methods to replace
LLM self-reported confidence scores:

1. TokenPerplexity - via Ollama logprobs API (lower = more certain)
2. SemanticEntropy - multiple generations clustered by meaning (lower = more certain)
3. EmbeddingConsistency - cosine similarity of answer embeddings (higher = more certain)

Usage:
    from uncertainty_metrics import UncertaintyCalculator, UncertaintyScores

    calculator = UncertaintyCalculator(llm, embeddings, ollama_host, model)
    scores = calculator.compute_all(question, answer, context, n_generations=3)
    print(f"Combined confidence: {scores.combined_confidence:.2%}")
"""

import math
import logging
import requests
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyScores:
    """Container for all uncertainty metrics."""
    # Raw scores
    perplexity: float  # Lower = more certain (typically 1-100+)
    semantic_entropy: float  # Lower = more certain (0-1 normalized)
    embedding_consistency: float  # Higher = more certain (0-1)

    # Normalized scores (all 0-1, higher = more certain)
    perplexity_normalized: float
    semantic_entropy_normalized: float  # Inverted: 1 - entropy

    # Combined score
    combined_confidence: float  # Weighted average (0-1)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "perplexity": self.perplexity,
            "perplexity_normalized": self.perplexity_normalized,
            "semantic_entropy": self.semantic_entropy,
            "semantic_entropy_normalized": self.semantic_entropy_normalized,
            "embedding_consistency": self.embedding_consistency,
            "combined_confidence": self.combined_confidence,
        }


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def cluster_by_similarity(
    vectors: List[List[float]],
    threshold: float = 0.9,
) -> List[List[int]]:
    """Cluster vectors by cosine similarity using simple greedy clustering.

    Args:
        vectors: List of embedding vectors
        threshold: Similarity threshold for clustering (default: 0.9)

    Returns:
        List of clusters, where each cluster is a list of indices
    """
    n = len(vectors)
    if n == 0:
        return []

    assigned = [False] * n
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start a new cluster with vector i
        cluster = [i]
        assigned[i] = True

        # Find all similar vectors
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True

        clusters.append(cluster)

    return clusters


class UncertaintyCalculator:
    """Unified calculator for all uncertainty metrics.

    Uses main LLM (nemotron-3-nano:30b) with diversity-optimized settings
    for generating candidate answers.
    """

    # Nemotron-3-Nano settings for diverse generation (uncertainty sampling)
    DIVERSITY_SETTINGS = {
        "temperature": 1.0,  # Balanced sampling for diverse outputs
        "top_p": 1.0,  # Full probability mass
        "top_k": 20,  # Prevent repetitive loops (community tested)
    }

    # Perplexity normalization (typical range for coherent text is 1-50)
    MAX_PERPLEXITY = 100.0

    # Weights for combined confidence
    DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)  # (perplexity, entropy, consistency)

    def __init__(
        self,
        llm: ChatOllama,
        embeddings: OllamaEmbeddings,
        ollama_host: str,
        model: str = "nemotron-3-nano:30b",
    ):
        """Initialize uncertainty calculator.

        Args:
            llm: Main LLM for reference (not used for sampling)
            embeddings: Embedding model for semantic analysis
            ollama_host: Ollama API host URL
            model: Model name for perplexity and sampling
        """
        self.llm = llm
        self.embeddings = embeddings
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model

        # Create diversity-focused LLM for sampling (same model, different temp)
        self.sampling_llm = ChatOllama(
            model=model,
            base_url=ollama_host,
            **self.DIVERSITY_SETTINGS,
        )

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity via Ollama logprobs API.

        Perplexity = exp(mean negative log probability)
        Lower perplexity = model is more confident in the text.

        Args:
            text: Text to compute perplexity for

        Returns:
            Perplexity score (typically 1-100+ for coherent text)
        """
        try:
            # Note: Ollama's logprobs are for GENERATED tokens, not input evaluation.
            # We generate a few continuation tokens and use their logprobs as a proxy
            # for how well the model "understands" the input text.
            # Low logprobs (close to 0) = model confidently continues the text.
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": text,
                    "logprobs": True,  # Must be top-level, not inside options
                    "stream": False,
                    "options": {"num_predict": 10},  # Only generate 10 tokens for speed
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # Extract log probabilities from response
            # Ollama returns logprobs as list of {token, logprob, bytes} objects
            log_probs = []
            if isinstance(data.get("logprobs"), list):
                for token_data in data["logprobs"]:
                    if "logprob" in token_data:
                        log_probs.append(token_data["logprob"])

            if not log_probs:
                logger.warning("No logprobs returned from Ollama API")
                return float("inf")

            # Perplexity = exp(mean negative log probability)
            avg_neg_log_prob = -sum(log_probs) / len(log_probs)
            perplexity = math.exp(avg_neg_log_prob)

            logger.debug(f"Perplexity: {perplexity:.2f} ({len(log_probs)} tokens)")
            return perplexity

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return float("inf")
        except Exception as e:
            logger.error(f"Perplexity calculation failed: {e}")
            return float("inf")

    def normalize_perplexity(self, perplexity: float) -> float:
        """Convert perplexity to 0-1 confidence (1 = most certain).

        Args:
            perplexity: Raw perplexity score

        Returns:
            Normalized score (0-1, higher = more certain)
        """
        if perplexity == float("inf") or perplexity <= 0:
            return 0.0
        return max(0.0, 1.0 - (perplexity / self.MAX_PERPLEXITY))

    def generate_answers(
        self,
        question: str,
        context: str,
        n_generations: int = 3,
    ) -> List[str]:
        """Generate n diverse answers using sampling LLM.

        Uses Nemotron-optimized settings:
        - temperature=1.0, top_p=1.0 for diversity
        - top_k=20 to prevent repetitive loops
        - System prompt: "detailed thinking off" (skip <think> blocks)

        Args:
            question: The question to answer
            context: Retrieved context for answering
            n_generations: Number of answers to generate

        Returns:
            List of generated answers
        """
        # System prompt without thinking mode for faster sampling
        system_msg = SystemMessage(
            content="You are a helpful assistant. detailed thinking off. "
            "Answer the question based on the provided context. Be concise."
        )

        # User prompt with context and question
        user_msg = HumanMessage(
            content=f"Context:\n{context[:4000]}\n\nQuestion: {question}\n\nAnswer:"
        )

        answers = []
        for i in range(n_generations):
            try:
                response = self.sampling_llm.invoke([system_msg, user_msg])
                answer = response.content.strip()
                if answer:
                    answers.append(answer)
                    logger.debug(f"Generated answer {i+1}: {answer[:100]}...")
            except Exception as e:
                logger.warning(f"Answer generation {i+1} failed: {e}")

        if not answers:
            logger.warning("No answers generated for uncertainty calculation")

        return answers

    def compute_semantic_entropy(self, answers: List[str]) -> float:
        """Compute entropy over semantic clusters of answers.

        Clusters answers by embedding similarity, then computes Shannon entropy
        over the cluster distribution. Lower entropy = more consistent answers.

        Args:
            answers: List of generated answers

        Returns:
            Normalized entropy (0-1, lower = more certain)
        """
        if len(answers) < 2:
            return 0.0  # Single answer = no uncertainty

        try:
            # Embed all answers
            vectors = self.embeddings.embed_documents(answers)

            # Cluster by cosine similarity (threshold: 0.9)
            clusters = cluster_by_similarity(vectors, threshold=0.9)

            # Compute entropy over cluster sizes
            n = len(answers)
            probs = [len(c) / n for c in clusters]

            entropy = 0.0
            for p in probs:
                if p > 0:
                    entropy -= p * math.log(p)

            # Normalize by max entropy (log(n) when all answers in different clusters)
            max_entropy = math.log(n)
            normalized = entropy / max_entropy if max_entropy > 0 else 0.0

            logger.debug(
                f"Semantic entropy: {normalized:.3f} "
                f"({len(clusters)} clusters from {n} answers)"
            )
            return normalized

        except Exception as e:
            logger.error(f"Semantic entropy calculation failed: {e}")
            return 0.5  # Default to medium uncertainty

    def compute_embedding_consistency(self, answers: List[str]) -> float:
        """Compute pairwise cosine similarity of answer embeddings.

        Higher consistency = more similar answers = higher confidence.

        Args:
            answers: List of generated answers

        Returns:
            Mean pairwise cosine similarity (0-1, higher = more certain)
        """
        if len(answers) < 2:
            return 1.0  # Single answer = fully consistent

        try:
            # Embed all answers
            vectors = self.embeddings.embed_documents(answers)

            # Compute pairwise cosine similarities
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sim = cosine_similarity(vectors[i], vectors[j])
                    similarities.append(sim)

            mean_similarity = (
                sum(similarities) / len(similarities) if similarities else 0.0
            )

            logger.debug(
                f"Embedding consistency: {mean_similarity:.3f} "
                f"({len(similarities)} pairs)"
            )
            return mean_similarity

        except Exception as e:
            logger.error(f"Embedding consistency calculation failed: {e}")
            return 0.5  # Default to medium consistency

    def combine_uncertainties(
        self,
        perplexity_norm: float,
        semantic_entropy_norm: float,
        embedding_consistency: float,
        weights: Optional[Tuple[float, float, float]] = None,
        perplexity_available: bool = True,
    ) -> float:
        """Compute weighted combination of uncertainty metrics.

        Args:
            perplexity_norm: Normalized perplexity (0-1, higher = more certain)
            semantic_entropy_norm: Normalized entropy (0-1, lower = more certain)
            embedding_consistency: Consistency score (0-1, higher = more certain)
            weights: Tuple of weights (perplexity, entropy, consistency)
            perplexity_available: Whether perplexity metric is available

        Returns:
            Combined confidence score (0-1)
        """
        weights = weights or self.DEFAULT_WEIGHTS

        # Invert entropy (lower entropy = higher confidence)
        entropy_confidence = 1.0 - semantic_entropy_norm

        if not perplexity_available:
            # Re-weight when perplexity is unavailable (just entropy + consistency)
            total_weight = weights[1] + weights[2]
            if total_weight > 0:
                combined = (
                    (weights[1] / total_weight) * entropy_confidence
                    + (weights[2] / total_weight) * embedding_consistency
                )
            else:
                combined = (entropy_confidence + embedding_consistency) / 2.0
        else:
            combined = (
                weights[0] * perplexity_norm
                + weights[1] * entropy_confidence
                + weights[2] * embedding_consistency
            )

        return max(0.0, min(1.0, combined))

    def compute_all(
        self,
        question: str,
        answer: str,
        context: str,
        n_generations: int = 3,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> UncertaintyScores:
        """Compute all uncertainty metrics for a given answer.

        Args:
            question: The original question
            answer: The answer to evaluate
            context: Retrieved context used for answering
            n_generations: Number of answer generations for entropy/consistency
            weights: Optional custom weights for combining metrics

        Returns:
            UncertaintyScores with all metrics
        """
        logger.info(f"Computing uncertainty metrics (n_generations={n_generations})")

        # 1. Compute perplexity on the original answer (may not be available)
        perplexity = self.compute_perplexity(answer)
        perplexity_available = perplexity != float("inf")
        perplexity_norm = self.normalize_perplexity(perplexity) if perplexity_available else 0.0

        if not perplexity_available:
            logger.info("Perplexity unavailable (logprobs not supported), using entropy + consistency only")

        # 2. Generate diverse answers (shared for entropy & consistency)
        generated_answers = self.generate_answers(question, context, n_generations)

        # 3. Compute semantic entropy
        semantic_entropy = self.compute_semantic_entropy(generated_answers)
        semantic_entropy_norm = 1.0 - semantic_entropy  # Invert for confidence

        # 4. Compute embedding consistency
        embedding_consistency = self.compute_embedding_consistency(generated_answers)

        # 5. Combine into final confidence score
        combined = self.combine_uncertainties(
            perplexity_norm,
            semantic_entropy,  # Note: passed as raw entropy, combined inverts it
            embedding_consistency,
            weights,
            perplexity_available=perplexity_available,
        )

        scores = UncertaintyScores(
            perplexity=perplexity if perplexity_available else -1.0,  # -1 indicates unavailable
            perplexity_normalized=perplexity_norm,
            semantic_entropy=semantic_entropy,
            semantic_entropy_normalized=semantic_entropy_norm,
            embedding_consistency=embedding_consistency,
            combined_confidence=combined,
        )

        logger.info(
            f"Uncertainty scores: PPL={'N/A' if not perplexity_available else f'{perplexity:.1f}'} ({perplexity_norm:.2f}), "
            f"Entropy={semantic_entropy:.2f}, Consistency={embedding_consistency:.2f}, "
            f"Combined={combined:.2f}"
        )

        return scores


def compute_all_uncertainties(
    question: str,
    answer: str,
    context: str,
    llm: ChatOllama,
    embeddings: OllamaEmbeddings,
    model: str = "nemotron-3-nano:30b",
    ollama_host: str = "http://host.docker.internal:11434",
    n_generations: int = 3,
) -> UncertaintyScores:
    """Convenience function to compute all uncertainty metrics.

    Args:
        question: The original question
        answer: The answer to evaluate
        context: Retrieved context used for answering
        llm: Main LLM instance
        embeddings: Embedding model
        model: Model name for API calls
        ollama_host: Ollama API host URL
        n_generations: Number of answer generations

    Returns:
        UncertaintyScores with all metrics
    """
    calculator = UncertaintyCalculator(llm, embeddings, ollama_host, model)
    return calculator.compute_all(question, answer, context, n_generations)


if __name__ == "__main__":
    # Quick test
    import os

    logging.basicConfig(level=logging.INFO)

    host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    model = os.getenv("OLLAMA_MAIN_MODEL", "nemotron-3-nano:30b")

    llm = ChatOllama(model=model, base_url=host, temperature=0)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b", base_url=host)

    calculator = UncertaintyCalculator(llm, embeddings, host, model)

    # Test with a simple question
    question = "What is inflation?"
    context = "Inflation is when prices rise across the economy over time."
    answer = "Inflation refers to the general increase in prices of goods and services."

    scores = calculator.compute_all(question, answer, context, n_generations=3)

    print("\n=== Uncertainty Scores ===")
    print(f"Perplexity: {scores.perplexity:.2f} (normalized: {scores.perplexity_normalized:.2f})")
    print(f"Semantic Entropy: {scores.semantic_entropy:.2f} (normalized: {scores.semantic_entropy_normalized:.2f})")
    print(f"Embedding Consistency: {scores.embedding_consistency:.2f}")
    print(f"Combined Confidence: {scores.combined_confidence:.2%}")
