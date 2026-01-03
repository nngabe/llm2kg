"""
Citation accuracy checker.

Validates that citations in answers are accurate and traceable.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationStatus(str, Enum):
    """Status of a citation check."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    HALLUCINATED = "hallucinated"


class CitationResult(BaseModel):
    """Result of checking a single citation."""

    source_id: str
    source_type: str = "unknown"
    status: CitationStatus = CitationStatus.UNVERIFIED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    excerpt: str = ""
    matched_text: str = ""
    reason: str = ""


class CitationChecker:
    """
    Checker for citation accuracy.

    Verifies that claims in answers are supported by cited sources.
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.75,
    ):
        """
        Initialize citation checker.

        Args:
            embedding_model: Model for semantic similarity.
            similarity_threshold: Threshold for verification.
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

    def verify_citation(
        self,
        claim: str,
        cited_source: str,
        source_content: Optional[str] = None,
    ) -> CitationResult:
        """
        Verify a single citation.

        Args:
            claim: The claim being cited.
            cited_source: The source being cited.
            source_content: Full content of the source.

        Returns:
            CitationResult with verification status.
        """
        result = CitationResult(
            source_id=cited_source,
            excerpt=claim[:200] if claim else "",
        )

        if not source_content:
            result.status = CitationStatus.UNVERIFIED
            result.reason = "Source content not available"
            return result

        # Check exact match
        if claim.lower() in source_content.lower():
            result.status = CitationStatus.VERIFIED
            result.confidence = 1.0
            result.matched_text = claim
            result.reason = "Exact match found"
            return result

        # Check semantic similarity
        self._ensure_embedding_model()

        try:
            claim_embedding = self._embedding_model.embed_query(claim)

            # Split source into chunks
            chunks = self._split_into_chunks(source_content, chunk_size=500)

            best_similarity = 0.0
            best_chunk = ""

            for chunk in chunks:
                chunk_embedding = self._embedding_model.embed_query(chunk)

                import numpy as np
                claim_vec = np.array(claim_embedding)
                chunk_vec = np.array(chunk_embedding)

                similarity = np.dot(claim_vec, chunk_vec) / (
                    np.linalg.norm(claim_vec) * np.linalg.norm(chunk_vec)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_chunk = chunk

            result.confidence = float(best_similarity)
            result.matched_text = best_chunk[:200]

            if best_similarity >= self.similarity_threshold:
                result.status = CitationStatus.VERIFIED
                result.reason = f"Semantic match ({best_similarity:.2f})"
            elif best_similarity >= 0.5:
                result.status = CitationStatus.PARTIALLY_VERIFIED
                result.reason = f"Partial match ({best_similarity:.2f})"
            else:
                result.status = CitationStatus.HALLUCINATED
                result.reason = f"No match found ({best_similarity:.2f})"

        except Exception as e:
            logger.warning(f"Citation verification failed: {e}")
            result.status = CitationStatus.UNVERIFIED
            result.reason = str(e)

        return result

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size // 2):
            chunk = " ".join(words[i:i + chunk_size // 5])
            if chunk:
                chunks.append(chunk)

        return chunks if chunks else [text]

    def check_all_citations(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        sources: Dict[str, str],
    ) -> List[CitationResult]:
        """
        Check all citations in an answer.

        Args:
            answer: The answer text.
            citations: List of citation dicts.
            sources: Dict mapping source_id to content.

        Returns:
            List of citation results.
        """
        results = []

        for citation in citations:
            source_id = citation.get("source_id", "")
            excerpt = citation.get("excerpt", "")
            source_type = citation.get("source_type", "unknown")

            source_content = sources.get(source_id, "")

            result = self.verify_citation(
                claim=excerpt,
                cited_source=source_id,
                source_content=source_content,
            )
            result.source_type = source_type

            results.append(result)

        return results

    def compute_citation_accuracy(
        self,
        results: List[CitationResult],
    ) -> float:
        """
        Compute overall citation accuracy.

        Args:
            results: List of citation results.

        Returns:
            Accuracy score 0-1.
        """
        if not results:
            return 1.0  # No citations = perfect accuracy

        verified = sum(
            1 for r in results
            if r.status in [CitationStatus.VERIFIED, CitationStatus.PARTIALLY_VERIFIED]
        )

        return verified / len(results)

    def check_hallucinated_citations(
        self,
        citations: List[Dict[str, Any]],
        available_sources: List[str],
    ) -> List[str]:
        """
        Find citations to non-existent sources.

        Args:
            citations: List of citation dicts.
            available_sources: List of available source IDs.

        Returns:
            List of hallucinated source IDs.
        """
        available_set = set(s.lower() for s in available_sources)
        hallucinated = []

        for citation in citations:
            source_id = citation.get("source_id", "").lower()
            if source_id and source_id not in available_set:
                hallucinated.append(source_id)

        return hallucinated
