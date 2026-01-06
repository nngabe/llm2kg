"""
Citation Recall Metric.

Custom LLM-as-judge metric measuring how well claims are supported by citations.
"""

import json
import logging
from typing import Any, Dict, List

from ..base import LLMJudgeMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer.
Return a JSON array of claim strings.

Answer: {answer}

Return ONLY a JSON object in this format:
{{
    "claims": ["claim 1", "claim 2", ...]
}}
"""

CITATION_CHECK_PROMPT = """Determine if the following claim is supported by any of the provided citations.

Claim: {claim}

Citations:
{citations}

Evaluate whether the claim can be verified from the citation content.
Return a JSON object with your assessment:
{{
    "supported": true/false,
    "supporting_citation": "citation ID or null if not supported",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""


class CitationRecallMetric(LLMJudgeMetric):
    """Measures citation recall using LLM-as-judge.

    Citation Recall = (Claims with valid citations) / (Total claims)

    This metric uses a two-step LLM evaluation:
    1. Extract factual claims from the answer
    2. Check each claim against provided citations
    """

    layer = EvaluationLayer.GENERATION
    name = "Citation Recall"

    def __init__(
        self,
        threshold: float = 0.60,
        judge_model: str = "gpt-5.2",
        temperature: float = 0.0,
    ):
        super().__init__(threshold, judge_model, temperature)

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer using LLM.

        Args:
            answer: The answer text

        Returns:
            List of claim strings
        """
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        result = self._invoke_judge(prompt)

        if "error" in result:
            logger.warning(f"Claim extraction failed: {result['error']}")
            # Fallback: split answer into sentences as claims
            sentences = [s.strip() for s in answer.replace("!", ".").replace("?", ".").split(".")]
            return [s for s in sentences if len(s) > 10]

        return result.get("claims", [])

    def _check_citation_support(
        self,
        claim: str,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check if a claim is supported by citations.

        Args:
            claim: The claim to verify
            citations: List of citation dictionaries

        Returns:
            Dictionary with support status and details
        """
        if not citations:
            return {
                "supported": False,
                "supporting_citation": None,
                "confidence": 1.0,
                "reasoning": "No citations provided",
            }

        # Format citations for the prompt
        citations_text = "\n".join([
            f"[{i+1}] Source: {c.get('source_title', c.get('source_id', 'Unknown'))}\n"
            f"    Type: {c.get('source_type', 'unknown')}\n"
            f"    Content: {c.get('excerpt', c.get('content', 'No content'))[:500]}"
            for i, c in enumerate(citations)
        ])

        prompt = CITATION_CHECK_PROMPT.format(
            claim=claim,
            citations=citations_text,
        )

        result = self._invoke_judge(prompt)

        if "error" in result:
            logger.warning(f"Citation check failed: {result['error']}")
            return {
                "supported": False,
                "supporting_citation": None,
                "confidence": 0.0,
                "reasoning": f"Judge error: {result['error']}",
            }

        return result

    def _fallback_citation_recall(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
    ) -> float:
        """Fallback citation recall without LLM judge.

        Simple heuristic: check if citations contain relevant content.
        """
        if not answer or not citations:
            return 0.0

        answer_lower = answer.lower()
        citation_content = " ".join([
            c.get("excerpt", c.get("content", "")).lower()
            for c in citations
        ])

        # Extract key phrases from answer
        words = answer_lower.split()
        # Look for 3-grams that appear in citations
        supported = 0
        total = 0

        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            total += 1
            if phrase in citation_content:
                supported += 1

        return supported / total if total > 0 else 0.0

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure citation recall.

        Args:
            test_case: Test case
            agent_output: Agent output with answer and citations

        Returns:
            MetricResult with citation recall score
        """
        answer = agent_output.answer
        citations = agent_output.citations

        if not answer:
            return self._create_result(
                score=0.0,
                details={"note": "Empty answer"},
            )

        # Try LLM-based evaluation
        try:
            # Step 1: Extract claims
            claims = self._extract_claims(answer)

            if not claims:
                return self._create_result(
                    score=1.0,
                    details={"note": "No factual claims extracted from answer"},
                )

            # Step 2: Check each claim
            claim_results = []
            supported_count = 0

            for claim in claims:
                result = self._check_citation_support(claim, citations)
                claim_results.append({
                    "claim": claim,
                    **result,
                })
                if result.get("supported", False):
                    supported_count += 1

            # Calculate recall
            recall = supported_count / len(claims)

            return self._create_result(
                score=recall,
                details={
                    "total_claims": len(claims),
                    "supported_claims": supported_count,
                    "citation_count": len(citations),
                    "claim_results": claim_results,
                    "method": "llm_judge",
                },
            )

        except Exception as e:
            logger.warning(f"LLM-based citation recall failed, using fallback: {e}")

            # Fallback implementation
            score = self._fallback_citation_recall(answer, citations)

            return self._create_result(
                score=score,
                details={
                    "citation_count": len(citations),
                    "method": "fallback",
                    "error": str(e),
                },
            )
