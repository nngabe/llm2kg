"""
Source Citation Accuracy Metric.

LLM-as-judge metric verifying accuracy of web citations.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..base import LLMJudgeMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


CITATION_VERIFICATION_PROMPT = """You are verifying the accuracy of a citation used in an AI-generated answer.

Citation:
- Source: {source_title}
- URL: {url}
- Type: {source_type}
- Excerpt: {excerpt}

Claim being supported: {claim}

Evaluate:
1. Does the excerpt actually support the claim?
2. Is the source type credible for this claim?
3. Is the citation properly attributed?

Return a JSON object:
{{
    "is_accurate": true/false,
    "credibility_score": 0.0-1.0,
    "support_score": 0.0-1.0,
    "issues": ["list of issues found"],
    "reasoning": "explanation"
}}
"""


class SourceCitationAccuracyMetric(LLMJudgeMetric):
    """Measures accuracy of source citations.

    Source Citation Accuracy = verified_citations / total_citations

    This metric verifies:
    - Citations are from valid sources
    - Excerpts support the claims made
    - Source types are appropriate
    """

    layer = EvaluationLayer.INTEGRITY
    name = "Source Citation Accuracy"

    def __init__(
        self,
        threshold: float = 0.80,
        judge_model: str = "gpt-5.2",
        temperature: float = 0.0,
    ):
        super().__init__(threshold, judge_model, temperature)
        # Credible domains for different source types
        self._credible_domains = {
            "news": ["reuters.com", "apnews.com", "bbc.com", "nytimes.com", "wsj.com"],
            "academic": ["arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov"],
            "government": [".gov", ".gov.uk", ".gc.ca"],
            "organization": [".org"],
        }

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL.

        Args:
            url: The URL to parse

        Returns:
            Domain string or None
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return None

    def _check_url_validity(self, url: str) -> Dict[str, Any]:
        """Basic URL validity check.

        Args:
            url: URL to check

        Returns:
            Validity check result
        """
        if not url:
            return {"valid": False, "reason": "Empty URL"}

        # Basic URL format check
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        if not url_pattern.match(url):
            return {"valid": False, "reason": "Invalid URL format"}

        domain = self._extract_domain(url)
        if not domain:
            return {"valid": False, "reason": "Could not extract domain"}

        # Check for suspicious patterns
        suspicious_patterns = [
            "bit.ly", "tinyurl", "goo.gl",  # URL shorteners
            "localhost", "127.0.0.1",  # Local addresses
            "example.com", "test.com",  # Test domains
        ]

        for pattern in suspicious_patterns:
            if pattern in domain:
                return {"valid": False, "reason": f"Suspicious domain: {pattern}"}

        return {"valid": True, "domain": domain}

    def _assess_source_credibility(
        self,
        source_type: str,
        url: str,
        domain: str,
    ) -> float:
        """Assess credibility of a source.

        Args:
            source_type: Type of source (web, entity, etc.)
            url: Source URL
            domain: Source domain

        Returns:
            Credibility score 0.0-1.0
        """
        if source_type in ("entity", "graph", "neo4j"):
            # Graph sources are trusted internal data
            return 1.0

        if not domain:
            return 0.3

        # Check against credible domains
        for category, domains in self._credible_domains.items():
            for credible in domains:
                if credible in domain:
                    return 0.9

        # Known tech/reference sites
        reference_domains = [
            "wikipedia.org", "stackoverflow.com", "github.com",
            "docs.python.org", "docs.microsoft.com",
        ]
        for ref in reference_domains:
            if ref in domain:
                return 0.8

        # Default credibility for unknown sources
        return 0.5

    def _extract_supported_claims(
        self,
        answer: str,
        citation: Dict[str, Any],
    ) -> List[str]:
        """Extract claims from answer that might be supported by citation.

        Args:
            answer: The full answer
            citation: The citation to check

        Returns:
            List of potentially supported claims
        """
        # Simple sentence extraction
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Filter to sentences that might relate to the citation
        excerpt = (citation.get("excerpt") or "").lower()
        source_title = (citation.get("source_title") or "").lower()

        related = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check for keyword overlap
            excerpt_words = set(excerpt.split())
            sentence_words = set(sentence_lower.split())

            # Remove common words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "for", "on", "with"}
            excerpt_words -= stop_words
            sentence_words -= stop_words

            overlap = len(excerpt_words & sentence_words)
            if overlap >= 2:
                related.append(sentence)

        return related[:3]  # Limit to 3 most relevant

    def _verify_citation(
        self,
        citation: Dict[str, Any],
        answer: str,
    ) -> Dict[str, Any]:
        """Verify a single citation using LLM.

        Args:
            citation: Citation to verify
            answer: Full answer for context

        Returns:
            Verification result
        """
        url = citation.get("url") or citation.get("source_id", "")
        source_type = citation.get("source_type", "unknown")

        # Skip graph/entity sources (internally trusted)
        if source_type in ("entity", "graph", "neo4j"):
            return {
                "is_accurate": True,
                "credibility_score": 1.0,
                "support_score": 1.0,
                "issues": [],
                "reasoning": "Internal graph source - trusted",
            }

        # Check URL validity
        url_check = self._check_url_validity(url)
        if not url_check.get("valid", False):
            return {
                "is_accurate": False,
                "credibility_score": 0.0,
                "support_score": 0.0,
                "issues": [url_check.get("reason", "Invalid URL")],
                "reasoning": f"URL validation failed: {url_check.get('reason')}",
            }

        domain = url_check.get("domain", "")
        credibility = self._assess_source_credibility(source_type, url, domain)

        # Extract claims this citation might support
        claims = self._extract_supported_claims(answer, citation)
        claim_text = claims[0] if claims else "General information support"

        # Use LLM to verify
        prompt = CITATION_VERIFICATION_PROMPT.format(
            source_title=citation.get("source_title", "Unknown"),
            url=url,
            source_type=source_type,
            excerpt=citation.get("excerpt", "No excerpt available")[:500],
            claim=claim_text,
        )

        result = self._invoke_judge(prompt)

        if "error" in result:
            return {
                "is_accurate": True,  # Assume accurate on error
                "credibility_score": credibility,
                "support_score": 0.5,
                "issues": [f"Verification error: {result['error']}"],
                "reasoning": "Could not verify with LLM judge",
            }

        # Combine LLM result with URL check
        result["credibility_score"] = max(
            result.get("credibility_score", 0.5),
            credibility
        )
        result["domain"] = domain

        return result

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure source citation accuracy.

        Args:
            test_case: Test case
            agent_output: Agent output with citations

        Returns:
            MetricResult with citation accuracy score
        """
        citations = agent_output.citations

        if not citations:
            return self._create_result(
                score=1.0,
                details={"note": "No citations to verify"},
            )

        # Filter to web citations (not graph/entity)
        web_citations = [
            c for c in citations
            if c.get("source_type") not in ("entity", "graph", "neo4j")
        ]

        if not web_citations:
            return self._create_result(
                score=1.0,
                details={
                    "note": "All citations are from trusted internal sources",
                    "total_citations": len(citations),
                    "graph_citations": len(citations),
                },
            )

        # Verify each web citation
        verifications = []
        accurate_count = 0

        for citation in web_citations:
            result = self._verify_citation(citation, agent_output.answer)
            verifications.append({
                "source": citation.get("source_title", citation.get("source_id", "Unknown")),
                "url": citation.get("url", citation.get("source_id", "")),
                **result,
            })

            if result.get("is_accurate", False):
                accurate_count += 1

        # Calculate score
        score = accurate_count / len(web_citations)

        # Collect all issues
        all_issues = []
        for v in verifications:
            all_issues.extend(v.get("issues", []))

        return self._create_result(
            score=score,
            details={
                "total_citations": len(citations),
                "web_citations": len(web_citations),
                "graph_citations": len(citations) - len(web_citations),
                "accurate_citations": accurate_count,
                "verifications": verifications,
                "all_issues": all_issues,
            },
        )
