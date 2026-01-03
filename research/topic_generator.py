"""
Research topic generator from identified knowledge gaps.

Generates prioritized research topics numbered 1-10 for user approval.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .gap_detector import Gap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchTopic(BaseModel):
    """A research topic generated from gaps."""

    number: int = Field(ge=1, le=10, description="Topic number 1-10")
    title: str = Field(description="Short title for the topic")
    description: str = Field(description="Detailed description of what to research")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_documents: int = Field(default=5, description="Estimated documents needed")
    search_queries: List[str] = Field(default_factory=list)
    related_gaps: List[str] = Field(default_factory=list)
    approved: bool = Field(default=False)


TOPIC_GENERATION_PROMPT = """Based on the following knowledge gaps, generate {num_topics} research topics.
Each topic should address one or more gaps and be actionable for web research.

Knowledge Gaps:
{gaps}

Generate topics in JSON format:
[
    {{
        "number": 1,
        "title": "Short descriptive title",
        "description": "What to research and why",
        "priority": 0.0-1.0,
        "estimated_documents": 3-10,
        "search_queries": ["query 1", "query 2"]
    }},
    ...
]

Guidelines:
- Order by priority (most important first)
- Each topic should be specific and searchable
- Include 2-3 search queries per topic
- Estimate documents needed (3-10 per topic)
- Combine related gaps into single topics where appropriate"""


class ResearchTopicGenerator:
    """
    Generator for research topics from knowledge gaps.

    Uses LLM to synthesize gaps into actionable research topics.
    """

    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the topic generator.

        Args:
            llm: Language model to use. Defaults to GPT-4.
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.3)

    def generate_topics(
        self,
        gaps: List[Gap],
        num_topics: int = 10,
    ) -> List[ResearchTopic]:
        """
        Generate research topics from gaps.

        Args:
            gaps: List of identified knowledge gaps.
            num_topics: Number of topics to generate (1-10).

        Returns:
            List of research topics numbered 1 to num_topics.
        """
        num_topics = min(max(num_topics, 1), 10)

        if not gaps:
            return self._generate_default_topics(num_topics)

        # Format gaps for prompt
        gap_descriptions = []
        for i, gap in enumerate(gaps[:20], 1):  # Limit to 20 gaps
            gap_descriptions.append(
                f"{i}. [{gap.gap_type}] {gap.description} "
                f"(importance: {gap.importance_score:.2f})"
            )

        gaps_text = "\n".join(gap_descriptions)

        prompt = ChatPromptTemplate.from_template(TOPIC_GENERATION_PROMPT)

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "num_topics": num_topics,
                "gaps": gaps_text,
            })

            # Parse response
            import json
            content = response.content

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            topics_data = json.loads(content)

            topics = []
            for i, data in enumerate(topics_data[:num_topics], 1):
                topic = ResearchTopic(
                    number=i,
                    title=data.get("title", f"Topic {i}"),
                    description=data.get("description", ""),
                    priority=data.get("priority", 0.5),
                    estimated_documents=data.get("estimated_documents", 5),
                    search_queries=data.get("search_queries", []),
                )
                topics.append(topic)

            return topics

        except Exception as e:
            logger.error(f"Topic generation failed: {e}")
            return self._generate_default_topics(num_topics)

    def _generate_default_topics(self, num_topics: int) -> List[ResearchTopic]:
        """Generate default topics when LLM generation fails."""
        return [
            ResearchTopic(
                number=i,
                title=f"Research Area {i}",
                description=f"General research topic {i} - requires manual review",
                priority=1.0 - (i * 0.1),
                estimated_documents=5,
                search_queries=[f"topic {i} research"],
            )
            for i in range(1, num_topics + 1)
        ]

    def prioritize_topics(
        self,
        topics: List[ResearchTopic],
    ) -> List[ResearchTopic]:
        """
        Re-prioritize topics based on criteria.

        Args:
            topics: List of topics to prioritize.

        Returns:
            Sorted list of topics by priority.
        """
        sorted_topics = sorted(
            topics,
            key=lambda t: t.priority,
            reverse=True,
        )

        # Renumber after sorting
        for i, topic in enumerate(sorted_topics, 1):
            topic.number = i

        return sorted_topics

    def format_for_approval(
        self,
        topics: List[ResearchTopic],
        include_queries: bool = False,
    ) -> str:
        """
        Format topics for user approval display.

        Args:
            topics: Topics to format.
            include_queries: Whether to include search queries.

        Returns:
            Formatted string for display.
        """
        lines = ["Research Topics for Approval:", ""]

        for topic in topics:
            priority_bar = "█" * int(topic.priority * 5) + "░" * (5 - int(topic.priority * 5))
            lines.append(f"{topic.number}. {topic.title}")
            lines.append(f"   Priority: {priority_bar} | Est. docs: {topic.estimated_documents}")
            lines.append(f"   {topic.description}")

            if include_queries and topic.search_queries:
                queries = ", ".join(topic.search_queries[:3])
                lines.append(f"   Queries: {queries}")

            lines.append("")

        lines.append("Reply with topic numbers to approve (e.g., 1,2,4,7,9,10)")

        return "\n".join(lines)

    def format_for_email(self, topics: List[ResearchTopic]) -> str:
        """
        Format topics for email approval.

        Args:
            topics: Topics to format.

        Returns:
            Email-formatted string.
        """
        lines = [
            "Knowledge Graph Research Topics",
            "=" * 40,
            "",
            "Please review the following research topics and reply with the",
            "numbers you want to approve (e.g., 1,2,4,7,9,10):",
            "",
        ]

        for topic in topics:
            lines.append(f"{topic.number}. {topic.title}")
            lines.append(f"   {topic.description}")
            lines.append(f"   Priority: {topic.priority:.0%} | Documents: ~{topic.estimated_documents}")
            lines.append("")

        lines.append("=" * 40)
        lines.append("Reply to this email with your approved topic numbers.")

        return "\n".join(lines)
