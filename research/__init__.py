"""
Research package for autonomous knowledge gap filling.

Components:
- GapDetector: Identifies gaps in the knowledge graph
- TopicGenerator: Generates research topics from gaps
- ResearchAgent: Autonomous research execution
"""

from .gap_detector import KnowledgeGapDetector, Gap
from .topic_generator import ResearchTopicGenerator, ResearchTopic
from .research_agent import ResearchAgent, ResearchState

__all__ = [
    "KnowledgeGapDetector",
    "Gap",
    "ResearchTopicGenerator",
    "ResearchTopic",
    "ResearchAgent",
    "ResearchState",
]
