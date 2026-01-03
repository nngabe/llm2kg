"""
Frontend mode handlers for different interaction styles.

Modes:
- Q&A Mode: Answer questions using hybrid GraphRAG + web search
- Research Mode: Autonomous research to fill knowledge gaps
"""

from .qa_mode import QAModeHandler
from .research_mode import ResearchModeHandler

__all__ = ["QAModeHandler", "ResearchModeHandler"]
