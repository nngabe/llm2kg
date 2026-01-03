"""
Web search package for external information retrieval.

Provides Tavily-based web search with trusted source validation.
"""

from .search_client import WebSearchClient, SearchResult, DocumentContent
from .source_manager import TrustedSourceManager, SourceMetadata

__all__ = [
    "WebSearchClient",
    "SearchResult",
    "DocumentContent",
    "TrustedSourceManager",
    "SourceMetadata",
]
