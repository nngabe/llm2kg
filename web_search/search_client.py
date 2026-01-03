"""
Web search client using Tavily API.

Provides web search functionality with trusted source filtering and content extraction.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """A single search result from web search."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Snippet/content from the search result")
    score: float = Field(default=0.0, description="Relevance score")
    domain: str = Field(default="", description="Domain of the URL")
    published_date: Optional[str] = Field(default=None, description="Publication date if available")

    def __post_init__(self):
        """Extract domain from URL if not provided."""
        if not self.domain and self.url:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            self.domain = parsed.netloc


class DocumentContent(BaseModel):
    """Full content extracted from a web page."""

    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    content: str = Field(description="Full text content")
    domain: str = Field(description="Domain of the URL")
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    word_count: int = Field(default=0, description="Word count of content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __post_init__(self):
        """Calculate word count."""
        if self.content and self.word_count == 0:
            self.word_count = len(self.content.split())


class WebSearchClient:
    """
    Web search client using Tavily API.

    Provides search functionality with optional trusted source filtering.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the web search client.

        Args:
            api_key: Tavily API key. If not provided, uses TAVILY_API_KEY env var.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("No Tavily API key provided. Web search will be unavailable.")

        self.base_url = "https://api.tavily.com"
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()

    def search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_raw_content: bool = False,
    ) -> List[SearchResult]:
        """
        Perform a web search.

        Args:
            query: Search query string.
            num_results: Number of results to return (1-10).
            search_depth: "basic" or "advanced" (advanced is slower but more thorough).
            include_answer: Whether to include AI-generated answer.
            include_raw_content: Whether to include raw HTML content.

        Returns:
            List of SearchResult objects.
        """
        if not self.api_key:
            logger.error("Cannot search: No API key configured")
            return []

        num_results = max(1, min(10, num_results))

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        try:
            response = self._client.post(
                f"{self.base_url}/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                from urllib.parse import urlparse
                domain = urlparse(item.get("url", "")).netloc

                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    domain=domain,
                    published_date=item.get("published_date"),
                )
                results.append(result)

            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def search_trusted_only(
        self,
        query: str,
        trusted_domains: List[str],
        num_results: int = 5,
    ) -> List[SearchResult]:
        """
        Search and filter results to trusted domains only.

        Args:
            query: Search query string.
            trusted_domains: List of trusted domain patterns.
            num_results: Desired number of results.

        Returns:
            List of SearchResult objects from trusted domains.
        """
        # Request more results to account for filtering
        raw_results = self.search(
            query=query,
            num_results=min(num_results * 3, 10),
            search_depth="advanced",
        )

        filtered = []
        for result in raw_results:
            if self._is_trusted_domain(result.domain, trusted_domains):
                filtered.append(result)
                if len(filtered) >= num_results:
                    break

        logger.info(
            f"Filtered {len(raw_results)} results to {len(filtered)} trusted results"
        )
        return filtered

    def _is_trusted_domain(self, domain: str, trusted_domains: List[str]) -> bool:
        """Check if domain matches any trusted domain pattern."""
        domain_lower = domain.lower()
        for trusted in trusted_domains:
            trusted_lower = trusted.lower()
            # Handle suffix matching (e.g., ".gov")
            if trusted_lower.startswith("."):
                if domain_lower.endswith(trusted_lower):
                    return True
            # Handle exact or subdomain matching
            elif domain_lower == trusted_lower or domain_lower.endswith("." + trusted_lower):
                return True
        return False

    def fetch_content(
        self,
        url: str,
        extract_main_content: bool = True,
    ) -> Optional[DocumentContent]:
        """
        Fetch and extract content from a URL.

        Uses Tavily's extract endpoint for clean content extraction.

        Args:
            url: URL to fetch content from.
            extract_main_content: Whether to extract only main content.

        Returns:
            DocumentContent object or None if extraction fails.
        """
        if not self.api_key:
            logger.error("Cannot fetch content: No API key configured")
            return None

        payload = {
            "api_key": self.api_key,
            "urls": [url],
        }

        try:
            response = self._client.post(
                f"{self.base_url}/extract",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                logger.warning(f"No content extracted from {url}")
                return None

            item = results[0]
            from urllib.parse import urlparse
            domain = urlparse(url).netloc

            content = DocumentContent(
                url=url,
                title=item.get("title", ""),
                content=item.get("raw_content", ""),
                domain=domain,
                metadata={
                    "extracted_at": datetime.now().isoformat(),
                },
            )

            logger.info(f"Extracted {content.word_count} words from {url}")
            return content

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching content: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            return None

    def search_and_extract(
        self,
        query: str,
        num_results: int = 3,
        trusted_domains: Optional[List[str]] = None,
    ) -> List[DocumentContent]:
        """
        Search and extract full content from results.

        Args:
            query: Search query string.
            num_results: Number of documents to extract.
            trusted_domains: Optional list of trusted domains to filter by.

        Returns:
            List of DocumentContent objects with full content.
        """
        if trusted_domains:
            search_results = self.search_trusted_only(
                query=query,
                trusted_domains=trusted_domains,
                num_results=num_results,
            )
        else:
            search_results = self.search(
                query=query,
                num_results=num_results,
            )

        documents = []
        for result in search_results:
            content = self.fetch_content(result.url)
            if content:
                documents.append(content)

        return documents


class AsyncWebSearchClient:
    """
    Async version of web search client for use with Chainlit/async frameworks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize async web search client."""
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com"
        self.timeout = timeout

    async def search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic",
    ) -> List[SearchResult]:
        """Async version of search."""
        if not self.api_key:
            logger.error("Cannot search: No API key configured")
            return []

        num_results = max(1, min(10, num_results))

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("results", []):
                from urllib.parse import urlparse
                domain = urlparse(item.get("url", "")).netloc

                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    domain=domain,
                    published_date=item.get("published_date"),
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error during async search: {e}")
            return []

    async def search_trusted_only(
        self,
        query: str,
        trusted_domains: List[str],
        num_results: int = 5,
    ) -> List[SearchResult]:
        """Async version of trusted-only search."""
        raw_results = await self.search(
            query=query,
            num_results=min(num_results * 3, 10),
            search_depth="advanced",
        )

        filtered = []
        for result in raw_results:
            if self._is_trusted_domain(result.domain, trusted_domains):
                filtered.append(result)
                if len(filtered) >= num_results:
                    break

        return filtered

    def _is_trusted_domain(self, domain: str, trusted_domains: List[str]) -> bool:
        """Check if domain matches any trusted domain pattern."""
        domain_lower = domain.lower()
        for trusted in trusted_domains:
            trusted_lower = trusted.lower()
            if trusted_lower.startswith("."):
                if domain_lower.endswith(trusted_lower):
                    return True
            elif domain_lower == trusted_lower or domain_lower.endswith("." + trusted_lower):
                return True
        return False

    async def fetch_content(self, url: str) -> Optional[DocumentContent]:
        """Async version of content extraction."""
        if not self.api_key:
            return None

        payload = {
            "api_key": self.api_key,
            "urls": [url],
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/extract",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])
            if not results:
                return None

            item = results[0]
            from urllib.parse import urlparse
            domain = urlparse(url).netloc

            return DocumentContent(
                url=url,
                title=item.get("title", ""),
                content=item.get("raw_content", ""),
                domain=domain,
            )

        except Exception as e:
            logger.error(f"Error during async content fetch: {e}")
            return None
