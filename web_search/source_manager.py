"""
Trusted source management for web search.

Provides configuration and validation for trusted domains.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Trust level for sources."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceCategory(str, Enum):
    """Categories of trusted sources."""
    ACADEMIC = "academic"
    REFERENCE = "reference"
    NEWS = "news"
    GOVERNMENT = "government"
    CUSTOM = "custom"


class SourceMetadata(BaseModel):
    """Metadata for a trusted source."""

    domain: str = Field(description="Domain pattern (e.g., 'arxiv.org', '.gov')")
    trust_level: TrustLevel = Field(default=TrustLevel.MEDIUM)
    category: SourceCategory = Field(default=SourceCategory.CUSTOM)
    match_type: str = Field(
        default="exact",
        description="Match type: 'exact', 'suffix', or 'contains'"
    )
    description: Optional[str] = Field(default=None)
    enabled: bool = Field(default=True)


class TrustedSourceManager:
    """
    Manager for trusted source configuration and validation.

    Loads trusted sources from YAML config and provides domain validation.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Initialize the trusted source manager.

        Args:
            config_path: Path to trusted_sources.yaml config file.
            auto_load: Whether to automatically load config on init.
        """
        self.config_path = config_path or self._default_config_path()
        self.sources: Dict[str, SourceMetadata] = {}
        self.blocked_domains: Set[str] = set()
        self._domain_cache: Dict[str, bool] = {}

        if auto_load:
            self.load_config()

    def _default_config_path(self) -> str:
        """Get default config path."""
        base_dir = Path(__file__).parent.parent
        return str(base_dir / "config" / "trusted_sources.yaml")

    def load_config(self, config_path: Optional[str] = None) -> None:
        """
        Load trusted sources from YAML config file.

        Args:
            config_path: Optional override path.
        """
        path = config_path or self.config_path

        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}. Using defaults.")
            self._load_defaults()
            return

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            self.sources.clear()
            self.blocked_domains.clear()
            self._domain_cache.clear()

            # Load trusted sources by category
            trusted = config.get("trusted_sources", {})
            for category, sources in trusted.items():
                if isinstance(sources, list):
                    for source_config in sources:
                        self._add_source_from_config(source_config, category)

            # Load blocked domains
            blocked = config.get("blocked_domains", [])
            for domain in blocked:
                if isinstance(domain, str):
                    self.blocked_domains.add(domain.lower())
                elif isinstance(domain, dict):
                    self.blocked_domains.add(domain.get("domain", "").lower())

            logger.info(
                f"Loaded {len(self.sources)} trusted sources, "
                f"{len(self.blocked_domains)} blocked domains"
            )

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._load_defaults()

    def _add_source_from_config(
        self,
        config: Dict[str, Any],
        category: str,
    ) -> None:
        """Add a source from config dict."""
        domain = config.get("domain", "")
        if not domain:
            return

        try:
            source_category = SourceCategory(category)
        except ValueError:
            source_category = SourceCategory.CUSTOM

        try:
            trust_level = TrustLevel(config.get("trust_level", "medium"))
        except ValueError:
            trust_level = TrustLevel.MEDIUM

        metadata = SourceMetadata(
            domain=domain,
            trust_level=trust_level,
            category=source_category,
            match_type=config.get("match_type", "exact"),
            description=config.get("description"),
            enabled=config.get("enabled", True),
        )

        self.sources[domain.lower()] = metadata

    def _load_defaults(self) -> None:
        """Load default trusted sources."""
        defaults = [
            # Academic
            SourceMetadata(
                domain="arxiv.org",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.ACADEMIC,
            ),
            SourceMetadata(
                domain="scholar.google.com",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.ACADEMIC,
            ),
            SourceMetadata(
                domain="pubmed.gov",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.ACADEMIC,
            ),
            SourceMetadata(
                domain="semanticscholar.org",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.ACADEMIC,
            ),
            # Reference
            SourceMetadata(
                domain="wikipedia.org",
                trust_level=TrustLevel.MEDIUM,
                category=SourceCategory.REFERENCE,
            ),
            SourceMetadata(
                domain="britannica.com",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.REFERENCE,
            ),
            SourceMetadata(
                domain="stanford.edu",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.REFERENCE,
            ),
            # News
            SourceMetadata(
                domain="reuters.com",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.NEWS,
            ),
            SourceMetadata(
                domain="apnews.com",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.NEWS,
            ),
            SourceMetadata(
                domain="bbc.com",
                trust_level=TrustLevel.MEDIUM,
                category=SourceCategory.NEWS,
            ),
            # Government
            SourceMetadata(
                domain=".gov",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.GOVERNMENT,
                match_type="suffix",
            ),
            SourceMetadata(
                domain="who.int",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.GOVERNMENT,
            ),
            SourceMetadata(
                domain="un.org",
                trust_level=TrustLevel.HIGH,
                category=SourceCategory.GOVERNMENT,
            ),
        ]

        for source in defaults:
            self.sources[source.domain.lower()] = source

        logger.info(f"Loaded {len(self.sources)} default trusted sources")

    def is_trusted(self, url: str) -> bool:
        """
        Check if a URL is from a trusted source.

        Args:
            url: URL or domain to check.

        Returns:
            True if the domain is trusted and not blocked.
        """
        domain = self._extract_domain(url)
        if not domain:
            return False

        # Check cache
        if domain in self._domain_cache:
            return self._domain_cache[domain]

        # Check blocked first
        if self._is_blocked(domain):
            self._domain_cache[domain] = False
            return False

        # Check trusted sources
        is_trusted = self._matches_trusted(domain)
        self._domain_cache[domain] = is_trusted
        return is_trusted

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL or return if already a domain."""
        if "://" in url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        return url.lower()

    def _is_blocked(self, domain: str) -> bool:
        """Check if domain is blocked."""
        for blocked in self.blocked_domains:
            if blocked.startswith("."):
                if domain.endswith(blocked):
                    return True
            elif domain == blocked or domain.endswith("." + blocked):
                return True
        return False

    def _matches_trusted(self, domain: str) -> bool:
        """Check if domain matches any trusted source."""
        for pattern, metadata in self.sources.items():
            if not metadata.enabled:
                continue

            if metadata.match_type == "suffix":
                if domain.endswith(pattern):
                    return True
            elif metadata.match_type == "contains":
                if pattern in domain:
                    return True
            else:  # exact match (including subdomains)
                if domain == pattern or domain.endswith("." + pattern):
                    return True

        return False

    def get_trust_level(self, url: str) -> Optional[TrustLevel]:
        """
        Get trust level for a URL.

        Args:
            url: URL or domain to check.

        Returns:
            TrustLevel if trusted, None if not trusted.
        """
        domain = self._extract_domain(url)
        if not domain or self._is_blocked(domain):
            return None

        for pattern, metadata in self.sources.items():
            if not metadata.enabled:
                continue

            matches = False
            if metadata.match_type == "suffix":
                matches = domain.endswith(pattern)
            elif metadata.match_type == "contains":
                matches = pattern in domain
            else:
                matches = domain == pattern or domain.endswith("." + pattern)

            if matches:
                return metadata.trust_level

        return None

    def get_category(self, url: str) -> Optional[SourceCategory]:
        """
        Get source category for a URL.

        Args:
            url: URL or domain to check.

        Returns:
            SourceCategory if trusted, None if not trusted.
        """
        domain = self._extract_domain(url)
        if not domain or self._is_blocked(domain):
            return None

        for pattern, metadata in self.sources.items():
            if not metadata.enabled:
                continue

            matches = False
            if metadata.match_type == "suffix":
                matches = domain.endswith(pattern)
            elif metadata.match_type == "contains":
                matches = pattern in domain
            else:
                matches = domain == pattern or domain.endswith("." + pattern)

            if matches:
                return metadata.category

        return None

    def get_metadata(self, url: str) -> Optional[SourceMetadata]:
        """
        Get full metadata for a URL.

        Args:
            url: URL or domain to check.

        Returns:
            SourceMetadata if trusted, None if not trusted.
        """
        domain = self._extract_domain(url)
        if not domain or self._is_blocked(domain):
            return None

        for pattern, metadata in self.sources.items():
            if not metadata.enabled:
                continue

            matches = False
            if metadata.match_type == "suffix":
                matches = domain.endswith(pattern)
            elif metadata.match_type == "contains":
                matches = pattern in domain
            else:
                matches = domain == pattern or domain.endswith("." + pattern)

            if matches:
                return metadata

        return None

    def add_trusted_source(
        self,
        domain: str,
        trust_level: TrustLevel = TrustLevel.MEDIUM,
        category: SourceCategory = SourceCategory.CUSTOM,
        match_type: str = "exact",
        description: Optional[str] = None,
    ) -> None:
        """
        Add a new trusted source.

        Args:
            domain: Domain pattern to trust.
            trust_level: Trust level for the source.
            category: Category of the source.
            match_type: How to match the domain.
            description: Optional description.
        """
        metadata = SourceMetadata(
            domain=domain,
            trust_level=trust_level,
            category=category,
            match_type=match_type,
            description=description,
            enabled=True,
        )
        self.sources[domain.lower()] = metadata
        self._domain_cache.clear()

    def block_domain(self, domain: str) -> None:
        """Add a domain to the blocked list."""
        self.blocked_domains.add(domain.lower())
        self._domain_cache.clear()

    def get_trusted_domains(
        self,
        category: Optional[SourceCategory] = None,
        trust_level: Optional[TrustLevel] = None,
    ) -> List[str]:
        """
        Get list of trusted domain patterns.

        Args:
            category: Optional filter by category.
            trust_level: Optional filter by minimum trust level.

        Returns:
            List of domain patterns.
        """
        domains = []
        trust_order = [TrustLevel.HIGH, TrustLevel.MEDIUM, TrustLevel.LOW]

        for pattern, metadata in self.sources.items():
            if not metadata.enabled:
                continue

            if category and metadata.category != category:
                continue

            if trust_level:
                pattern_idx = trust_order.index(metadata.trust_level)
                required_idx = trust_order.index(trust_level)
                if pattern_idx > required_idx:
                    continue

            domains.append(pattern)

        return domains

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.

        Args:
            config_path: Optional override path.
        """
        path = config_path or self.config_path

        # Organize by category
        by_category: Dict[str, List[Dict]] = {}
        for domain, metadata in self.sources.items():
            cat = metadata.category.value
            if cat not in by_category:
                by_category[cat] = []

            source_dict = {
                "domain": metadata.domain,
                "trust_level": metadata.trust_level.value,
            }
            if metadata.match_type != "exact":
                source_dict["match_type"] = metadata.match_type
            if metadata.description:
                source_dict["description"] = metadata.description
            if not metadata.enabled:
                source_dict["enabled"] = False

            by_category[cat].append(source_dict)

        config = {
            "trusted_sources": by_category,
            "blocked_domains": list(self.blocked_domains),
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {path}")
