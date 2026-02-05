"""
Sitemap Parser and Hierarchical Indexer for GE Vernova Web Pages.

Parses XML sitemaps and builds a hierarchical URL index for web page crawling.

Hierarchy levels:
- Level 0: Root domain (e.g., gevernova.com)
- Level 1: Business divisions (e.g., /gas-power/, /wind-power/)
- Level 2: Categories (e.g., /equipment/, /services/)
- Level 3+: Subcategories and product pages

Usage:
    python -m knowledge_graph.sitemap_parser --output hierarchy.json --max-depth 3
    python -m knowledge_graph.sitemap_parser --stats-only
"""

import os
import json
import logging
import argparse
import requests
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class URLEntry:
    """Represents a URL entry from a sitemap."""
    url: str
    domain: str
    division: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    depth: int = 0
    lastmod: Optional[str] = None
    changefreq: Optional[str] = None
    priority: Optional[float] = None


@dataclass
class HierarchyNode:
    """Node in the URL hierarchy tree."""
    path: str
    url: str
    depth: int
    children: Dict[str, "HierarchyNode"] = field(default_factory=dict)
    urls: List[str] = field(default_factory=list)


class SitemapParser:
    """
    Parse XML sitemaps and build hierarchical URL index for GE Vernova sites.

    Supports:
    - Standard sitemap XML format
    - Sitemap index files
    - Multiple divisions/domains
    """

    # Known GE Vernova division sitemaps
    SITEMAPS = {
        "gas-power": "https://www.gevernova.com/gas-power/sitemap.xml",
        "steam-power": "https://www.gevernova.com/steam-power/sitemap.xml",
        "wind-power": "https://www.gevernova.com/wind-power/sitemap.xml",
        "hydropower": "https://www.gevernova.com/hydropower/sitemap.xml",
        "grid-solutions": "https://www.gevernova.com/grid-solutions/sitemap.xml",
        "solar-storage": "https://www.gevernova.com/solar-storage/sitemap.xml",
        "power-conversion": "https://www.gevernova.com/power-conversion/sitemap.xml",
        "consulting": "https://www.gevernova.com/consulting/sitemap.xml",
        "lm-wind-power": "https://www.lmwindpower.com/sitemap.xml",
    }

    # XML namespaces used in sitemaps
    NAMESPACES = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "xhtml": "http://www.w3.org/1999/xhtml",
    }

    def __init__(
        self,
        timeout: int = 30,
        user_agent: str = "GEVernovaKGBot/1.0",
    ):
        """
        Initialize the sitemap parser.

        Args:
            timeout: Request timeout in seconds.
            user_agent: User agent string for requests.
        """
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
        self._session = None

    def _get_session(self) -> requests.Session:
        """Lazy load requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.headers)
        return self._session

    def fetch_sitemap(self, url: str) -> Optional[str]:
        """
        Fetch sitemap XML content from URL.

        Args:
            url: Sitemap URL.

        Returns:
            XML content string or None if fetch failed.
        """
        session = self._get_session()
        try:
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch sitemap {url}: {e}")
            return None

    def parse_sitemap(self, url: str) -> List[str]:
        """
        Parse sitemap XML and return all URLs.

        Handles both regular sitemaps and sitemap index files.

        Args:
            url: Sitemap URL.

        Returns:
            List of URLs found in the sitemap.
        """
        content = self.fetch_sitemap(url)
        if not content:
            return []

        urls = []
        try:
            root = ET.fromstring(content)

            # Check if this is a sitemap index
            sitemap_tags = root.findall(".//sm:sitemap/sm:loc", self.NAMESPACES)
            if sitemap_tags:
                # This is a sitemap index, recursively parse each sitemap
                logger.info(f"Found sitemap index with {len(sitemap_tags)} sitemaps")
                for sitemap_loc in sitemap_tags:
                    child_url = sitemap_loc.text
                    if child_url:
                        urls.extend(self.parse_sitemap(child_url))
            else:
                # Regular sitemap - extract all URLs
                url_tags = root.findall(".//sm:url/sm:loc", self.NAMESPACES)
                for url_tag in url_tags:
                    if url_tag.text:
                        urls.append(url_tag.text)

            logger.debug(f"Parsed {len(urls)} URLs from {url}")

        except ET.ParseError as e:
            logger.warning(f"Failed to parse sitemap XML {url}: {e}")

        return urls

    def parse_sitemap_with_metadata(self, url: str) -> List[URLEntry]:
        """
        Parse sitemap and return URLs with metadata.

        Args:
            url: Sitemap URL.

        Returns:
            List of URLEntry objects with full metadata.
        """
        content = self.fetch_sitemap(url)
        if not content:
            return []

        entries = []
        try:
            root = ET.fromstring(content)

            # Check if this is a sitemap index
            sitemap_tags = root.findall(".//sm:sitemap", self.NAMESPACES)
            if sitemap_tags:
                # Recursively parse each child sitemap
                for sitemap in sitemap_tags:
                    loc = sitemap.find("sm:loc", self.NAMESPACES)
                    if loc is not None and loc.text:
                        entries.extend(self.parse_sitemap_with_metadata(loc.text))
            else:
                # Regular sitemap - extract URLs with metadata
                url_elements = root.findall(".//sm:url", self.NAMESPACES)
                for url_elem in url_elements:
                    loc = url_elem.find("sm:loc", self.NAMESPACES)
                    if loc is None or not loc.text:
                        continue

                    page_url = loc.text
                    lastmod = url_elem.find("sm:lastmod", self.NAMESPACES)
                    changefreq = url_elem.find("sm:changefreq", self.NAMESPACES)
                    priority = url_elem.find("sm:priority", self.NAMESPACES)

                    entry = self._create_url_entry(
                        page_url,
                        lastmod=lastmod.text if lastmod is not None else None,
                        changefreq=changefreq.text if changefreq is not None else None,
                        priority=float(priority.text) if priority is not None else None,
                    )
                    entries.append(entry)

        except ET.ParseError as e:
            logger.warning(f"Failed to parse sitemap XML {url}: {e}")

        return entries

    def _create_url_entry(
        self,
        url: str,
        lastmod: Optional[str] = None,
        changefreq: Optional[str] = None,
        priority: Optional[float] = None,
    ) -> URLEntry:
        """
        Create a URLEntry from a URL string with hierarchy info.

        Args:
            url: The URL to process.
            lastmod: Last modification date.
            changefreq: Change frequency.
            priority: URL priority.

        Returns:
            URLEntry with hierarchy information.
        """
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        # Determine hierarchy level and division
        depth = len(path_parts)
        division = path_parts[0] if path_parts else ""
        category = path_parts[1] if len(path_parts) > 1 else None
        subcategory = "/".join(path_parts[2:]) if len(path_parts) > 2 else None

        return URLEntry(
            url=url,
            domain=domain,
            division=division,
            category=category,
            subcategory=subcategory,
            depth=depth,
            lastmod=lastmod,
            changefreq=changefreq,
            priority=priority,
        )

    def infer_depth(self, url: str) -> int:
        """
        Infer hierarchy depth from URL structure.

        Args:
            url: URL to analyze.

        Returns:
            Depth level (0 = root, 1 = division, 2 = category, etc.)
        """
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        return len(path_parts)

    def build_hierarchy(self, urls: List[str]) -> HierarchyNode:
        """
        Build hierarchical tree from URL paths.

        Args:
            urls: List of URLs to organize.

        Returns:
            Root HierarchyNode containing the full tree.
        """
        root = HierarchyNode(path="/", url="", depth=0)

        for url in urls:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.strip("/").split("/") if p]

            current = root
            for i, part in enumerate(path_parts):
                if part not in current.children:
                    child_path = "/" + "/".join(path_parts[: i + 1])
                    child_url = f"{parsed.scheme}://{parsed.netloc}{child_path}"
                    current.children[part] = HierarchyNode(
                        path=child_path,
                        url=child_url,
                        depth=i + 1,
                    )
                current = current.children[part]

            current.urls.append(url)

        return root

    def parse_all_divisions(
        self,
        divisions: Optional[List[str]] = None,
        max_depth: int = 10,
    ) -> Dict[str, List[URLEntry]]:
        """
        Parse sitemaps for all (or specified) divisions.

        Args:
            divisions: List of division names to parse (None = all).
            max_depth: Maximum URL depth to include.

        Returns:
            Dictionary mapping division names to URL entries.
        """
        divisions = divisions or list(self.SITEMAPS.keys())
        results = {}

        for division in divisions:
            if division not in self.SITEMAPS:
                logger.warning(f"Unknown division: {division}")
                continue

            sitemap_url = self.SITEMAPS[division]
            logger.info(f"Parsing sitemap for {division}: {sitemap_url}")

            entries = self.parse_sitemap_with_metadata(sitemap_url)

            # Filter by max_depth
            entries = [e for e in entries if e.depth <= max_depth]

            results[division] = entries
            logger.info(f"  Found {len(entries)} URLs (max_depth={max_depth})")

        return results

    def get_stats(self, results: Dict[str, List[URLEntry]]) -> Dict[str, Any]:
        """
        Calculate statistics from parsed results.

        Args:
            results: Dictionary from parse_all_divisions().

        Returns:
            Statistics dictionary.
        """
        stats = {
            "total_urls": 0,
            "divisions": {},
            "depth_distribution": {},
            "domains": set(),
        }

        for division, entries in results.items():
            stats["divisions"][division] = {
                "url_count": len(entries),
                "categories": set(),
            }
            stats["total_urls"] += len(entries)

            for entry in entries:
                stats["domains"].add(entry.domain)
                if entry.category:
                    stats["divisions"][division]["categories"].add(entry.category)

                depth_key = f"depth_{entry.depth}"
                stats["depth_distribution"][depth_key] = (
                    stats["depth_distribution"].get(depth_key, 0) + 1
                )

        # Convert sets to lists for JSON serialization
        stats["domains"] = list(stats["domains"])
        for division in stats["divisions"]:
            stats["divisions"][division]["categories"] = list(
                stats["divisions"][division]["categories"]
            )

        return stats

    def export_hierarchy(
        self,
        results: Dict[str, List[URLEntry]],
        output_path: str,
    ) -> None:
        """
        Export hierarchy to JSON file.

        Args:
            results: Dictionary from parse_all_divisions().
            output_path: Path for output JSON file.
        """
        # Convert URLEntry objects to dicts
        export_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "divisions": {},
            "stats": self.get_stats(results),
        }

        for division, entries in results.items():
            export_data["divisions"][division] = [asdict(e) for e in entries]

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported hierarchy to {output_path}")

    def load_hierarchy(self, input_path: str) -> Dict[str, List[URLEntry]]:
        """
        Load hierarchy from JSON file.

        Args:
            input_path: Path to input JSON file.

        Returns:
            Dictionary mapping division names to URL entries.
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        results = {}
        for division, entries in data.get("divisions", {}).items():
            results[division] = [URLEntry(**e) for e in entries]

        return results


def main():
    """CLI entry point for sitemap parser."""
    parser = argparse.ArgumentParser(
        description="Parse GE Vernova sitemaps and build hierarchical URL index"
    )

    parser.add_argument(
        "--divisions",
        type=str,
        nargs="+",
        help="Specific divisions to parse (default: all)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum URL depth to include (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hierarchy.json",
        help="Output JSON file path (default: hierarchy.json)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics without saving",
    )
    parser.add_argument(
        "--list-divisions",
        action="store_true",
        help="List available divisions and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sitemap_parser = SitemapParser()

    # List divisions mode
    if args.list_divisions:
        print("\nAvailable GE Vernova Divisions:")
        print("-" * 50)
        for division, url in SitemapParser.SITEMAPS.items():
            print(f"  {division:20s} {url}")
        return

    print("\n" + "=" * 60)
    print("GE VERNOVA SITEMAP PARSER")
    print("=" * 60)

    # Parse sitemaps
    results = sitemap_parser.parse_all_divisions(
        divisions=args.divisions,
        max_depth=args.max_depth,
    )

    # Calculate and display stats
    stats = sitemap_parser.get_stats(results)

    print("\n" + "=" * 60)
    print("PARSING RESULTS")
    print("=" * 60)
    print(f"Total URLs:     {stats['total_urls']}")
    print(f"Domains:        {', '.join(stats['domains'])}")
    print(f"Max Depth:      {args.max_depth}")

    print("\nURLs per Division:")
    for division, info in stats["divisions"].items():
        categories = info["categories"][:5]  # Show first 5
        cat_str = ", ".join(categories) if categories else "(none)"
        if len(info["categories"]) > 5:
            cat_str += f" (+{len(info['categories']) - 5} more)"
        print(f"  {division:20s}: {info['url_count']:5d} URLs  [{cat_str}]")

    print("\nDepth Distribution:")
    for depth, count in sorted(stats["depth_distribution"].items()):
        print(f"  {depth}: {count}")

    # Export if not stats-only
    if not args.stats_only:
        sitemap_parser.export_hierarchy(results, args.output)
        print(f"\nHierarchy exported to: {args.output}")


if __name__ == "__main__":
    main()
