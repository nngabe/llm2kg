"""
Web Page Processing Pipeline for GE Vernova Sites.

Scrapes and processes web pages from GE Vernova divisions:
1. Parse sitemaps to discover URLs (via SitemapParser)
2. Scrape page content using requests/BeautifulSoup
3. Store as WebPage nodes in FalkorDB
4. Chunk content using existing strategies (recursive, semantic, RAPTOR)
5. Link DocumentChunks to WebPage nodes

Usage:
    python -m knowledge_graph.webpage_pipeline --graph test_gev_kg \
        --division gas-power --max-pages 50 --chunking-strategy raptor
"""

import os
import re
import time
import json
import logging
import argparse
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

from .models import (
    WebPage,
    DocumentChunk,
    ChunkStatus,
    ChunkType,
    SourceType,
)
from .sitemap_parser import SitemapParser, URLEntry
from .content_formatter import (
    ContentFormatterFactory,
    FormattingStrategy,
    EntityContext,
    FormattedContent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")


class WebPagePipeline:
    """
    Process GE Vernova web pages into the knowledge graph.

    Pipeline stages:
    1. Discover URLs from sitemaps
    2. Scrape page content
    3. Store WebPage nodes
    4. Chunk content (reuses chunking strategies from WikipediaArticlePipeline)
    5. Embed and store DocumentChunks
    """

    def __init__(
        self,
        graph_name: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        embedding_model: str = "qwen3-embedding:8b",
        falkordb_host: str = FALKORDB_HOST,
        falkordb_port: int = FALKORDB_PORT,
        ollama_host: str = OLLAMA_HOST,
        batch_size: int = 10,
        chunking_strategy: str = "recursive",
        request_delay: float = 1.0,
        content_format: str = "raw",
    ):
        """
        Initialize the web page pipeline.

        Args:
            graph_name: Name of the FalkorDB graph.
            chunk_size: Target size for each chunk (characters).
            chunk_overlap: Overlap between consecutive chunks.
            embedding_model: Ollama embedding model name.
            falkordb_host: FalkorDB host.
            falkordb_port: FalkorDB port.
            ollama_host: Ollama API host.
            batch_size: Batch size for database operations.
            chunking_strategy: Strategy for chunking (recursive, semantic, raptor).
            request_delay: Delay between requests (be nice to servers).
            content_format: Content formatting strategy (raw, synthesized, abstract, llm).
        """
        self.graph_name = graph_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.batch_size = batch_size
        self.chunking_strategy = chunking_strategy
        self.request_delay = request_delay
        self.content_format = content_format

        # Initialize FalkorDB connection
        from falkordb import FalkorDB

        self.client = FalkorDB(host=falkordb_host, port=falkordb_port)
        self.graph = self.client.select_graph(graph_name)

        # Initialize sitemap parser
        self.sitemap_parser = SitemapParser()

        # Lazy-loaded components
        self._embeddings = None
        self._splitter = None
        self._semantic_chunker = None
        self._llm = None
        self._session = None
        self._content_formatter = None

    def _get_session(self) -> requests.Session:
        """Lazy load requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": "GEVernovaKGBot/1.0 (Knowledge Graph Builder)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                }
            )
        return self._session

    def _get_embeddings(self):
        """Lazy load embedding model."""
        if self._embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            self._embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.ollama_host,
                num_ctx=4096,
            )
        return self._embeddings

    def _get_splitter(self):
        """Lazy load text splitter."""
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        return self._splitter

    def _get_semantic_chunker(self):
        """Lazy load semantic chunker."""
        if self._semantic_chunker is None:
            from langchain_experimental.text_splitter import SemanticChunker

            self._semantic_chunker = SemanticChunker(
                embeddings=self._get_embeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
                add_start_index=True,
            )
        return self._semantic_chunker

    def _get_llm(self):
        """Lazy load LLM for RAPTOR summarization."""
        if self._llm is None:
            from langchain_ollama import ChatOllama

            self._llm = ChatOllama(
                model="qwen3:8b",
                base_url=self.ollama_host,
                temperature=0.0,
            )
        return self._llm

    def _get_content_formatter(self):
        """Lazy load content formatter."""
        if self._content_formatter is None:
            # Get LLM if needed for llm formatting strategy
            llm = self._get_llm() if self.content_format == "llm" else None
            self._content_formatter = ContentFormatterFactory.create(
                strategy=self.content_format,
                graph=self.graph,
                llm=llm,
            )
        return self._content_formatter

    def format_content(self, webpage: WebPage, entity_context: EntityContext = None) -> FormattedContent:
        """
        Apply content formatting to a scraped webpage.

        Args:
            webpage: WebPage object with raw content.
            entity_context: Optional entity context (fetched if not provided for synthesized strategy).

        Returns:
            FormattedContent with processed content.
        """
        if self.content_format == "raw":
            # Skip formatting overhead for raw strategy
            return FormattedContent(
                content=webpage.content,
                abstract=None,
                strategy_used="raw",
            )

        formatter = self._get_content_formatter()

        # Build entity context if not provided
        if entity_context is None:
            entity_context = EntityContext(
                qid=webpage.webpage_id,
                name=webpage.title,
                description=None,
                wikipedia_url=None,
                raw_content=webpage.content,
            )

        # Apply formatting
        formatted = formatter.format(webpage.content, entity_context)

        logger.debug(
            f"Formatted content for {webpage.title}: "
            f"{len(webpage.content)} -> {len(formatted.content)} chars "
            f"(strategy={formatted.strategy_used})"
        )

        return formatted

    def init_schema(self):
        """Initialize FalkorDB schema for WebPage and DocumentChunk nodes."""
        # Create index on WebPage.webpage_id
        try:
            self.graph.query("CREATE INDEX FOR (w:WebPage) ON (w.webpage_id)")
            logger.info("Created index on WebPage.webpage_id")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on WebPage.url
        try:
            self.graph.query("CREATE INDEX FOR (w:WebPage) ON (w.url)")
            logger.info("Created index on WebPage.url")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on WebPage.division
        try:
            self.graph.query("CREATE INDEX FOR (w:WebPage) ON (w.division)")
            logger.info("Created index on WebPage.division")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on WebPage.parent_url for hierarchy joins
        try:
            self.graph.query("CREATE INDEX FOR (w:WebPage) ON (w.parent_url)")
            logger.info("Created index on WebPage.parent_url")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on DocumentChunk.source_id
        try:
            self.graph.query("CREATE INDEX FOR (c:DocumentChunk) ON (c.source_id)")
            logger.info("Created index on DocumentChunk.source_id")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on DocumentChunk.source_type
        try:
            self.graph.query("CREATE INDEX FOR (c:DocumentChunk) ON (c.source_type)")
            logger.info("Created index on DocumentChunk.source_type")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        logger.info("Schema initialized for WebPage nodes")

    def scrape_page(self, url: str) -> Optional[WebPage]:
        """
        Scrape a single web page.

        Args:
            url: URL to scrape.

        Returns:
            WebPage object with content, or None if scraping failed.
        """
        session = self._get_session()

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract main content
            main_content = ""

            # Try common content containers
            content_selectors = [
                "main",
                "article",
                '[role="main"]',
                ".main-content",
                ".content",
                "#content",
                ".page-content",
                ".article-content",
            ]

            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text(separator="\n", strip=True)
                    break

            # Fallback to body if no specific content container found
            if not main_content:
                body = soup.find("body")
                if body:
                    main_content = body.get_text(separator="\n", strip=True)

            # Clean up content
            main_content = self._clean_content(main_content)

            if not main_content or len(main_content) < 100:
                logger.warning(f"Insufficient content from {url}")
                return None

            # Parse URL for hierarchy info
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            path_parts = [p for p in parsed.path.strip("/").split("/") if p]

            division = path_parts[0] if path_parts else ""
            category = path_parts[1] if len(path_parts) > 1 else None
            subcategory = "/".join(path_parts[2:]) if len(path_parts) > 2 else None
            depth = len(path_parts)

            # Compute parent URL for hierarchy
            parent_url = self._compute_parent_url(url)

            # Create WebPage object
            webpage = WebPage(
                url=url,
                title=title,
                domain=domain,
                division=division,
                category=category,
                subcategory=subcategory,
                depth=depth,
                parent_url=parent_url,
                content=main_content,
                crawled_at=datetime.now(timezone.utc),
                content_hash=hashlib.sha256(main_content.encode()).hexdigest()[:16],
            )

            logger.debug(f"Scraped {url}: {len(main_content)} chars")
            return webpage

        except requests.RequestException as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None

    def _clean_content(self, content: str) -> str:
        """
        Clean scraped content.

        Args:
            content: Raw scraped text.

        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r" {2,}", " ", content)

        # Remove common noise patterns
        noise_patterns = [
            r"Cookie\s*(Settings|Policy|Preferences).*?\n",
            r"Accept\s*(All)?\s*Cookies.*?\n",
            r"Skip to (main )?content.*?\n",
            r"Share this (page|article).*?\n",
            r"Print this page.*?\n",
            r"Follow us on.*?\n",
        ]
        for pattern in noise_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE)

        return content.strip()

    def _compute_parent_url(self, url: str) -> Optional[str]:
        """
        Compute parent URL by removing the last path segment.

        Args:
            url: The URL to compute parent for.

        Returns:
            Parent URL or None if at root/division level.
        """
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(path_parts) <= 1:
            # Root or division level - no parent (or parent is domain root)
            return None

        # Remove last segment to get parent path
        parent_path = "/" + "/".join(path_parts[:-1])
        parent_url = f"{parsed.scheme}://{parsed.netloc}{parent_path}"

        # Add trailing slash if original had one
        if url.endswith("/") and not parent_url.endswith("/"):
            parent_url += "/"

        return parent_url

    def save_webpage(self, webpage: WebPage) -> bool:
        """
        Save WebPage node to FalkorDB.

        Args:
            webpage: WebPage object to save.

        Returns:
            True if saved successfully, False otherwise.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Escape strings for Cypher
            title = webpage.title.replace("'", "\\'").replace("\n", " ")
            url = webpage.url.replace("'", "\\'")
            division = webpage.division.replace("'", "\\'") if webpage.division else ""
            category = (
                webpage.category.replace("'", "\\'") if webpage.category else ""
            )
            subcategory = (
                webpage.subcategory.replace("'", "\\'") if webpage.subcategory else ""
            )
            parent_url = (
                webpage.parent_url.replace("'", "\\'") if webpage.parent_url else ""
            )

            # MERGE WebPage node
            self.graph.query(
                f"""
                MERGE (w:WebPage {{url: '{url}'}})
                ON CREATE SET
                    w.webpage_id = '{webpage.webpage_id}',
                    w.title = '{title}',
                    w.domain = '{webpage.domain}',
                    w.division = '{division}',
                    w.category = '{category}',
                    w.subcategory = '{subcategory}',
                    w.depth = {webpage.depth},
                    w.parent_url = '{parent_url}',
                    w.crawled_at = '{timestamp}',
                    w.content_hash = '{webpage.content_hash or ""}',
                    w.source_type = 'webpage'
                ON MATCH SET
                    w.title = '{title}',
                    w.crawled_at = '{timestamp}',
                    w.content_hash = '{webpage.content_hash or ""}'
            """
            )

            # Create CHILD_OF relationship to parent WebPage if parent exists
            if webpage.parent_url:
                try:
                    self.graph.query(
                        f"""
                        MATCH (child:WebPage {{url: '{url}'}})
                        MATCH (parent:WebPage {{url: '{parent_url}'}})
                        MERGE (child)-[:CHILD_OF]->(parent)
                    """
                    )
                    logger.debug(f"Linked {webpage.url} to parent {webpage.parent_url}")
                except Exception as e:
                    # Parent may not be crawled yet - this is OK
                    logger.debug(f"Parent not yet crawled for {webpage.url}: {e}")

            logger.debug(f"Saved WebPage: {webpage.url}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save WebPage {webpage.url}: {e}")
            return False

    def link_hierarchy(self) -> int:
        """
        Create CHILD_OF relationships for all WebPage nodes based on parent_url.

        This method is useful for linking pages that were crawled out of order,
        where the parent page was not yet in the graph when the child was saved.

        Returns:
            Number of CHILD_OF relationships created.
        """
        try:
            result = self.graph.query(
                """
                MATCH (child:WebPage)
                WHERE child.parent_url IS NOT NULL AND child.parent_url <> ''
                MATCH (parent:WebPage {url: child.parent_url})
                WHERE NOT (child)-[:CHILD_OF]->(parent)
                MERGE (child)-[:CHILD_OF]->(parent)
                RETURN count(*) as linked
            """
            )
            count = result.result_set[0][0] if result.result_set else 0
            logger.info(f"Created {count} CHILD_OF relationships")
            return count
        except Exception as e:
            logger.error(f"Failed to link hierarchy: {e}")
            return 0

    def chunk_content(self, webpage: WebPage) -> List[DocumentChunk]:
        """
        Split webpage content into chunks.

        Args:
            webpage: WebPage with content.

        Returns:
            List of DocumentChunk objects.
        """
        if not webpage.content:
            return []

        if self.chunking_strategy == "raptor":
            return self._chunk_content_raptor(webpage)
        elif self.chunking_strategy == "semantic":
            return self._chunk_content_semantic(webpage)
        else:
            return self._chunk_content_recursive(webpage)

    def _chunk_content_recursive(self, webpage: WebPage) -> List[DocumentChunk]:
        """Chunk using recursive character splitting."""
        splitter = self._get_splitter()
        texts = splitter.split_text(webpage.content)

        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                content=text,
                chunk_index=i,
                source_id=webpage.webpage_id,
                source_type=SourceType.WEBPAGE,
                source_url=webpage.url,
                status=ChunkStatus.PENDING,
                metadata={
                    "division": webpage.division,
                    "category": webpage.category,
                    "chunking_strategy": "recursive",
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_content_semantic(self, webpage: WebPage) -> List[DocumentChunk]:
        """Chunk using embedding-based semantic splitting."""
        chunker = self._get_semantic_chunker()

        try:
            docs = chunker.create_documents([webpage.content])
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to recursive: {e}")
            return self._chunk_content_recursive(webpage)

        chunks = []
        for i, doc in enumerate(docs):
            chunk = DocumentChunk(
                content=doc.page_content,
                chunk_index=i,
                source_id=webpage.webpage_id,
                source_type=SourceType.WEBPAGE,
                source_url=webpage.url,
                chunk_type=ChunkType.LEAF,
                level=2,
                status=ChunkStatus.PENDING,
                metadata={
                    "division": webpage.division,
                    "category": webpage.category,
                    "start_index": doc.metadata.get("start_index", 0),
                    "chunking_strategy": "semantic",
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_content_raptor(self, webpage: WebPage) -> List[DocumentChunk]:
        """RAPTOR-style hierarchical chunking."""
        import numpy as np

        # Step 1: Create semantic chunks (leaves)
        leaf_chunks = self._chunk_content_semantic(webpage)
        if len(leaf_chunks) < 3:
            return leaf_chunks

        # Step 2: Embed all chunks
        embeddings = self._get_embeddings()
        texts = [c.content for c in leaf_chunks]

        try:
            vectors = embeddings.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Failed to embed for RAPTOR: {e}")
            return leaf_chunks

        for chunk, vector in zip(leaf_chunks, vectors):
            chunk.embedding = vector
            chunk.status = ChunkStatus.EMBEDDED

        # Step 3: Cluster chunks
        from sklearn.cluster import KMeans

        n_clusters = min(max(2, len(leaf_chunks) // 3), 8)

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(np.array(vectors))
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return leaf_chunks

        for chunk, label in zip(leaf_chunks, cluster_labels):
            chunk.cluster_id = int(label)

        # Step 4: Create cluster summaries
        cluster_chunks = []
        for cluster_id in range(n_clusters):
            cluster_members = [c for c in leaf_chunks if c.cluster_id == cluster_id]
            if not cluster_members:
                continue

            cluster_text = "\n\n".join([c.content for c in cluster_members])
            summary = self._generate_summary(cluster_text, max_length=300)

            cluster_chunk = DocumentChunk(
                content=summary,
                chunk_index=len(leaf_chunks) + len(cluster_chunks),
                source_id=webpage.webpage_id,
                source_type=SourceType.WEBPAGE,
                source_url=webpage.url,
                chunk_type=ChunkType.CLUSTER,
                level=1,
                cluster_id=cluster_id,
                status=ChunkStatus.PENDING,
                metadata={
                    "division": webpage.division,
                    "member_count": len(cluster_members),
                    "chunking_strategy": "raptor",
                },
            )
            cluster_chunks.append(cluster_chunk)

            for member in cluster_members:
                member.parent_chunk_id = cluster_chunk.chunk_id

        # Step 5: Create root summary
        if cluster_chunks:
            all_cluster_text = "\n\n".join([c.content for c in cluster_chunks])
            root_summary = self._generate_summary(all_cluster_text, max_length=500)

            root_chunk = DocumentChunk(
                content=root_summary,
                chunk_index=len(leaf_chunks) + len(cluster_chunks),
                source_id=webpage.webpage_id,
                source_type=SourceType.WEBPAGE,
                source_url=webpage.url,
                chunk_type=ChunkType.ROOT,
                level=0,
                status=ChunkStatus.PENDING,
                metadata={
                    "division": webpage.division,
                    "cluster_count": len(cluster_chunks),
                    "leaf_count": len(leaf_chunks),
                    "chunking_strategy": "raptor",
                },
            )

            for cc in cluster_chunks:
                cc.parent_chunk_id = root_chunk.chunk_id

            all_chunks = leaf_chunks + cluster_chunks + [root_chunk]
        else:
            all_chunks = leaf_chunks

        logger.info(
            f"RAPTOR: {len(leaf_chunks)} leaves, {len(cluster_chunks)} clusters for {webpage.url}"
        )
        return all_chunks

    def _generate_summary(self, text: str, max_length: int = 300) -> str:
        """Generate LLM summary with extractive fallback."""
        try:
            llm = self._get_llm()
            truncated = text[:4000] if len(text) > 4000 else text

            prompt = f"""Summarize the following text in {max_length} characters or less.
Focus on key concepts and facts.

Text:
{truncated}

Summary:"""

            response = llm.invoke(prompt)
            summary = response.content.strip()

            if len(summary) > max_length:
                summary = summary[: max_length - 3] + "..."

            return summary

        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return text[:max_length - 3] + "..." if len(text) > max_length else text

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Compute embeddings for chunks."""
        embeddings = self._get_embeddings()
        texts = [c.content for c in chunks]

        try:
            vectors = embeddings.embed_documents(texts)

            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
                chunk.status = ChunkStatus.EMBEDDED

            return chunks

        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            for chunk in chunks:
                chunk.status = ChunkStatus.FAILED
            return chunks

    def save_chunks(self, chunks: List[DocumentChunk], webpage: WebPage) -> int:
        """
        Save DocumentChunk nodes to FalkorDB with HAS_CHUNK relationships.

        Args:
            chunks: List of DocumentChunks to save.
            webpage: Source WebPage.

        Returns:
            Number of chunks saved.
        """
        if not chunks:
            return 0

        timestamp = datetime.now(timezone.utc).isoformat()
        saved_count = 0

        for chunk in chunks:
            try:
                # Escape content
                escaped_content = chunk.content.replace("'", "\\'").replace("\n", "\\n")
                metadata_str = json.dumps(chunk.metadata) if chunk.metadata else "{}"
                escaped_metadata = metadata_str.replace("'", "\\'")

                chunk_type_val = chunk.chunk_type.value if chunk.chunk_type else "leaf"
                level_val = chunk.level if chunk.level is not None else 2
                cluster_id_val = (
                    chunk.cluster_id if chunk.cluster_id is not None else "null"
                )
                source_url = (
                    chunk.source_url.replace("'", "\\'") if chunk.source_url else ""
                )

                # MERGE DocumentChunk with multi-source fields
                if chunk.embedding:
                    embed_str = str(chunk.embedding)
                    self.graph.query(
                        f"""
                        MERGE (c:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                        ON CREATE SET
                            c.content = '{escaped_content}',
                            c.chunk_index = {chunk.chunk_index},
                            c.source_id = '{chunk.source_id}',
                            c.source_type = '{chunk.source_type.value}',
                            c.source_url = '{source_url}',
                            c.status = '{chunk.status.value}',
                            c.created_at = '{timestamp}',
                            c.embedding = {embed_str},
                            c.metadata = '{escaped_metadata}',
                            c.chunk_type = '{chunk_type_val}',
                            c.level = {level_val},
                            c.cluster_id = {cluster_id_val}
                        ON MATCH SET
                            c.status = '{chunk.status.value}'
                    """
                    )
                else:
                    self.graph.query(
                        f"""
                        MERGE (c:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                        ON CREATE SET
                            c.content = '{escaped_content}',
                            c.chunk_index = {chunk.chunk_index},
                            c.source_id = '{chunk.source_id}',
                            c.source_type = '{chunk.source_type.value}',
                            c.source_url = '{source_url}',
                            c.status = '{chunk.status.value}',
                            c.created_at = '{timestamp}',
                            c.metadata = '{escaped_metadata}',
                            c.chunk_type = '{chunk_type_val}',
                            c.level = {level_val},
                            c.cluster_id = {cluster_id_val}
                        ON MATCH SET
                            c.status = '{chunk.status.value}'
                    """
                    )

                # Create HAS_CHUNK relationship to WebPage
                url_escaped = webpage.url.replace("'", "\\'")
                self.graph.query(
                    f"""
                    MATCH (w:WebPage {{url: '{url_escaped}'}})
                    MATCH (c:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                    MERGE (w)-[r:HAS_CHUNK]->(c)
                    ON CREATE SET r.position = {chunk.chunk_index}
                """
                )

                saved_count += 1

            except Exception as e:
                logger.warning(f"Failed to save chunk {chunk.chunk_id}: {e}")

        # Create CHILD_OF relationships for RAPTOR hierarchy
        child_of_count = 0
        for chunk in chunks:
            if chunk.parent_chunk_id:
                try:
                    self.graph.query(
                        f"""
                        MATCH (child:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                        MATCH (parent:DocumentChunk {{chunk_id: '{chunk.parent_chunk_id}'}})
                        MERGE (child)-[:CHILD_OF]->(parent)
                    """
                    )
                    child_of_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create CHILD_OF: {e}")

        if child_of_count > 0:
            logger.info(f"Created {child_of_count} CHILD_OF relationships")

        return saved_count

    def process_webpage(
        self,
        url: str,
        with_embeddings: bool = True,
    ) -> Tuple[Optional[WebPage], int]:
        """
        Process a single web page through the full pipeline.

        Args:
            url: URL to process.
            with_embeddings: Whether to compute embeddings.

        Returns:
            Tuple of (WebPage or None, chunks_saved).
        """
        # Scrape page
        webpage = self.scrape_page(url)
        if not webpage:
            return None, 0

        # Apply content formatting if not raw
        if self.content_format != "raw":
            formatted = self.format_content(webpage)
            # Replace content with formatted version
            webpage.content = formatted.content
            # Store formatting metadata
            if webpage.metadata is None:
                webpage.metadata = {}
            webpage.metadata["content_format"] = formatted.strategy_used
            if formatted.abstract:
                webpage.metadata["abstract"] = formatted.abstract[:500]  # Truncate for storage
            logger.debug(f"Applied {formatted.strategy_used} formatting to {url}")

        # Save WebPage node
        if not self.save_webpage(webpage):
            return webpage, 0

        # Chunk content
        chunks = self.chunk_content(webpage)
        if not chunks:
            return webpage, 0

        # Embed chunks
        if with_embeddings:
            chunks = self.embed_chunks(chunks)

        # Save chunks
        saved = self.save_chunks(chunks, webpage)
        logger.info(f"Processed {url}: {saved} chunks")

        return webpage, saved

    def crawl_division(
        self,
        division: str,
        max_pages: int = 50,
        max_depth: int = 3,
        with_embeddings: bool = True,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Crawl and process all pages for a division.

        Args:
            division: Division name (e.g., "gas-power").
            max_pages: Maximum number of pages to process.
            max_depth: Maximum URL depth to crawl.
            with_embeddings: Whether to compute embeddings.
            skip_existing: Skip URLs already in the graph.

        Returns:
            Dictionary with crawl statistics.
        """
        start_time = time.time()

        # Initialize schema
        self.init_schema()

        # Get URLs from sitemap
        results = self.sitemap_parser.parse_all_divisions(
            divisions=[division],
            max_depth=max_depth,
        )

        if division not in results:
            return {"error": f"Division {division} not found in sitemaps"}

        urls = [entry.url for entry in results[division]]
        logger.info(f"Found {len(urls)} URLs for division {division}")

        # Get already processed URLs if skip_existing
        processed_urls = set()
        if skip_existing:
            try:
                result = self.graph.query(
                    """
                    MATCH (w:WebPage)-[:HAS_CHUNK]->(c:DocumentChunk)
                    RETURN DISTINCT w.url as url
                """
                )
                for row in result.result_set:
                    processed_urls.add(row[0])
                logger.info(f"Found {len(processed_urls)} already processed URLs")
            except Exception:
                pass

            urls = [u for u in urls if u not in processed_urls]

        # Limit to max_pages
        urls = urls[:max_pages]
        logger.info(f"Processing {len(urls)} URLs...")

        # Process each URL
        stats = {
            "division": division,
            "pages_processed": 0,
            "pages_failed": 0,
            "total_chunks": 0,
            "pages": [],
        }

        for url in urls:
            try:
                webpage, chunks_saved = self.process_webpage(url, with_embeddings)
                if webpage and chunks_saved > 0:
                    stats["pages_processed"] += 1
                    stats["total_chunks"] += chunks_saved
                    stats["pages"].append(
                        {
                            "url": url,
                            "title": webpage.title,
                            "chunks": chunks_saved,
                        }
                    )
                else:
                    stats["pages_failed"] += 1

                # Be nice to servers
                time.sleep(self.request_delay)

            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                stats["pages_failed"] += 1

        stats["processing_time_seconds"] = time.time() - start_time
        stats["graph_name"] = self.graph_name

        return stats

    def run(
        self,
        divisions: Optional[List[str]] = None,
        max_pages: int = 50,
        max_depth: int = 3,
        with_embeddings: bool = True,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full web page pipeline for multiple divisions.

        Args:
            divisions: List of divisions to crawl (None = all).
            max_pages: Maximum pages per division.
            max_depth: Maximum URL depth.
            with_embeddings: Whether to compute embeddings.
            skip_existing: Skip already processed URLs.

        Returns:
            Combined statistics dictionary.
        """
        divisions = divisions or list(SitemapParser.SITEMAPS.keys())

        all_stats = {
            "divisions": {},
            "total_pages_processed": 0,
            "total_pages_failed": 0,
            "total_chunks": 0,
        }

        for division in divisions:
            logger.info(f"\n{'='*60}\nProcessing division: {division}\n{'='*60}")

            stats = self.crawl_division(
                division=division,
                max_pages=max_pages,
                max_depth=max_depth,
                with_embeddings=with_embeddings,
                skip_existing=skip_existing,
            )

            all_stats["divisions"][division] = stats
            all_stats["total_pages_processed"] += stats.get("pages_processed", 0)
            all_stats["total_pages_failed"] += stats.get("pages_failed", 0)
            all_stats["total_chunks"] += stats.get("total_chunks", 0)

        return all_stats

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about WebPage and DocumentChunk nodes."""
        stats = {}

        try:
            # Count WebPages
            result = self.graph.query("MATCH (w:WebPage) RETURN count(w) as count")
            stats["total_webpages"] = (
                result.result_set[0][0] if result.result_set else 0
            )

            # Count by division
            result = self.graph.query(
                """
                MATCH (w:WebPage)
                RETURN w.division as division, count(w) as count
                ORDER BY count DESC
            """
            )
            stats["webpages_by_division"] = {}
            for row in result.result_set:
                stats["webpages_by_division"][row[0] or "unknown"] = row[1]

            # Count DocumentChunks from WebPages
            result = self.graph.query(
                """
                MATCH (c:DocumentChunk)
                WHERE c.source_type = 'webpage'
                RETURN count(c) as count
            """
            )
            stats["webpage_chunks"] = (
                result.result_set[0][0] if result.result_set else 0
            )

            # Count HAS_CHUNK relationships from WebPage
            result = self.graph.query(
                """
                MATCH (w:WebPage)-[r:HAS_CHUNK]->(c:DocumentChunk)
                RETURN count(r) as count
            """
            )
            stats["webpage_has_chunk_relationships"] = (
                result.result_set[0][0] if result.result_set else 0
            )

            # Count CHILD_OF relationships between WebPages
            result = self.graph.query(
                """
                MATCH (child:WebPage)-[r:CHILD_OF]->(parent:WebPage)
                RETURN count(r) as count
            """
            )
            stats["webpage_child_of_relationships"] = (
                result.result_set[0][0] if result.result_set else 0
            )

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats["error"] = str(e)

        return stats


def main():
    """CLI entry point for web page pipeline."""
    parser = argparse.ArgumentParser(
        description="Process GE Vernova web pages into knowledge graph"
    )

    parser.add_argument(
        "--graph",
        type=str,
        default="wikidata",
        help="FalkorDB graph name (default: wikidata)",
    )
    parser.add_argument(
        "--division",
        type=str,
        help="Specific division to crawl (e.g., gas-power)",
    )
    parser.add_argument(
        "--divisions",
        type=str,
        nargs="+",
        help="Multiple divisions to crawl",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum pages per division (default: 50)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum URL depth (default: 3)",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="recursive",
        choices=["recursive", "semantic", "raptor"],
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--webpage-format",
        type=str,
        default="raw",
        choices=["raw", "synthesized", "abstract", "llm"],
        help="Content formatting strategy: raw (no formatting), synthesized (hierarchical with children), abstract (extract clean description), llm (LLM-generated Wikipedia-style). Default: raw",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding computation",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess pages that already have chunks",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show current statistics",
    )
    parser.add_argument(
        "--link-hierarchy",
        action="store_true",
        help="Create CHILD_OF relationships for all WebPage nodes based on parent_url",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--detect-communities",
        action="store_true",
        help="Run Louvain community detection after processing",
    )
    parser.add_argument(
        "--temporal-stats",
        action="store_true",
        help="Show temporal layer statistics (Episodes, Communities)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize pipeline
    pipeline = WebPagePipeline(
        graph_name=args.graph,
        chunking_strategy=args.chunking_strategy,
        content_format=args.webpage_format,
    )

    # Stats-only mode
    if args.stats_only:
        stats = pipeline.get_stats()
        print(f"\nWebPage Statistics ({args.graph}):")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        return

    # Link hierarchy mode
    if args.link_hierarchy:
        print(f"\nLinking WebPage hierarchy in graph: {args.graph}")
        print("-" * 40)
        pipeline.init_schema()  # Ensure parent_url index exists
        count = pipeline.link_hierarchy()
        print(f"Created {count} CHILD_OF relationships")
        return

    # Determine divisions to process
    divisions = None
    if args.division:
        divisions = [args.division]
    elif args.divisions:
        divisions = args.divisions

    print("\n" + "=" * 60)
    print("GE VERNOVA WEB PAGE PIPELINE")
    print("=" * 60)
    print(f"Graph:           {args.graph}")
    print(f"Divisions:       {divisions or 'all'}")
    print(f"Max Pages:       {args.max_pages}")
    print(f"Max Depth:       {args.max_depth}")
    print(f"Chunking:        {args.chunking_strategy}")
    print(f"Content Format:  {args.webpage_format}")
    print(f"With Embeddings: {not args.no_embeddings}")
    print(f"Skip Existing:   {not args.reprocess}")
    print("=" * 60)

    if divisions and len(divisions) == 1:
        # Single division
        stats = pipeline.crawl_division(
            division=divisions[0],
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            with_embeddings=not args.no_embeddings,
            skip_existing=not args.reprocess,
        )
    else:
        # Multiple divisions
        stats = pipeline.run(
            divisions=divisions,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            with_embeddings=not args.no_embeddings,
            skip_existing=not args.reprocess,
        )

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return

    if "divisions" in stats:
        print(f"Total pages processed: {stats['total_pages_processed']}")
        print(f"Total pages failed: {stats['total_pages_failed']}")
        print(f"Total chunks: {stats['total_chunks']}")

        for division, div_stats in stats["divisions"].items():
            print(f"\n{division}:")
            print(f"  Pages: {div_stats.get('pages_processed', 0)}")
            print(f"  Chunks: {div_stats.get('total_chunks', 0)}")
    else:
        print(f"Pages processed: {stats.get('pages_processed', 0)}")
        print(f"Pages failed: {stats.get('pages_failed', 0)}")
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
        print(f"Processing time: {stats.get('processing_time_seconds', 0):.2f}s")

    # Show temporal stats if requested
    if args.temporal_stats:
        from .temporal_queries import get_temporal_statistics
        temporal_stats = get_temporal_statistics(pipeline.graph)
        print("\n" + "=" * 60)
        print("TEMPORAL LAYER STATISTICS")
        print("=" * 60)
        print(f"Episodes: {temporal_stats.get('episodes', 0)}")
        print(f"Communities: {temporal_stats.get('communities', 0)}")
        print(f"CONTAINS relationships: {temporal_stats.get('contains_relationships', 0)}")
        print(f"BELONGS_TO relationships: {temporal_stats.get('belongs_to_relationships', 0)}")

    # Run community detection if requested
    if args.detect_communities:
        from .community_detection import EntityCommunityDetector
        print("\n" + "=" * 60)
        print("COMMUNITY DETECTION")
        print("=" * 60)
        detector = EntityCommunityDetector(pipeline.graph, min_community_size=3)
        count = detector.run()
        print(f"Created {count} entity communities")


if __name__ == "__main__":
    main()
