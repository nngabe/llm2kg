"""
Wikipedia Article Processing Pipeline (Stage 2).

Processes Wikipedia articles from the Wikidata backbone graph:
1. Rank articles by usefulness (connectivity + abstraction score)
2. Load content via LangChain WikipediaLoader
3. Chunk with semantic-aware splitting
4. Embed with qwen3-embedding:8b
5. Store as DocumentChunk nodes linked to WikiPage

Usage:
    python -m knowledge_graph.wikipedia_pipeline --graph ge_vernova_test \
        --max-articles 10 --chunk-size 1500
"""

import os
import time
import logging
import argparse
import heapq
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from enum import Enum

from .models import (
    DocumentChunk,
    ArticleRankingResult,
    PipelineConfig,
    ChunkStatus,
    ChunkType,
)


class ChunkingStrategy(str, Enum):
    """Chunking strategy for document splitting."""
    RECURSIVE = "recursive"      # Current behavior - recursive character splitting
    SECTION = "section"          # Markdown header-based splitting
    HYBRID = "hybrid"            # Section + size control
    SEMANTIC = "semantic"        # Embedding-based semantic boundary detection
    RAPTOR = "raptor"            # Semantic + clustering + LLM summaries (full SOTA)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")


class WikipediaArticlePipeline:
    """
    Process Wikipedia articles from backbone graph.

    Pipeline stages:
    1. Rank articles by usefulness (connectivity + abstraction score)
    2. Load content via LangChain WikipediaLoader
    3. Chunk with semantic-aware splitting
    4. Embed with qwen3-embedding:8b
    5. Store as DocumentChunk nodes linked to WikiPage
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
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ):
        """
        Initialize the Wikipedia article pipeline.

        Args:
            graph_name: Name of the FalkorDB graph.
            chunk_size: Target size for each chunk (characters).
            chunk_overlap: Overlap between consecutive chunks.
            embedding_model: Ollama embedding model name.
            falkordb_host: FalkorDB host.
            falkordb_port: FalkorDB port.
            ollama_host: Ollama API host.
            batch_size: Batch size for database operations.
            chunking_strategy: Strategy for chunking documents.
        """
        self.graph_name = graph_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.batch_size = batch_size
        self.chunking_strategy = chunking_strategy

        # Initialize FalkorDB connection
        from falkordb import FalkorDB
        self.client = FalkorDB(host=falkordb_host, port=falkordb_port)
        self.graph = self.client.select_graph(graph_name)

        # Lazy-loaded components
        self._embeddings = None
        self._splitter = None
        self._header_splitter = None
        self._size_splitter = None
        self._semantic_chunker = None
        self._llm = None
        self._wiki_loader_cache: Dict[str, str] = {}

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

    def _get_hybrid_splitter(self):
        """
        Lazy load hybrid splitter components.

        Returns tuple of (header_splitter, size_splitter) for:
        - Step 1: Split by section headers (preserves section metadata)
        - Step 2: Size control for large sections
        """
        if self._header_splitter is None:
            from langchain_text_splitters import (
                MarkdownHeaderTextSplitter,
                RecursiveCharacterTextSplitter,
            )

            # Step 1: Split by section headers
            self._header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("##", "section"),
                    ("###", "subsection"),
                ]
            )

            # Step 2: Size control for large sections
            self._size_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )

        return self._header_splitter, self._size_splitter

    def _get_semantic_chunker(self):
        """
        Lazy load semantic chunker with Ollama embeddings.

        Uses LangChain's SemanticChunker which splits text based on
        embedding similarity - detecting topic shifts by measuring
        cosine similarity between consecutive sentences.
        """
        if self._semantic_chunker is None:
            from langchain_experimental.text_splitter import SemanticChunker
            self._semantic_chunker = SemanticChunker(
                embeddings=self._get_embeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,  # Higher = fewer splits
                add_start_index=True,
            )
        return self._semantic_chunker

    def _get_llm(self):
        """Lazy load LLM for summarization (used in RAPTOR strategy)."""
        if self._llm is None:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model="qwen3:8b",
                base_url=self.ollama_host,
                temperature=0.0,
            )
        return self._llm

    def init_schema(self):
        """
        Initialize FalkorDB schema for DocumentChunk nodes.

        Creates indexes for:
        - chunk_id: Primary lookup
        - source_qid: WikiPage joins
        - level: RAPTOR hierarchy queries
        - chunk_type: Filter by node type
        """
        # Create index on chunk_id
        try:
            self.graph.query("CREATE INDEX FOR (c:DocumentChunk) ON (c.chunk_id)")
            logger.info("Created index on DocumentChunk.chunk_id")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on source_qid for efficient joins
        try:
            self.graph.query("CREATE INDEX FOR (c:DocumentChunk) ON (c.source_qid)")
            logger.info("Created index on DocumentChunk.source_qid")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on level for RAPTOR hierarchy queries
        try:
            self.graph.query("CREATE INDEX FOR (c:DocumentChunk) ON (c.level)")
            logger.info("Created index on DocumentChunk.level")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create index on chunk_type for filtering by node type
        try:
            self.graph.query("CREATE INDEX FOR (c:DocumentChunk) ON (c.chunk_type)")
            logger.info("Created index on DocumentChunk.chunk_type")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        # Create vector index for embeddings
        # Note: FalkorDB vector index is created implicitly when storing vectors
        # For explicit index, we'd need Redis Search integration
        # For now, we skip explicit vector index creation as FalkorDB handles it
        logger.info("Schema initialized (vector search via embedding property)")

    def get_wikipages_with_urls(
        self,
        min_score: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get WikiPage nodes with wikipedia_url for Stage 2 processing.

        Args:
            min_score: Minimum connectivity score (if available).
            limit: Maximum number of results.

        Returns:
            List of dictionaries with qid, name, url, description, connectivity_score.
        """
        query = """
            MATCH (w:WikiPage)
            WHERE w.wikipedia_url IS NOT NULL AND w.wikipedia_url <> ''
            RETURN w.wikidata_id as qid, w.name as name,
                   w.wikipedia_url as url, w.description as description,
                   COALESCE(w.connectivity_score, 0) as connectivity_score
            ORDER BY connectivity_score DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        result = self.graph.query(query)
        wikipages = []

        for row in result.result_set:
            wikipages.append({
                "qid": row[0],
                "name": row[1],
                "url": row[2],
                "description": row[3],
                "connectivity_score": row[4] if row[4] else 0.0,
            })

        logger.info(f"Found {len(wikipages)} WikiPage nodes with URLs")
        return wikipages

    def rank_articles(
        self,
        wikipages: List[Dict[str, Any]],
        abstraction_scores: Optional[Dict[str, float]] = None,
    ) -> List[ArticleRankingResult]:
        """
        Rank articles by usefulness for processing.

        Score formula:
          0.4 * connectivity + 0.3 * abstraction + 0.2 * has_description + 0.1 * url_quality

        Args:
            wikipages: List of WikiPage dictionaries from get_wikipages_with_urls().
            abstraction_scores: Optional dict mapping QID to abstraction score.

        Returns:
            List of ArticleRankingResult sorted by score (highest first).
        """
        abstraction_scores = abstraction_scores or {}
        ranked = []

        for wp in wikipages:
            result = ArticleRankingResult.compute_score(
                qid=wp["qid"],
                name=wp["name"],
                url=wp["url"],
                description=wp["description"],
                connectivity=wp.get("connectivity_score", 0.0),
                abstraction=abstraction_scores.get(wp["qid"], 0.0),
            )
            ranked.append(result)

        # Sort by score (highest first)
        ranked.sort()
        logger.info(f"Ranked {len(ranked)} articles for processing")

        if ranked:
            logger.info(f"  Top 3: {[(r.name, f'{r.score:.3f}') for r in ranked[:3]]}")

        return ranked

    def load_article(self, title: str) -> Optional[str]:
        """
        Load Wikipedia article content by title.

        Uses LangChain WikipediaLoader.

        Args:
            title: Wikipedia article title.

        Returns:
            Article content as string, or None if loading failed.
        """
        # Check cache first
        if title in self._wiki_loader_cache:
            return self._wiki_loader_cache[title]

        try:
            from langchain_community.document_loaders import WikipediaLoader

            loader = WikipediaLoader(
                query=title,
                load_max_docs=1,
                doc_content_chars_max=50000,  # Limit content size
            )
            docs = loader.load()

            if docs and docs[0].page_content:
                content = docs[0].page_content
                self._wiki_loader_cache[title] = content
                logger.debug(f"Loaded article '{title}' ({len(content)} chars)")
                return content
            else:
                logger.warning(f"No content found for article '{title}'")
                return None

        except Exception as e:
            logger.warning(f"Failed to load article '{title}': {e}")
            return None

    def extract_title_from_url(self, url: str) -> Optional[str]:
        """
        Extract Wikipedia article title from URL.

        Args:
            url: Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Gas_turbine)

        Returns:
            Article title (e.g., "Gas turbine") or None.
        """
        if not url or "wikipedia.org/wiki/" not in url:
            return None

        try:
            # Extract title from URL path
            import urllib.parse
            path = urllib.parse.urlparse(url).path
            title = path.split("/wiki/")[-1]
            # Decode URL encoding (e.g., %20 -> space)
            title = urllib.parse.unquote(title)
            # Replace underscores with spaces
            title = title.replace("_", " ")
            return title
        except Exception:
            return None

    def chunk_content(self, content: str, source_qid: str) -> List[DocumentChunk]:
        """
        Split content into chunks based on configured strategy.

        Args:
            content: Full article content.
            source_qid: Wikidata QID of the source WikiPage.

        Returns:
            List of DocumentChunk objects.
        """
        if self.chunking_strategy == ChunkingStrategy.RAPTOR:
            return self._chunk_content_raptor(content, source_qid)
        elif self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_content_semantic(content, source_qid)
        elif self.chunking_strategy == ChunkingStrategy.HYBRID:
            return self._chunk_content_hybrid(content, source_qid)
        elif self.chunking_strategy == ChunkingStrategy.SECTION:
            return self._chunk_content_section(content, source_qid)
        else:
            return self._chunk_content_recursive(content, source_qid)

    def _chunk_content_recursive(self, content: str, source_qid: str) -> List[DocumentChunk]:
        """
        Split content using recursive character splitting (original behavior).

        Args:
            content: Full article content.
            source_qid: Wikidata QID of the source WikiPage.

        Returns:
            List of DocumentChunk objects.
        """
        splitter = self._get_splitter()
        texts = splitter.split_text(content)

        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                content=text,
                source_qid=source_qid,
                chunk_index=i,
                status=ChunkStatus.PENDING,
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks for {source_qid} (recursive)")
        return chunks

    def _chunk_content_section(self, content: str, source_qid: str) -> List[DocumentChunk]:
        """
        Split content by markdown section headers only.

        Args:
            content: Full article content.
            source_qid: Wikidata QID of the source WikiPage.

        Returns:
            List of DocumentChunk objects with section metadata.
        """
        header_splitter, _ = self._get_hybrid_splitter()
        header_docs = header_splitter.split_text(content)

        chunks = []
        for i, section in enumerate(header_docs):
            section_text = section.page_content if hasattr(section, 'page_content') else str(section)
            section_metadata = section.metadata if hasattr(section, 'metadata') else {}

            chunk = DocumentChunk(
                content=section_text,
                source_qid=source_qid,
                chunk_index=i,
                metadata={
                    **section_metadata,
                    "section_index": i,
                    "chunking_strategy": "section",
                },
                status=ChunkStatus.PENDING,
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks for {source_qid} (section)")
        return chunks

    def _chunk_content_hybrid(self, content: str, source_qid: str) -> List[DocumentChunk]:
        """
        Hybrid chunking: section headers → size control.

        First splits by markdown headers to preserve section structure,
        then applies size control to large sections.

        Args:
            content: Full article content.
            source_qid: Wikidata QID of the source WikiPage.

        Returns:
            List of DocumentChunk objects with section metadata.
        """
        header_splitter, size_splitter = self._get_hybrid_splitter()

        # Step 1: Split by headers (preserves section metadata)
        header_docs = header_splitter.split_text(content)

        # Step 2: For each section, apply size control if needed
        final_chunks = []
        for i, section in enumerate(header_docs):
            section_text = section.page_content if hasattr(section, 'page_content') else str(section)
            section_metadata = section.metadata if hasattr(section, 'metadata') else {}

            if len(section_text) > self.chunk_size * 1.5:
                # Large section: split further by size
                sub_chunks = size_splitter.split_text(section_text)
                for j, sub_text in enumerate(sub_chunks):
                    chunk = DocumentChunk(
                        content=sub_text,
                        source_qid=source_qid,
                        chunk_index=len(final_chunks),
                        metadata={
                            **section_metadata,
                            "section_index": i,
                            "sub_chunk_index": j,
                            "chunking_strategy": "hybrid",
                        },
                        status=ChunkStatus.PENDING,
                    )
                    final_chunks.append(chunk)
            else:
                # Small section: keep as-is
                chunk = DocumentChunk(
                    content=section_text,
                    source_qid=source_qid,
                    chunk_index=len(final_chunks),
                    metadata={
                        **section_metadata,
                        "section_index": i,
                        "chunking_strategy": "hybrid",
                    },
                    status=ChunkStatus.PENDING,
                )
                final_chunks.append(chunk)

        logger.debug(f"Created {len(final_chunks)} chunks for {source_qid} (hybrid: {len(header_docs)} sections)")
        return final_chunks

    def _chunk_content_semantic(self, content: str, source_qid: str) -> List[DocumentChunk]:
        """
        Split content using embedding-based semantic boundary detection.

        Uses LangChain's SemanticChunker which:
        1. Embeds consecutive sentences
        2. Computes cosine similarity between neighbors
        3. Splits at points where similarity drops below threshold

        Args:
            content: Full article content.
            source_qid: Wikidata QID of the source WikiPage.

        Returns:
            List of DocumentChunk objects (all LEAF type).
        """
        chunker = self._get_semantic_chunker()

        try:
            docs = chunker.create_documents([content])
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to recursive: {e}")
            return self._chunk_content_recursive(content, source_qid)

        chunks = []
        for i, doc in enumerate(docs):
            chunk = DocumentChunk(
                content=doc.page_content,
                source_qid=source_qid,
                chunk_index=i,
                chunk_type=ChunkType.LEAF,
                level=2,
                metadata={
                    "start_index": doc.metadata.get("start_index", 0),
                    "chunking_strategy": "semantic",
                },
                status=ChunkStatus.PENDING,
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} semantic chunks for {source_qid}")
        return chunks

    def _chunk_content_raptor(self, content: str, source_qid: str) -> List[DocumentChunk]:
        """
        RAPTOR-style hierarchical chunking: semantic chunks + clustering + LLM summaries.

        Creates a 3-level hierarchy:
        1. Level 2 (LEAF): Semantic chunks from embedding-based splitting
        2. Level 1 (CLUSTER): K-means clustered groups with LLM summaries
        3. Level 0 (ROOT): Single document summary

        Pipeline:
        1. Create semantic chunks (leaf nodes)
        2. Embed all chunks
        3. Cluster by similarity (k-means)
        4. Generate LLM summary for each cluster
        5. Create root summary from cluster summaries

        Args:
            content: Full article content.
            source_qid: Wikidata QID of the source WikiPage.

        Returns:
            List of DocumentChunk objects (leaves + clusters + root).
        """
        import numpy as np

        # Step 1: Create semantic chunks (leaf nodes)
        leaf_chunks = self._chunk_content_semantic(content, source_qid)
        if len(leaf_chunks) < 3:
            # Too few chunks to cluster meaningfully
            logger.debug(f"Only {len(leaf_chunks)} chunks, skipping RAPTOR clustering")
            return leaf_chunks

        # Step 2: Embed all chunks (needed for clustering)
        embeddings = self._get_embeddings()
        texts = [c.content for c in leaf_chunks]

        try:
            vectors = embeddings.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Failed to embed chunks for RAPTOR: {e}")
            return leaf_chunks

        for chunk, vector in zip(leaf_chunks, vectors):
            chunk.embedding = vector
            chunk.status = ChunkStatus.EMBEDDED

        # Step 3: Cluster chunks using K-means
        from sklearn.cluster import KMeans

        # Auto-determine cluster count: ~3 chunks per cluster, min 2, max 8 clusters
        n_clusters = min(max(2, len(leaf_chunks) // 3), 8)

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(np.array(vectors))
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}")
            return leaf_chunks

        # Assign cluster IDs to leaf chunks
        for chunk, label in zip(leaf_chunks, cluster_labels):
            chunk.cluster_id = int(label)

        # Step 4: Create cluster summary nodes
        cluster_chunks = []
        for cluster_id in range(n_clusters):
            cluster_members = [c for c in leaf_chunks if c.cluster_id == cluster_id]
            if not cluster_members:
                continue

            # Generate LLM summary for this cluster
            cluster_text = "\n\n".join([c.content for c in cluster_members])
            summary = self._generate_summary(
                cluster_text,
                max_length=300,
                context=f"cluster of {len(cluster_members)} related text segments",
            )

            cluster_chunk = DocumentChunk(
                content=summary,
                source_qid=source_qid,
                chunk_index=len(leaf_chunks) + len(cluster_chunks),
                chunk_type=ChunkType.CLUSTER,
                level=1,
                cluster_id=cluster_id,
                metadata={
                    "member_count": len(cluster_members),
                    "chunking_strategy": "raptor",
                },
                status=ChunkStatus.PENDING,
            )
            cluster_chunks.append(cluster_chunk)

            # Link leaf chunks to their cluster parent
            for member in cluster_members:
                member.parent_chunk_id = cluster_chunk.chunk_id

        # Step 5: Create root summary node
        if cluster_chunks:
            all_cluster_text = "\n\n".join([c.content for c in cluster_chunks])
            root_summary = self._generate_summary(
                all_cluster_text,
                max_length=500,
                context="document overview combining all topic clusters",
            )

            root_chunk = DocumentChunk(
                content=root_summary,
                source_qid=source_qid,
                chunk_index=len(leaf_chunks) + len(cluster_chunks),
                chunk_type=ChunkType.ROOT,
                level=0,
                metadata={
                    "cluster_count": len(cluster_chunks),
                    "leaf_count": len(leaf_chunks),
                    "chunking_strategy": "raptor",
                },
                status=ChunkStatus.PENDING,
            )

            # Link cluster chunks to root
            for cc in cluster_chunks:
                cc.parent_chunk_id = root_chunk.chunk_id

            all_chunks = leaf_chunks + cluster_chunks + [root_chunk]
        else:
            all_chunks = leaf_chunks

        logger.info(
            f"RAPTOR: {len(leaf_chunks)} leaves, {len(cluster_chunks)} clusters, "
            f"1 root for {source_qid}"
        )
        return all_chunks

    def _generate_summary(
        self,
        text: str,
        max_length: int = 300,
        context: str = "text",
    ) -> str:
        """
        Generate a summary using the LLM.

        Falls back to extractive summary (first N characters) if LLM fails.

        Args:
            text: Text to summarize.
            max_length: Maximum length of summary in characters.
            context: Description of what's being summarized (for the prompt).

        Returns:
            Summary string.
        """
        try:
            llm = self._get_llm()
            # Truncate input to avoid context length issues
            truncated_text = text[:4000] if len(text) > 4000 else text

            prompt = f"""Summarize the following {context} in {max_length} characters or less.
Focus on the key concepts, facts, and relationships. Be concise and informative.

Text to summarize:
{truncated_text}

Summary:"""

            response = llm.invoke(prompt)
            summary = response.content.strip()

            # Ensure we don't exceed max_length
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."

            return summary

        except Exception as e:
            logger.warning(f"LLM summary failed, using extractive fallback: {e}")
            # Extractive fallback: first N characters
            if len(text) <= max_length:
                return text
            return text[:max_length - 3] + "..."

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Compute embeddings for chunks.

        Args:
            chunks: List of DocumentChunk objects without embeddings.

        Returns:
            Same chunks with embeddings populated.
        """
        embeddings = self._get_embeddings()
        texts = [c.content for c in chunks]

        try:
            vectors = embeddings.embed_documents(texts)

            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
                chunk.status = ChunkStatus.EMBEDDED

            logger.debug(f"Embedded {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            for chunk in chunks:
                chunk.status = ChunkStatus.FAILED
            return chunks

    def save_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Save DocumentChunk nodes to FalkorDB with HAS_CHUNK and CHILD_OF relationships.

        For RAPTOR hierarchical chunks:
        - Creates CHILD_OF relationships between leaves -> clusters -> root
        - Stores chunk_type, level, and cluster_id for hierarchy queries

        Args:
            chunks: List of DocumentChunk objects to save.

        Returns:
            Number of chunks saved.
        """
        import json

        if not chunks:
            return 0

        saved_count = 0
        # Get current timestamp as ISO string (FalkorDB doesn't support datetime())
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        for chunk in chunks:
            try:
                # Escape content for Cypher
                escaped_content = chunk.content.replace("'", "\\'").replace("\n", "\\n")

                # Serialize metadata to JSON string
                metadata_str = json.dumps(chunk.metadata) if chunk.metadata else "{}"
                escaped_metadata = metadata_str.replace("'", "\\'")

                # Prepare hierarchy fields
                chunk_type_val = chunk.chunk_type.value if chunk.chunk_type else "leaf"
                level_val = chunk.level if chunk.level is not None else 2
                cluster_id_val = chunk.cluster_id if chunk.cluster_id is not None else "null"

                # MERGE the DocumentChunk node with hierarchy fields
                if chunk.embedding:
                    embed_str = str(chunk.embedding)
                    self.graph.query(f"""
                        MERGE (c:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                        ON CREATE SET
                            c.content = '{escaped_content}',
                            c.chunk_index = {chunk.chunk_index},
                            c.source_qid = '{chunk.source_qid}',
                            c.status = '{chunk.status.value}',
                            c.created_at = '{timestamp}',
                            c.embedding = {embed_str},
                            c.metadata = '{escaped_metadata}',
                            c.chunk_type = '{chunk_type_val}',
                            c.level = {level_val},
                            c.cluster_id = {cluster_id_val}
                        ON MATCH SET
                            c.status = '{chunk.status.value}'
                    """)
                else:
                    self.graph.query(f"""
                        MERGE (c:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                        ON CREATE SET
                            c.content = '{escaped_content}',
                            c.chunk_index = {chunk.chunk_index},
                            c.source_qid = '{chunk.source_qid}',
                            c.status = '{chunk.status.value}',
                            c.created_at = '{timestamp}',
                            c.metadata = '{escaped_metadata}',
                            c.chunk_type = '{chunk_type_val}',
                            c.level = {level_val},
                            c.cluster_id = {cluster_id_val}
                        ON MATCH SET
                            c.status = '{chunk.status.value}'
                    """)

                # Create HAS_CHUNK relationship to WikiPage
                self.graph.query(f"""
                    MATCH (w:WikiPage {{wikidata_id: '{chunk.source_qid}'}})
                    MATCH (c:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                    MERGE (w)-[r:HAS_CHUNK]->(c)
                    ON CREATE SET r.position = {chunk.chunk_index}
                """)

                saved_count += 1

            except Exception as e:
                logger.warning(f"Failed to save chunk {chunk.chunk_id}: {e}")

        # Create CHILD_OF relationships for RAPTOR hierarchy
        # (done after all chunks are saved to ensure parents exist)
        child_of_count = 0
        for chunk in chunks:
            if chunk.parent_chunk_id:
                try:
                    self.graph.query(f"""
                        MATCH (child:DocumentChunk {{chunk_id: '{chunk.chunk_id}'}})
                        MATCH (parent:DocumentChunk {{chunk_id: '{chunk.parent_chunk_id}'}})
                        MERGE (child)-[:CHILD_OF]->(parent)
                    """)
                    child_of_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create CHILD_OF for {chunk.chunk_id}: {e}")

        if child_of_count > 0:
            logger.info(f"Created {child_of_count} CHILD_OF relationships")

        logger.info(f"Saved {saved_count}/{len(chunks)} chunks to FalkorDB")
        return saved_count

    def process_article(
        self,
        article: ArticleRankingResult,
        with_embeddings: bool = True,
    ) -> Tuple[int, List[DocumentChunk]]:
        """
        Process a single Wikipedia article through the full pipeline.

        Args:
            article: ArticleRankingResult with article metadata.
            with_embeddings: Whether to compute embeddings.

        Returns:
            Tuple of (chunks_saved, chunks_list).
        """
        # Extract title from URL
        title = self.extract_title_from_url(article.wikipedia_url)
        if not title:
            title = article.name  # Fallback to entity name

        logger.info(f"Processing article: {title} ({article.qid})")

        # Load article content
        content = self.load_article(title)
        if not content:
            logger.warning(f"  Skipping: Could not load content")
            return 0, []

        # Chunk the content
        chunks = self.chunk_content(content, article.qid)
        if not chunks:
            logger.warning(f"  Skipping: No chunks generated")
            return 0, []

        # Embed chunks if requested
        if with_embeddings:
            chunks = self.embed_chunks(chunks)

        # Save to FalkorDB
        saved = self.save_chunks(chunks)
        logger.info(f"  Created {saved} chunks ({len(content)} chars)")

        return saved, chunks

    def run(
        self,
        max_articles: int = 100,
        min_score: float = 0.0,
        with_embeddings: bool = True,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full Wikipedia article pipeline.

        Args:
            max_articles: Maximum number of articles to process.
            min_score: Minimum ranking score to process.
            with_embeddings: Whether to compute embeddings.
            skip_existing: Skip articles that already have chunks.

        Returns:
            Dictionary with pipeline statistics.
        """
        start_time = time.time()

        # Initialize schema
        self.init_schema()

        # Get WikiPages with URLs
        wikipages = self.get_wikipages_with_urls(
            min_score=min_score,
            limit=max_articles * 2,  # Get extra in case some fail
        )

        if not wikipages:
            return {"error": "No WikiPage nodes found with URLs"}

        # Rank articles
        ranked = self.rank_articles(wikipages)

        # Filter by min score
        if min_score > 0:
            ranked = [r for r in ranked if r.score >= min_score]

        # Get already processed QIDs if skip_existing
        processed_qids = set()
        if skip_existing:
            result = self.graph.query("""
                MATCH (w:WikiPage)-[:HAS_CHUNK]->(c:DocumentChunk)
                RETURN DISTINCT w.wikidata_id as qid
            """)
            for row in result.result_set:
                processed_qids.add(row[0])
            logger.info(f"Found {len(processed_qids)} already processed articles")

            # Filter out already processed
            ranked = [r for r in ranked if r.qid not in processed_qids]

        # Limit to max_articles
        ranked = ranked[:max_articles]

        logger.info(f"Processing {len(ranked)} articles...")

        # Process each article
        stats = {
            "articles_processed": 0,
            "articles_failed": 0,
            "total_chunks": 0,
            "articles": [],
        }

        for article in ranked:
            try:
                saved, chunks = self.process_article(article, with_embeddings)
                stats["articles_processed"] += 1
                stats["total_chunks"] += saved
                stats["articles"].append({
                    "qid": article.qid,
                    "name": article.name,
                    "chunks": saved,
                    "score": article.score,
                })
            except Exception as e:
                logger.error(f"Failed to process {article.name}: {e}")
                stats["articles_failed"] += 1

        stats["processing_time_seconds"] = time.time() - start_time
        stats["graph_name"] = self.graph_name

        # Get final counts
        try:
            result = self.graph.query("MATCH (c:DocumentChunk) RETURN count(c) as count")
            stats["total_document_chunks"] = result.result_set[0][0] if result.result_set else 0
        except Exception:
            pass

        return stats

    def similarity_search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks using vector similarity.

        Uses text-based search as fallback since FalkorDB vector index
        API may vary by version.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            min_score: Minimum similarity score (0-1).

        Returns:
            List of matching chunks with similarity scores.
        """
        # Get query embedding
        embeddings = self._get_embeddings()
        query_vector = embeddings.embed_query(query)

        # Try FalkorDB vector search with different syntax options
        try:
            # Option 1: Try the KNN query syntax
            vector_str = str(query_vector)
            result = self.graph.query(f"""
                CALL db.idx.vector.queryNodes('DocumentChunk', 'embedding', {limit * 2}, vecf32({vector_str}))
                YIELD node, score
                WHERE score >= {min_score}
                MATCH (w:WikiPage)-[:HAS_CHUNK]->(node)
                RETURN node.chunk_id as chunk_id,
                       node.content as content,
                       node.chunk_index as chunk_index,
                       w.name as source_name,
                       w.wikidata_id as source_qid,
                       score
                ORDER BY score DESC
                LIMIT {limit}
            """)

            results = []
            for row in result.result_set:
                results.append({
                    "chunk_id": row[0],
                    "content": row[1],
                    "chunk_index": row[2],
                    "source_name": row[3],
                    "source_qid": row[4],
                    "score": row[5],
                })

            return results

        except Exception as e:
            logger.warning(f"Vector search failed, falling back to text search: {e}")

            # Fallback: text-based search using CONTAINS
            try:
                # Split query into keywords
                keywords = query.lower().split()
                if not keywords:
                    return []

                # Search by content containing keywords
                keyword = keywords[0]  # Use first keyword
                result = self.graph.query(f"""
                    MATCH (w:WikiPage)-[:HAS_CHUNK]->(c:DocumentChunk)
                    WHERE toLower(c.content) CONTAINS '{keyword}'
                    RETURN c.chunk_id as chunk_id,
                           c.content as content,
                           c.chunk_index as chunk_index,
                           w.name as source_name,
                           w.wikidata_id as source_qid,
                           1.0 as score
                    LIMIT {limit}
                """)

                results = []
                for row in result.result_set:
                    results.append({
                        "chunk_id": row[0],
                        "content": row[1][:500] if row[1] else "",
                        "chunk_index": row[2],
                        "source_name": row[3],
                        "source_qid": row[4],
                        "score": row[5],
                    })
                return results

            except Exception as e2:
                logger.error(f"Fallback text search also failed: {e2}")
                return []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about DocumentChunk nodes including RAPTOR hierarchy."""
        stats = {}

        try:
            # Count chunks
            result = self.graph.query("MATCH (c:DocumentChunk) RETURN count(c) as count")
            stats["total_chunks"] = result.result_set[0][0] if result.result_set else 0

            # Count chunks by status
            result = self.graph.query("""
                MATCH (c:DocumentChunk)
                RETURN c.status as status, count(c) as count
            """)
            stats["chunks_by_status"] = {}
            for row in result.result_set:
                stats["chunks_by_status"][row[0] or "unknown"] = row[1]

            # Count WikiPages with chunks
            result = self.graph.query("""
                MATCH (w:WikiPage)-[:HAS_CHUNK]->(c:DocumentChunk)
                RETURN count(DISTINCT w) as count
            """)
            stats["wikipages_with_chunks"] = result.result_set[0][0] if result.result_set else 0

            # Count HAS_CHUNK relationships
            result = self.graph.query("""
                MATCH ()-[r:HAS_CHUNK]->()
                RETURN count(r) as count
            """)
            stats["has_chunk_relationships"] = result.result_set[0][0] if result.result_set else 0

            # RAPTOR hierarchy stats: chunks by level
            result = self.graph.query("""
                MATCH (c:DocumentChunk)
                WHERE c.level IS NOT NULL
                RETURN c.level as level, count(c) as count
                ORDER BY c.level
            """)
            stats["chunks_by_level"] = {}
            for row in result.result_set:
                level_name = {0: "root", 1: "cluster", 2: "leaf"}.get(row[0], f"level_{row[0]}")
                stats["chunks_by_level"][level_name] = row[1]

            # RAPTOR hierarchy stats: chunks by type
            result = self.graph.query("""
                MATCH (c:DocumentChunk)
                WHERE c.chunk_type IS NOT NULL
                RETURN c.chunk_type as chunk_type, count(c) as count
            """)
            stats["chunks_by_type"] = {}
            for row in result.result_set:
                stats["chunks_by_type"][row[0] or "unknown"] = row[1]

            # Count CHILD_OF relationships (RAPTOR hierarchy)
            result = self.graph.query("""
                MATCH ()-[r:CHILD_OF]->()
                RETURN count(r) as count
            """)
            stats["child_of_relationships"] = result.result_set[0][0] if result.result_set else 0

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats["error"] = str(e)

        return stats


def main():
    """CLI entry point for Wikipedia article pipeline."""
    parser = argparse.ArgumentParser(
        description="Process Wikipedia articles from Wikidata backbone graph"
    )

    parser.add_argument(
        "--graph",
        type=str,
        default="wikidata",
        help="FalkorDB graph name (default: wikidata)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="Maximum number of articles to process (default: 100)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Chunk size in characters (default: 1500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum article ranking score (default: 0.0)"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding computation"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess articles that already have chunks"
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="recursive",
        choices=["recursive", "section", "hybrid", "semantic", "raptor"],
        help="Chunking strategy: recursive (default), section, hybrid, semantic (embedding-based), or raptor (semantic + clustering + LLM summaries)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show current statistics"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Run similarity search with this query"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse chunking strategy
    chunking_strategy = ChunkingStrategy(args.chunking_strategy)

    # Initialize pipeline
    pipeline = WikipediaArticlePipeline(
        graph_name=args.graph,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=chunking_strategy,
    )

    # Stats-only mode
    if args.stats_only:
        stats = pipeline.get_stats()
        print(f"\nDocumentChunk Statistics ({args.graph}):")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        return

    # Search mode
    if args.search:
        print(f"\nSearching for: '{args.search}'")
        results = pipeline.similarity_search(args.search, limit=5)
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r['score']:.3f}] {r['source_name']} (chunk {r['chunk_index']})")
            print(f"   {r['content'][:200]}...")
        return

    # Run pipeline
    print("\n" + "=" * 60)
    print("WIKIPEDIA ARTICLE PIPELINE")
    print("=" * 60)
    print(f"Graph:           {args.graph}")
    print(f"Max Articles:    {args.max_articles}")
    print(f"Chunk Size:      {args.chunk_size}")
    print(f"Chunk Overlap:   {args.chunk_overlap}")
    print(f"Chunking:        {args.chunking_strategy}")
    print(f"Min Score:       {args.min_score}")
    print(f"With Embeddings: {not args.no_embeddings}")
    print(f"Skip Existing:   {not args.reprocess}")
    print("=" * 60)

    stats = pipeline.run(
        max_articles=args.max_articles,
        min_score=args.min_score,
        with_embeddings=not args.no_embeddings,
        skip_existing=not args.reprocess,
    )

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return

    print(f"Articles processed: {stats['articles_processed']}")
    print(f"Articles failed: {stats['articles_failed']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Processing time: {stats['processing_time_seconds']:.2f}s")

    if stats.get("total_document_chunks"):
        print(f"\nTotal DocumentChunks in graph: {stats['total_document_chunks']}")

    if stats.get("articles"):
        print("\nProcessed articles:")
        for a in stats["articles"][:10]:
            print(f"  - {a['name']}: {a['chunks']} chunks (score: {a['score']:.3f})")
        if len(stats["articles"]) > 10:
            print(f"  ... and {len(stats['articles']) - 10} more")


if __name__ == "__main__":
    main()
