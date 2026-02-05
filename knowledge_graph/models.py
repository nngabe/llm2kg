"""
Data models for the Knowledge Graph Pipeline.

This module defines dataclasses for:
- DocumentChunk: Chunked content with embeddings (multi-source: Wikipedia/WebPage)
- WebPage: Web page source node (analogous to WikiPage)
- ExtractionResult: Results from entity/relationship extraction
- ExtractedEntity: Individual extracted entity with bi-temporal metadata
- ExtractedRelationship: Extracted relationship between entities
- Episode: Episodic memory node linking chunks to entities (Zep temporal layer)
- Community: Entity community detected via Louvain algorithm (Zep community layer)
- PipelineConfig: Configuration for the entire pipeline
- ArticleRankingResult: Scored Wikipedia article for processing prioritization

Zep Architecture Layers:
- Episodic Layer: Episodes from document chunks
- Semantic Layer: Entities with full bi-temporal support
- Community Layer: Graph-aware entity communities
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class SourceType(str, Enum):
    """Type of content source."""
    WIKIPEDIA = "wikipedia"      # Wikipedia/Wikidata source
    WEBPAGE = "webpage"          # Web page source (e.g., GE Vernova)


class ChunkStatus(str, Enum):
    """Status of a document chunk in the pipeline."""
    PENDING = "pending"           # Not yet processed
    EMBEDDED = "embedded"         # Has embedding computed
    EXTRACTED = "extracted"       # Entities extracted
    FAILED = "failed"             # Processing failed


class ChunkType(str, Enum):
    """Type of chunk in the RAPTOR hierarchy."""
    LEAF = "leaf"           # Semantic chunks (searchable, level=2)
    CLUSTER = "cluster"     # Cluster summary (intermediate, level=1)
    ROOT = "root"           # Document summary (top level, level=0)


class EntityStatus(str, Enum):
    """Status of an extracted entity."""
    CANDIDATE = "candidate"       # Newly extracted, not yet verified
    VERIFIED = "verified"         # Verified against existing entities
    MERGED = "merged"             # Merged with existing entity
    REJECTED = "rejected"         # Rejected as invalid/duplicate


class FactStatus(str, Enum):
    """Lifecycle status of a fact (entity or relationship) - Zep-style temporal tracking."""
    CANDIDATE = "candidate"       # Newly extracted, not verified
    ACTIVE = "active"             # Currently valid
    SUPERSEDED = "superseded"     # Replaced by newer version
    OBSOLETE = "obsolete"         # Fact became false
    DELETED = "deleted"           # Soft-deleted


@dataclass
class WebPage:
    """
    Represents a web page source node (analogous to WikiPage).

    Stored in FalkorDB as:
    (:WebPage {
        webpage_id: "uuid",
        url: "https://www.gevernova.com/gas-power/equipment/turbines",
        title: "Gas Turbines | GE Vernova",
        domain: "gevernova.com",
        division: "gas-power",
        category: "equipment",
        depth: 2,
        source_type: "webpage",
        crawled_at: datetime
    })

    Hierarchy levels:
    - Level 0: Root domain (e.g., gevernova.com)
    - Level 1: Business divisions (e.g., gas-power, wind-power)
    - Level 2: Categories (e.g., equipment, services)
    - Level 3+: Subcategories and product pages
    """
    url: str                                   # Full URL
    title: str                                 # Page title
    domain: str                                # Domain (e.g., "gevernova.com")
    division: str                              # Level 1 division (e.g., "gas-power")
    webpage_id: Optional[str] = None           # UUID
    category: Optional[str] = None             # Level 2 category
    subcategory: Optional[str] = None          # Level 3+ subcategory
    depth: int = 0                             # Hierarchy level
    parent_url: Optional[str] = None           # Parent page URL
    content: Optional[str] = None              # Page content (after scraping)
    crawled_at: Optional[datetime] = None      # When page was crawled
    content_hash: Optional[str] = None         # For change detection
    source_type: SourceType = SourceType.WEBPAGE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate webpage_id if not provided."""
        if self.webpage_id is None:
            self.webpage_id = str(uuid.uuid4())[:12]
        if self.crawled_at is None:
            self.crawled_at = datetime.now(timezone.utc)

    def _compute_content_hash(self) -> Optional[str]:
        """Compute hash of content for change detection."""
        if self.content:
            return hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return None

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to parameters for Cypher queries."""
        return {
            "webpage_id": self.webpage_id,
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "division": self.division,
            "category": self.category,
            "subcategory": self.subcategory,
            "depth": self.depth,
            "parent_url": self.parent_url,
            "crawled_at": self.crawled_at.isoformat() if self.crawled_at else None,
            "content_hash": self.content_hash or self._compute_content_hash(),
            "source_type": self.source_type.value,
            "metadata": self.metadata,
        }

    def __hash__(self):
        return hash(self.webpage_id)

    def __eq__(self, other):
        if isinstance(other, WebPage):
            return self.webpage_id == other.webpage_id
        return False


@dataclass
class DocumentChunk:
    """
    Represents a chunked portion of content from Wikipedia or WebPage sources.

    Stored in FalkorDB as:
    (c:DocumentChunk {
        chunk_id: "hash",
        content: "text",
        chunk_index: 0,
        source_id: "Q123" or "uuid",
        source_type: "wikipedia" or "webpage",
        source_url: "https://...",
        embedding: [4096 floats],
        created_at: datetime,
        chunk_type: "leaf",
        level: 2,
        cluster_id: 0
    })
    -[:HAS_CHUNK {position: 0}]->
    (w:WikiPage {wikidata_id: "Q123"}) | (wp:WebPage {webpage_id: "uuid"})

    Multi-source support:
    - source_id: Unique identifier (wikidata_id for Wikipedia, webpage_id for WebPage)
    - source_type: "wikipedia" or "webpage"
    - source_url: Original URL for provenance
    - source_qid: Deprecated, kept for backwards compatibility (use source_id)

    RAPTOR Hierarchy (when using raptor chunking strategy):
    - Level 0 (ROOT): Document summary node
    - Level 1 (CLUSTER): Cluster summary nodes (children of root)
    - Level 2 (LEAF): Semantic chunks (children of clusters)

    Relationships:
    - (leaf)-[:CHILD_OF]->(cluster)
    - (cluster)-[:CHILD_OF]->(root)
    """
    content: str                              # The actual text content
    chunk_index: int                          # Position in the article (0-indexed)

    # Multi-source support
    source_id: Optional[str] = None           # webpage_id OR wikidata_id
    source_type: SourceType = SourceType.WIKIPEDIA  # "wikipedia" | "webpage"
    source_url: Optional[str] = None          # Original URL for provenance

    # Backwards compatibility (deprecated, use source_id)
    source_qid: Optional[str] = None          # Wikidata QID (deprecated)

    chunk_id: Optional[str] = None            # Computed hash of content
    embedding: Optional[List[float]] = None   # Vector embedding (4096 dim for qwen3)
    status: ChunkStatus = ChunkStatus.PENDING
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # RAPTOR hierarchy fields
    chunk_type: ChunkType = ChunkType.LEAF    # leaf, cluster, or root
    level: int = 2                            # 0=root, 1=cluster, 2=leaf
    parent_chunk_id: Optional[str] = None     # ID of parent chunk (for CHILD_OF relationship)
    cluster_id: Optional[int] = None          # Cluster assignment for RAPTOR

    def __post_init__(self):
        """Generate chunk_id and handle backwards compatibility."""
        # Backwards compatibility: if source_qid is set but source_id is not
        if self.source_qid and not self.source_id:
            self.source_id = self.source_qid
            self.source_type = SourceType.WIKIPEDIA
        # Forward compatibility: if source_id is set but source_qid is not
        if self.source_id and not self.source_qid and self.source_type == SourceType.WIKIPEDIA:
            self.source_qid = self.source_id

        if self.chunk_id is None:
            self.chunk_id = self._compute_id()
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def _compute_id(self) -> str:
        """Compute a unique ID based on content and source."""
        source = self.source_id or self.source_qid or "unknown"
        hash_input = f"{source}:{self.chunk_index}:{self.content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to parameters for Cypher queries."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            # Multi-source fields
            "source_id": self.source_id,
            "source_type": self.source_type.value if self.source_type else "wikipedia",
            "source_url": self.source_url,
            # Backwards compatibility
            "source_qid": self.source_qid,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
            # RAPTOR hierarchy fields
            "chunk_type": self.chunk_type.value,
            "level": self.level,
            "parent_chunk_id": self.parent_chunk_id,
            "cluster_id": self.cluster_id,
        }

    def __hash__(self):
        return hash(self.chunk_id)

    def __eq__(self, other):
        if isinstance(other, DocumentChunk):
            return self.chunk_id == other.chunk_id
        return False


@dataclass
class ExtractedEntity:
    """
    An entity extracted from document chunks.

    Stored in FalkorDB as:
    (e:Entity {
        id: "uuid",
        name: "Gas Turbine",
        ontology_type: "Equipment",
        description: "...",
        embedding: [4096],
        source_urls: ["https://en.wikipedia.org/...", "https://gevernova.com/..."],
        source_types: ["wikipedia", "webpage"],
        valid_from: datetime,
        valid_to: datetime,
        created_at: datetime
    })
    -[:EXTRACTED_FROM {confidence: 0.95}]->
    (c:DocumentChunk {chunk_id: "..."})

    Multi-source provenance:
    - source_urls: Set of all source URLs (Wikipedia + WebPage)
    - source_types: Set of source types ("wikipedia", "webpage")
    """
    name: str                                  # Canonical entity name
    ontology_type: str                         # Type from ontology (e.g., "Equipment")
    entity_id: Optional[str] = None            # UUID for this entity
    description: Optional[str] = None          # Rich description
    embedding: Optional[List[float]] = None    # Vector embedding
    aliases: List[str] = field(default_factory=list)  # Alternative names
    source_chunk_ids: Set[str] = field(default_factory=set)  # Provenance (chunk IDs)
    confidence: float = 1.0                    # Extraction confidence [0-1]
    status: EntityStatus = EntityStatus.CANDIDATE
    # Multi-source provenance
    source_urls: Set[str] = field(default_factory=set)     # All source URLs
    source_types: Set[str] = field(default_factory=set)    # "wikipedia", "webpage"
    # Enhanced bi-temporal model (Zep-style)
    valid_from: Optional[datetime] = None      # When this fact became true (event time)
    valid_to: Optional[datetime] = None        # When this fact ceased to be true (None = current)
    created_at: Optional[datetime] = None      # First ingestion time (transaction time)
    modified_at: Optional[datetime] = None     # Last update time
    obsoleted_at: Optional[datetime] = None    # Soft-delete time
    merged_at: Optional[datetime] = None       # When merged with another entity
    fact_status: str = "active"                # Lifecycle state (FactStatus value)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate entity_id if not provided."""
        if self.entity_id is None:
            self.entity_id = str(uuid.uuid4())[:12]
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to parameters for Cypher queries."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "ontology_type": self.ontology_type,
            "description": self.description,
            "aliases": list(self.aliases),
            "confidence": self.confidence,
            "status": self.status.value,
            # Multi-source provenance
            "source_urls": list(self.source_urls),
            "source_types": list(self.source_types),
            # Enhanced bi-temporal fields (Zep-style)
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "obsoleted_at": self.obsoleted_at.isoformat() if self.obsoleted_at else None,
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            "fact_status": self.fact_status,
            "metadata": self.metadata,
        }

    def merge_with(self, other: "ExtractedEntity") -> "ExtractedEntity":
        """
        Merge this entity with another (for deduplication).

        Combines descriptions, aliases, source chunk IDs, and source URLs.
        Uses the higher confidence score.
        Preserves all source URLs for multi-source provenance.
        """
        merged_aliases = set(self.aliases) | set(other.aliases)
        merged_aliases.add(other.name)  # Add other's name as alias
        merged_sources = self.source_chunk_ids | other.source_chunk_ids
        merged_urls = self.source_urls | other.source_urls  # UNION URLs
        merged_types = self.source_types | other.source_types  # UNION types

        # Combine descriptions if both exist
        if self.description and other.description:
            if other.description not in self.description:
                merged_desc = f"{self.description} {other.description}"
            else:
                merged_desc = self.description
        else:
            merged_desc = self.description or other.description

        return ExtractedEntity(
            name=self.name,  # Keep original name
            ontology_type=self.ontology_type,
            entity_id=self.entity_id,
            description=merged_desc,
            embedding=self.embedding,  # Keep original embedding
            aliases=list(merged_aliases),
            source_chunk_ids=merged_sources,
            confidence=max(self.confidence, other.confidence),
            status=EntityStatus.MERGED,
            source_urls=merged_urls,
            source_types=merged_types,
            valid_from=self.valid_from or other.valid_from,
            valid_to=self.valid_to or other.valid_to,
            created_at=self.created_at,
            modified_at=datetime.now(timezone.utc),
            merged_at=datetime.now(timezone.utc),  # Track merge time
            fact_status="active",
            metadata={**self.metadata, **other.metadata},
        )

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if isinstance(other, ExtractedEntity):
            return self.entity_id == other.entity_id
        return False


@dataclass
class ExtractedRelationship:
    """
    A relationship extracted between two entities.

    Stored in FalkorDB as:
    (source:Entity)-[r:RELATIONSHIP_TYPE {
        confidence: 0.95,
        valid_from: datetime,
        valid_to: datetime,
        source_chunk_id: "..."
    }]->(target:Entity)
    """
    source_entity_id: str                     # Source entity ID
    target_entity_id: str                     # Target entity ID
    relationship_type: str                    # Type from ontology (e.g., "PART_OF")
    relationship_id: Optional[str] = None     # Unique relationship ID
    description: Optional[str] = None         # Description of this specific relationship
    confidence: float = 1.0                   # Extraction confidence [0-1]
    source_chunk_ids: Set[str] = field(default_factory=set)  # Provenance
    # Temporal metadata
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate relationship_id if not provided."""
        if self.relationship_id is None:
            import uuid
            self.relationship_id = str(uuid.uuid4())[:12]
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to parameters for Cypher queries."""
        return {
            "relationship_id": self.relationship_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }

    def __hash__(self):
        return hash((self.source_entity_id, self.target_entity_id, self.relationship_type))

    def __eq__(self, other):
        if isinstance(other, ExtractedRelationship):
            return (self.source_entity_id == other.source_entity_id and
                    self.target_entity_id == other.target_entity_id and
                    self.relationship_type == other.relationship_type)
        return False


@dataclass
class Episode:
    """
    Episodic memory node - links document chunks to extracted entities.
    Each RAPTOR chunk becomes an Episode in the temporal layer.

    This implements Zep's episodic layer architecture:
    - Episodes are created from document chunks
    - Entities extracted from episodes link back via CONTAINS relationships
    - Provides temporal context for when facts were observed

    Stored in FalkorDB as:
    (ep:Episode {
        episode_id: "uuid",
        name: "chunk_abc123",
        content: "...",
        source_chunk_id: "...",
        source_type: "webpage",
        source_url: "https://...",
        reference_time: datetime,
        created_at: datetime
    })
    -[:DERIVED_FROM]-> (c:DocumentChunk)
    -[:CONTAINS]-> (e:Entity)
    """
    episode_id: str = ""
    name: str = ""                                # e.g., "chunk_abc123"
    content: str = ""                             # Chunk content (truncated for storage)
    source_chunk_id: str = ""                     # Link to DocumentChunk
    source_type: str = "text"                     # text, webpage, wikipedia
    source_url: Optional[str] = None
    reference_time: Optional[datetime] = None     # Event time of content
    created_at: Optional[datetime] = None         # Ingestion time
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    entity_ids: Set[str] = field(default_factory=set)  # Entities extracted from this episode

    def __post_init__(self):
        if not self.episode_id:
            self.episode_id = str(uuid.uuid4())[:12]
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.valid_from is None:
            self.valid_from = self.reference_time or self.created_at

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to parameters for Cypher queries."""
        return {
            "episode_id": self.episode_id,
            "name": self.name,
            "content": self.content[:2000] if self.content else "",  # Truncate for storage
            "source_chunk_id": self.source_chunk_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "reference_time": self.reference_time.isoformat() if self.reference_time else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }

    def __hash__(self):
        return hash(self.episode_id)

    def __eq__(self, other):
        if isinstance(other, Episode):
            return self.episode_id == other.episode_id
        return False


@dataclass
class Community:
    """
    Entity community detected via graph algorithms (Louvain).

    This implements Zep's community layer architecture:
    - Communities group semantically related entities
    - Detected using Louvain modularity optimization
    - Entities link to communities via BELONGS_TO relationships

    Stored in FalkorDB as:
    (c:Community {
        community_id: "uuid",
        name: "Community_abc123",
        summary: "...",
        algorithm: "louvain",
        modularity_score: 0.75,
        member_count: 15,
        created_at: datetime
    })
    <-[:BELONGS_TO]- (e:Entity)
    """
    community_id: str = ""
    name: str = ""                                # LLM-generated or auto name
    summary: Optional[str] = None                 # LLM-generated description
    algorithm: str = "louvain"
    modularity_score: float = 0.0
    member_count: int = 0
    representative_entities: List[str] = field(default_factory=list)  # Top entity IDs
    created_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    def __post_init__(self):
        if not self.community_id:
            self.community_id = str(uuid.uuid4())[:12]
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if not self.name:
            self.name = f"Community_{self.community_id[:6]}"

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to parameters for Cypher queries."""
        return {
            "community_id": self.community_id,
            "name": self.name,
            "summary": self.summary,
            "algorithm": self.algorithm,
            "modularity_score": self.modularity_score,
            "member_count": self.member_count,
            "representative_entities": self.representative_entities,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }

    def __hash__(self):
        return hash(self.community_id)

    def __eq__(self, other):
        if isinstance(other, Community):
            return self.community_id == other.community_id
        return False


@dataclass
class ExtractionResult:
    """
    Results from extracting entities and relationships from document chunks.

    Produced by Stage 3 extraction pipelines.
    """
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    source_chunk_ids: Set[str] = field(default_factory=set)
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def entity_count(self) -> int:
        """Number of extracted entities."""
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        """Number of extracted relationships."""
        return len(self.relationships)

    @property
    def success(self) -> bool:
        """Whether extraction completed without errors."""
        return len(self.errors) == 0

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        """Merge two extraction results."""
        return ExtractionResult(
            entities=self.entities + other.entities,
            relationships=self.relationships + other.relationships,
            source_chunk_ids=self.source_chunk_ids | other.source_chunk_ids,
            processing_time_seconds=self.processing_time_seconds + other.processing_time_seconds,
            errors=self.errors + other.errors,
            metadata={**self.metadata, **other.metadata},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "source_chunk_ids": list(self.source_chunk_ids),
            "processing_time_seconds": self.processing_time_seconds,
            "errors": self.errors,
            "success": self.success,
        }


@dataclass
class ArticleRankingResult:
    """
    Result from ranking Wikipedia articles for processing priority.

    Used by Stage 2 to determine which articles to process first.
    """
    qid: str                                  # Wikidata QID
    name: str                                 # Entity name
    wikipedia_url: Optional[str]              # Wikipedia URL
    description: Optional[str]                # Wikidata description
    score: float                              # Composite ranking score
    connectivity_score: float = 0.0           # How connected in the backbone graph
    abstraction_score: float = 0.0            # How abstract/general the concept is
    has_description: bool = False             # Whether it has a description
    url_quality: float = 0.0                  # URL validity score

    def __lt__(self, other: "ArticleRankingResult") -> bool:
        """For sorting (higher score = higher priority)."""
        return self.score > other.score

    @classmethod
    def compute_score(
        cls,
        qid: str,
        name: str,
        url: Optional[str],
        description: Optional[str],
        connectivity: float = 0.0,
        abstraction: float = 0.0,
    ) -> "ArticleRankingResult":
        """
        Compute ranking score for an article.

        Score formula:
          0.4 * connectivity + 0.3 * abstraction + 0.2 * has_description + 0.1 * url_quality
        """
        has_desc = 1.0 if description and len(description) > 10 else 0.0

        # URL quality: valid Wikipedia URL = 1.0, otherwise 0.0
        url_q = 0.0
        if url and "wikipedia.org" in url:
            url_q = 1.0

        # Normalize scores to [0, 1] range
        norm_connectivity = min(connectivity / 100.0, 1.0) if connectivity > 0 else 0.0
        norm_abstraction = min(abstraction / 50.0, 1.0) if abstraction > 0 else 0.0

        score = (
            0.4 * norm_connectivity +
            0.3 * norm_abstraction +
            0.2 * has_desc +
            0.1 * url_q
        )

        return cls(
            qid=qid,
            name=name,
            wikipedia_url=url,
            description=description,
            score=score,
            connectivity_score=connectivity,
            abstraction_score=abstraction,
            has_description=has_desc > 0,
            url_quality=url_q,
        )


@dataclass
class PipelineConfig:
    """
    Configuration for the Knowledge Graph Pipeline.

    Combines settings for all three stages.
    """
    # General settings
    graph_name: str = "wikidata"
    falkordb_host: str = "host.docker.internal"
    falkordb_port: int = 6379
    ollama_host: str = "http://host.docker.internal:11434"
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dimensions: int = 4096

    # Stage 2: Wikipedia pipeline settings
    max_articles: int = 100                   # Max articles to process
    chunk_size: int = 1500                    # Characters per chunk
    chunk_overlap: int = 200                  # Overlap between chunks
    min_article_score: float = 0.0            # Minimum score to process
    batch_size: int = 10                      # Batch size for operations

    # Stage 3: Entity extraction settings
    ontology_name: str = "medium"             # Ontology size (small/medium/large)
    discover_ontology: bool = False           # Auto-discover from chunks
    ontology_sample_size: int = 50            # Chunks to sample for discovery
    extract_entities: bool = True             # Run entity extraction
    use_graphiti: bool = True                 # Use graphiti for temporal extraction
    use_graphrag_sdk: bool = True             # Use graphrag_sdk for ontology
    dedup_threshold: float = 0.85             # Similarity threshold for deduplication

    # LLM settings (for extraction)
    llm_provider: str = "ollama"              # ollama, openai, anthropic, etc.
    llm_model: str = "qwen3:8b"               # Model name
    llm_temperature: float = 0.0              # Temperature for extraction

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph_name": self.graph_name,
            "falkordb_host": self.falkordb_host,
            "falkordb_port": self.falkordb_port,
            "ollama_host": self.ollama_host,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "max_articles": self.max_articles,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_article_score": self.min_article_score,
            "batch_size": self.batch_size,
            "ontology_name": self.ontology_name,
            "discover_ontology": self.discover_ontology,
            "ontology_sample_size": self.ontology_sample_size,
            "extract_entities": self.extract_entities,
            "use_graphiti": self.use_graphiti,
            "use_graphrag_sdk": self.use_graphrag_sdk,
            "dedup_threshold": self.dedup_threshold,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
