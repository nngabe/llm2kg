"""
Data models for the Knowledge Graph Pipeline.

This module defines dataclasses for:
- DocumentChunk: Chunked Wikipedia article content with embeddings
- ExtractionResult: Results from entity/relationship extraction
- ExtractedEntity: Individual extracted entity with temporal metadata
- ExtractedRelationship: Extracted relationship between entities
- PipelineConfig: Configuration for the entire pipeline
- ArticleRankingResult: Scored Wikipedia article for processing prioritization
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from enum import Enum


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


@dataclass
class DocumentChunk:
    """
    Represents a chunked portion of a Wikipedia article.

    Stored in FalkorDB as:
    (c:DocumentChunk {
        chunk_id: "hash",
        content: "text",
        chunk_index: 0,
        embedding: [4096 floats],
        created_at: datetime,
        chunk_type: "leaf",
        level: 2,
        cluster_id: 0
    })
    -[:HAS_CHUNK {position: 0}]->
    (w:WikiPage {wikidata_id: "Q123"})

    RAPTOR Hierarchy (when using raptor chunking strategy):
    - Level 0 (ROOT): Document summary node
    - Level 1 (CLUSTER): Cluster summary nodes (children of root)
    - Level 2 (LEAF): Semantic chunks (children of clusters)

    Relationships:
    - (leaf)-[:CHILD_OF]->(cluster)
    - (cluster)-[:CHILD_OF]->(root)
    """
    content: str                              # The actual text content
    source_qid: str                           # Wikidata QID of source WikiPage
    chunk_index: int                          # Position in the article (0-indexed)
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
        """Generate chunk_id if not provided."""
        if self.chunk_id is None:
            self.chunk_id = self._compute_id()
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def _compute_id(self) -> str:
        """Compute a unique ID based on content and source."""
        hash_input = f"{self.source_qid}:{self.chunk_index}:{self.content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def to_cypher_params(self) -> Dict[str, Any]:
        """Convert to parameters for Cypher queries."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
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
        valid_from: datetime,
        valid_to: datetime,
        created_at: datetime
    })
    -[:EXTRACTED_FROM {confidence: 0.95}]->
    (c:DocumentChunk {chunk_id: "..."})
    """
    name: str                                  # Canonical entity name
    ontology_type: str                         # Type from ontology (e.g., "Equipment")
    entity_id: Optional[str] = None            # UUID for this entity
    description: Optional[str] = None          # Rich description
    embedding: Optional[List[float]] = None    # Vector embedding
    aliases: List[str] = field(default_factory=list)  # Alternative names
    source_chunk_ids: Set[str] = field(default_factory=set)  # Provenance
    confidence: float = 1.0                    # Extraction confidence [0-1]
    status: EntityStatus = EntityStatus.CANDIDATE
    # Temporal bi-data model (graphiti-style)
    valid_from: Optional[datetime] = None      # When this fact became true
    valid_to: Optional[datetime] = None        # When this fact ceased to be true (None = current)
    created_at: Optional[datetime] = None      # When we ingested this
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate entity_id if not provided."""
        if self.entity_id is None:
            import uuid
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
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }

    def merge_with(self, other: "ExtractedEntity") -> "ExtractedEntity":
        """
        Merge this entity with another (for deduplication).

        Combines descriptions, aliases, and source chunk IDs.
        Uses the higher confidence score.
        """
        merged_aliases = set(self.aliases) | set(other.aliases)
        merged_aliases.add(other.name)  # Add other's name as alias
        merged_sources = self.source_chunk_ids | other.source_chunk_ids

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
            valid_from=self.valid_from or other.valid_from,
            valid_to=self.valid_to or other.valid_to,
            created_at=self.created_at,
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
