"""
Knowledge Graph Pipeline Package.

This package provides a 3-stage pipeline for building knowledge graphs from
Wikidata and Wikipedia:

Stage 1: Wikidata Backbone (existing wikidata_kg_builder.py)
    - WikidataSPARQLClient queries Wikidata for entities
    - WikidataKGBuilder traverses the entity graph
    - FalkorDBPageLoader stores WikiPage nodes in FalkorDB

Stage 2: Wikipedia Article Processing (wikipedia_pipeline.py)
    - Rank articles by usefulness (connectivity, abstraction)
    - Load content via WikipediaLoader
    - Chunk with semantic-aware splitting
    - Embed with qwen3-embedding:8b
    - Store as DocumentChunk nodes linked to WikiPage

Stage 3: Entity/Relationship Extraction (entity_extraction.py)
    - Ontology discovery using graphrag_sdk
    - Temporal entity extraction using graphiti-core
    - Entity merging and deduplication

Key Components:
    - WikipediaArticlePipeline: Stage 2 orchestrator
    - OntologyDiscoveryPipeline: Auto-discover entity types
    - GraphitiEntityExtractor: Temporal entity extraction
    - HybridEntityExtractor: Combined extraction approach
"""

from .models import (
    DocumentChunk,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    PipelineConfig,
    ArticleRankingResult,
)

from .wikipedia_pipeline import WikipediaArticlePipeline

from .entity_extraction import (
    OntologyDiscoveryPipeline,
    GraphitiEntityExtractor,
    HybridEntityExtractor,
)

__all__ = [
    # Models
    "DocumentChunk",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "PipelineConfig",
    "ArticleRankingResult",
    # Stage 2
    "WikipediaArticlePipeline",
    # Stage 3
    "OntologyDiscoveryPipeline",
    "GraphitiEntityExtractor",
    "HybridEntityExtractor",
]

__version__ = "0.1.0"
