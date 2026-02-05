"""
Knowledge Graph Pipeline Package.

This package provides a multi-stage pipeline for building knowledge graphs from
Wikidata, Wikipedia, and GE Vernova web pages:

Stage 1: Wikidata Backbone (existing wikidata_kg_builder.py)
    - WikidataSPARQLClient queries Wikidata for entities
    - WikidataKGBuilder traverses the entity graph
    - FalkorDBPageLoader stores WikiPage nodes in FalkorDB

Stage 2a: Wikipedia Article Processing (wikipedia_pipeline.py)
    - Rank articles by usefulness (connectivity, abstraction)
    - Load content via WikipediaLoader
    - Chunk with semantic-aware splitting
    - Embed with qwen3-embedding:8b
    - Store as DocumentChunk nodes linked to WikiPage

Stage 2b: GE Vernova Web Page Processing (webpage_pipeline.py)
    - Parse sitemaps for URL discovery (sitemap_parser.py)
    - Scrape web page content
    - Store as WebPage nodes in FalkorDB
    - Chunk and embed content
    - Store as DocumentChunk nodes linked to WebPage

Stage 3: Entity/Relationship Extraction (entity_extraction.py)
    - Ontology discovery using graphrag_sdk
    - Temporal entity extraction using graphiti-core
    - Entity merging and deduplication

Stage 4: Entity Resolution (entity_resolution.py)
    - Cross-source entity matching using embeddings
    - Merge entities while preserving source URLs
    - Unified provenance across Wikipedia and WebPage sources

Key Components:
    - WikipediaArticlePipeline: Stage 2a orchestrator
    - SitemapParser: Parse GE Vernova sitemaps
    - WebPagePipeline: Stage 2b orchestrator
    - OntologyDiscoveryPipeline: Auto-discover entity types
    - GraphitiEntityExtractor: Temporal entity extraction
    - HybridEntityExtractor: Combined extraction approach
    - CrossSourceEntityResolver: Entity resolution across sources
"""

from .models import (
    DocumentChunk,
    WebPage,
    SourceType,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    PipelineConfig,
    ArticleRankingResult,
)

from .wikipedia_pipeline import WikipediaArticlePipeline

from .sitemap_parser import SitemapParser, URLEntry

from .webpage_pipeline import WebPagePipeline

from .entity_extraction import (
    OntologyDiscoveryPipeline,
    GraphitiEntityExtractor,
    HybridEntityExtractor,
)

from .entity_resolution import CrossSourceEntityResolver

__all__ = [
    # Models
    "DocumentChunk",
    "WebPage",
    "SourceType",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "PipelineConfig",
    "ArticleRankingResult",
    # Stage 2a: Wikipedia
    "WikipediaArticlePipeline",
    # Stage 2b: Web Pages
    "SitemapParser",
    "URLEntry",
    "WebPagePipeline",
    # Stage 3: Entity Extraction
    "OntologyDiscoveryPipeline",
    "GraphitiEntityExtractor",
    "HybridEntityExtractor",
    # Stage 4: Entity Resolution
    "CrossSourceEntityResolver",
]

__version__ = "0.2.0"
