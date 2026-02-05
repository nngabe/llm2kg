#!/usr/bin/env python3
"""
Unified Knowledge Graph Building Pipeline.

Orchestrates all stages of KG building with configurable parameters.
All configuration is defined at the top of this file as CAPS_VARS.

Usage:
    python run_pipeline.py                        # Run full pipeline
    python run_pipeline.py --stages 1,2a          # Run specific stages
    python run_pipeline.py --graph test_kg        # Override graph name
    python run_pipeline.py --dry-run              # Show config without executing
    python run_pipeline.py --stats-only           # Show current graph stats only
"""

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

# --- Graph Settings ---
GRAPH_NAME = "ge_vernova_kg"
BACKEND = "falkordb"  # "falkordb" or "neo4j"

# --- Stage 1: Wikidata Backbone ---
WIKIDATA_SEEDS_FILE = "seed/ge_vernova_seeds_small.json"
WIKIDATA_MAX_DEPTH = 2
WIKIDATA_MAX_PER_LEVEL = 500
WIKIDATA_STRATEGY = "root_first"  # "bfs" or "root_first"
WIKIDATA_REQUIRE_LABEL = True
WIKIDATA_WITH_EMBEDDINGS = True

# --- Stage 2a: Wikipedia Articles ---
WIKIPEDIA_MAX_ARTICLES = 100
WIKIPEDIA_MIN_SCORE = 0.1
WIKIPEDIA_CHUNKING_STRATEGY = "raptor"  # "recursive", "semantic", "raptor"

# --- Stage 2b: Web Pages ---
# Available divisions: gas-power, steam-power, wind-power, hydropower,
#                      grid-solutions, solar-storage, power-conversion,
#                      consulting, lm-wind-power
# Use "all" to crawl all divisions, or a list like ["gas-power", "wind-power"]
WEBPAGE_DIVISIONS = "all" # ["gas-power"]
WEBPAGE_MAX_PAGES = 50
WEBPAGE_MAX_DEPTH = 5
WEBPAGE_CHUNKING_STRATEGY = "raptor"
WEBPAGE_CONTENT_FORMAT = "abstract"  # "raw", "synthesized", "abstract", "llm"

# --- Stage 3: Entity Extraction ---
ENTITY_MAX_CHUNKS = 500
ENTITY_ONTOLOGY = "medium"  # "small", "medium", "large"
ENTITY_DISCOVER_ONTOLOGY = False
ENTITY_WITH_DEDUP = True

# --- Stage 4: Community Detection ---
COMMUNITY_MIN_SIZE = 3
COMMUNITY_RESOLUTION = 1.0
COMMUNITY_GENERATE_SUMMARIES = True

# --- General Settings ---
WITH_EMBEDDINGS = True
SKIP_EXISTING = True
VERBOSE = False

# --- Stage Control ---
RUN_STAGE_1_WIKIDATA = True
RUN_STAGE_2A_WIKIPEDIA = True
RUN_STAGE_2B_WEBPAGES = True
RUN_STAGE_3_ENTITIES = True
RUN_STAGE_4_COMMUNITIES = True


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
import logging
import asyncio
import argparse
from typing import Dict, Any, Optional, List

# Stage 1
from wikidata_kg_builder import WikidataKGBuilder, ExplorationStrategy

# Stages 2-4
from knowledge_graph.wikipedia_pipeline import WikipediaArticlePipeline, ChunkingStrategy
from knowledge_graph.webpage_pipeline import WebPagePipeline
from knowledge_graph.entity_extraction import HybridEntityExtractor
from knowledge_graph.community_detection import EntityCommunityDetector

# Ontology support
from ontologies import get_ontology

# FalkorDB connection
from falkordb import FalkorDB

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))


def get_falkordb_connection():
    """Get FalkorDB graph connection."""
    db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    return db.select_graph(GRAPH_NAME)


def print_header(title: str, char: str = "=", width: int = 60):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def print_config():
    """Print current pipeline configuration."""
    print_header("KG BUILDING PIPELINE")
    print(f"Graph:           {GRAPH_NAME}")
    print(f"Backend:         {BACKEND}")

    stages = []
    if RUN_STAGE_1_WIKIDATA:
        stages.append("1")
    if RUN_STAGE_2A_WIKIPEDIA:
        stages.append("2a")
    if RUN_STAGE_2B_WEBPAGES:
        stages.append("2b")
    if RUN_STAGE_3_ENTITIES:
        stages.append("3")
    if RUN_STAGE_4_COMMUNITIES:
        stages.append("4")

    print(f"Stages:          {', '.join(stages) if stages else 'None'}")
    print("=" * 60)


# =============================================================================
# STAGE FUNCTIONS
# =============================================================================

def run_wikidata_stage() -> Dict[str, Any]:
    """
    Stage 1: Build Wikidata backbone from seed entities.

    Returns:
        Statistics dictionary from the Wikidata build.
    """
    print_header("[Stage 1: Wikidata Backbone]", char="-")
    print(f"  Seeds file:    {WIKIDATA_SEEDS_FILE}")
    print(f"  Strategy:      {WIKIDATA_STRATEGY}")
    print(f"  Max depth:     {WIKIDATA_MAX_DEPTH}")
    print(f"  Max per level: {WIKIDATA_MAX_PER_LEVEL}")
    print(f"  Require label: {WIKIDATA_REQUIRE_LABEL}")
    print(f"  Embeddings:    {WIKIDATA_WITH_EMBEDDINGS}")
    print()

    # Map strategy string to enum
    strategy = (
        ExplorationStrategy.ROOT_FIRST
        if WIKIDATA_STRATEGY == "root_first"
        else ExplorationStrategy.BFS
    )

    # Create builder
    builder = WikidataKGBuilder(
        backend=BACKEND,
        exploration_strategy=strategy,
        falkordb_graph_name=GRAPH_NAME,
        require_label=WIKIDATA_REQUIRE_LABEL,
    )

    try:
        # Build from seeds
        stats = builder.build_from_seeds(
            seeds_file=WIKIDATA_SEEDS_FILE,
            max_depth=WIKIDATA_MAX_DEPTH,
            max_entities_per_level=WIKIDATA_MAX_PER_LEVEL,
            with_embeddings=WIKIDATA_WITH_EMBEDDINGS,
            exploration_strategy=strategy,
        )

        # Extract summary stats
        totals = stats.get("totals", {})
        entities = totals.get("total_entities", 0)
        relationships = totals.get("total_relationships", 0)
        seeds_processed = totals.get("seeds_processed", 0)
        seeds_failed = totals.get("seeds_failed", 0)

        print(f"  Result: {entities} entities, {relationships} relationships")
        print(f"          {seeds_processed} seeds processed, {seeds_failed} failed")

        return {
            "status": "success",
            "entities": entities,
            "relationships": relationships,
            "seeds_processed": seeds_processed,
            "seeds_failed": seeds_failed,
            "full_stats": stats,
        }

    except Exception as e:
        logger.error(f"Wikidata stage failed: {e}")
        raise
    finally:
        builder.close()


def run_wikipedia_stage() -> Dict[str, Any]:
    """
    Stage 2a: Process Wikipedia articles linked from WikiPage nodes.

    Returns:
        Statistics dictionary from Wikipedia processing.
    """
    print_header("[Stage 2a: Wikipedia Articles]", char="-")
    print(f"  Max articles:  {WIKIPEDIA_MAX_ARTICLES}")
    print(f"  Min score:     {WIKIPEDIA_MIN_SCORE}")
    print(f"  Chunking:      {WIKIPEDIA_CHUNKING_STRATEGY}")
    print(f"  Embeddings:    {WITH_EMBEDDINGS}")
    print(f"  Skip existing: {SKIP_EXISTING}")
    print()

    # Map chunking strategy
    chunking = ChunkingStrategy(WIKIPEDIA_CHUNKING_STRATEGY)

    # Create pipeline
    pipeline = WikipediaArticlePipeline(
        graph_name=GRAPH_NAME,
        chunking_strategy=chunking,
    )

    try:
        # Run pipeline
        stats = pipeline.run(
            max_articles=WIKIPEDIA_MAX_ARTICLES,
            min_score=WIKIPEDIA_MIN_SCORE,
            with_embeddings=WITH_EMBEDDINGS,
            skip_existing=SKIP_EXISTING,
        )

        if "error" in stats:
            logger.warning(f"Wikipedia stage warning: {stats['error']}")
            return {"status": "warning", "message": stats["error"]}

        articles = stats.get("articles_processed", 0)
        chunks = stats.get("total_chunks", 0)

        print(f"  Result: {articles} articles, {chunks} chunks")

        return {
            "status": "success",
            "articles_processed": articles,
            "total_chunks": chunks,
            "full_stats": stats,
        }

    except Exception as e:
        logger.error(f"Wikipedia stage failed: {e}")
        raise


def run_webpage_stage() -> Dict[str, Any]:
    """
    Stage 2b: Crawl and process web pages from GE Vernova divisions.

    Returns:
        Statistics dictionary from web page processing.
    """
    # Resolve divisions: "all" means None (pipeline uses all), otherwise use the list
    if WEBPAGE_DIVISIONS == "all":
        divisions = None
        divisions_display = "all"
    else:
        divisions = WEBPAGE_DIVISIONS
        divisions_display = WEBPAGE_DIVISIONS

    print_header("[Stage 2b: Web Pages]", char="-")
    print(f"  Divisions:     {divisions_display}")
    print(f"  Max pages:     {WEBPAGE_MAX_PAGES}")
    print(f"  Max depth:     {WEBPAGE_MAX_DEPTH}")
    print(f"  Chunking:      {WEBPAGE_CHUNKING_STRATEGY}")
    print(f"  Content fmt:   {WEBPAGE_CONTENT_FORMAT}")
    print(f"  Embeddings:    {WITH_EMBEDDINGS}")
    print(f"  Skip existing: {SKIP_EXISTING}")
    print()

    # Create pipeline
    pipeline = WebPagePipeline(
        graph_name=GRAPH_NAME,
        chunking_strategy=WEBPAGE_CHUNKING_STRATEGY,
        content_format=WEBPAGE_CONTENT_FORMAT,
    )

    try:
        # Run pipeline
        stats = pipeline.run(
            divisions=divisions,
            max_pages=WEBPAGE_MAX_PAGES,
            max_depth=WEBPAGE_MAX_DEPTH,
            with_embeddings=WITH_EMBEDDINGS,
            skip_existing=SKIP_EXISTING,
        )

        pages = stats.get("total_pages_processed", 0)
        chunks = stats.get("total_chunks", 0)
        failed = stats.get("total_pages_failed", 0)

        print(f"  Result: {pages} pages, {chunks} chunks, {failed} failed")

        return {
            "status": "success",
            "pages_processed": pages,
            "total_chunks": chunks,
            "pages_failed": failed,
            "full_stats": stats,
        }

    except Exception as e:
        logger.error(f"Webpage stage failed: {e}")
        raise


def run_entity_extraction_stage() -> Dict[str, Any]:
    """
    Stage 3: Extract entities and relationships from document chunks.

    Returns:
        Statistics dictionary from entity extraction.
    """
    print_header("[Stage 3: Entity Extraction]", char="-")
    print(f"  Max chunks:    {ENTITY_MAX_CHUNKS}")
    print(f"  Ontology:      {ENTITY_ONTOLOGY}")
    print(f"  Discover ont:  {ENTITY_DISCOVER_ONTOLOGY}")
    print(f"  With dedup:    {ENTITY_WITH_DEDUP}")
    print()

    # Get ontology configuration
    ontology_config = get_ontology(ENTITY_ONTOLOGY)
    ontology_dict = {
        "entity_types": ontology_config.entity_types,
        "relationship_types": ontology_config.relationship_types,
    }

    # Create extractor
    extractor = HybridEntityExtractor(
        graph_name=GRAPH_NAME,
        ontology=ontology_dict,
    )

    try:
        # Run extraction (async)
        stats = asyncio.run(
            extractor.run(
                max_chunks=ENTITY_MAX_CHUNKS,
                with_dedup=ENTITY_WITH_DEDUP,
                discover_ontology=ENTITY_DISCOVER_ONTOLOGY,
            )
        )

        if "error" in stats:
            logger.warning(f"Entity extraction warning: {stats['error']}")
            return {"status": "warning", "message": stats["error"]}

        entities = stats.get("total_entities_saved", stats.get("total_entities_extracted", 0))
        relationships = stats.get("total_relationships_saved", stats.get("total_relationships_extracted", 0))
        episodes = stats.get("episodes_created", 0)

        print(f"  Result: {entities} entities, {relationships} relationships, {episodes} episodes")

        return {
            "status": "success",
            "entities": entities,
            "relationships": relationships,
            "episodes": episodes,
            "full_stats": stats,
        }

    except Exception as e:
        logger.error(f"Entity extraction stage failed: {e}")
        raise


def run_community_detection_stage() -> Dict[str, Any]:
    """
    Stage 4: Detect communities among extracted entities.

    Returns:
        Statistics dictionary from community detection.
    """
    print_header("[Stage 4: Community Detection]", char="-")
    print(f"  Min size:      {COMMUNITY_MIN_SIZE}")
    print(f"  Resolution:    {COMMUNITY_RESOLUTION}")
    print(f"  Summaries:     {COMMUNITY_GENERATE_SUMMARIES}")
    print()

    # Get graph connection
    graph = get_falkordb_connection()

    # Create detector
    detector = EntityCommunityDetector(
        graph=graph,
        min_community_size=COMMUNITY_MIN_SIZE,
        resolution=COMMUNITY_RESOLUTION,
    )

    try:
        # Run detection
        num_communities = detector.run(
            clear_existing=True,
            generate_summaries=COMMUNITY_GENERATE_SUMMARIES,
        )

        print(f"  Result: {num_communities} communities")

        # Get detailed stats
        detailed_stats = detector.get_stats()

        return {
            "status": "success",
            "communities": num_communities,
            "full_stats": detailed_stats,
        }

    except Exception as e:
        logger.error(f"Community detection stage failed: {e}")
        raise


# =============================================================================
# STATS FUNCTIONS
# =============================================================================

def get_graph_stats() -> Dict[str, Any]:
    """Get current graph statistics."""
    graph = get_falkordb_connection()

    stats = {}

    # Node counts by label
    node_labels = ["WikiPage", "DocumentChunk", "Entity", "Episode", "Community", "WebPage"]
    for label in node_labels:
        try:
            result = graph.query(f"MATCH (n:{label}) RETURN count(n) as count")
            stats[label] = result.result_set[0][0] if result.result_set else 0
        except Exception:
            stats[label] = 0

    # Relationship count
    try:
        result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
        stats["total_relationships"] = result.result_set[0][0] if result.result_set else 0
    except Exception:
        stats["total_relationships"] = 0

    return stats


def print_stats():
    """Print current graph statistics."""
    print_header("GRAPH STATISTICS")
    stats = get_graph_stats()

    for label, count in stats.items():
        print(f"  {label:20s} {count:,}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_pipeline() -> Dict[str, Any]:
    """
    Execute all enabled pipeline stages in order.

    Returns:
        Dictionary with results from all stages.
    """
    results = {
        "stages": {},
        "total_time": 0,
        "status": "success",
    }
    start = time.time()

    try:
        if RUN_STAGE_1_WIKIDATA:
            results["stages"]["wikidata"] = run_wikidata_stage()

        if RUN_STAGE_2A_WIKIPEDIA:
            results["stages"]["wikipedia"] = run_wikipedia_stage()

        if RUN_STAGE_2B_WEBPAGES:
            results["stages"]["webpages"] = run_webpage_stage()

        if RUN_STAGE_3_ENTITIES:
            results["stages"]["entities"] = run_entity_extraction_stage()

        if RUN_STAGE_4_COMMUNITIES:
            results["stages"]["communities"] = run_community_detection_stage()

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        logger.error(f"Pipeline failed: {e}")
        raise

    results["total_time"] = time.time() - start

    # Print summary
    print_header("PIPELINE COMPLETE")
    print(f"Total time:      {results['total_time']:.1f}s")
    print(f"Status:          {results['status']}")

    # Get final graph stats
    stats = get_graph_stats()
    print()
    for label, count in stats.items():
        print(f"{label:20s} {count:,}")

    print("=" * 60)

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_stages(stages_str: str) -> None:
    """Parse stages string and update global flags."""
    global RUN_STAGE_1_WIKIDATA, RUN_STAGE_2A_WIKIPEDIA, RUN_STAGE_2B_WEBPAGES
    global RUN_STAGE_3_ENTITIES, RUN_STAGE_4_COMMUNITIES

    # Disable all stages first
    RUN_STAGE_1_WIKIDATA = False
    RUN_STAGE_2A_WIKIPEDIA = False
    RUN_STAGE_2B_WEBPAGES = False
    RUN_STAGE_3_ENTITIES = False
    RUN_STAGE_4_COMMUNITIES = False

    # Enable requested stages
    stages = [s.strip().lower() for s in stages_str.split(",")]
    for stage in stages:
        if stage == "1":
            RUN_STAGE_1_WIKIDATA = True
        elif stage == "2a":
            RUN_STAGE_2A_WIKIPEDIA = True
        elif stage == "2b":
            RUN_STAGE_2B_WEBPAGES = True
        elif stage == "3":
            RUN_STAGE_3_ENTITIES = True
        elif stage == "4":
            RUN_STAGE_4_COMMUNITIES = True
        else:
            logger.warning(f"Unknown stage: {stage}")


def main():
    """Main entry point with CLI argument parsing."""
    global GRAPH_NAME

    parser = argparse.ArgumentParser(
        description="Unified Knowledge Graph Building Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                        Run full pipeline
  python run_pipeline.py --stages 1,2a          Run Wikidata + Wikipedia only
  python run_pipeline.py --graph test_kg        Override graph name
  python run_pipeline.py --dry-run              Show config without executing
  python run_pipeline.py --stats-only           Show current graph stats only
        """,
    )

    parser.add_argument(
        "--graph",
        help="Override GRAPH_NAME configuration",
    )
    parser.add_argument(
        "--stages",
        help="Comma-separated stages to run: 1,2a,2b,3,4",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without executing",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show current graph statistics only",
    )

    args = parser.parse_args()

    # Override globals if CLI args provided
    if args.graph:
        GRAPH_NAME = args.graph

    if args.stages:
        parse_stages(args.stages)

    # Print configuration
    print_config()

    # Handle special modes
    if args.stats_only:
        print_stats()
        return

    if args.dry_run:
        print("\n[DRY RUN - No execution]")

        if RUN_STAGE_1_WIKIDATA:
            print(f"\nStage 1: Wikidata")
            print(f"  Seeds: {WIKIDATA_SEEDS_FILE}")
            print(f"  Depth: {WIKIDATA_MAX_DEPTH}, Strategy: {WIKIDATA_STRATEGY}")

        if RUN_STAGE_2A_WIKIPEDIA:
            print(f"\nStage 2a: Wikipedia")
            print(f"  Max articles: {WIKIPEDIA_MAX_ARTICLES}")
            print(f"  Chunking: {WIKIPEDIA_CHUNKING_STRATEGY}")

        if RUN_STAGE_2B_WEBPAGES:
            print(f"\nStage 2b: Web Pages")
            divisions_str = "all" if WEBPAGE_DIVISIONS == "all" else WEBPAGE_DIVISIONS
            print(f"  Divisions: {divisions_str}")
            print(f"  Max pages: {WEBPAGE_MAX_PAGES}")

        if RUN_STAGE_3_ENTITIES:
            print(f"\nStage 3: Entity Extraction")
            print(f"  Max chunks: {ENTITY_MAX_CHUNKS}")
            print(f"  Ontology: {ENTITY_ONTOLOGY}")

        if RUN_STAGE_4_COMMUNITIES:
            print(f"\nStage 4: Community Detection")
            print(f"  Min size: {COMMUNITY_MIN_SIZE}")
            print(f"  Resolution: {COMMUNITY_RESOLUTION}")

        return

    # Run the pipeline
    run_pipeline()


if __name__ == "__main__":
    main()
