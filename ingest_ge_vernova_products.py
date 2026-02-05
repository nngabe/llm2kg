#!/usr/bin/env python3
"""
Ingest GE Vernova Product Specification Pages.

This script ingests key product pages that were missing from the knowledge graph
to improve QA benchmark accuracy on technical specification questions.

Target products:
- 9HA Gas Turbine (efficiency, hydrogen blending)
- LM2500XPRESS (fast start capability)
- HVDC Systems (voltage ratings)
- Haliade-X Offshore Wind (power output, rotor diameter)
- H-Class Gas Turbines Overview (combined cycle specs)

Usage:
    python ingest_ge_vernova_products.py [--graph test_kg] [--verify-only]
"""

import argparse
import logging
import sys
from datetime import datetime

from knowledge_graph.webpage_pipeline import WebPagePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Target URLs for product specs missing from KG
PRODUCT_URLS = [
    # 9HA Gas Turbine - efficiency (61-64%), hydrogen (50%)
    "https://www.gevernova.com/gas-power/products/gas-turbines/9ha",
    # LM2500XPRESS - fast start (5 min)
    "https://www.gevernova.com/gas-power/products/gas-turbines/lm2500",
    # HVDC Systems - voltage ratings (800kV DC, 1200kV AC)
    "https://www.gevernova.com/grid-solutions/systems/high-voltage-direct-current-hvdc-systems",
    # Haliade-X - power output (12-18 MW), rotor (220m)
    "https://www.gevernova.com/wind-power/wind-turbines/offshore-wind-turbines",
    # H-Class Overview - combined cycle specs
    "https://www.gevernova.com/gas-power/products/gas-turbines/h-class-gas-turbines",
]


def ingest_product_pages(
    graph_name: str = "test_kg",
    chunking_strategy: str = "raptor",
    content_format: str = "abstract",
) -> dict:
    """
    Ingest GE Vernova product specification pages into the knowledge graph.

    Args:
        graph_name: Target FalkorDB graph name.
        chunking_strategy: Chunking strategy (recursive, semantic, raptor).
        content_format: Content formatting (raw, synthesized, abstract, llm).

    Returns:
        Statistics dictionary with ingestion results.
    """
    print("\n" + "=" * 60)
    print("GE VERNOVA PRODUCT SPECS INGESTION")
    print("=" * 60)
    print(f"Graph:           {graph_name}")
    print(f"Chunking:        {chunking_strategy}")
    print(f"Content Format:  {content_format}")
    print(f"URLs to ingest:  {len(PRODUCT_URLS)}")
    print("=" * 60)

    # Initialize pipeline
    pipeline = WebPagePipeline(
        graph_name=graph_name,
        chunking_strategy=chunking_strategy,
        content_format=content_format,
    )

    # Initialize schema
    pipeline.init_schema()

    # Process each product page
    stats = {
        "start_time": datetime.now().isoformat(),
        "graph_name": graph_name,
        "pages_processed": 0,
        "pages_failed": 0,
        "total_chunks": 0,
        "pages": [],
    }

    for i, url in enumerate(PRODUCT_URLS, 1):
        print(f"\n[{i}/{len(PRODUCT_URLS)}] Processing: {url}")

        try:
            webpage, chunks_saved = pipeline.process_webpage(url, with_embeddings=True)

            if webpage and chunks_saved > 0:
                stats["pages_processed"] += 1
                stats["total_chunks"] += chunks_saved
                stats["pages"].append({
                    "url": url,
                    "title": webpage.title,
                    "chunks": chunks_saved,
                    "status": "success",
                })
                print(f"    Title: {webpage.title}")
                print(f"    Chunks created: {chunks_saved}")
            else:
                stats["pages_failed"] += 1
                stats["pages"].append({
                    "url": url,
                    "status": "failed",
                    "reason": "No content or chunks",
                })
                print(f"    FAILED: No content or chunks created")

        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            stats["pages_failed"] += 1
            stats["pages"].append({
                "url": url,
                "status": "error",
                "error": str(e),
            })
            print(f"    ERROR: {e}")

    stats["end_time"] = datetime.now().isoformat()

    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Pages processed: {stats['pages_processed']}")
    print(f"Pages failed:    {stats['pages_failed']}")
    print(f"Total chunks:    {stats['total_chunks']}")
    print("=" * 60)

    return stats


def verify_ingestion(graph_name: str = "test_kg") -> dict:
    """
    Verify ingestion by querying for new DocumentChunks.

    Args:
        graph_name: FalkorDB graph name to query.

    Returns:
        Verification statistics.
    """
    from falkordb import FalkorDB
    import os

    FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
    FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

    client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    graph = client.select_graph(graph_name)

    print("\n" + "=" * 60)
    print("VERIFICATION QUERIES")
    print("=" * 60)

    verification = {}

    # Count DocumentChunks from GE Vernova URLs
    query = """
    MATCH (d:DocumentChunk)
    WHERE d.source_url CONTAINS 'gevernova.com/gas-power'
       OR d.source_url CONTAINS 'gevernova.com/grid-solutions'
       OR d.source_url CONTAINS 'gevernova.com/wind-power'
    RETURN d.source_url, count(d) as chunks
    ORDER BY chunks DESC
    """

    try:
        result = graph.query(query)
        print("\nDocumentChunks by source URL:")
        print("-" * 50)
        total_chunks = 0
        urls_found = []
        for row in result.result_set:
            url, count = row[0], row[1]
            print(f"  {url}: {count} chunks")
            total_chunks += count
            urls_found.append({"url": url, "chunks": count})

        verification["chunks_by_url"] = urls_found
        verification["total_chunks"] = total_chunks
        print(f"\nTotal chunks from GE Vernova: {total_chunks}")

    except Exception as e:
        logger.error(f"Verification query failed: {e}")
        verification["error"] = str(e)

    # Count WebPage nodes
    try:
        result = graph.query("""
            MATCH (w:WebPage)
            WHERE w.url CONTAINS 'gevernova.com'
            RETURN count(w) as count
        """)
        webpage_count = result.result_set[0][0] if result.result_set else 0
        verification["webpage_nodes"] = webpage_count
        print(f"WebPage nodes: {webpage_count}")
    except Exception as e:
        logger.error(f"WebPage count failed: {e}")

    # Check for specific product pages
    product_checks = [
        ("9HA", "9ha"),
        ("LM2500", "lm2500"),
        ("HVDC", "hvdc"),
        ("Offshore Wind", "offshore-wind"),
        ("H-Class", "h-class"),
    ]

    print("\nProduct page checks:")
    print("-" * 50)
    verification["product_checks"] = {}

    for name, url_fragment in product_checks:
        try:
            result = graph.query(f"""
                MATCH (w:WebPage)-[:HAS_CHUNK]->(d:DocumentChunk)
                WHERE w.url CONTAINS '{url_fragment}'
                RETURN w.url, count(d) as chunks
            """)
            if result.result_set:
                url, chunks = result.result_set[0]
                print(f"  {name}: {chunks} chunks ({url})")
                verification["product_checks"][name] = {
                    "found": True,
                    "chunks": chunks,
                    "url": url,
                }
            else:
                print(f"  {name}: NOT FOUND")
                verification["product_checks"][name] = {"found": False}
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            verification["product_checks"][name] = {"error": str(e)}

    print("=" * 60)

    return verification


def main():
    parser = argparse.ArgumentParser(
        description="Ingest GE Vernova product specification pages"
    )
    parser.add_argument(
        "--graph",
        default="test_kg",
        help="FalkorDB graph name (default: test_kg)",
    )
    parser.add_argument(
        "--chunking-strategy",
        default="raptor",
        choices=["recursive", "semantic", "raptor"],
        help="Chunking strategy (default: raptor)",
    )
    parser.add_argument(
        "--content-format",
        default="abstract",
        choices=["raw", "synthesized", "abstract", "llm"],
        help="Content formatting strategy (default: abstract)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification queries, skip ingestion",
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_ingestion(args.graph)
        return 0

    # Run ingestion
    stats = ingest_product_pages(
        graph_name=args.graph,
        chunking_strategy=args.chunking_strategy,
        content_format=args.content_format,
    )

    # Verify results
    verify_ingestion(args.graph)

    return 0 if stats["pages_processed"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
