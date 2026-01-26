#!/usr/bin/env python3
"""
Benchmark Script for Neo4j vs FalkorDB Backend Comparison.

This script benchmarks the Wikidata Knowledge Graph Builder across both
Neo4j and FalkorDB backends, measuring:
- Write performance (entity and relationship creation)
- Read performance (stats retrieval)
- Search performance (name-based queries)
- Query performance (traversal queries)

Usage:
    python benchmarks/wikidata_backend_benchmark.py --nodes 1000
    python benchmarks/wikidata_backend_benchmark.py --nodes 5000 --runs 3
    python benchmarks/wikidata_backend_benchmark.py --output results.json
"""

import os
import sys
import time
import json
import logging
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, '/app')

from wikidata_kg_builder import (
    WikidataEntity,
    WikidataRelationship,
    WikidataSPARQLClient,
    WikiPageLoader,
    FalkorDBPageLoader,
    WikidataKGBuilder,
    PROPERTY_TO_REL_TYPE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    """Timing metrics for a single benchmark operation."""
    operation: str
    backend: str
    times: List[float] = field(default_factory=list)

    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0

    @property
    def avg_time(self) -> float:
        return statistics.mean(self.times) if self.times else 0

    @property
    def std_dev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0


@dataclass
class BackendBenchmarkResult:
    """Complete benchmark results for a single backend."""
    backend: str
    timestamp: str
    success: bool
    error: Optional[str] = None

    # Data metrics
    total_nodes: int = 0
    total_relationships: int = 0

    # Write metrics
    entity_write_time: float = 0.0
    relationship_write_time: float = 0.0
    total_write_time: float = 0.0
    entities_per_second: float = 0.0
    relationships_per_second: float = 0.0

    # Read metrics
    stats_read_times: List[float] = field(default_factory=list)
    avg_stats_read_time: float = 0.0

    # Search metrics
    search_times: List[float] = field(default_factory=list)
    avg_search_time: float = 0.0

    # Traversal query metrics
    traversal_times: List[float] = field(default_factory=list)
    avg_traversal_time: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report comparing backends."""
    timestamp: str
    config: Dict[str, Any]
    wikidata_fetch_time: float
    entities_fetched: int
    relationships_fetched: int
    results: Dict[str, BackendBenchmarkResult] = field(default_factory=dict)


class WikidataBackendBenchmark:
    """Benchmarks Neo4j and FalkorDB backends for the Wikidata KG Builder."""

    def __init__(
        self,
        target_nodes: int = 1000,
        root_qid: str = "Q1101",
        max_depth: int = 4,
        max_per_level: int = 500,
        num_runs: int = 3,
        search_queries: Optional[List[str]] = None,
    ):
        """
        Initialize the benchmark.

        Args:
            target_nodes: Target number of nodes to create.
            root_qid: Root Wikidata QID for building the graph.
            max_depth: Maximum BFS depth.
            max_per_level: Maximum entities per level.
            num_runs: Number of runs for averaging timing measurements.
            search_queries: List of search queries to test.
        """
        self.target_nodes = target_nodes
        self.root_qid = root_qid
        self.max_depth = max_depth
        self.max_per_level = max_per_level
        self.num_runs = num_runs
        self.search_queries = search_queries or [
            "machine learning",
            "computer",
            "software",
            "network",
            "data",
            "algorithm",
            "artificial intelligence",
            "programming",
        ]

        self.sparql_client = WikidataSPARQLClient()
        self.entities: List[WikidataEntity] = []
        self.relationships: List[WikidataRelationship] = []

    def fetch_wikidata(self) -> float:
        """
        Fetch data from Wikidata.

        Returns:
            Time taken in seconds.
        """
        logger.info("Fetching data from Wikidata...")
        start = time.time()

        # Use dry run builder to fetch data
        builder = WikidataKGBuilder(sparql_client=self.sparql_client, dry_run=True)
        builder.build_from_root(
            root_qid=self.root_qid,
            max_depth=self.max_depth,
            max_entities_per_level=self.max_per_level,
        )
        builder.close()

        # Re-fetch to get actual entities and relationships
        # This is a workaround - ideally we'd modify the builder to expose the data
        visited = set()
        self.entities = []
        self.relationships = []

        # Fetch root
        root = self.sparql_client.fetch_entity(self.root_qid)
        if root:
            self.entities.append(root)
            visited.add(self.root_qid)

        # BFS to collect data
        from collections import deque
        queue = deque([[self.root_qid]])
        depth = 0

        while queue and depth < self.max_depth:
            current_qids = queue.popleft()
            result = self.sparql_client.fetch_children(
                parent_qids=current_qids,
                limit=self.max_per_level,
            )

            new_entities = [e for e in result.entities if e.qid not in visited]
            for e in new_entities:
                visited.add(e.qid)

            self.entities.extend(new_entities)
            self.relationships.extend(result.relationships)

            if new_entities and depth + 1 < self.max_depth:
                queue.append([e.qid for e in new_entities])
            depth += 1

        elapsed = time.time() - start
        logger.info(f"Fetched {len(self.entities)} entities, {len(self.relationships)} relationships in {elapsed:.2f}s")
        return elapsed

    def _clear_backend(self, loader, backend: str):
        """Clear all data from a backend."""
        try:
            if backend == "falkordb":
                loader.clear_graph()
            else:
                with loader.driver.session() as session:
                    session.run("MATCH (w:WikiPage) DETACH DELETE w")
            logger.debug(f"Cleared {backend} data")
        except Exception as e:
            logger.warning(f"Failed to clear {backend}: {e}")

    def benchmark_backend(self, backend: str) -> BackendBenchmarkResult:
        """
        Benchmark a single backend.

        Args:
            backend: Backend name ("neo4j" or "falkordb").

        Returns:
            BackendBenchmarkResult with all metrics.
        """
        result = BackendBenchmarkResult(
            backend=backend,
            timestamp=datetime.now().isoformat(),
            success=False,
        )

        try:
            # Initialize loader
            if backend == "neo4j":
                loader = WikiPageLoader()
            else:
                loader = FalkorDBPageLoader()

            # Clear existing data
            self._clear_backend(loader, backend)

            # Initialize schema
            loader.init_schema()

            # Write entities
            logger.info(f"[{backend}] Writing {len(self.entities)} entities...")
            entity_start = time.time()
            loader.batch_create_entities(self.entities)
            result.entity_write_time = time.time() - entity_start

            # Write relationships
            logger.info(f"[{backend}] Writing {len(self.relationships)} relationships...")
            rel_start = time.time()
            loader.batch_create_relationships(self.relationships)
            result.relationship_write_time = time.time() - rel_start

            result.total_write_time = result.entity_write_time + result.relationship_write_time
            result.entities_per_second = len(self.entities) / result.entity_write_time if result.entity_write_time > 0 else 0
            result.relationships_per_second = len(self.relationships) / result.relationship_write_time if result.relationship_write_time > 0 else 0

            # Get final stats
            stats = loader.get_stats()
            result.total_nodes = stats.get("total_nodes", 0)
            result.total_relationships = stats.get("total_relationships", 0)

            # Benchmark read operations (stats)
            logger.info(f"[{backend}] Benchmarking read operations...")
            for _ in range(self.num_runs):
                start = time.time()
                loader.get_stats()
                result.stats_read_times.append(time.time() - start)
            result.avg_stats_read_time = statistics.mean(result.stats_read_times)

            # Benchmark search operations
            logger.info(f"[{backend}] Benchmarking search operations...")
            for query in self.search_queries:
                start = time.time()
                loader.search_by_name(query, limit=10)
                result.search_times.append(time.time() - start)
            result.avg_search_time = statistics.mean(result.search_times)

            # Benchmark traversal queries
            logger.info(f"[{backend}] Benchmarking traversal queries...")
            result.traversal_times = self._benchmark_traversal(loader, backend)
            result.avg_traversal_time = statistics.mean(result.traversal_times) if result.traversal_times else 0

            loader.close()
            result.success = True

        except Exception as e:
            logger.error(f"[{backend}] Benchmark failed: {e}")
            result.error = str(e)

        return result

    def _benchmark_traversal(self, loader, backend: str) -> List[float]:
        """Benchmark traversal queries on the backend."""
        times = []

        # Sample queries for traversal
        queries = [
            # 1-hop traversal
            "MATCH (w:WikiPage)-[:SUBCLASS_OF]->(parent:WikiPage) RETURN w.name, parent.name LIMIT 100",
            # 2-hop traversal
            "MATCH (w:WikiPage)-[:SUBCLASS_OF*1..2]->(ancestor:WikiPage) RETURN w.name, count(ancestor) as ancestors LIMIT 50",
            # Relationship count
            "MATCH (w:WikiPage) RETURN w.name, size((w)-[:SUBCLASS_OF]->()) as children LIMIT 50",
        ]

        for query in queries:
            try:
                start = time.time()
                if backend == "neo4j":
                    with loader.driver.session() as session:
                        list(session.run(query))
                else:
                    loader.graph.query(query)
                times.append(time.time() - start)
            except Exception as e:
                logger.warning(f"[{backend}] Traversal query failed: {e}")
                times.append(float('inf'))

        return times

    def run(self) -> BenchmarkReport:
        """
        Run the complete benchmark.

        Returns:
            BenchmarkReport with all results.
        """
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            config={
                "target_nodes": self.target_nodes,
                "root_qid": self.root_qid,
                "max_depth": self.max_depth,
                "max_per_level": self.max_per_level,
                "num_runs": self.num_runs,
                "search_queries": self.search_queries,
            },
            wikidata_fetch_time=0,
            entities_fetched=0,
            relationships_fetched=0,
        )

        # Fetch Wikidata
        report.wikidata_fetch_time = self.fetch_wikidata()
        report.entities_fetched = len(self.entities)
        report.relationships_fetched = len(self.relationships)

        # Benchmark each backend
        for backend in ["neo4j", "falkordb"]:
            logger.info(f"\n{'='*50}")
            logger.info(f"Benchmarking {backend.upper()}")
            logger.info(f"{'='*50}")
            report.results[backend] = self.benchmark_backend(backend)

        return report

    def print_report(self, report: BenchmarkReport):
        """Print a formatted benchmark report."""
        print("\n" + "=" * 80)
        print("WIKIDATA KNOWLEDGE GRAPH BUILDER - BACKEND BENCHMARK REPORT")
        print("=" * 80)
        print(f"\nTimestamp: {report.timestamp}")
        print(f"\nConfiguration:")
        print(f"  Root QID:        {report.config['root_qid']}")
        print(f"  Max Depth:       {report.config['max_depth']}")
        print(f"  Max Per Level:   {report.config['max_per_level']}")
        print(f"  Num Runs:        {report.config['num_runs']}")
        print(f"\nWikidata Fetch:")
        print(f"  Time:            {report.wikidata_fetch_time:.2f}s")
        print(f"  Entities:        {report.entities_fetched}")
        print(f"  Relationships:   {report.relationships_fetched}")

        # Comparison table
        print("\n" + "-" * 80)
        print(f"{'Metric':<35} {'Neo4j':>18} {'FalkorDB':>18} {'Winner':>10}")
        print("-" * 80)

        neo = report.results.get("neo4j")
        falk = report.results.get("falkordb")

        def row(name: str, neo_val: Any, falk_val: Any, lower_better: bool = True, unit: str = ""):
            if neo is None or not neo.success:
                neo_str = "FAILED"
                winner = "FalkorDB" if falk and falk.success else "N/A"
            elif falk is None or not falk.success:
                neo_str = f"{neo_val:.4f}{unit}" if isinstance(neo_val, float) else str(neo_val)
                falk_str = "FAILED"
                winner = "Neo4j"
            else:
                neo_str = f"{neo_val:.4f}{unit}" if isinstance(neo_val, float) else str(neo_val)
                falk_str = f"{falk_val:.4f}{unit}" if isinstance(falk_val, float) else str(falk_val)
                if lower_better:
                    winner = "Neo4j" if neo_val <= falk_val else "FalkorDB"
                else:
                    winner = "Neo4j" if neo_val >= falk_val else "FalkorDB"

            falk_str = "FAILED" if falk is None or not falk.success else f"{falk_val:.4f}{unit}" if isinstance(falk_val, float) else str(falk_val)
            print(f"{name:<35} {neo_str:>18} {falk_str:>18} {winner:>10}")

        # Write performance
        print("\n-- Write Performance --")
        row("Entity Write (s)", neo.entity_write_time if neo else 0, falk.entity_write_time if falk else 0, True)
        row("Relationship Write (s)", neo.relationship_write_time if neo else 0, falk.relationship_write_time if falk else 0, True)
        row("Total Write (s)", neo.total_write_time if neo else 0, falk.total_write_time if falk else 0, True)
        row("Entities/second", neo.entities_per_second if neo else 0, falk.entities_per_second if falk else 0, False)
        row("Relationships/second", neo.relationships_per_second if neo else 0, falk.relationships_per_second if falk else 0, False)

        # Read performance
        print("\n-- Read Performance --")
        row("Avg Stats Read (ms)", (neo.avg_stats_read_time * 1000) if neo else 0, (falk.avg_stats_read_time * 1000) if falk else 0, True)
        row("Avg Search (ms)", (neo.avg_search_time * 1000) if neo else 0, (falk.avg_search_time * 1000) if falk else 0, True)
        row("Avg Traversal (ms)", (neo.avg_traversal_time * 1000) if neo else 0, (falk.avg_traversal_time * 1000) if falk else 0, True)

        # Data verification
        print("\n-- Data Verification --")
        row("Total Nodes", neo.total_nodes if neo else 0, falk.total_nodes if falk else 0, False)
        row("Total Relationships", neo.total_relationships if neo else 0, falk.total_relationships if falk else 0, False)

        # Errors
        if neo and neo.error:
            print(f"\nNeo4j Error: {neo.error}")
        if falk and falk.error:
            print(f"\nFalkorDB Error: {falk.error}")

        print("\n" + "=" * 80)


def main():
    """CLI entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark Neo4j vs FalkorDB for Wikidata KG Builder"
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=1000,
        help="Target number of nodes (default: 1000)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="Q1101",
        help="Root Wikidata QID (default: Q1101 'Technology')"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum BFS depth (default: 4)"
    )
    parser.add_argument(
        "--max-per-level",
        type=int,
        default=500,
        help="Maximum entities per level (default: 500)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for timing measurements (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    benchmark = WikidataBackendBenchmark(
        target_nodes=args.nodes,
        root_qid=args.root,
        max_depth=args.max_depth,
        max_per_level=args.max_per_level,
        num_runs=args.runs,
    )

    report = benchmark.run()
    benchmark.print_report(report)

    # Save to file if requested
    if args.output:
        # Convert dataclasses to dicts
        report_dict = {
            "timestamp": report.timestamp,
            "config": report.config,
            "wikidata_fetch_time": report.wikidata_fetch_time,
            "entities_fetched": report.entities_fetched,
            "relationships_fetched": report.relationships_fetched,
            "results": {
                k: asdict(v) for k, v in report.results.items()
            },
        }

        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
