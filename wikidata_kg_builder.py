#!/usr/bin/env python3
"""
Wikipedia Knowledge Graph Builder from Wikidata.

Builds a lightweight Knowledge Graph from Wikidata's Technology tree,
storing structure/relationships in Neo4j while using WikipediaLoader
at runtime for full content.

Usage:
    python wikidata_kg_builder.py --max-depth 3
    python wikidata_kg_builder.py --root Q21198 --max-depth 4
    python wikidata_kg_builder.py --dry-run --max-depth 2
    python wikidata_kg_builder.py --with-embeddings
"""

import os
import time
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque
from enum import Enum
import heapq

import requests
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# FalkorDB Configuration
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH_NAME = os.getenv("FALKORDB_GRAPH_NAME", "wikidata")

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Property IDs for relationships
PROPERTY_SUBCLASS_OF = "P279"
PROPERTY_INSTANCE_OF = "P31"
PROPERTY_PART_OF = "P361"

# Relationship type mapping
PROPERTY_TO_REL_TYPE = {
    PROPERTY_SUBCLASS_OF: "SUBCLASS_OF",
    PROPERTY_INSTANCE_OF: "INSTANCE_OF",
    PROPERTY_PART_OF: "PART_OF",
}


# --- DATA MODELS ---

@dataclass
class WikidataEntity:
    """Represents a Wikidata entity."""
    qid: str                      # "Q1101"
    name: str                     # "Technology"
    wikipedia_url: Optional[str] = None
    description: Optional[str] = None

    def __hash__(self):
        return hash(self.qid)

    def __eq__(self, other):
        if isinstance(other, WikidataEntity):
            return self.qid == other.qid
        return False


@dataclass
class WikidataRelationship:
    """Represents a relationship between Wikidata entities."""
    source_qid: str
    target_qid: str
    property_id: str              # P279, P31, P361
    property_label: str           # subclass_of, instance_of, part_of

    def __hash__(self):
        return hash((self.source_qid, self.target_qid, self.property_id))

    def __eq__(self, other):
        if isinstance(other, WikidataRelationship):
            return (self.source_qid == other.source_qid and
                    self.target_qid == other.target_qid and
                    self.property_id == other.property_id)
        return False


@dataclass
class WikidataFetchResult:
    """Result from a Wikidata SPARQL fetch."""
    entities: List[WikidataEntity] = field(default_factory=list)
    relationships: List[WikidataRelationship] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# --- EXPLORATION STRATEGY DATA STRUCTURES ---

class ExplorationStrategy(str, Enum):
    """Strategy for graph exploration."""
    BFS = "bfs"                # Original behavior
    ROOT_FIRST = "root_first"  # Two-phase strategy: roots first, then leaves


@dataclass
class NodeMetrics:
    """Metrics for classifying a node as root-like or leaf-like."""
    qid: str
    name: str
    parent_count: int = 0    # P279 parents (this node is a subclass of...)
    child_count: int = 0     # P279 children (things that are subclasses of this)
    instance_count: int = 0  # P31 instances pointing to this (things that are instances of this)

    @property
    def abstraction_score(self) -> float:
        """
        Compute abstraction score.
        Higher = more abstract/root-like.
        Formula: child_count - (0.5 * parent_count) + (10 if has_instances else 0)
        """
        instance_bonus = 10.0 if self.instance_count > 0 else 0.0
        return self.child_count - (0.5 * self.parent_count) + instance_bonus

    def is_root_like(self, threshold: float = 5.0) -> bool:
        """
        Determine if this node is root-like (abstract concept).

        Root indicators:
        - Has instances pointing to it (it's a class others instantiate)
        - Few parents and at least one child
        - High abstraction score
        """
        # If others are instances of this, it's definitely a class/root
        if self.instance_count > 0:
            return True
        # Few parents and at least some children = top-level concept
        if self.parent_count <= 2 and self.child_count >= 1:
            return True
        # Otherwise use the score threshold
        return self.abstraction_score >= threshold


@dataclass
class ScoredNode:
    """A node with a connectivity score for prioritization."""
    qid: str
    name: str
    score: float  # Connectivity to roots (higher = better connected)
    connected_roots: Set[str] = field(default_factory=set)

    def __lt__(self, other: "ScoredNode") -> bool:
        # For max-heap, invert comparison (higher score = higher priority)
        return self.score > other.score


@dataclass
class ExplorationState:
    """State for root-first exploration."""
    phase: int = 1  # 1 = collecting roots, 2 = collecting leaves
    root_qids: Set[str] = field(default_factory=set)
    leaf_candidates: List[ScoredNode] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)


# --- SPARQL CLIENT ---

class WikidataSPARQLClient:
    """Client for querying Wikidata SPARQL endpoint."""

    def __init__(
        self,
        endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize the SPARQL client.

        Args:
            endpoint: Wikidata SPARQL endpoint URL.
            rate_limit_delay: Delay between requests in seconds.
            max_retries: Maximum number of retries for failed requests.
            timeout: Request timeout in seconds.
        """
        self.endpoint = endpoint
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Wait to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def execute_query(self, sparql: str) -> Dict[str, Any]:
        """
        Execute a SPARQL query against Wikidata.

        Args:
            sparql: SPARQL query string.

        Returns:
            Query results as a dictionary.
        """
        self._wait_for_rate_limit()

        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "WikidataKGBuilder/1.0 (Python; Neo4j integration)",
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.endpoint,
                    params={"query": sparql},
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning(f"SPARQL request attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return {}

    def _build_combined_query(
        self,
        parent_qids: List[str],
        include_subclasses: bool = True,
        include_instances: bool = True,
        include_parts: bool = True,
        limit: int = 1000,
    ) -> str:
        """
        Build a combined SPARQL query for fetching related entities.

        Args:
            parent_qids: List of parent QIDs to fetch children for.
            include_subclasses: Include subclass_of relationships (P279).
            include_instances: Include instance_of relationships (P31).
            include_parts: Include part_of relationships (P361).
            limit: Maximum number of results.

        Returns:
            SPARQL query string.
        """
        # Build VALUES clause for parent QIDs
        values_str = " ".join(f"wd:{qid}" for qid in parent_qids)

        # Build UNION clauses for relationship types
        union_clauses = []
        if include_subclasses:
            union_clauses.append('''
    {
        ?item wdt:P279 ?parent .
        BIND("subclass_of" AS ?relation)
        BIND("P279" AS ?propertyId)
    }''')
        if include_instances:
            union_clauses.append('''
    {
        ?item wdt:P31 ?parent .
        BIND("instance_of" AS ?relation)
        BIND("P31" AS ?propertyId)
    }''')
        if include_parts:
            union_clauses.append('''
    {
        ?item wdt:P361 ?parent .
        BIND("part_of" AS ?relation)
        BIND("P361" AS ?propertyId)
    }''')

        if not union_clauses:
            raise ValueError("At least one relationship type must be included")

        union_block = " UNION ".join(union_clauses)

        query = f'''
SELECT DISTINCT ?item ?itemLabel ?itemDescription ?sitelink ?parentQID ?relation ?propertyId WHERE {{
    VALUES ?parent {{ {values_str} }}

    {union_block}

    BIND(REPLACE(STR(?parent), ".*/(Q\\\\d+)$", "$1") AS ?parentQID)

    OPTIONAL {{
        ?sitelink schema:about ?item ;
                  schema:isPartOf <https://en.wikipedia.org/> .
    }}

    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT {limit}
'''
        return query

    def fetch_children(
        self,
        parent_qids: List[str],
        limit: int = 1000,
        include_subclasses: bool = True,
        include_instances: bool = True,
        include_parts: bool = True,
    ) -> WikidataFetchResult:
        """
        Fetch all children (subclasses, instances, parts) of given parents.

        Args:
            parent_qids: List of parent QIDs.
            limit: Maximum number of results.
            include_subclasses: Include subclass_of relationships.
            include_instances: Include instance_of relationships.
            include_parts: Include part_of relationships.

        Returns:
            WikidataFetchResult with entities and relationships.
        """
        result = WikidataFetchResult()

        if not parent_qids:
            return result

        try:
            query = self._build_combined_query(
                parent_qids,
                include_subclasses=include_subclasses,
                include_instances=include_instances,
                include_parts=include_parts,
                limit=limit,
            )
            data = self.execute_query(query)

            seen_entities: Set[str] = set()
            seen_rels: Set[Tuple[str, str, str]] = set()

            for binding in data.get("results", {}).get("bindings", []):
                # Extract QID from item URI
                item_uri = binding.get("item", {}).get("value", "")
                qid_match = item_uri.split("/")[-1] if item_uri else ""
                if not qid_match.startswith("Q"):
                    continue
                qid = qid_match

                # Extract entity data
                name = binding.get("itemLabel", {}).get("value", qid)
                description = binding.get("itemDescription", {}).get("value")
                sitelink = binding.get("sitelink", {}).get("value")

                # Add entity if not seen
                if qid not in seen_entities:
                    entity = WikidataEntity(
                        qid=qid,
                        name=name,
                        wikipedia_url=sitelink,
                        description=description,
                    )
                    result.entities.append(entity)
                    seen_entities.add(qid)

                # Extract relationship
                parent_qid = binding.get("parentQID", {}).get("value", "")
                relation = binding.get("relation", {}).get("value", "")
                property_id = binding.get("propertyId", {}).get("value", "")

                if parent_qid and relation:
                    rel_key = (qid, parent_qid, property_id)
                    if rel_key not in seen_rels:
                        rel = WikidataRelationship(
                            source_qid=qid,
                            target_qid=parent_qid,
                            property_id=property_id,
                            property_label=relation,
                        )
                        result.relationships.append(rel)
                        seen_rels.add(rel_key)

            logger.info(f"Fetched {len(result.entities)} entities, {len(result.relationships)} relationships from {len(parent_qids)} parents")

        except Exception as e:
            logger.error(f"Error fetching children: {e}")
            result.errors.append(str(e))

        return result

    def fetch_entity(self, qid: str) -> Optional[WikidataEntity]:
        """
        Fetch a single entity by QID.

        Args:
            qid: Wikidata QID (e.g., "Q1101").

        Returns:
            WikidataEntity or None if not found.
        """
        query = f'''
SELECT ?itemLabel ?itemDescription ?sitelink WHERE {{
    BIND(wd:{qid} AS ?item)

    OPTIONAL {{
        ?sitelink schema:about ?item ;
                  schema:isPartOf <https://en.wikipedia.org/> .
    }}

    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT 1
'''
        try:
            data = self.execute_query(query)
            bindings = data.get("results", {}).get("bindings", [])

            if not bindings:
                return None

            binding = bindings[0]
            name = binding.get("itemLabel", {}).get("value", qid)
            description = binding.get("itemDescription", {}).get("value")
            sitelink = binding.get("sitelink", {}).get("value")

            return WikidataEntity(
                qid=qid,
                name=name,
                wikipedia_url=sitelink,
                description=description,
            )

        except Exception as e:
            logger.error(f"Error fetching entity {qid}: {e}")
            return None

    def fetch_node_metrics(self, qids: List[str]) -> Dict[str, NodeMetrics]:
        """
        Fetch parent/child/instance counts for classifying nodes as root-like or leaf-like.

        Args:
            qids: List of Wikidata QIDs to fetch metrics for.

        Returns:
            Dictionary mapping QID to NodeMetrics.
        """
        if not qids:
            return {}

        # Build VALUES clause for QIDs
        values_str = " ".join(f"wd:{qid}" for qid in qids)

        # Query counts parents (P279 targets), children (P279 sources), and instances (P31 sources)
        query = f'''
SELECT ?item ?itemLabel
       (COUNT(DISTINCT ?parent) AS ?parentCount)
       (COUNT(DISTINCT ?child) AS ?childCount)
       (COUNT(DISTINCT ?instance) AS ?instanceCount)
WHERE {{
    VALUES ?item {{ {values_str} }}

    # Count P279 parents (this item is subclass of parent)
    OPTIONAL {{ ?item wdt:P279 ?parent . }}

    # Count P279 children (child is subclass of this item)
    OPTIONAL {{ ?child wdt:P279 ?item . }}

    # Count P31 instances (instance is an instance of this item)
    OPTIONAL {{ ?instance wdt:P31 ?item . }}

    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
GROUP BY ?item ?itemLabel
'''
        try:
            data = self.execute_query(query)
            results: Dict[str, NodeMetrics] = {}

            for binding in data.get("results", {}).get("bindings", []):
                # Extract QID from item URI
                item_uri = binding.get("item", {}).get("value", "")
                qid = item_uri.split("/")[-1] if item_uri else ""
                if not qid.startswith("Q"):
                    continue

                name = binding.get("itemLabel", {}).get("value", qid)
                parent_count = int(binding.get("parentCount", {}).get("value", "0"))
                child_count = int(binding.get("childCount", {}).get("value", "0"))
                instance_count = int(binding.get("instanceCount", {}).get("value", "0"))

                results[qid] = NodeMetrics(
                    qid=qid,
                    name=name,
                    parent_count=parent_count,
                    child_count=child_count,
                    instance_count=instance_count,
                )

            logger.debug(f"Fetched metrics for {len(results)}/{len(qids)} QIDs")
            return results

        except Exception as e:
            logger.error(f"Error fetching node metrics: {e}")
            return {}


# --- NEO4J LOADER ---

class WikiPageLoader:
    """Loads WikiPage nodes and relationships into Neo4j."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        batch_size: int = 100,
    ):
        """
        Initialize the WikiPage loader.

        Args:
            uri: Neo4j connection URI.
            user: Neo4j username.
            password: Neo4j password.
            batch_size: Batch size for Neo4j operations.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size
        self._embedding_model = None

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from langchain_ollama import OllamaEmbeddings
            self._embedding_model = OllamaEmbeddings(
                model="qwen3-embedding:8b",
                base_url=OLLAMA_HOST,
                num_ctx=4096,
            )
        return self._embedding_model

    def init_schema(self):
        """
        Initialize Neo4j schema for WikiPage nodes.

        Creates:
        - Unique constraint on wikidata_id
        - Text index on name for search
        """
        with self.driver.session() as session:
            # Create unique constraint on wikidata_id
            session.run("""
                CREATE CONSTRAINT wikipage_qid_unique IF NOT EXISTS
                FOR (w:WikiPage) REQUIRE w.wikidata_id IS UNIQUE
            """)

            # Create text index on name for search
            session.run("""
                CREATE TEXT INDEX wikipage_name_idx IF NOT EXISTS
                FOR (w:WikiPage) ON (w.name)
            """)

            logger.info("Neo4j schema initialized for WikiPage nodes")

    def init_vector_index(self, dimensions: int = 4096):
        """
        Initialize vector index for WikiPage embeddings (optional).

        Args:
            dimensions: Embedding dimensions (default: 4096 for qwen3-embedding).
        """
        with self.driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX wikipage_embeddings IF NOT EXISTS
                FOR (w:WikiPage) ON (w.embedding)
                OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}
            """)

            logger.info(f"WikiPage vector index created (dimensions={dimensions})")

    def batch_create_entities(
        self,
        entities: List[WikidataEntity],
        with_embeddings: bool = False,
    ) -> int:
        """
        Create WikiPage nodes in batches.

        Args:
            entities: List of WikidataEntity objects.
            with_embeddings: Whether to compute and store embeddings.

        Returns:
            Number of entities created/merged.
        """
        if not entities:
            return 0

        total_created = 0

        # Process in batches
        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i + self.batch_size]

            # Prepare batch data
            batch_data = []
            for entity in batch:
                data = {
                    "wikidata_id": entity.qid,
                    "name": entity.name,
                    "wikipedia_url": entity.wikipedia_url,
                    "description": entity.description,
                }

                if with_embeddings:
                    try:
                        embedding_model = self._get_embedding_model()
                        embedding = embedding_model.embed_query(entity.name)
                        data["embedding"] = embedding
                    except Exception as e:
                        logger.warning(f"Failed to embed {entity.name}: {e}")

                batch_data.append(data)

            # Execute batch merge
            with self.driver.session() as session:
                # Main merge query without embedding filtering
                result = session.run("""
                    UNWIND $batch AS entity
                    MERGE (w:WikiPage {wikidata_id: entity.wikidata_id})
                    ON CREATE SET
                        w.name = entity.name,
                        w.wikipedia_url = entity.wikipedia_url,
                        w.description = entity.description,
                        w.created_at = datetime()
                    ON MATCH SET
                        w.name = entity.name,
                        w.wikipedia_url = COALESCE(entity.wikipedia_url, w.wikipedia_url),
                        w.description = COALESCE(entity.description, w.description)
                    RETURN count(w) as count
                """, batch=batch_data)

                record = result.single()
                batch_count = record["count"] if record else len(batch)
                total_created += batch_count

                # Separate query for embeddings if any entities have them
                if with_embeddings:
                    embed_batch = [d for d in batch_data if d.get("embedding") is not None]
                    if embed_batch:
                        session.run("""
                            UNWIND $batch AS entity
                            MATCH (w:WikiPage {wikidata_id: entity.wikidata_id})
                            SET w.embedding = entity.embedding
                        """, batch=embed_batch)

            logger.debug(f"Batch {i // self.batch_size + 1}: {batch_count} entities")

        logger.info(f"Created/merged {total_created} WikiPage nodes")
        return total_created

    def batch_create_relationships(
        self,
        relationships: List[WikidataRelationship],
    ) -> int:
        """
        Create relationships between WikiPage nodes in batches.

        Args:
            relationships: List of WikidataRelationship objects.

        Returns:
            Number of relationships created/merged.
        """
        if not relationships:
            return 0

        total_created = 0

        # Group relationships by type for efficient batch processing
        rels_by_type: Dict[str, List[WikidataRelationship]] = {}
        for rel in relationships:
            rel_type = PROPERTY_TO_REL_TYPE.get(rel.property_id, "RELATED_TO")
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)

        # Process each relationship type
        for rel_type, type_rels in rels_by_type.items():
            for i in range(0, len(type_rels), self.batch_size):
                batch = type_rels[i:i + self.batch_size]

                batch_data = [
                    {
                        "source_qid": rel.source_qid,
                        "target_qid": rel.target_qid,
                    }
                    for rel in batch
                ]

                # Note: We need to use string interpolation for relationship type
                # because Cypher doesn't support parameterized relationship types
                with self.driver.session() as session:
                    result = session.run(f"""
                        UNWIND $batch AS rel
                        MATCH (source:WikiPage {{wikidata_id: rel.source_qid}})
                        MATCH (target:WikiPage {{wikidata_id: rel.target_qid}})
                        MERGE (source)-[r:{rel_type}]->(target)
                        RETURN count(r) as count
                    """, batch=batch_data)

                    record = result.single()
                    batch_count = record["count"] if record else len(batch)
                    total_created += batch_count

            logger.debug(f"Created {len(type_rels)} {rel_type} relationships")

        logger.info(f"Created/merged {total_created} relationships")
        return total_created

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about WikiPage nodes and relationships.

        Returns:
            Dictionary with node and relationship counts.
        """
        with self.driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (w:WikiPage) RETURN count(w) as count")
            node_count = node_result.single()["count"]

            # Count relationships by type
            rel_result = session.run("""
                MATCH (w1:WikiPage)-[r]->(w2:WikiPage)
                RETURN type(r) as rel_type, count(r) as count
            """)

            rel_counts = {}
            total_rels = 0
            for record in rel_result:
                rel_counts[record["rel_type"]] = record["count"]
                total_rels += record["count"]

            return {
                "total_nodes": node_count,
                "total_relationships": total_rels,
                **{f"rel_{k}": v for k, v in rel_counts.items()},
            }

    def search_by_name(self, search_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search WikiPage nodes by name.

        Args:
            search_query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching WikiPage nodes.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (w:WikiPage)
                WHERE toLower(w.name) CONTAINS toLower($search_term)
                RETURN w.name as name, w.wikidata_id as qid, w.wikipedia_url as url
                ORDER BY size(w.name)
                LIMIT $max_results
            """, search_term=search_query, max_results=limit)

            return [dict(record) for record in result]


# --- FALKORDB LOADER ---

class FalkorDBPageLoader:
    """Loads WikiPage nodes and relationships into FalkorDB (Redis-based graph DB)."""

    def __init__(
        self,
        host: str = FALKORDB_HOST,
        port: int = FALKORDB_PORT,
        graph_name: str = FALKORDB_GRAPH_NAME,
        batch_size: int = 100,
    ):
        """
        Initialize the FalkorDB loader.

        Args:
            host: FalkorDB/Redis host.
            port: FalkorDB/Redis port.
            graph_name: Name of the graph in FalkorDB.
            batch_size: Batch size for operations.
        """
        try:
            from falkordb import FalkorDB
            self.client = FalkorDB(host=host, port=port)
            self.graph = self.client.select_graph(graph_name)
        except ImportError:
            raise ImportError("FalkorDB package not installed. Install with: pip install falkordb")

        self.batch_size = batch_size
        self._embedding_model = None
        self.graph_name = graph_name

    def close(self):
        """Close the FalkorDB connection."""
        # FalkorDB connections are managed by the client
        pass

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from langchain_ollama import OllamaEmbeddings
            self._embedding_model = OllamaEmbeddings(
                model="qwen3-embedding:8b",
                base_url=OLLAMA_HOST,
                num_ctx=4096,
            )
        return self._embedding_model

    def init_schema(self):
        """
        Initialize FalkorDB schema for WikiPage nodes.

        Creates:
        - Index on wikidata_id for efficient lookups
        - Index on name for search
        """
        try:
            # Create index on wikidata_id
            self.graph.query("CREATE INDEX FOR (w:WikiPage) ON (w.wikidata_id)")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        try:
            # Create index on name
            self.graph.query("CREATE INDEX FOR (w:WikiPage) ON (w.name)")
        except Exception as e:
            logger.debug(f"Index may already exist: {e}")

        logger.info("FalkorDB schema initialized for WikiPage nodes")

    def init_vector_index(self, dimensions: int = 4096):
        """
        Initialize vector index for WikiPage embeddings (optional).

        Note: FalkorDB vector index support may be limited compared to Neo4j.

        Args:
            dimensions: Embedding dimensions.
        """
        try:
            # FalkorDB vector index creation (if supported)
            self.graph.query(f"""
                CREATE VECTOR INDEX wikipage_vec_idx
                FOR (w:WikiPage) ON (w.embedding)
                OPTIONS {{dimension: {dimensions}, similarityFunction: 'cosine'}}
            """)
            logger.info(f"FalkorDB vector index created (dimensions={dimensions})")
        except Exception as e:
            logger.warning(f"FalkorDB vector index creation failed (may not be supported): {e}")

    def batch_create_entities(
        self,
        entities: List[WikidataEntity],
        with_embeddings: bool = False,
    ) -> int:
        """
        Create WikiPage nodes in batches.

        Args:
            entities: List of WikidataEntity objects.
            with_embeddings: Whether to compute and store embeddings.

        Returns:
            Number of entities created/merged.
        """
        if not entities:
            return 0

        total_created = 0

        # Process in batches
        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i + self.batch_size]

            for entity in batch:
                try:
                    # Escape single quotes in strings
                    name = entity.name.replace("'", "\\'") if entity.name else ""
                    description = entity.description.replace("'", "\\'") if entity.description else ""
                    url = entity.wikipedia_url.replace("'", "\\'") if entity.wikipedia_url else ""

                    # MERGE query for FalkorDB
                    query = f"""
                        MERGE (w:WikiPage {{wikidata_id: '{entity.qid}'}})
                        ON CREATE SET
                            w.name = '{name}',
                            w.wikipedia_url = '{url}',
                            w.description = '{description}'
                        ON MATCH SET
                            w.name = '{name}'
                        RETURN w
                    """
                    self.graph.query(query)

                    # Add embedding if requested
                    if with_embeddings:
                        try:
                            embedding_model = self._get_embedding_model()
                            embedding = embedding_model.embed_query(entity.name)
                            # Store embedding as a property
                            embed_str = str(embedding)
                            self.graph.query(f"""
                                MATCH (w:WikiPage {{wikidata_id: '{entity.qid}'}})
                                SET w.embedding = {embed_str}
                            """)
                        except Exception as e:
                            logger.warning(f"Failed to embed {entity.name}: {e}")

                    total_created += 1

                except Exception as e:
                    logger.warning(f"Failed to create entity {entity.qid}: {e}")

            logger.debug(f"Batch {i // self.batch_size + 1}: {len(batch)} entities")

        logger.info(f"Created/merged {total_created} WikiPage nodes in FalkorDB")
        return total_created

    def batch_create_relationships(
        self,
        relationships: List[WikidataRelationship],
    ) -> int:
        """
        Create relationships between WikiPage nodes in batches.

        Args:
            relationships: List of WikidataRelationship objects.

        Returns:
            Number of relationships created/merged.
        """
        if not relationships:
            return 0

        total_created = 0

        for rel in relationships:
            try:
                rel_type = PROPERTY_TO_REL_TYPE.get(rel.property_id, "RELATED_TO")

                query = f"""
                    MATCH (source:WikiPage {{wikidata_id: '{rel.source_qid}'}})
                    MATCH (target:WikiPage {{wikidata_id: '{rel.target_qid}'}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    RETURN r
                """
                self.graph.query(query)
                total_created += 1

            except Exception as e:
                logger.warning(f"Failed to create relationship {rel.source_qid} -> {rel.target_qid}: {e}")

        logger.info(f"Created/merged {total_created} relationships in FalkorDB")
        return total_created

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about WikiPage nodes and relationships.

        Returns:
            Dictionary with node and relationship counts.
        """
        try:
            # Count nodes
            node_result = self.graph.query("MATCH (w:WikiPage) RETURN count(w) as count")
            node_count = node_result.result_set[0][0] if node_result.result_set else 0

            # Count relationships by type
            rel_counts = {}
            total_rels = 0

            for rel_type in PROPERTY_TO_REL_TYPE.values():
                try:
                    rel_result = self.graph.query(f"""
                        MATCH (:WikiPage)-[r:{rel_type}]->(:WikiPage)
                        RETURN count(r) as count
                    """)
                    count = rel_result.result_set[0][0] if rel_result.result_set else 0
                    if count > 0:
                        rel_counts[rel_type] = count
                        total_rels += count
                except Exception:
                    pass

            return {
                "total_nodes": node_count,
                "total_relationships": total_rels,
                **{f"rel_{k}": v for k, v in rel_counts.items()},
            }

        except Exception as e:
            logger.error(f"Error getting FalkorDB stats: {e}")
            return {"total_nodes": 0, "total_relationships": 0}

    def search_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search WikiPage nodes by name.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching WikiPage nodes.
        """
        try:
            # Escape query string
            safe_query = query.replace("'", "\\'").lower()

            result = self.graph.query(f"""
                MATCH (w:WikiPage)
                WHERE toLower(w.name) CONTAINS '{safe_query}'
                RETURN w.name as name, w.wikidata_id as qid, w.wikipedia_url as url
                ORDER BY size(w.name)
                LIMIT {limit}
            """)

            results = []
            for row in result.result_set:
                results.append({
                    "name": row[0],
                    "qid": row[1],
                    "url": row[2],
                })
            return results

        except Exception as e:
            logger.error(f"FalkorDB search error: {e}")
            return []

    def clear_graph(self):
        """Clear all data from the graph."""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            logger.info("FalkorDB graph cleared")
        except Exception as e:
            logger.error(f"Error clearing FalkorDB graph: {e}")

    def get_wikipages_with_urls(
        self,
        min_score: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get WikiPage nodes with wikipedia_url for Stage 2 processing.

        This method is used by the Wikipedia article pipeline to retrieve
        WikiPage nodes that have associated Wikipedia URLs for content loading.

        Args:
            min_score: Minimum connectivity score to filter by (default: 0.0).
            limit: Maximum number of results to return (default: None = all).

        Returns:
            List of dictionaries containing:
                - qid: Wikidata QID (e.g., "Q123")
                - name: Entity name
                - url: Wikipedia URL
                - description: Entity description
                - connectivity_score: Score indicating graph connectivity
        """
        try:
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
                score = row[4] if row[4] else 0.0
                if score >= min_score:
                    wikipages.append({
                        "qid": row[0],
                        "name": row[1],
                        "url": row[2],
                        "description": row[3],
                        "connectivity_score": score,
                    })

            logger.info(f"Found {len(wikipages)} WikiPage nodes with URLs")
            return wikipages

        except Exception as e:
            logger.error(f"Error getting WikiPages with URLs: {e}")
            return []

    def update_connectivity_scores(self) -> int:
        """
        Update connectivity_score for all WikiPage nodes.

        Computes connectivity based on the number of relationships
        each node has (both incoming and outgoing).

        Returns:
            Number of nodes updated.
        """
        try:
            # Compute and update connectivity scores
            self.graph.query("""
                MATCH (w:WikiPage)
                OPTIONAL MATCH (w)-[r1]->()
                OPTIONAL MATCH ()-[r2]->(w)
                WITH w, count(DISTINCT r1) + count(DISTINCT r2) as score
                SET w.connectivity_score = score
            """)

            # Count updated nodes
            result = self.graph.query("""
                MATCH (w:WikiPage)
                WHERE w.connectivity_score IS NOT NULL
                RETURN count(w) as count
            """)
            count = result.result_set[0][0] if result.result_set else 0

            logger.info(f"Updated connectivity scores for {count} WikiPage nodes")
            return count

        except Exception as e:
            logger.error(f"Error updating connectivity scores: {e}")
            return 0


# --- BFS BUILDER ---

class WikidataKGBuilder:
    """Builds a knowledge graph from Wikidata using BFS or root-first traversal."""

    def __init__(
        self,
        sparql_client: Optional[WikidataSPARQLClient] = None,
        loader: Optional[Any] = None,
        backend: str = "neo4j",
        dry_run: bool = False,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.BFS,
        root_threshold: float = 5.0,
        max_leaves: Optional[int] = None,
        falkordb_graph_name: Optional[str] = None,
        require_label: bool = False,
    ):
        """
        Initialize the KG builder.

        Args:
            sparql_client: SPARQL client for Wikidata queries.
            loader: Graph database loader (WikiPageLoader or FalkorDBPageLoader).
            backend: Backend type: "neo4j" or "falkordb".
            dry_run: If True, only query Wikidata without writing to database.
            exploration_strategy: Traversal strategy (BFS or ROOT_FIRST).
            root_threshold: Abstraction score threshold for root classification.
            max_leaves: Maximum leaf nodes to collect in phase 2 (None = no limit).
            falkordb_graph_name: Custom graph name for FalkorDB (default: env var or 'wikidata').
            require_label: If True, skip entities without English labels (QID-only names).
        """
        self.sparql_client = sparql_client or WikidataSPARQLClient()
        self.loader = loader
        self.backend = backend
        self.dry_run = dry_run
        self.exploration_strategy = exploration_strategy
        self.root_threshold = root_threshold
        self.max_leaves = max_leaves
        self.require_label = require_label
        self.skipped_no_label = 0  # Track entities skipped due to missing labels

        if not dry_run and loader is None:
            if backend == "falkordb":
                self.loader = FalkorDBPageLoader(
                    graph_name=falkordb_graph_name or FALKORDB_GRAPH_NAME
                )
            else:
                self.loader = WikiPageLoader()

        # Alias for backwards compatibility
        self.neo4j_loader = self.loader if backend == "neo4j" else None

    def close(self):
        """Close connections."""
        if self.loader:
            self.loader.close()

    def _compute_connectivity_score(
        self,
        qid: str,
        name: str,
        root_qids: Set[str],
        relationships: List[WikidataRelationship],
    ) -> ScoredNode:
        """
        Compute connectivity score for a leaf candidate.

        Score = 3.0 * direct_connections_to_roots
              + 2.0 * distinct_roots_reachable
              + 1.5 * subclass_connections

        Args:
            qid: The node's QID.
            name: The node's name.
            root_qids: Set of collected root QIDs.
            relationships: All relationships collected so far.

        Returns:
            ScoredNode with computed score and connected roots.
        """
        connected_roots: Set[str] = set()
        direct_connections = 0
        subclass_connections = 0

        for rel in relationships:
            # Check if this node has a direct relationship with a root
            if rel.source_qid == qid and rel.target_qid in root_qids:
                direct_connections += 1
                connected_roots.add(rel.target_qid)
            elif rel.target_qid == qid and rel.source_qid in root_qids:
                direct_connections += 1
                connected_roots.add(rel.source_qid)

            # Count subclass relationships involving this node
            if rel.property_id == PROPERTY_SUBCLASS_OF:
                if rel.source_qid == qid or rel.target_qid == qid:
                    subclass_connections += 1

        score = (
            3.0 * direct_connections
            + 2.0 * len(connected_roots)
            + 1.5 * subclass_connections
        )

        return ScoredNode(
            qid=qid,
            name=name,
            score=score,
            connected_roots=connected_roots,
        )

    def build_from_root(
        self,
        root_qid: str = "Q1101",
        max_depth: int = 3,
        max_entities_per_level: int = 500,
        include_instances: bool = True,
        include_parts: bool = True,
        with_embeddings: bool = False,
        exploration_strategy: Optional[ExplorationStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from a root entity.

        Args:
            root_qid: Root Wikidata QID (default: Q1101 "Technology").
            max_depth: Maximum traversal depth.
            max_entities_per_level: Maximum entities to fetch per level.
            include_instances: Include instance_of relationships.
            include_parts: Include part_of relationships.
            with_embeddings: Compute and store name embeddings.
            exploration_strategy: Override default exploration strategy.

        Returns:
            Statistics about the build process.
        """
        strategy = exploration_strategy or self.exploration_strategy

        logger.info(f"Building KG from root {root_qid} (strategy={strategy.value}, "
                    f"depth={max_depth}, limit={max_entities_per_level})")

        # Initialize schema if not dry run
        if not self.dry_run and self.loader:
            self.loader.init_schema()
            if with_embeddings:
                self.loader.init_vector_index()

        # Dispatch to appropriate strategy
        if strategy == ExplorationStrategy.ROOT_FIRST:
            return self._build_root_first(
                root_qid=root_qid,
                max_depth=max_depth,
                max_entities_per_level=max_entities_per_level,
                include_instances=include_instances,
                include_parts=include_parts,
                with_embeddings=with_embeddings,
            )
        else:
            return self._build_bfs(
                root_qid=root_qid,
                max_depth=max_depth,
                max_entities_per_level=max_entities_per_level,
                include_instances=include_instances,
                include_parts=include_parts,
                with_embeddings=with_embeddings,
            )

    def _build_bfs(
        self,
        root_qid: str,
        max_depth: int,
        max_entities_per_level: int,
        include_instances: bool,
        include_parts: bool,
        with_embeddings: bool,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph using standard BFS traversal.

        This is the original algorithm that explores all children at each level
        before moving to the next depth.
        """
        # Reset skipped counter for this build
        self.skipped_no_label = 0

        # Track visited entities and all collected data
        visited: Set[str] = set()
        all_entities: List[WikidataEntity] = []
        all_relationships: List[WikidataRelationship] = []

        # Fetch root entity
        root_entity = self.sparql_client.fetch_entity(root_qid)
        if root_entity:
            all_entities.append(root_entity)
            visited.add(root_qid)
            logger.info(f"Root entity: {root_entity.name} ({root_qid})")
        else:
            logger.error(f"Could not fetch root entity {root_qid}")
            return {"error": f"Root entity {root_qid} not found"}

        # BFS queue: (qids_at_level, current_depth)
        queue: deque = deque()
        queue.append(([root_qid], 0))

        stats = {
            "root_qid": root_qid,
            "max_depth": max_depth,
            "strategy": "bfs",
            "levels": [],
        }

        while queue:
            current_qids, depth = queue.popleft()

            if depth >= max_depth:
                continue

            logger.info(f"BFS Level {depth + 1}: Processing {len(current_qids)} parent(s)")

            # Fetch children of current level
            result = self.sparql_client.fetch_children(
                parent_qids=current_qids,
                limit=max_entities_per_level,
                include_subclasses=True,
                include_instances=include_instances,
                include_parts=include_parts,
            )

            # Filter out already visited entities
            new_entities = [e for e in result.entities if e.qid not in visited]

            # Filter out entities without English labels if require_label is set
            if self.require_label:
                filtered_entities = []
                for entity in new_entities:
                    if entity.name == entity.qid:
                        logger.debug(f"Skipping {entity.qid} - no English label available")
                        self.skipped_no_label += 1
                    else:
                        filtered_entities.append(entity)
                new_entities = filtered_entities

            new_qids = [e.qid for e in new_entities]

            # Update visited set
            for qid in new_qids:
                visited.add(qid)

            # Collect entities and relationships
            all_entities.extend(new_entities)
            all_relationships.extend(result.relationships)

            level_stats = {
                "depth": depth + 1,
                "parents": len(current_qids),
                "new_entities": len(new_entities),
                "relationships": len(result.relationships),
                "errors": result.errors,
            }
            stats["levels"].append(level_stats)

            logger.info(f"  Found {len(new_entities)} new entities, {len(result.relationships)} relationships")

            # Queue next level if we have new entities
            if new_qids and depth + 1 < max_depth:
                queue.append((new_qids, depth + 1))

        # Write to database if not dry run
        if not self.dry_run and self.loader:
            logger.info(f"Writing {len(all_entities)} entities to {self.backend}...")
            entities_created = self.loader.batch_create_entities(
                all_entities,
                with_embeddings=with_embeddings,
            )

            logger.info(f"Writing {len(all_relationships)} relationships to {self.backend}...")
            rels_created = self.loader.batch_create_relationships(all_relationships)

            stats["entities_created"] = entities_created
            stats["relationships_created"] = rels_created
            stats["backend"] = self.backend
            stats["db_stats"] = self.loader.get_stats()
        else:
            stats["dry_run"] = True
            stats["entities_found"] = len(all_entities)
            stats["relationships_found"] = len(all_relationships)

        stats["total_visited"] = len(visited)
        if self.require_label:
            stats["skipped_no_label"] = self.skipped_no_label

        return stats

    def _build_root_first(
        self,
        root_qid: str,
        max_depth: int,
        max_entities_per_level: int,
        include_instances: bool,
        include_parts: bool,
        with_embeddings: bool,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph using two-phase root-first traversal.

        Phase 1: Collect abstract "root" nodes first (nodes with high abstraction scores)
        Phase 2: Collect "leaf" nodes prioritized by their connectivity to collected roots
        """
        logger.info(f"Using ROOT_FIRST strategy (threshold={self.root_threshold})")

        # Reset skipped counter for this build
        self.skipped_no_label = 0

        # Initialize exploration state
        state = ExplorationState()
        all_entities: List[WikidataEntity] = []
        all_relationships: List[WikidataRelationship] = []
        entity_map: Dict[str, WikidataEntity] = {}  # QID -> Entity for lookup

        # Fetch root entity
        root_entity = self.sparql_client.fetch_entity(root_qid)
        if root_entity:
            all_entities.append(root_entity)
            entity_map[root_qid] = root_entity
            state.visited.add(root_qid)
            state.root_qids.add(root_qid)  # Starting point is always a root
            logger.info(f"Root entity: {root_entity.name} ({root_qid})")
        else:
            logger.error(f"Could not fetch root entity {root_qid}")
            return {"error": f"Root entity {root_qid} not found"}

        stats = {
            "root_qid": root_qid,
            "max_depth": max_depth,
            "strategy": "root_first",
            "root_threshold": self.root_threshold,
            "phase1_roots": 0,
            "phase2_leaves": 0,
            "levels": [],
        }

        # ===== PHASE 1: Collect root nodes =====
        logger.info("=" * 50)
        logger.info("PHASE 1: Collecting abstract root nodes")
        logger.info("=" * 50)

        root_queue: deque = deque()
        root_queue.append(([root_qid], 0))

        while root_queue:
            current_qids, depth = root_queue.popleft()

            if depth >= max_depth:
                continue

            logger.info(f"Phase 1 - Level {depth + 1}: Processing {len(current_qids)} parent(s)")

            # Fetch children of current level
            result = self.sparql_client.fetch_children(
                parent_qids=current_qids,
                limit=max_entities_per_level,
                include_subclasses=True,
                include_instances=include_instances,
                include_parts=include_parts,
            )

            # Collect relationships
            all_relationships.extend(result.relationships)

            # Filter to unvisited entities
            new_entities = [e for e in result.entities if e.qid not in state.visited]

            # Filter out entities without English labels if require_label is set
            if self.require_label:
                filtered_entities = []
                for entity in new_entities:
                    if entity.name == entity.qid:
                        logger.debug(f"Skipping {entity.qid} - no English label available")
                        self.skipped_no_label += 1
                    else:
                        filtered_entities.append(entity)
                new_entities = filtered_entities

            if not new_entities:
                continue

            # Fetch metrics to classify as root or leaf
            new_qids = [e.qid for e in new_entities]
            metrics = self.sparql_client.fetch_node_metrics(new_qids)

            roots_this_level = []
            leaves_this_level = []

            for entity in new_entities:
                state.visited.add(entity.qid)
                entity_map[entity.qid] = entity

                node_metrics = metrics.get(entity.qid)
                if node_metrics and node_metrics.is_root_like(self.root_threshold):
                    # This is a root - add to results and queue for expansion
                    all_entities.append(entity)
                    state.root_qids.add(entity.qid)
                    roots_this_level.append(entity.qid)
                    logger.debug(f"  ROOT: {entity.name} ({entity.qid}) "
                                 f"[parents={node_metrics.parent_count}, "
                                 f"children={node_metrics.child_count}, "
                                 f"instances={node_metrics.instance_count}, "
                                 f"score={node_metrics.abstraction_score:.1f}]")
                else:
                    # This is a leaf candidate - defer for phase 2
                    leaves_this_level.append(entity.qid)
                    if node_metrics:
                        logger.debug(f"  LEAF: {entity.name} ({entity.qid}) "
                                     f"[score={node_metrics.abstraction_score:.1f}]")

            level_stats = {
                "phase": 1,
                "depth": depth + 1,
                "parents": len(current_qids),
                "roots_found": len(roots_this_level),
                "leaves_deferred": len(leaves_this_level),
            }
            stats["levels"].append(level_stats)

            logger.info(f"  Found {len(roots_this_level)} roots, "
                        f"deferred {len(leaves_this_level)} leaves")

            # Queue roots for next level expansion
            if roots_this_level and depth + 1 < max_depth:
                root_queue.append((roots_this_level, depth + 1))

        stats["phase1_roots"] = len(state.root_qids)
        logger.info(f"Phase 1 complete: {len(state.root_qids)} root nodes collected")

        # ===== PHASE 2: Collect leaf nodes by connectivity =====
        logger.info("=" * 50)
        logger.info("PHASE 2: Collecting leaf nodes by connectivity")
        logger.info("=" * 50)

        # Score all visited but not yet added entities (leaves)
        leaf_qids = state.visited - state.root_qids
        logger.info(f"Scoring {len(leaf_qids)} leaf candidates...")

        scored_leaves: List[ScoredNode] = []
        for qid in leaf_qids:
            entity = entity_map.get(qid)
            if entity:
                scored = self._compute_connectivity_score(
                    qid=qid,
                    name=entity.name,
                    root_qids=state.root_qids,
                    relationships=all_relationships,
                )
                scored_leaves.append(scored)

        # Sort by score (highest first) using heapq
        heapq.heapify(scored_leaves)

        # Determine how many leaves to collect
        max_leaves = self.max_leaves
        if max_leaves is None:
            # Default: collect all leaves
            max_leaves = len(scored_leaves)

        leaves_collected = 0
        while scored_leaves and leaves_collected < max_leaves:
            best = heapq.heappop(scored_leaves)
            entity = entity_map.get(best.qid)
            if entity:
                all_entities.append(entity)
                leaves_collected += 1
                if leaves_collected <= 10:  # Log first 10
                    logger.info(f"  Added leaf: {entity.name} ({entity.qid}) "
                                f"[score={best.score:.1f}, "
                                f"connected to {len(best.connected_roots)} roots]")

        if leaves_collected > 10:
            logger.info(f"  ... and {leaves_collected - 10} more leaves")

        stats["phase2_leaves"] = leaves_collected
        logger.info(f"Phase 2 complete: {leaves_collected} leaf nodes collected")

        # ===== Write to database =====
        if not self.dry_run and self.loader:
            logger.info(f"Writing {len(all_entities)} entities to {self.backend}...")
            entities_created = self.loader.batch_create_entities(
                all_entities,
                with_embeddings=with_embeddings,
            )

            logger.info(f"Writing {len(all_relationships)} relationships to {self.backend}...")
            rels_created = self.loader.batch_create_relationships(all_relationships)

            stats["entities_created"] = entities_created
            stats["relationships_created"] = rels_created
            stats["backend"] = self.backend
            stats["db_stats"] = self.loader.get_stats()
        else:
            stats["dry_run"] = True
            stats["entities_found"] = len(all_entities)
            stats["relationships_found"] = len(all_relationships)

        stats["total_visited"] = len(state.visited)
        if self.require_label:
            stats["skipped_no_label"] = self.skipped_no_label

        return stats

    def build_from_seeds(
        self,
        seeds_file: str,
        max_depth: int = 3,
        max_entities_per_level: int = 500,
        include_instances: bool = True,
        include_parts: bool = True,
        with_embeddings: bool = False,
        category: Optional[str] = None,
        exploration_strategy: Optional[ExplorationStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from multiple seed entities defined in a JSON file.

        Args:
            seeds_file: Path to JSON file containing seed configurations.
            max_depth: Maximum traversal depth for each seed.
            max_entities_per_level: Maximum entities to fetch per level.
            include_instances: Include instance_of relationships.
            include_parts: Include part_of relationships.
            with_embeddings: Compute and store name embeddings.
            category: Optional category filter - only process seeds in this category.
            exploration_strategy: Override default exploration strategy.

        Returns:
            Aggregated statistics from all seed builds.
        """
        # Load seeds configuration
        with open(seeds_file, 'r') as f:
            config = json.load(f)

        seeds = config.get("seeds", [])
        if not seeds:
            return {"error": "No seeds found in configuration file"}

        # Filter by category if specified
        if category:
            seeds = [s for s in seeds if s.get("category") == category]
            if not seeds:
                return {"error": f"No seeds found for category '{category}'"}
            logger.info(f"Filtered to {len(seeds)} seeds in category '{category}'")

        logger.info(f"Building KG from {len(seeds)} seed entities")
        logger.info(f"Config: {config.get('name', 'Unknown')}")

        # Initialize schema once before processing seeds
        if not self.dry_run and self.loader:
            self.loader.init_schema()
            if with_embeddings:
                self.loader.init_vector_index()

        # Track aggregated results
        all_stats = {
            "config_name": config.get("name", "Unknown"),
            "config_description": config.get("description", ""),
            "seeds_file": seeds_file,
            "category_filter": category,
            "seeds": [],
            "totals": {
                "seeds_processed": 0,
                "seeds_failed": 0,
                "total_entities": 0,
                "total_relationships": 0,
                "total_visited": 0,
                "total_skipped_no_label": 0,
            },
        }

        # Process each seed
        for i, seed in enumerate(seeds, 1):
            qid = seed.get("qid")
            name = seed.get("name", qid)
            seed_category = seed.get("category", "unknown")

            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(seeds)}] Processing seed: {name} ({qid}) - {seed_category}")
            logger.info(f"{'='*60}")

            try:
                # Build from this seed (schema already initialized)
                # We need a modified version that doesn't re-init schema
                seed_stats = self._build_from_root_no_schema_init(
                    root_qid=qid,
                    max_depth=max_depth,
                    max_entities_per_level=max_entities_per_level,
                    include_instances=include_instances,
                    include_parts=include_parts,
                    with_embeddings=with_embeddings,
                    exploration_strategy=exploration_strategy,
                )

                seed_stats["seed_name"] = name
                seed_stats["seed_category"] = seed_category
                all_stats["seeds"].append(seed_stats)

                # Update totals
                all_stats["totals"]["seeds_processed"] += 1
                if self.dry_run:
                    all_stats["totals"]["total_entities"] += seed_stats.get("entities_found", 0)
                    all_stats["totals"]["total_relationships"] += seed_stats.get("relationships_found", 0)
                else:
                    all_stats["totals"]["total_entities"] += seed_stats.get("entities_created", 0)
                    all_stats["totals"]["total_relationships"] += seed_stats.get("relationships_created", 0)
                all_stats["totals"]["total_visited"] += seed_stats.get("total_visited", 0)
                all_stats["totals"]["total_skipped_no_label"] += seed_stats.get("skipped_no_label", 0)

            except Exception as e:
                logger.error(f"Failed to process seed {qid}: {e}")
                all_stats["seeds"].append({
                    "root_qid": qid,
                    "seed_name": name,
                    "seed_category": seed_category,
                    "error": str(e),
                })
                all_stats["totals"]["seeds_failed"] += 1

        # Get final database stats
        if not self.dry_run and self.loader:
            all_stats["db_stats"] = self.loader.get_stats()
            all_stats["backend"] = self.backend

        return all_stats

    def _build_from_root_no_schema_init(
        self,
        root_qid: str,
        max_depth: int = 3,
        max_entities_per_level: int = 500,
        include_instances: bool = True,
        include_parts: bool = True,
        with_embeddings: bool = False,
        exploration_strategy: Optional[ExplorationStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from a root entity without re-initializing schema.

        This is used internally by build_from_seeds() to avoid repeated schema init.
        """
        strategy = exploration_strategy or self.exploration_strategy

        logger.info(f"Building from root {root_qid} (strategy={strategy.value}, "
                    f"depth={max_depth}, limit={max_entities_per_level})")

        # Dispatch to appropriate strategy
        if strategy == ExplorationStrategy.ROOT_FIRST:
            return self._build_root_first(
                root_qid=root_qid,
                max_depth=max_depth,
                max_entities_per_level=max_entities_per_level,
                include_instances=include_instances,
                include_parts=include_parts,
                with_embeddings=with_embeddings,
            )
        else:
            return self._build_bfs(
                root_qid=root_qid,
                max_depth=max_depth,
                max_entities_per_level=max_entities_per_level,
                include_instances=include_instances,
                include_parts=include_parts,
                with_embeddings=with_embeddings,
            )


# --- BENCHMARK ---

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    total_nodes: int
    total_relationships: int
    write_time_seconds: float
    read_time_seconds: float
    search_time_seconds: float
    success: bool
    error: Optional[str] = None


def run_benchmark(
    target_nodes: int = 1000,
    root_qid: str = "Q1101",
    max_depth: int = 4,
    max_per_level: int = 500,
) -> Dict[str, Any]:
    """
    Run benchmark comparing Neo4j and FalkorDB backends.

    Args:
        target_nodes: Target number of nodes to create.
        root_qid: Root Wikidata QID for building the graph.
        max_depth: Maximum BFS depth.
        max_per_level: Maximum entities per level.

    Returns:
        Dictionary with benchmark results.
    """
    print("\n" + "=" * 70)
    print("WIKIDATA KG BUILDER - BACKEND BENCHMARK")
    print("=" * 70)
    print(f"Target Nodes:   ~{target_nodes}")
    print(f"Root QID:       {root_qid}")
    print(f"Max Depth:      {max_depth}")
    print(f"Max Per Level:  {max_per_level}")
    print("=" * 70)

    results: Dict[str, BenchmarkResult] = {}

    # First, fetch data from Wikidata (shared between backends)
    print("\n[1/3] Fetching data from Wikidata...")
    sparql_client = WikidataSPARQLClient()

    # Build with dry run to get data
    dry_builder = WikidataKGBuilder(sparql_client=sparql_client, dry_run=True)
    fetch_start = time.time()
    dry_stats = dry_builder.build_from_root(
        root_qid=root_qid,
        max_depth=max_depth,
        max_entities_per_level=max_per_level,
    )
    fetch_time = time.time() - fetch_start
    dry_builder.close()

    entities_count = dry_stats.get("entities_found", 0)
    rels_count = dry_stats.get("relationships_found", 0)
    print(f"    Fetched {entities_count} entities, {rels_count} relationships in {fetch_time:.2f}s")

    # Helper function to test a backend
    def benchmark_backend(backend: str) -> BenchmarkResult:
        print(f"\n[2/3] Testing {backend.upper()} backend...")

        try:
            # Initialize loader
            if backend == "neo4j":
                loader = WikiPageLoader()
            else:
                loader = FalkorDBPageLoader()

            # Clear existing data (for fair comparison)
            if backend == "falkordb":
                try:
                    loader.clear_graph()
                except Exception:
                    pass
            else:
                try:
                    with loader.driver.session() as session:
                        session.run("MATCH (w:WikiPage) DETACH DELETE w")
                except Exception:
                    pass

            # Build the graph
            builder = WikidataKGBuilder(
                sparql_client=sparql_client,
                loader=loader,
                backend=backend,
            )

            write_start = time.time()
            stats = builder.build_from_root(
                root_qid=root_qid,
                max_depth=max_depth,
                max_entities_per_level=max_per_level,
            )
            write_time = time.time() - write_start
            print(f"    Write: {stats.get('entities_created', 0)} nodes, "
                  f"{stats.get('relationships_created', 0)} rels in {write_time:.2f}s")

            # Test read performance (get stats)
            read_start = time.time()
            for _ in range(10):
                loader.get_stats()
            read_time = (time.time() - read_start) / 10
            print(f"    Read (get_stats): avg {read_time*1000:.1f}ms")

            # Test search performance
            search_queries = ["machine learning", "computer", "software", "network", "data"]
            search_start = time.time()
            for q in search_queries:
                loader.search_by_name(q, limit=5)
            search_time = (time.time() - search_start) / len(search_queries)
            print(f"    Search: avg {search_time*1000:.1f}ms per query")

            # Get final stats
            final_stats = loader.get_stats()

            builder.close()

            return BenchmarkResult(
                backend=backend,
                total_nodes=final_stats.get("total_nodes", 0),
                total_relationships=final_stats.get("total_relationships", 0),
                write_time_seconds=write_time,
                read_time_seconds=read_time,
                search_time_seconds=search_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"{backend} benchmark failed: {e}")
            return BenchmarkResult(
                backend=backend,
                total_nodes=0,
                total_relationships=0,
                write_time_seconds=0,
                read_time_seconds=0,
                search_time_seconds=0,
                success=False,
                error=str(e),
            )

    # Run Neo4j benchmark
    results["neo4j"] = benchmark_backend("neo4j")

    # Run FalkorDB benchmark
    results["falkordb"] = benchmark_backend("falkordb")

    # Print comparison
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Neo4j':>15} {'FalkorDB':>15} {'Winner':>12}")
    print("-" * 72)

    neo4j = results["neo4j"]
    falkor = results["falkordb"]

    def compare(name: str, neo_val: float, falk_val: float, lower_better: bool = True):
        if not neo4j.success:
            winner = "FalkorDB" if falkor.success else "N/A"
        elif not falkor.success:
            winner = "Neo4j"
        elif lower_better:
            winner = "Neo4j" if neo_val <= falk_val else "FalkorDB"
        else:
            winner = "Neo4j" if neo_val >= falk_val else "FalkorDB"

        neo_str = f"{neo_val:.3f}" if neo4j.success else "FAILED"
        falk_str = f"{falk_val:.3f}" if falkor.success else "FAILED"
        print(f"{name:<30} {neo_str:>15} {falk_str:>15} {winner:>12}")

    compare("Write Time (s)", neo4j.write_time_seconds, falkor.write_time_seconds, True)
    compare("Read Time (s)", neo4j.read_time_seconds, falkor.read_time_seconds, True)
    compare("Search Time (s)", neo4j.search_time_seconds, falkor.search_time_seconds, True)

    print("-" * 72)
    print(f"{'Total Nodes':<30} {neo4j.total_nodes:>15} {falkor.total_nodes:>15}")
    print(f"{'Total Relationships':<30} {neo4j.total_relationships:>15} {falkor.total_relationships:>15}")

    if not neo4j.success:
        print(f"\nNeo4j Error: {neo4j.error}")
    if not falkor.success:
        print(f"\nFalkorDB Error: {falkor.error}")

    print("\n" + "=" * 70)

    return {
        "neo4j": vars(neo4j),
        "falkordb": vars(falkor),
        "wikidata_fetch_time": fetch_time,
        "entities_fetched": entities_count,
        "relationships_fetched": rels_count,
    }


# --- CLI ---

def main():
    """CLI entry point for building the Wikidata KG."""
    parser = argparse.ArgumentParser(
        description="Build a Knowledge Graph from Wikidata's Technology tree"
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
        default=3,
        help="Maximum BFS traversal depth (default: 3)"
    )
    parser.add_argument(
        "--max-per-level",
        type=int,
        default=500,
        help="Maximum entities per BFS level (default: 500)"
    )
    parser.add_argument(
        "--no-instances",
        action="store_true",
        help="Exclude instance_of relationships"
    )
    parser.add_argument(
        "--no-parts",
        action="store_true",
        help="Exclude part_of relationships"
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Compute and store name embeddings"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query only, don't write to Neo4j"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show current Neo4j stats"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search WikiPage nodes by name"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["neo4j", "falkordb"],
        default="falkordb",
        help="Database backend (default: falkordb)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparing Neo4j and FalkorDB"
    )
    parser.add_argument(
        "--benchmark-nodes",
        type=int,
        default=1000,
        help="Target number of nodes for benchmark (default: 1000)"
    )
    parser.add_argument(
        "--seeds-file",
        type=str,
        help="JSON file with multiple seed QIDs to build from"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Only process seeds from this category (requires --seeds-file)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["bfs", "root_first"],
        default="bfs",
        help="Exploration strategy: 'bfs' (original) or 'root_first' (two-phase)"
    )
    parser.add_argument(
        "--root-threshold",
        type=float,
        default=5.0,
        help="Abstraction score threshold for root classification (default: 5.0)"
    )
    parser.add_argument(
        "--max-leaves",
        type=int,
        default=None,
        help="Maximum leaf nodes to collect in phase 2 (default: unlimited)"
    )
    parser.add_argument(
        "--falkordb-graph",
        type=str,
        default=None,
        help="FalkorDB graph name for this run (default: 'wikidata' or FALKORDB_GRAPH_NAME env var)"
    )
    parser.add_argument(
        "--require-label",
        action="store_true",
        help="Skip entities without English labels (filters QID-only names)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Benchmark mode
    if args.benchmark:
        run_benchmark(
            target_nodes=args.benchmark_nodes,
            root_qid=args.root,
            max_depth=args.max_depth,
            max_per_level=args.max_per_level,
        )
        return

    # Helper to get loader based on backend
    def get_loader(backend: str, falkordb_graph: str = None):
        if backend == "falkordb":
            if falkordb_graph:
                return FalkorDBPageLoader(graph_name=falkordb_graph)
            return FalkorDBPageLoader()
        return WikiPageLoader()

    # Stats-only mode
    if args.stats_only:
        loader = get_loader(args.backend, args.falkordb_graph)
        try:
            stats = loader.get_stats()
            print(f"\n{args.backend.title()} WikiPage Statistics:")
            print("-" * 40)
            for key, value in stats.items():
                print(f"  {key}: {value}")
        finally:
            loader.close()
        return

    # Search mode
    if args.search:
        loader = get_loader(args.backend, args.falkordb_graph)
        try:
            results = loader.search_by_name(args.search, limit=10)
            print(f"\nSearch results for '{args.search}' ({args.backend}):")
            print("-" * 40)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['name']} ({r['qid']})")
                if r['url']:
                    print(f"   URL: {r['url']}")
        finally:
            loader.close()
        return

    # Build mode
    # Parse exploration strategy
    strategy = ExplorationStrategy(args.strategy)

    print("\n" + "=" * 60)
    print("WIKIDATA KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)
    print(f"Backend:       {args.backend}")
    if args.seeds_file:
        print(f"Seeds File:    {args.seeds_file}")
        if args.category:
            print(f"Category:      {args.category}")
    else:
        print(f"Root QID:      {args.root}")
    print(f"Strategy:      {args.strategy}")
    if args.strategy == "root_first":
        print(f"Root Threshold: {args.root_threshold}")
        if args.max_leaves is not None:
            print(f"Max Leaves:    {args.max_leaves}")
    print(f"Max Depth:     {args.max_depth}")
    print(f"Max Per Level: {args.max_per_level}")
    print(f"Include Instances: {not args.no_instances}")
    print(f"Include Parts: {not args.no_parts}")
    print(f"With Embeddings: {args.with_embeddings}")
    print(f"Require Label: {args.require_label}")
    print(f"Dry Run:       {args.dry_run}")
    print("=" * 60)

    builder = WikidataKGBuilder(
        backend=args.backend,
        dry_run=args.dry_run,
        exploration_strategy=strategy,
        root_threshold=args.root_threshold,
        max_leaves=args.max_leaves,
        falkordb_graph_name=args.falkordb_graph,
        require_label=args.require_label,
    )

    try:
        # Choose build method based on arguments
        if args.seeds_file:
            stats = builder.build_from_seeds(
                seeds_file=args.seeds_file,
                max_depth=args.max_depth,
                max_entities_per_level=args.max_per_level,
                include_instances=not args.no_instances,
                include_parts=not args.no_parts,
                with_embeddings=args.with_embeddings,
                category=args.category,
                exploration_strategy=strategy,
            )
        else:
            stats = builder.build_from_root(
                root_qid=args.root,
                max_depth=args.max_depth,
                max_entities_per_level=args.max_per_level,
                include_instances=not args.no_instances,
                include_parts=not args.no_parts,
                with_embeddings=args.with_embeddings,
                exploration_strategy=strategy,
            )

        print("\n" + "=" * 60)
        print("BUILD RESULTS")
        print("=" * 60)

        if "error" in stats:
            print(f"ERROR: {stats['error']}")
            return

        # Handle multi-seed results
        if args.seeds_file:
            totals = stats.get("totals", {})
            print(f"Config: {stats.get('config_name', 'Unknown')}")
            print(f"Seeds processed: {totals.get('seeds_processed', 0)}")
            if totals.get("seeds_failed", 0) > 0:
                print(f"Seeds failed: {totals.get('seeds_failed', 0)}")
            print(f"Total entities visited: {totals.get('total_visited', 0)}")
            if totals.get("total_skipped_no_label", 0) > 0:
                print(f"Entities skipped (no label): {totals['total_skipped_no_label']}")

            print("\nPer-seed breakdown:")
            for seed_stats in stats.get("seeds", []):
                seed_name = seed_stats.get("seed_name", seed_stats.get("root_qid", "?"))
                seed_cat = seed_stats.get("seed_category", "?")
                if "error" in seed_stats:
                    print(f"  {seed_name} ({seed_cat}): ERROR - {seed_stats['error']}")
                else:
                    visited = seed_stats.get("total_visited", 0)
                    if args.dry_run:
                        entities = seed_stats.get("entities_found", 0)
                        rels = seed_stats.get("relationships_found", 0)
                    else:
                        entities = seed_stats.get("entities_created", 0)
                        rels = seed_stats.get("relationships_created", 0)
                    print(f"  {seed_name} ({seed_cat}): {entities} entities, {rels} relationships")

            if args.dry_run:
                print(f"\n[DRY RUN] Would create:")
                print(f"  Total Entities: {totals.get('total_entities', 0)}")
                print(f"  Total Relationships: {totals.get('total_relationships', 0)}")
            else:
                print(f"\n{args.backend.title()} Results:")
                print(f"  Total Entities created/merged: {totals.get('total_entities', 0)}")
                print(f"  Total Relationships created/merged: {totals.get('total_relationships', 0)}")

                db_stats = stats.get("db_stats", {})
                if db_stats:
                    print(f"\nCurrent {args.backend.title()} Stats:")
                    for key, value in db_stats.items():
                        print(f"    {key}: {value}")
        else:
            # Single root results
            print(f"Strategy: {stats.get('strategy', 'bfs')}")
            print(f"Total entities visited: {stats['total_visited']}")
            if stats.get("skipped_no_label", 0) > 0:
                print(f"Entities skipped (no label): {stats['skipped_no_label']}")

            # Handle different stats formats for BFS vs root_first
            if stats.get("strategy") == "root_first":
                print(f"Phase 1 roots: {stats.get('phase1_roots', 0)}")
                print(f"Phase 2 leaves: {stats.get('phase2_leaves', 0)}")
                print(f"Root threshold: {stats.get('root_threshold', 5.0)}")

                print("\nPhase breakdown:")
                for level in stats.get("levels", []):
                    phase = level.get("phase", 1)
                    depth = level.get("depth", "?")
                    roots = level.get("roots_found", 0)
                    leaves = level.get("leaves_deferred", 0)
                    print(f"  Phase {phase} Depth {depth}: {roots} roots, {leaves} leaves deferred")
            else:
                print("\nLevel breakdown:")
                for level in stats.get("levels", []):
                    print(f"  Depth {level['depth']}: {level['new_entities']} entities, "
                          f"{level['relationships']} relationships")
                    if level.get("errors"):
                        print(f"    Errors: {level['errors']}")

            if args.dry_run:
                print(f"\n[DRY RUN] Would create:")
                print(f"  Entities: {stats.get('entities_found', 0)}")
                print(f"  Relationships: {stats.get('relationships_found', 0)}")
            else:
                print(f"\n{args.backend.title()} Results:")
                print(f"  Entities created/merged: {stats.get('entities_created', 0)}")
                print(f"  Relationships created/merged: {stats.get('relationships_created', 0)}")

                db_stats = stats.get("db_stats", {})
                if db_stats:
                    print(f"\nCurrent {args.backend.title()} Stats:")
                    for key, value in db_stats.items():
                        print(f"    {key}: {value}")

    finally:
        builder.close()


if __name__ == "__main__":
    main()
