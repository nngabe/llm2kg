"""
Cross-Source Entity Resolution.

Resolves and merges entities across Wikipedia and WebPage sources:
1. Find similar entities using embedding similarity
2. Merge entities while preserving all source URLs
3. Update graph with merged entities

Usage:
    python -m knowledge_graph.entity_resolution --graph test_gev_kg --threshold 0.85
"""

import os
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
import numpy as np

from .models import (
    ExtractedEntity,
    EntityStatus,
    SourceType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")


class CrossSourceEntityResolver:
    """
    Resolve entities across Wikipedia and WebPage sources.

    Uses embedding similarity to find matching entities and merges them
    while preserving all source URLs for provenance.
    """

    def __init__(
        self,
        graph_name: str,
        similarity_threshold: float = 0.85,
        embedding_model: str = "qwen3-embedding:8b",
        falkordb_host: str = FALKORDB_HOST,
        falkordb_port: int = FALKORDB_PORT,
        ollama_host: str = OLLAMA_HOST,
    ):
        """
        Initialize the entity resolver.

        Args:
            graph_name: Name of the FalkorDB graph.
            similarity_threshold: Minimum similarity score for entity matching.
            embedding_model: Ollama embedding model name.
            falkordb_host: FalkorDB host.
            falkordb_port: FalkorDB port.
            ollama_host: Ollama API host.
        """
        self.graph_name = graph_name
        self.threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host

        # Initialize FalkorDB connection
        from falkordb import FalkorDB

        self.client = FalkorDB(host=falkordb_host, port=falkordb_port)
        self.graph = self.client.select_graph(graph_name)

        # Lazy-loaded embedding model
        self._embeddings = None

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

    def get_all_entities(self, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all entities from the graph.

        Args:
            source_type: Optional filter by source type ("wikipedia" or "webpage").

        Returns:
            List of entity dictionaries.
        """
        source_filter = ""
        if source_type:
            source_filter = f"WHERE '{source_type}' IN e.source_types"

        query = f"""
            MATCH (e:Entity)
            {source_filter}
            RETURN e.entity_id as entity_id,
                   e.name as name,
                   e.ontology_type as ontology_type,
                   e.description as description,
                   e.aliases as aliases,
                   e.confidence as confidence,
                   e.source_urls as source_urls,
                   e.source_types as source_types,
                   e.embedding as embedding
        """

        result = self.graph.query(query)
        entities = []

        for row in result.result_set:
            entities.append({
                "entity_id": row[0],
                "name": row[1],
                "ontology_type": row[2],
                "description": row[3],
                "aliases": row[4] or [],
                "confidence": row[5] or 1.0,
                "source_urls": set(row[6]) if row[6] else set(),
                "source_types": set(row[7]) if row[7] else set(),
                "embedding": row[8],
            })

        logger.info(f"Retrieved {len(entities)} entities")
        return entities

    def compute_embeddings(
        self,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compute embeddings for entities that don't have them.

        Args:
            entities: List of entity dictionaries.

        Returns:
            Updated entity list with embeddings.
        """
        embeddings_model = self._get_embeddings()

        # Find entities without embeddings
        entities_without_embeddings = [
            e for e in entities if e.get("embedding") is None
        ]

        if not entities_without_embeddings:
            logger.info("All entities already have embeddings")
            return entities

        logger.info(
            f"Computing embeddings for {len(entities_without_embeddings)} entities"
        )

        # Compute embeddings for names
        names = [e["name"] for e in entities_without_embeddings]
        try:
            vectors = embeddings_model.embed_documents(names)
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return entities

        # Update entities with embeddings
        for entity, vector in zip(entities_without_embeddings, vectors):
            entity["embedding"] = vector

            # Also update in graph
            try:
                embed_str = str(vector)
                self.graph.query(
                    f"""
                    MATCH (e:Entity {{entity_id: '{entity['entity_id']}'}})
                    SET e.embedding = {embed_str}
                """
                )
            except Exception as e:
                logger.warning(f"Failed to save embedding: {e}")

        return entities

    def find_matches(
        self,
        entity: Dict[str, Any],
        all_entities: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar entities for a given entity.

        Args:
            entity: The entity to find matches for.
            all_entities: List of all entities to search.

        Returns:
            List of (entity, similarity_score) tuples above threshold.
        """
        if entity.get("embedding") is None:
            return []

        entity_vector = np.array(entity["embedding"])
        matches = []

        for other in all_entities:
            if other["entity_id"] == entity["entity_id"]:
                continue

            if other.get("embedding") is None:
                continue

            other_vector = np.array(other["embedding"])

            # Cosine similarity
            similarity = np.dot(entity_vector, other_vector) / (
                np.linalg.norm(entity_vector) * np.linalg.norm(other_vector) + 1e-8
            )

            if similarity >= self.threshold:
                matches.append((other, float(similarity)))

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def find_cross_source_matches(
        self,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Find entities that appear in multiple sources (Wikipedia + WebPage).

        Args:
            entities: List of all entities.

        Returns:
            List of match groups (entities that should be merged).
        """
        # Separate entities by source type
        wikipedia_entities = [
            e for e in entities if "wikipedia" in e.get("source_types", set())
        ]
        webpage_entities = [
            e for e in entities if "webpage" in e.get("source_types", set())
        ]

        logger.info(
            f"Cross-source matching: {len(wikipedia_entities)} Wikipedia, "
            f"{len(webpage_entities)} WebPage entities"
        )

        match_groups = []

        # For each Wikipedia entity, find matching WebPage entities
        for wiki_entity in wikipedia_entities:
            matches = self.find_matches(wiki_entity, webpage_entities)
            if matches:
                match_groups.append({
                    "primary": wiki_entity,
                    "matches": matches,
                    "source": "wikipedia",
                })

        # Also check reverse (WebPage entities not matched)
        matched_webpage_ids = set()
        for group in match_groups:
            for match, _ in group["matches"]:
                matched_webpage_ids.add(match["entity_id"])

        # Find WebPage entities with no Wikipedia match
        unmatched_webpage = [
            e for e in webpage_entities if e["entity_id"] not in matched_webpage_ids
        ]

        # For unmatched WebPage entities, find other WebPage matches
        for webpage_entity in unmatched_webpage:
            matches = self.find_matches(webpage_entity, webpage_entities)
            if matches:
                match_groups.append({
                    "primary": webpage_entity,
                    "matches": matches,
                    "source": "webpage",
                })

        logger.info(f"Found {len(match_groups)} match groups")
        return match_groups

    def merge_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge multiple entities into one, preserving all URLs.

        Args:
            entities: List of entity dictionaries to merge.

        Returns:
            Merged entity dictionary.
        """
        if not entities:
            return {}

        # Use first entity as base
        merged = entities[0].copy()

        # Merge all others
        for other in entities[1:]:
            # Add name to aliases
            if other["name"] != merged["name"]:
                merged.setdefault("aliases", [])
                if other["name"] not in merged["aliases"]:
                    merged["aliases"].append(other["name"])

            # Merge aliases
            for alias in other.get("aliases", []):
                if alias not in merged.get("aliases", []) and alias != merged["name"]:
                    merged.setdefault("aliases", [])
                    merged["aliases"].append(alias)

            # Union source URLs
            merged["source_urls"] = merged.get("source_urls", set()) | other.get(
                "source_urls", set()
            )

            # Union source types
            merged["source_types"] = merged.get("source_types", set()) | other.get(
                "source_types", set()
            )

            # Combine descriptions
            if other.get("description"):
                if not merged.get("description"):
                    merged["description"] = other["description"]
                elif other["description"] not in merged["description"]:
                    merged["description"] = (
                        f"{merged['description']} {other['description']}"
                    )

            # Use max confidence
            merged["confidence"] = max(
                merged.get("confidence", 1.0), other.get("confidence", 1.0)
            )

        return merged

    def update_entity_in_graph(self, entity: Dict[str, Any]) -> bool:
        """
        Update an entity in the graph with merged information.

        Args:
            entity: Merged entity dictionary.

        Returns:
            True if updated successfully.
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            entity_id = entity["entity_id"]

            # Escape strings
            name = entity["name"].replace("'", "\\'")
            desc = (entity.get("description") or "").replace("'", "\\'")
            aliases_str = str(list(entity.get("aliases", [])))
            source_urls_str = str(list(entity.get("source_urls", [])))
            source_types_str = str(list(entity.get("source_types", [])))

            self.graph.query(
                f"""
                MATCH (e:Entity {{entity_id: '{entity_id}'}})
                SET e.name = '{name}',
                    e.description = '{desc}',
                    e.aliases = {aliases_str},
                    e.source_urls = {source_urls_str},
                    e.source_types = {source_types_str},
                    e.confidence = {entity.get('confidence', 1.0)},
                    e.status = 'merged',
                    e.merged_at = '{timestamp}'
            """
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to update entity {entity.get('entity_id')}: {e}")
            return False

    def delete_merged_entities(
        self,
        primary_id: str,
        merged_ids: List[str],
    ) -> int:
        """
        Delete entities that were merged into the primary entity.

        Transfers relationships to the primary entity before deletion.

        Args:
            primary_id: ID of the primary (kept) entity.
            merged_ids: IDs of entities to delete.

        Returns:
            Number of entities deleted.
        """
        deleted = 0

        for merged_id in merged_ids:
            if merged_id == primary_id:
                continue

            try:
                # Transfer EXTRACTED_FROM relationships
                self.graph.query(
                    f"""
                    MATCH (old:Entity {{entity_id: '{merged_id}'}})-[r:EXTRACTED_FROM]->(c:DocumentChunk)
                    MATCH (primary:Entity {{entity_id: '{primary_id}'}})
                    MERGE (primary)-[:EXTRACTED_FROM]->(c)
                """
                )

                # Transfer other relationships (source side)
                self.graph.query(
                    f"""
                    MATCH (old:Entity {{entity_id: '{merged_id}'}})-[r]->(target)
                    WHERE type(r) <> 'EXTRACTED_FROM'
                    MATCH (primary:Entity {{entity_id: '{primary_id}'}})
                    CALL {{
                        WITH old, r, target, primary
                        CREATE (primary)-[nr:RELATED_TO]->(target)
                        SET nr = properties(r)
                        RETURN 1 as cnt
                    }}
                    RETURN count(*) as cnt
                """
                )

                # Transfer relationships (target side)
                self.graph.query(
                    f"""
                    MATCH (source)-[r]->(old:Entity {{entity_id: '{merged_id}'}})
                    MATCH (primary:Entity {{entity_id: '{primary_id}'}})
                    CALL {{
                        WITH source, r, old, primary
                        CREATE (source)-[nr:RELATED_TO]->(primary)
                        SET nr = properties(r)
                        RETURN 1 as cnt
                    }}
                    RETURN count(*) as cnt
                """
                )

                # Delete the old entity and its relationships
                self.graph.query(
                    f"""
                    MATCH (e:Entity {{entity_id: '{merged_id}'}})
                    DETACH DELETE e
                """
                )

                deleted += 1
                logger.debug(f"Deleted merged entity: {merged_id}")

            except Exception as e:
                logger.warning(f"Failed to delete entity {merged_id}: {e}")

        return deleted

    def resolve_all(
        self,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run entity resolution across entire graph.

        Args:
            dry_run: If True, don't modify the graph.

        Returns:
            Dictionary with resolution statistics.
        """
        start_time = time.time()

        # Get all entities
        entities = self.get_all_entities()
        if not entities:
            return {"error": "No entities found in graph"}

        # Compute embeddings for entities without them
        entities = self.compute_embeddings(entities)

        # Find match groups
        match_groups = self.find_cross_source_matches(entities)

        stats = {
            "total_entities": len(entities),
            "match_groups": len(match_groups),
            "entities_merged": 0,
            "entities_deleted": 0,
            "merge_details": [],
        }

        # Process each match group
        for group in match_groups:
            primary = group["primary"]
            matches = group["matches"]

            if not matches:
                continue

            # Collect all entities to merge
            all_to_merge = [primary] + [m[0] for m in matches]

            # Merge entities
            merged = self.merge_entities(all_to_merge)

            merge_detail = {
                "primary_id": primary["entity_id"],
                "primary_name": primary["name"],
                "merged_count": len(matches),
                "merged_names": [m[0]["name"] for m in matches],
                "source_urls": list(merged.get("source_urls", [])),
            }
            stats["merge_details"].append(merge_detail)

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would merge: {primary['name']} + "
                    f"{[m[0]['name'] for m in matches]}"
                )
                continue

            # Update primary entity with merged info
            if self.update_entity_in_graph(merged):
                stats["entities_merged"] += 1

                # Delete merged entities
                merged_ids = [m[0]["entity_id"] for m in matches]
                deleted = self.delete_merged_entities(
                    primary["entity_id"], merged_ids
                )
                stats["entities_deleted"] += deleted

        stats["processing_time_seconds"] = time.time() - start_time
        stats["graph_name"] = self.graph_name

        return stats

    def get_multi_source_entities(self) -> List[Dict[str, Any]]:
        """
        Get entities that appear in multiple sources.

        Returns:
            List of entities with both Wikipedia and WebPage sources.
        """
        query = """
            MATCH (e:Entity)
            WHERE 'wikipedia' IN e.source_types AND 'webpage' IN e.source_types
            RETURN e.entity_id as entity_id,
                   e.name as name,
                   e.ontology_type as ontology_type,
                   e.source_urls as source_urls,
                   e.source_types as source_types
            ORDER BY size(e.source_urls) DESC
        """

        result = self.graph.query(query)
        entities = []

        for row in result.result_set:
            entities.append({
                "entity_id": row[0],
                "name": row[1],
                "ontology_type": row[2],
                "source_urls": row[3] or [],
                "source_types": row[4] or [],
            })

        return entities

    def get_stats(self) -> Dict[str, Any]:
        """Get entity resolution statistics."""
        stats = {}

        try:
            # Total entities
            result = self.graph.query("MATCH (e:Entity) RETURN count(e) as count")
            stats["total_entities"] = (
                result.result_set[0][0] if result.result_set else 0
            )

            # Entities by source type
            result = self.graph.query(
                """
                MATCH (e:Entity)
                UNWIND e.source_types as source_type
                RETURN source_type, count(DISTINCT e) as count
            """
            )
            stats["entities_by_source_type"] = {}
            for row in result.result_set:
                stats["entities_by_source_type"][row[0] or "unknown"] = row[1]

            # Multi-source entities
            result = self.graph.query(
                """
                MATCH (e:Entity)
                WHERE size(e.source_types) > 1
                RETURN count(e) as count
            """
            )
            stats["multi_source_entities"] = (
                result.result_set[0][0] if result.result_set else 0
            )

            # Merged entities
            result = self.graph.query(
                """
                MATCH (e:Entity)
                WHERE e.status = 'merged'
                RETURN count(e) as count
            """
            )
            stats["merged_entities"] = (
                result.result_set[0][0] if result.result_set else 0
            )

            # Entities with embeddings
            result = self.graph.query(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN count(e) as count
            """
            )
            stats["entities_with_embeddings"] = (
                result.result_set[0][0] if result.result_set else 0
            )

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats["error"] = str(e)

        return stats


def main():
    """CLI entry point for entity resolution."""
    parser = argparse.ArgumentParser(
        description="Resolve entities across Wikipedia and WebPage sources"
    )

    parser.add_argument(
        "--graph",
        type=str,
        default="wikidata",
        help="FalkorDB graph name (default: wikidata)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for matching (default: 0.85)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without making changes",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show current statistics",
    )
    parser.add_argument(
        "--list-multi-source",
        action="store_true",
        help="List entities that appear in multiple sources",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize resolver
    resolver = CrossSourceEntityResolver(
        graph_name=args.graph,
        similarity_threshold=args.threshold,
    )

    # Stats-only mode
    if args.stats_only:
        stats = resolver.get_stats()
        print(f"\nEntity Resolution Statistics ({args.graph}):")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        return

    # List multi-source entities
    if args.list_multi_source:
        entities = resolver.get_multi_source_entities()
        print(f"\nMulti-Source Entities ({args.graph}):")
        print("-" * 60)
        for e in entities[:20]:  # Show first 20
            sources = ", ".join(e.get("source_types", []))
            urls = len(e.get("source_urls", []))
            print(f"  {e['name']:<40} [{sources}] ({urls} URLs)")
        if len(entities) > 20:
            print(f"  ... and {len(entities) - 20} more")
        return

    # Run resolution
    print("\n" + "=" * 60)
    print("CROSS-SOURCE ENTITY RESOLUTION")
    print("=" * 60)
    print(f"Graph:           {args.graph}")
    print(f"Threshold:       {args.threshold}")
    print(f"Dry Run:         {args.dry_run}")
    print("=" * 60)

    stats = resolver.resolve_all(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("RESOLUTION RESULTS")
    print("=" * 60)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return

    print(f"Total entities:    {stats['total_entities']}")
    print(f"Match groups:      {stats['match_groups']}")
    print(f"Entities merged:   {stats['entities_merged']}")
    print(f"Entities deleted:  {stats['entities_deleted']}")
    print(f"Processing time:   {stats['processing_time_seconds']:.2f}s")

    if stats.get("merge_details"):
        print("\nMerge Details:")
        for detail in stats["merge_details"][:10]:
            print(
                f"  {detail['primary_name']} <- {detail['merged_names']} "
                f"({len(detail['source_urls'])} URLs)"
            )
        if len(stats["merge_details"]) > 10:
            print(f"  ... and {len(stats['merge_details']) - 10} more merges")


if __name__ == "__main__":
    main()
