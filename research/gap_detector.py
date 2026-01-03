"""
Knowledge gap detector for identifying areas needing research.

Analyzes the knowledge graph to find:
- Sparse regions with few connections
- Failed queries indicating missing information
- Entities with incomplete descriptions
- Missing relationships between related entities
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class Gap(BaseModel):
    """A knowledge gap identified in the graph."""

    gap_type: str = Field(description="Type of gap: sparse_entity, missing_relation, incomplete_info, failed_query")
    entity_name: Optional[str] = Field(default=None, description="Related entity name")
    description: str = Field(description="Description of the gap")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_queries: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGapDetector:
    """
    Detector for knowledge gaps in the graph.

    Methods:
    - identify_gaps: Main entry point for gap detection
    - find_sparse_entities: Find entities with few connections
    - find_incomplete_entities: Find entities with missing descriptions
    - analyze_query_failures: Analyze failed query patterns
    """

    def __init__(
        self,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_password: str = NEO4J_PASSWORD,
    ):
        """Initialize the gap detector."""
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )

    def close(self):
        """Close database connection."""
        self.driver.close()

    def identify_gaps(
        self,
        max_gaps: int = 20,
        include_sparse: bool = True,
        include_incomplete: bool = True,
        min_importance: float = 0.3,
    ) -> List[Gap]:
        """
        Identify knowledge gaps in the graph.

        Args:
            max_gaps: Maximum number of gaps to return.
            include_sparse: Include sparse entity gaps.
            include_incomplete: Include incomplete entity gaps.
            min_importance: Minimum importance score threshold.

        Returns:
            List of identified gaps sorted by importance.
        """
        gaps = []

        if include_sparse:
            gaps.extend(self.find_sparse_entities(limit=max_gaps // 2))

        if include_incomplete:
            gaps.extend(self.find_incomplete_entities(limit=max_gaps // 2))

        # Filter and sort by importance
        gaps = [g for g in gaps if g.importance_score >= min_importance]
        gaps.sort(key=lambda g: g.importance_score, reverse=True)

        return gaps[:max_gaps]

    def find_sparse_entities(self, limit: int = 10) -> List[Gap]:
        """
        Find entities with few connections.

        These are entities that might need more context or relationships.
        """
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as rel_count
        WHERE rel_count < 3
        RETURN e.name as name, e.ontology_type as type, e.description as description, rel_count
        ORDER BY rel_count ASC
        LIMIT $limit
        """

        gaps = []
        try:
            with self.driver.session() as session:
                result = session.run(query, limit=limit)

                for record in result:
                    name = record["name"]
                    rel_count = record["rel_count"]

                    # Calculate importance based on isolation
                    importance = 1.0 - (rel_count / 3.0)

                    gap = Gap(
                        gap_type="sparse_entity",
                        entity_name=name,
                        description=f"Entity '{name}' has only {rel_count} relationships",
                        importance_score=importance,
                        suggested_queries=[
                            f"{name} relationships",
                            f"{name} connections",
                            f"What is related to {name}",
                        ],
                        metadata={
                            "entity_type": record["type"],
                            "relationship_count": rel_count,
                        },
                    )
                    gaps.append(gap)

        except Exception as e:
            logger.error(f"Error finding sparse entities: {e}")

        return gaps

    def find_incomplete_entities(self, limit: int = 10) -> List[Gap]:
        """
        Find entities with missing or short descriptions.
        """
        query = """
        MATCH (e:Entity)
        WHERE e.description IS NULL OR size(e.description) < 50
        RETURN e.name as name, e.ontology_type as type, e.description as description
        LIMIT $limit
        """

        gaps = []
        try:
            with self.driver.session() as session:
                result = session.run(query, limit=limit)

                for record in result:
                    name = record["name"]
                    desc = record["description"] or ""
                    entity_type = record["type"] or "Entity"

                    # Calculate importance based on description completeness
                    if not desc:
                        importance = 0.9
                    else:
                        importance = 0.7 - (len(desc) / 100.0)
                        importance = max(importance, 0.3)

                    gap = Gap(
                        gap_type="incomplete_info",
                        entity_name=name,
                        description=f"Entity '{name}' ({entity_type}) has incomplete description",
                        importance_score=importance,
                        suggested_queries=[
                            f"What is {name}",
                            f"{name} definition",
                            f"{name} overview",
                        ],
                        metadata={
                            "entity_type": entity_type,
                            "description_length": len(desc),
                        },
                    )
                    gaps.append(gap)

        except Exception as e:
            logger.error(f"Error finding incomplete entities: {e}")

        return gaps

    def find_missing_relationships(
        self,
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Gap]:
        """
        Find potentially missing relationships between related entities.
        """
        # Find entity pairs that share keywords but have no direct relationship
        query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE id(e1) < id(e2)
        AND NOT (e1)-[]-(e2)
        AND any(word IN split(toLower(e1.description), ' ')
                WHERE size(word) > 4 AND toLower(e2.description) CONTAINS word)
        RETURN e1.name as entity1, e2.name as entity2,
               e1.ontology_type as type1, e2.ontology_type as type2
        LIMIT $limit
        """

        gaps = []
        try:
            with self.driver.session() as session:
                result = session.run(query, limit=limit)

                for record in result:
                    e1 = record["entity1"]
                    e2 = record["entity2"]

                    gap = Gap(
                        gap_type="missing_relation",
                        description=f"Potential relationship missing between '{e1}' and '{e2}'",
                        importance_score=0.6,
                        suggested_queries=[
                            f"relationship between {e1} and {e2}",
                            f"{e1} {e2} connection",
                        ],
                        metadata={
                            "entity1": e1,
                            "entity2": e2,
                            "type1": record["type1"],
                            "type2": record["type2"],
                        },
                    )
                    gaps.append(gap)

        except Exception as e:
            logger.error(f"Error finding missing relationships: {e}")

        return gaps

    def get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types in the graph."""
        query = """
        MATCH (e:Entity)
        RETURN e.ontology_type as type, count(*) as count
        ORDER BY count DESC
        """

        distribution = {}
        try:
            with self.driver.session() as session:
                result = session.run(query)
                for record in result:
                    distribution[record["type"] or "Unknown"] = record["count"]
        except Exception as e:
            logger.error(f"Error getting entity distribution: {e}")

        return distribution

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        RETURN count(DISTINCT e) as entity_count,
               count(r) as relationship_count,
               avg(size(coalesce(e.description, ''))) as avg_description_length
        """

        stats = {}
        try:
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                if record:
                    stats = {
                        "entity_count": record["entity_count"],
                        "relationship_count": record["relationship_count"],
                        "avg_description_length": record["avg_description_length"],
                    }
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")

        return stats
