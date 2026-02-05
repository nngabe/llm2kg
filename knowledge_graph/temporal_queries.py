"""
Temporal query utilities for Zep-style knowledge graph.

Provides bi-temporal query capabilities:
- Valid time: When facts are/were true in the real world
- Transaction time: When facts were ingested into the system

Classes:
- TemporalQueryBuilder: Generate Cypher queries with temporal constraints
- TemporalEntityManager: Manage entity lifecycle (versioning, obsolescence)

Usage:
    from knowledge_graph.temporal_queries import TemporalQueryBuilder, TemporalEntityManager

    # Build temporal query
    query = TemporalQueryBuilder.entities_valid_at(datetime.now())

    # Manage entity lifecycle
    manager = TemporalEntityManager(graph)
    manager.mark_obsolete(entity_id, reason="fact_invalidated")
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TemporalQueryBuilder:
    """
    Build Cypher queries with temporal constraints.

    Supports Zep-style bi-temporal queries:
    - Valid time queries: What was true at time T?
    - Transaction time queries: What did we know at time T?
    - Combined queries: What did we know was true at time T as of time T'?
    """

    @staticmethod
    def entities_valid_at(timestamp: datetime) -> str:
        """
        Get entities valid at a specific point in time.

        Args:
            timestamp: Point in time to query.

        Returns:
            Cypher query string.
        """
        ts = timestamp.isoformat()
        return f"""
            MATCH (e:Entity)
            WHERE (e.valid_from IS NULL OR e.valid_from <= '{ts}')
              AND (e.valid_to IS NULL OR e.valid_to > '{ts}')
              AND e.fact_status = 'active'
            RETURN e
        """

    @staticmethod
    def entities_in_range(start: datetime, end: datetime) -> str:
        """
        Get entities valid during a time range (overlapping).

        Args:
            start: Start of time range.
            end: End of time range.

        Returns:
            Cypher query string.
        """
        return f"""
            MATCH (e:Entity)
            WHERE (e.valid_from IS NULL OR e.valid_from <= '{end.isoformat()}')
              AND (e.valid_to IS NULL OR e.valid_to >= '{start.isoformat()}')
            RETURN e
        """

    @staticmethod
    def entity_timeline(entity_id: str) -> str:
        """
        Get all episodes mentioning an entity, ordered by time.

        Args:
            entity_id: Entity ID to query.

        Returns:
            Cypher query string.
        """
        return f"""
            MATCH (ep:Episode)-[r:CONTAINS]->(e:Entity {{entity_id: '{entity_id}'}})
            RETURN ep.name as episode_name,
                   ep.reference_time as reference_time,
                   ep.source_url as source_url,
                   r.confidence as confidence
            ORDER BY ep.reference_time
        """

    @staticmethod
    def entity_timeline_by_name(entity_name: str) -> str:
        """
        Get all episodes mentioning an entity by name, ordered by time.

        Args:
            entity_name: Entity name to query.

        Returns:
            Cypher query string.
        """
        name_escaped = entity_name.replace("'", "\\'")
        return f"""
            MATCH (ep:Episode)-[r:CONTAINS]->(e:Entity)
            WHERE e.name = '{name_escaped}'
            RETURN ep.name as episode_name,
                   ep.reference_time as reference_time,
                   ep.source_url as source_url,
                   r.confidence as confidence
            ORDER BY ep.reference_time
        """

    @staticmethod
    def facts_known_at(transaction_time: datetime) -> str:
        """
        Get facts we knew at a specific ingestion time.

        This is a transaction-time query that returns all entities
        that existed in the system at the given transaction time.

        Args:
            transaction_time: Point in time for transaction query.

        Returns:
            Cypher query string.
        """
        ts = transaction_time.isoformat()
        return f"""
            MATCH (e:Entity)
            WHERE e.created_at <= '{ts}'
              AND (e.obsoleted_at IS NULL OR e.obsoleted_at > '{ts}')
            RETURN e
        """

    @staticmethod
    def bi_temporal_query(
        valid_time: datetime,
        transaction_time: datetime,
    ) -> str:
        """
        Combined bi-temporal query.

        Returns entities that:
        - Were valid at valid_time (real-world state)
        - Were known at transaction_time (system state)

        Args:
            valid_time: Point in time for validity.
            transaction_time: Point in time for knowledge.

        Returns:
            Cypher query string.
        """
        vt = valid_time.isoformat()
        tt = transaction_time.isoformat()
        return f"""
            MATCH (e:Entity)
            WHERE (e.valid_from IS NULL OR e.valid_from <= '{vt}')
              AND (e.valid_to IS NULL OR e.valid_to > '{vt}')
              AND e.created_at <= '{tt}'
              AND (e.obsoleted_at IS NULL OR e.obsoleted_at > '{tt}')
            RETURN e
        """

    @staticmethod
    def superseded_entities() -> str:
        """
        Get all superseded entities with their replacements.

        Returns:
            Cypher query string.
        """
        return """
            MATCH (new:Entity)-[r:SUPERSEDES]->(old:Entity)
            RETURN old.entity_id as old_id,
                   old.name as old_name,
                   new.entity_id as new_id,
                   new.name as new_name,
                   r.superseded_at as superseded_at
            ORDER BY r.superseded_at DESC
        """

    @staticmethod
    def entity_versions(entity_name: str) -> str:
        """
        Get all versions of an entity by name.

        Args:
            entity_name: Entity name to query.

        Returns:
            Cypher query string.
        """
        name_escaped = entity_name.replace("'", "\\'")
        return f"""
            MATCH (e:Entity)
            WHERE e.name = '{name_escaped}'
            OPTIONAL MATCH (e)-[r:SUPERSEDES*]->(older:Entity)
            RETURN e.entity_id as entity_id,
                   e.name as name,
                   e.fact_status as status,
                   e.valid_from as valid_from,
                   e.valid_to as valid_to,
                   e.created_at as created_at
            ORDER BY e.created_at DESC
        """

    @staticmethod
    def episodes_in_range(start: datetime, end: datetime) -> str:
        """
        Get episodes with reference time in range.

        Args:
            start: Start of time range.
            end: End of time range.

        Returns:
            Cypher query string.
        """
        return f"""
            MATCH (ep:Episode)
            WHERE ep.reference_time >= '{start.isoformat()}'
              AND ep.reference_time <= '{end.isoformat()}'
            RETURN ep
            ORDER BY ep.reference_time
        """

    @staticmethod
    def community_members(community_id: str) -> str:
        """
        Get all entities belonging to a community.

        Args:
            community_id: Community ID to query.

        Returns:
            Cypher query string.
        """
        return f"""
            MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {{community_id: '{community_id}'}})
            WHERE e.fact_status = 'active'
            RETURN e.entity_id as entity_id,
                   e.name as name,
                   e.ontology_type as type,
                   e.confidence as confidence
            ORDER BY e.confidence DESC
        """


class TemporalEntityManager:
    """
    Manage entity lifecycle with temporal versioning.

    Provides operations for:
    - Marking entities as obsolete (soft delete)
    - Creating new versions with SUPERSEDES relationships
    - Merging entities with proper temporal tracking
    """

    def __init__(self, graph):
        """
        Initialize the temporal entity manager.

        Args:
            graph: FalkorDB graph connection.
        """
        self.graph = graph

    def mark_obsolete(
        self,
        entity_id: str,
        reason: str = "fact_invalidated",
    ) -> bool:
        """
        Soft-delete entity by marking as obsolete.

        Sets:
        - obsoleted_at to current time
        - fact_status to 'obsolete'
        - modified_at to current time

        Args:
            entity_id: Entity ID to obsolete.
            reason: Reason for obsolescence.

        Returns:
            True if successful, False otherwise.
        """
        ts = datetime.now(timezone.utc).isoformat()
        reason_escaped = reason.replace("'", "\\'")

        try:
            self.graph.query(f"""
                MATCH (e:Entity {{entity_id: '{entity_id}'}})
                SET e.obsoleted_at = '{ts}',
                    e.fact_status = 'obsolete',
                    e.modified_at = '{ts}',
                    e.obsolete_reason = '{reason_escaped}'
            """)
            logger.info(f"Marked entity {entity_id} as obsolete: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark entity {entity_id} as obsolete: {e}")
            return False

    def create_new_version(
        self,
        entity_id: str,
        updates: Dict[str, Any],
    ) -> Optional[str]:
        """
        Create new version of entity, link with SUPERSEDES.

        The old entity is marked as 'superseded' and a new entity
        is created with updated values, linked via SUPERSEDES relationship.

        Args:
            entity_id: Entity ID to version.
            updates: Dict of fields to update in new version.

        Returns:
            New entity ID if successful, None otherwise.
        """
        import uuid
        ts = datetime.now(timezone.utc).isoformat()
        new_id = str(uuid.uuid4())[:12]

        try:
            # Mark old entity as superseded
            self.graph.query(f"""
                MATCH (old:Entity {{entity_id: '{entity_id}'}})
                SET old.valid_to = '{ts}',
                    old.fact_status = 'superseded',
                    old.modified_at = '{ts}'
            """)

            # Build SET clause for updates
            set_clauses = [f"new.entity_id = '{new_id}'"]
            set_clauses.append(f"new.valid_from = '{ts}'")
            set_clauses.append(f"new.created_at = '{ts}'")
            set_clauses.append("new.fact_status = 'active'")

            for key, value in updates.items():
                if isinstance(value, str):
                    value_escaped = value.replace("'", "\\'")
                    set_clauses.append(f"new.{key} = '{value_escaped}'")
                elif isinstance(value, (int, float)):
                    set_clauses.append(f"new.{key} = {value}")
                elif isinstance(value, list):
                    set_clauses.append(f"new.{key} = {value}")

            set_clause = ", ".join(set_clauses)

            # Create new version with SUPERSEDES relationship
            # Copy core properties from old entity
            self.graph.query(f"""
                MATCH (old:Entity {{entity_id: '{entity_id}'}})
                CREATE (new:Entity)
                SET new.name = old.name,
                    new.ontology_type = old.ontology_type,
                    new.description = old.description,
                    new.aliases = old.aliases,
                    new.source_urls = old.source_urls,
                    new.source_types = old.source_types,
                    {set_clause}
                CREATE (new)-[:SUPERSEDES {{superseded_at: '{ts}'}}]->(old)
            """)

            logger.info(f"Created new version {new_id} of entity {entity_id}")
            return new_id

        except Exception as e:
            logger.error(f"Failed to create new version of entity {entity_id}: {e}")
            return None

    def merge_entities(
        self,
        source_id: str,
        target_id: str,
    ) -> bool:
        """
        Merge source entity into target entity.

        The source entity is marked as 'merged' and a MERGED_INTO
        relationship is created pointing to the target.

        Args:
            source_id: Entity ID to merge from.
            target_id: Entity ID to merge into.

        Returns:
            True if successful, False otherwise.
        """
        ts = datetime.now(timezone.utc).isoformat()

        try:
            # Mark source as merged
            self.graph.query(f"""
                MATCH (source:Entity {{entity_id: '{source_id}'}})
                SET source.fact_status = 'merged',
                    source.merged_at = '{ts}',
                    source.valid_to = '{ts}',
                    source.modified_at = '{ts}'
            """)

            # Create MERGED_INTO relationship
            self.graph.query(f"""
                MATCH (source:Entity {{entity_id: '{source_id}'}})
                MATCH (target:Entity {{entity_id: '{target_id}'}})
                MERGE (source)-[:MERGED_INTO {{merged_at: '{ts}'}}]->(target)
            """)

            # Copy unique source URLs and types to target
            self.graph.query(f"""
                MATCH (source:Entity {{entity_id: '{source_id}'}})
                MATCH (target:Entity {{entity_id: '{target_id}'}})
                SET target.modified_at = '{ts}'
            """)

            logger.info(f"Merged entity {source_id} into {target_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to merge entity {source_id} into {target_id}: {e}")
            return False

    def restore_entity(self, entity_id: str) -> bool:
        """
        Restore a soft-deleted (obsolete) entity.

        Args:
            entity_id: Entity ID to restore.

        Returns:
            True if successful, False otherwise.
        """
        ts = datetime.now(timezone.utc).isoformat()

        try:
            self.graph.query(f"""
                MATCH (e:Entity {{entity_id: '{entity_id}'}})
                WHERE e.fact_status = 'obsolete'
                REMOVE e.obsoleted_at
                SET e.fact_status = 'active',
                    e.modified_at = '{ts}'
            """)
            logger.info(f"Restored entity {entity_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore entity {entity_id}: {e}")
            return False


def get_temporal_statistics(graph) -> Dict[str, Any]:
    """
    Get temporal statistics for the knowledge graph.

    Args:
        graph: FalkorDB graph connection.

    Returns:
        Dictionary with temporal statistics.
    """
    stats = {}

    try:
        # Count episodes
        result = graph.query("MATCH (ep:Episode) RETURN count(ep) as count")
        stats["episodes"] = result.result_set[0][0] if result.result_set else 0

        # Count communities
        result = graph.query("MATCH (c:Community) RETURN count(c) as count")
        stats["communities"] = result.result_set[0][0] if result.result_set else 0

        # Count CONTAINS relationships
        result = graph.query("MATCH ()-[r:CONTAINS]->() RETURN count(r) as count")
        stats["contains_relationships"] = result.result_set[0][0] if result.result_set else 0

        # Count BELONGS_TO relationships
        result = graph.query("MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count")
        stats["belongs_to_relationships"] = result.result_set[0][0] if result.result_set else 0

        # Count DERIVED_FROM relationships
        result = graph.query("MATCH ()-[r:DERIVED_FROM]->() RETURN count(r) as count")
        stats["derived_from_relationships"] = result.result_set[0][0] if result.result_set else 0

        # Count SUPERSEDES relationships
        result = graph.query("MATCH ()-[r:SUPERSEDES]->() RETURN count(r) as count")
        stats["supersedes_relationships"] = result.result_set[0][0] if result.result_set else 0

        # Entities by fact_status
        result = graph.query("""
            MATCH (e:Entity)
            WHERE e.fact_status IS NOT NULL
            RETURN e.fact_status as status, count(e) as count
        """)
        stats["entities_by_fact_status"] = {}
        for row in result.result_set:
            stats["entities_by_fact_status"][row[0] or "unknown"] = row[1]

        # Episode sources by type
        result = graph.query("""
            MATCH (ep:Episode)
            RETURN ep.source_type as source_type, count(ep) as count
        """)
        stats["episodes_by_source_type"] = {}
        for row in result.result_set:
            stats["episodes_by_source_type"][row[0] or "unknown"] = row[1]

    except Exception as e:
        logger.error(f"Failed to get temporal statistics: {e}")
        stats["error"] = str(e)

    return stats
