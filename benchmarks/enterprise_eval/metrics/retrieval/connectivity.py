"""
Subgraph Connectivity Metric.

Custom metric measuring how well connected the retrieved context is.
"""

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base import BaseMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


class SubgraphConnectivityMetric(BaseMetric):
    """Measures connectivity of retrieved subgraph.

    Subgraph Connectivity = connected_pairs / total_pairs

    This metric evaluates whether the retrieved context items form a
    coherent, connected subgraph rather than disconnected fragments.
    """

    layer = EvaluationLayer.RETRIEVAL
    name = "Subgraph Connectivity"

    def __init__(self, threshold: float = 0.50, max_path_length: int = 3):
        super().__init__(threshold)
        self.max_path_length = max_path_length

    def _extract_entities(self, agent_output: AgentOutput) -> List[str]:
        """Extract unique entities from agent output.

        Args:
            agent_output: Agent output with context

        Returns:
            List of entity names
        """
        entities = set()

        for item in agent_output.context_items:
            source_type = item.get("source_type", "")
            if source_type in ("entity", "graph", "neo4j"):
                source_id = item.get("source_id", "")
                if source_id:
                    entities.add(source_id)

                metadata = item.get("metadata", {})
                if "entity_name" in metadata:
                    entities.add(metadata["entity_name"])

        return list(entities)

    def _check_connectivity_neo4j(
        self,
        entity_pairs: List[Tuple[str, str]],
        neo4j_driver,
    ) -> Dict[Tuple[str, str], bool]:
        """Check connectivity between entity pairs using Neo4j.

        Args:
            entity_pairs: Pairs of entities to check
            neo4j_driver: Neo4j driver instance

        Returns:
            Dictionary mapping pairs to connectivity status
        """
        results = {}

        query = """
        MATCH (a:Entity {name: $e1}), (b:Entity {name: $e2})
        MATCH path = shortestPath((a)-[*1..$max_length]-(b))
        RETURN count(path) > 0 as connected, length(path) as path_length
        """

        with neo4j_driver.session() as session:
            for e1, e2 in entity_pairs:
                try:
                    result = session.run(
                        query,
                        e1=e1,
                        e2=e2,
                        max_length=self.max_path_length,
                    )
                    record = result.single()
                    if record:
                        results[(e1, e2)] = record["connected"]
                    else:
                        results[(e1, e2)] = False
                except Exception as e:
                    logger.warning(f"Neo4j query failed for ({e1}, {e2}): {e}")
                    results[(e1, e2)] = False

        return results

    def _check_connectivity_fallback(
        self,
        entity_pairs: List[Tuple[str, str]],
        agent_output: AgentOutput,
    ) -> Dict[Tuple[str, str], bool]:
        """Fallback connectivity check using context content.

        If an entity pair appears together in any context item,
        consider them connected.

        Args:
            entity_pairs: Pairs of entities to check
            agent_output: Agent output with context

        Returns:
            Dictionary mapping pairs to connectivity status
        """
        results = {}

        # Build content corpus for checking co-occurrence
        contexts = [
            item.get("content", "").lower()
            for item in agent_output.context_items
        ]

        for e1, e2 in entity_pairs:
            e1_lower = e1.lower()
            e2_lower = e2.lower()

            # Check if both entities appear in any single context
            connected = any(
                e1_lower in ctx and e2_lower in ctx
                for ctx in contexts
            )
            results[(e1, e2)] = connected

        return results

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        neo4j_driver: Optional[Any] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure subgraph connectivity.

        Args:
            test_case: Test case
            agent_output: Agent output with retrieved context
            neo4j_driver: Optional Neo4j driver for connectivity checks

        Returns:
            MetricResult with connectivity score
        """
        # Extract entities from retrieved context
        entities = self._extract_entities(agent_output)

        if len(entities) < 2:
            return self._create_result(
                score=1.0,
                details={
                    "entity_count": len(entities),
                    "note": "Less than 2 entities, connectivity trivially satisfied",
                },
            )

        # Generate all pairs
        entity_pairs = list(combinations(entities, 2))
        total_pairs = len(entity_pairs)

        # Check connectivity
        if neo4j_driver:
            connectivity = self._check_connectivity_neo4j(entity_pairs, neo4j_driver)
        else:
            logger.info("No Neo4j driver available, using fallback connectivity check")
            connectivity = self._check_connectivity_fallback(entity_pairs, agent_output)

        # Calculate score
        connected_pairs = sum(1 for connected in connectivity.values() if connected)
        score = connected_pairs / total_pairs if total_pairs > 0 else 1.0

        # Detailed breakdown
        connected_list = [
            f"{e1} <-> {e2}"
            for (e1, e2), connected in connectivity.items()
            if connected
        ]
        disconnected_list = [
            f"{e1} <-> {e2}"
            for (e1, e2), connected in connectivity.items()
            if not connected
        ]

        return self._create_result(
            score=score,
            details={
                "entities": entities,
                "entity_count": len(entities),
                "total_pairs": total_pairs,
                "connected_pairs": connected_pairs,
                "connected": connected_list[:10],  # Limit for readability
                "disconnected": disconnected_list[:10],
                "method": "neo4j" if neo4j_driver else "fallback",
            },
        )
