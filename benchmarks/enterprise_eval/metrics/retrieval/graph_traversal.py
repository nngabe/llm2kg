"""
Graph Traversal Efficiency Metric.

Custom metric measuring how efficiently the agent traverses the knowledge graph.
"""

import logging
from typing import Any, Dict, Set

from ..base import BaseMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


class GraphTraversalEfficiencyMetric(BaseMetric):
    """Measures efficiency of graph traversal.

    Graph Traversal Efficiency = |relevant ∩ visited| / |visited|

    This metric evaluates whether the agent visits mostly relevant nodes
    rather than wandering through irrelevant parts of the graph.
    """

    layer = EvaluationLayer.RETRIEVAL
    name = "Graph Traversal Efficiency"

    def __init__(self, threshold: float = 0.40):
        super().__init__(threshold)

    def _extract_visited_entities(self, agent_output: AgentOutput) -> Set[str]:
        """Extract entities visited during graph traversal.

        Args:
            agent_output: Agent output with context items

        Returns:
            Set of visited entity names/IDs
        """
        visited = set()

        # Extract from context items
        for item in agent_output.context_items:
            source_type = item.get("source_type", "")
            if source_type in ("entity", "graph", "neo4j"):
                # Try to extract entity identifier
                source_id = item.get("source_id", "")
                if source_id:
                    visited.add(source_id.lower())

                # Also check content for entity names
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                if "entity_name" in metadata:
                    visited.add(metadata["entity_name"].lower())

        # Extract from thought history (tool calls)
        for thought in agent_output.thought_history:
            action = thought.get("action")
            if action:
                tool_name = action.get("tool_name", "")
                args = action.get("arguments", {})

                # Check for entity-related tools
                if tool_name in ("graph_lookup", "entity_resolve", "get_entity"):
                    entity = args.get("entity") or args.get("entity_name") or args.get("name")
                    if entity:
                        visited.add(entity.lower())

                # Check Cypher queries for entity references
                if tool_name == "cypher_query":
                    query = args.get("query", "")
                    # Simple extraction of entity names from WHERE clauses
                    # This is a heuristic - could be improved with proper parsing
                    if "name:" in query.lower() or "name =" in query.lower():
                        # Extract quoted strings as potential entity names
                        import re
                        matches = re.findall(r"['\"]([^'\"]+)['\"]", query)
                        for match in matches:
                            if len(match) > 2:  # Skip short strings
                                visited.add(match.lower())

        return visited

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure graph traversal efficiency.

        Args:
            test_case: Test case with expected entities
            agent_output: Agent output with traversal data

        Returns:
            MetricResult with efficiency score
        """
        # Get expected relevant entities
        relevant = {e.lower() for e in test_case.expected_entities}

        if not relevant:
            return self._create_result(
                score=1.0,
                details={"note": "No expected entities defined for this test case"},
            )

        # Get visited entities
        visited = self._extract_visited_entities(agent_output)

        if not visited:
            return self._create_result(
                score=0.0,
                details={
                    "relevant_count": len(relevant),
                    "visited_count": 0,
                    "intersection_count": 0,
                    "note": "No graph entities were visited",
                },
            )

        # Calculate efficiency
        intersection = relevant & visited
        efficiency = len(intersection) / len(visited)

        # Also calculate recall for reference
        recall = len(intersection) / len(relevant) if relevant else 1.0

        return self._create_result(
            score=efficiency,
            details={
                "relevant_entities": list(relevant),
                "visited_entities": list(visited),
                "intersection": list(intersection),
                "relevant_count": len(relevant),
                "visited_count": len(visited),
                "intersection_count": len(intersection),
                "entity_recall": recall,
            },
        )
