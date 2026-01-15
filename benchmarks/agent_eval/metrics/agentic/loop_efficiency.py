"""
Loop Efficiency Metric.

Formula-based metric measuring how efficiently the agent reaches an answer.
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


class LoopEfficiencyMetric(BaseMetric):
    """Measures loop efficiency of the agent.

    Loop Efficiency = min(minimum_steps / actual_steps, 1.0)

    This metric evaluates whether the agent reaches the answer
    in a reasonable number of steps without unnecessary iterations.
    """

    layer = EvaluationLayer.AGENTIC
    name = "Loop Efficiency"

    def __init__(self, threshold: float = 0.60):
        super().__init__(threshold)

    def _estimate_minimum_steps(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
    ) -> int:
        """Estimate minimum steps needed to answer the question.

        If test_case.minimum_steps is provided, use that.
        Otherwise, estimate based on:
        - Number of expected entities to retrieve
        - Number of relationship queries needed
        - Planning step (1)
        - Final answer step (1)

        Args:
            test_case: Test case with optional minimum_steps
            agent_output: Agent output (for retrieval plan info)

        Returns:
            Estimated minimum steps
        """
        if test_case.minimum_steps is not None:
            return test_case.minimum_steps

        # Base: planning + answer
        min_steps = 2

        # Add steps for entities
        if test_case.expected_entities:
            # Assume batch retrieval, so ceiling of entities / 3
            min_steps += max(1, len(test_case.expected_entities) // 3)

        # Add steps for relationships
        if test_case.expected_relationships:
            min_steps += max(1, len(test_case.expected_relationships) // 2)

        # Check retrieval plan if available
        if agent_output.retrieval_plan:
            plan = agent_output.retrieval_plan
            entity_targets = plan.get("entity_targets", [])
            rel_queries = plan.get("relationship_queries", [])

            # Use plan info if more specific
            plan_steps = 2  # planning + answer
            if entity_targets:
                plan_steps += max(1, len(entity_targets) // 3)
            if rel_queries:
                plan_steps += max(1, len(rel_queries) // 2)

            min_steps = max(min_steps, plan_steps)

        return max(min_steps, 1)

    def _count_actual_steps(self, agent_output: AgentOutput) -> int:
        """Count actual steps taken by the agent.

        Steps include:
        - Tool calls
        - Thinking steps (if distinct from tool calls)
        - Iteration count

        Args:
            agent_output: Agent output with thought history

        Returns:
            Number of actual steps
        """
        # Use iteration count if available
        if agent_output.iteration_count > 0:
            return agent_output.iteration_count

        # Otherwise count from thought history
        thought_history = agent_output.thought_history

        if not thought_history:
            return 1  # At least one step to generate answer

        # Count distinct steps
        tool_calls = sum(
            1 for t in thought_history
            if t.get("action") and t["action"].get("tool_name")
        )

        # Add 1 for final answer generation
        return tool_calls + 1

    def _identify_redundant_steps(
        self,
        thought_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify potentially redundant steps.

        Args:
            thought_history: Agent's thought history

        Returns:
            List of redundant steps with reasons
        """
        redundant = []
        seen_queries = set()
        seen_entities = set()

        for i, thought in enumerate(thought_history):
            action = thought.get("action")
            if not action:
                continue

            tool_name = action.get("tool_name", "")
            args = action.get("arguments", {})

            # Check for duplicate queries
            query_key = f"{tool_name}:{str(sorted(args.items()))}"
            if query_key in seen_queries:
                redundant.append({
                    "step": i,
                    "tool": tool_name,
                    "reason": "Duplicate query",
                })
            seen_queries.add(query_key)

            # Check for re-fetching same entity
            if tool_name in ("graph_lookup", "get_entity_details"):
                entity = args.get("entity") or args.get("entity_id")
                if entity and entity.lower() in seen_entities:
                    redundant.append({
                        "step": i,
                        "tool": tool_name,
                        "reason": f"Entity '{entity}' already fetched",
                    })
                if entity:
                    seen_entities.add(entity.lower())

        return redundant

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure loop efficiency.

        Args:
            test_case: Test case with optional minimum_steps
            agent_output: Agent output with iteration data

        Returns:
            MetricResult with efficiency score
        """
        # Calculate steps
        min_steps = self._estimate_minimum_steps(test_case, agent_output)
        actual_steps = self._count_actual_steps(agent_output)

        # Calculate efficiency
        efficiency = min(min_steps / actual_steps, 1.0) if actual_steps > 0 else 0.0

        # Identify redundant steps
        redundant_steps = self._identify_redundant_steps(agent_output.thought_history)

        # Penalize for redundant steps
        if redundant_steps and actual_steps > min_steps:
            redundant_penalty = len(redundant_steps) * 0.1
            efficiency = max(0.0, efficiency - redundant_penalty)

        return self._create_result(
            score=efficiency,
            details={
                "minimum_steps": min_steps,
                "actual_steps": actual_steps,
                "redundant_steps": redundant_steps,
                "redundant_count": len(redundant_steps),
                "step_overhead": actual_steps - min_steps,
                "test_case_min_steps": test_case.minimum_steps,
                "iteration_count": agent_output.iteration_count,
            },
        )
