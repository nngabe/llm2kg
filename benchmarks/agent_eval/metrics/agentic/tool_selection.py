"""
Tool Selection Accuracy Metric.

LLM-as-judge metric evaluating whether the agent selects optimal tools.
"""

import logging
from typing import Any, Dict, List

from ..base import LLMJudgeMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


TOOL_SELECTION_PROMPT = """You are evaluating whether an AI agent made an optimal tool selection.

Available Tools:
- graph_lookup: Search for entities and relationships in the knowledge graph
- web_search: Search the internet for external information
- cypher_query: Execute a Cypher query against the Neo4j database
- entity_resolve: Disambiguate and resolve entity references
- get_entity_details: Get detailed information about a specific entity
- get_relationships: Get relationships for an entity

Question: {question}

Current Context (what the agent knows so far):
{context}

Tool Selected: {tool_name}
Arguments: {arguments}

Optimal Tool Sequence (if known): {optimal_sequence}

Evaluate:
1. Is this tool selection appropriate for the current state?
2. Would a different tool be more efficient?
3. Is this tool call necessary or redundant?

Return a JSON object:
{{
    "is_optimal": true/false,
    "score": 0.0-1.0,
    "reason": "explanation of your assessment",
    "better_alternative": "suggested better tool or null if optimal"
}}
"""


class ToolSelectionAccuracyMetric(LLMJudgeMetric):
    """Measures accuracy of tool selection decisions.

    Tool Selection Accuracy = avg(is_optimal) across all tool calls

    Uses LLM-as-judge to evaluate each tool selection against
    the optimal path.
    """

    layer = EvaluationLayer.AGENTIC
    name = "Tool Selection Accuracy"

    def __init__(
        self,
        threshold: float = 0.80,
        judge_provider: str = "google",
        judge_model: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        fallback_provider: str = "openai",
        fallback_model: str = "gpt-5.2",
    ):
        super().__init__(
            threshold=threshold,
            judge_provider=judge_provider,
            judge_model=judge_model,
            temperature=temperature,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )

    def _evaluate_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        question: str,
        context: str,
        optimal_sequence: List[str],
        call_index: int,
    ) -> Dict[str, Any]:
        """Evaluate a single tool call.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            question: Original question
            context: Context accumulated so far
            optimal_sequence: Expected optimal tool sequence
            call_index: Index of this call in the sequence

        Returns:
            Evaluation result dictionary
        """
        # Format optimal sequence info
        optimal_info = "Unknown"
        if optimal_sequence:
            if call_index < len(optimal_sequence):
                optimal_info = f"Expected: {optimal_sequence[call_index]} (step {call_index + 1} of {len(optimal_sequence)})"
            else:
                optimal_info = f"Optimal sequence has {len(optimal_sequence)} steps, this is step {call_index + 1}"

        prompt = TOOL_SELECTION_PROMPT.format(
            question=question,
            context=context[:1000] if context else "No context yet",
            tool_name=tool_name,
            arguments=str(arguments),
            optimal_sequence=optimal_info,
        )

        result = self._invoke_judge(prompt)

        if "error" in result:
            logger.warning(f"Tool selection evaluation failed: {result['error']}")
            return {
                "is_optimal": False,
                "score": 0.5,
                "reason": f"Judge error: {result['error']}",
                "better_alternative": None,
            }

        return result

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure tool selection accuracy.

        Args:
            test_case: Test case with optimal tool sequence
            agent_output: Agent output with thought history

        Returns:
            MetricResult with tool selection accuracy score
        """
        thought_history = agent_output.thought_history

        if not thought_history:
            return self._create_result(
                score=1.0,
                details={"note": "No tool calls to evaluate"},
            )

        # Extract tool calls from thought history
        tool_calls = []
        for thought in thought_history:
            action = thought.get("action")
            if action and action.get("tool_name"):
                tool_calls.append({
                    "tool_name": action["tool_name"],
                    "arguments": action.get("arguments", {}),
                    "observation": thought.get("observation", ""),
                })

        if not tool_calls:
            return self._create_result(
                score=1.0,
                details={"note": "No tool calls found in thought history"},
            )

        # Evaluate each tool call
        evaluations = []
        accumulated_context = ""
        total_score = 0.0

        for i, call in enumerate(tool_calls):
            eval_result = self._evaluate_tool_call(
                tool_name=call["tool_name"],
                arguments=call["arguments"],
                question=test_case.question,
                context=accumulated_context,
                optimal_sequence=test_case.optimal_tool_sequence,
                call_index=i,
            )

            evaluations.append({
                "call_index": i,
                "tool_name": call["tool_name"],
                "arguments": call["arguments"],
                **eval_result,
            })

            total_score += eval_result.get("score", 0.5)

            # Accumulate context
            accumulated_context += f"\n[{call['tool_name']}]: {call['observation'][:200]}"

        # Calculate average score
        avg_score = total_score / len(tool_calls)

        # Count optimal selections
        optimal_count = sum(1 for e in evaluations if e.get("is_optimal", False))

        return self._create_result(
            score=avg_score,
            details={
                "total_calls": len(tool_calls),
                "optimal_calls": optimal_count,
                "expected_sequence": test_case.optimal_tool_sequence,
                "actual_sequence": [c["tool_name"] for c in tool_calls],
                "evaluations": evaluations,
            },
        )
