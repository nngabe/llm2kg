"""
Argument Correctness Metric.

LLM-as-judge metric evaluating whether tool arguments are correct.
"""

import logging
from typing import Any, Dict, List

from ..base import LLMJudgeMetric, TestCase, AgentOutput
from ...config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


ARGUMENT_CORRECTNESS_PROMPT = """You are evaluating whether an AI agent provided correct arguments for a tool call.

Tool: {tool_name}

Tool Description:
{tool_description}

Question being answered: {question}

Arguments provided:
{arguments}

Previous context (what the agent knows):
{context}

Evaluate each argument:
1. Is the argument value appropriate for the task?
2. Is the argument correctly formatted/typed?
3. Are there any missing required arguments?
4. Are there unnecessary arguments?

Return a JSON object:
{{
    "overall_correct": true/false,
    "score": 0.0-1.0,
    "argument_evaluations": [
        {{"name": "arg_name", "correct": true/false, "reason": "explanation"}}
    ],
    "missing_arguments": ["list of missing required args"],
    "unnecessary_arguments": ["list of unnecessary args"],
    "reasoning": "overall assessment"
}}
"""


TOOL_DESCRIPTIONS = {
    "graph_lookup": """
Search for entities and relationships in the knowledge graph.
Required arguments:
- entity: string - The entity name or type to search for
Optional arguments:
- relationship_type: string - Filter by relationship type
- limit: int - Maximum results to return
""",
    "web_search": """
Search the internet for external information.
Required arguments:
- query: string - The search query
Optional arguments:
- num_results: int - Number of results to return
""",
    "cypher_query": """
Execute a Cypher query against the Neo4j database.
Required arguments:
- query: string - The Cypher query to execute
Optional arguments:
- parameters: object - Query parameters
""",
    "entity_resolve": """
Disambiguate and resolve entity references.
Required arguments:
- entity_name: string - The entity name to resolve
- context: string - Context for disambiguation
""",
    "get_entity_details": """
Get detailed information about a specific entity.
Required arguments:
- entity_id: string - The entity ID or name
""",
    "get_relationships": """
Get relationships for an entity.
Required arguments:
- entity_id: string - The entity ID or name
Optional arguments:
- direction: string - 'incoming', 'outgoing', or 'both'
- relationship_type: string - Filter by type
""",
}


class ArgumentCorrectnessMetric(LLMJudgeMetric):
    """Measures correctness of tool arguments.

    Argument Correctness = correct_args / total_args

    Uses LLM-as-judge to evaluate argument quality and correctness.
    """

    layer = EvaluationLayer.AGENTIC
    name = "Argument Correctness"

    def __init__(
        self,
        threshold: float = 0.85,
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

    def _evaluate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        question: str,
        context: str,
    ) -> Dict[str, Any]:
        """Evaluate arguments for a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Arguments passed to the tool
            question: Original question
            context: Accumulated context

        Returns:
            Evaluation result dictionary
        """
        tool_desc = TOOL_DESCRIPTIONS.get(
            tool_name,
            f"Unknown tool: {tool_name}. Evaluate arguments based on reasonable expectations."
        )

        prompt = ARGUMENT_CORRECTNESS_PROMPT.format(
            tool_name=tool_name,
            tool_description=tool_desc,
            question=question,
            arguments=str(arguments),
            context=context[:1000] if context else "No context yet",
        )

        result = self._invoke_judge(prompt)

        if "error" in result:
            logger.warning(f"Argument evaluation failed: {result['error']}")
            return {
                "overall_correct": True,  # Assume correct on error
                "score": 0.5,
                "argument_evaluations": [],
                "missing_arguments": [],
                "unnecessary_arguments": [],
                "reasoning": f"Judge error: {result['error']}",
            }

        return result

    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure argument correctness.

        Args:
            test_case: Test case
            agent_output: Agent output with thought history

        Returns:
            MetricResult with argument correctness score
        """
        thought_history = agent_output.thought_history

        if not thought_history:
            return self._create_result(
                score=1.0,
                details={"note": "No tool calls to evaluate"},
            )

        # Extract tool calls
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
                details={"note": "No tool calls found"},
            )

        # Evaluate arguments for each tool call
        evaluations = []
        total_score = 0.0
        total_args = 0
        correct_args = 0
        accumulated_context = ""

        for i, call in enumerate(tool_calls):
            eval_result = self._evaluate_arguments(
                tool_name=call["tool_name"],
                arguments=call["arguments"],
                question=test_case.question,
                context=accumulated_context,
            )

            evaluations.append({
                "call_index": i,
                "tool_name": call["tool_name"],
                "arguments": call["arguments"],
                **eval_result,
            })

            total_score += eval_result.get("score", 0.5)

            # Count arguments
            arg_evals = eval_result.get("argument_evaluations", [])
            for arg_eval in arg_evals:
                total_args += 1
                if arg_eval.get("correct", False):
                    correct_args += 1

            # Accumulate context
            accumulated_context += f"\n[{call['tool_name']}]: {call['observation'][:200]}"

        # Calculate score
        if total_args > 0:
            final_score = correct_args / total_args
        else:
            # Fall back to average judge score if no detailed arg evaluations
            final_score = total_score / len(tool_calls)

        # Count calls with correct arguments
        correct_calls = sum(1 for e in evaluations if e.get("overall_correct", False))

        return self._create_result(
            score=final_score,
            details={
                "total_calls": len(tool_calls),
                "correct_calls": correct_calls,
                "total_arguments": total_args,
                "correct_arguments": correct_args,
                "evaluations": evaluations,
            },
        )
