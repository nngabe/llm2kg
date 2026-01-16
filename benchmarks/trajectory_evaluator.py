"""
Trajectory Evaluator for ReActQAAgent Reasoning Quality.

This module evaluates the quality of agent reasoning trajectories,
inspired by NVIDIA NeMo-Agent-Toolkit's trajectory evaluation.

Metrics:
- Thought Relevance: Are thoughts relevant to the question?
- Tool Selection: Are tool choices appropriate?
- Reasoning Coherence: Is reasoning consistent across steps?
- Efficiency: Minimal steps to reach answer?
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Import agent types
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_qa import ThoughtStep, QAResponse, ToolCall

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MAIN_MODEL", "nemotron-3-nano:30b")


class TrajectoryMetric(Enum):
    """Available trajectory metrics."""
    THOUGHT_RELEVANCE = "thought_relevance"
    TOOL_SELECTION = "tool_selection"
    REASONING_COHERENCE = "reasoning_coherence"
    EFFICIENCY = "efficiency"


@dataclass
class TrajectoryScore:
    """Scores for trajectory evaluation."""
    thought_relevance: float = 0.0      # Are thoughts relevant to question?
    tool_selection: float = 0.0         # Are tool choices appropriate?
    reasoning_coherence: float = 0.0    # Is reasoning consistent across steps?
    efficiency: float = 0.0             # Minimal steps to reach answer?
    overall: float = 0.0

    # Detailed breakdowns
    step_scores: List[Dict[str, float]] = field(default_factory=list)
    tool_scores: List[Dict[str, float]] = field(default_factory=list)

    def calculate_overall(self, weights: Optional[Dict[str, float]] = None):
        """Calculate weighted overall score."""
        weights = weights or {
            "thought_relevance": 0.3,
            "tool_selection": 0.3,
            "reasoning_coherence": 0.25,
            "efficiency": 0.15,
        }

        self.overall = (
            weights["thought_relevance"] * self.thought_relevance +
            weights["tool_selection"] * self.tool_selection +
            weights["reasoning_coherence"] * self.reasoning_coherence +
            weights["efficiency"] * self.efficiency
        )
        return self.overall

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought_relevance": self.thought_relevance,
            "tool_selection": self.tool_selection,
            "reasoning_coherence": self.reasoning_coherence,
            "efficiency": self.efficiency,
            "overall": self.overall,
            "step_scores": self.step_scores,
            "tool_scores": self.tool_scores,
        }


# Prompts for LLM-based evaluation
THOUGHT_RELEVANCE_PROMPT = """Evaluate how relevant this thought is to answering the question.

Question: {question}

Thought: {thought}

Context so far: {context}

Rate the relevance from 0.0 to 1.0:
- 1.0: Directly addresses the question or identifies key information needs
- 0.7: Reasonably relevant, makes progress toward answer
- 0.5: Somewhat relevant but could be more focused
- 0.3: Tangential, doesn't directly help answer the question
- 0.0: Completely irrelevant

Respond with ONLY a number between 0.0 and 1.0."""

TOOL_SELECTION_PROMPT = """Evaluate if the selected tool is appropriate for the current situation.

Question: {question}

Current thought: {thought}

Available tools:
- graph_lookup: Look up entities in knowledge graph
- wiki_search: Search Wikipedia for encyclopedic information
- web_search: Search the web for current information
- cypher_query: Run database queries
- entity_resolve: Disambiguate entity names
- none: Ready to answer

Selected tool: {tool_name}
Tool arguments: {tool_args}

Rate the tool selection from 0.0 to 1.0:
- 1.0: Perfect choice, most efficient tool for this situation
- 0.7: Good choice, will help make progress
- 0.5: Acceptable but not optimal
- 0.3: Poor choice, better options available
- 0.0: Wrong tool, will not help

Respond with ONLY a number between 0.0 and 1.0."""

COHERENCE_PROMPT = """Evaluate the coherence between consecutive reasoning steps.

Question: {question}

Previous step:
Thought: {prev_thought}
Action: {prev_action}
Observation: {prev_observation}

Current step:
Thought: {curr_thought}

Rate the coherence from 0.0 to 1.0:
- 1.0: Current thought logically follows from previous, builds on observation
- 0.7: Generally coherent, reasonable progression
- 0.5: Some connection but could be clearer
- 0.3: Weak connection, seems to ignore previous observation
- 0.0: No logical connection, contradicts previous steps

Respond with ONLY a number between 0.0 and 1.0."""


class TrajectoryEvaluator:
    """
    Evaluate agent reasoning trajectory quality.

    Uses a combination of heuristics and LLM-based evaluation
    to score intermediate reasoning steps.
    """

    # Optimal step counts by question complexity
    OPTIMAL_STEPS = {
        "simple": 2,    # Direct lookup questions
        "medium": 3,    # Multi-hop questions
        "complex": 5,   # Research questions
    }

    def __init__(
        self,
        llm: Optional[Any] = None,
        use_llm_eval: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            llm: LLM for evaluation. Defaults to Ollama Nemotron.
            use_llm_eval: Whether to use LLM for evaluation (slower but more accurate)
        """
        self.llm = llm or ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
        )
        self.use_llm_eval = use_llm_eval

    def evaluate(
        self,
        question: str,
        response: QAResponse,
        expected_answer: Optional[str] = None,
        question_complexity: str = "medium",
    ) -> TrajectoryScore:
        """
        Evaluate the trajectory quality.

        Args:
            question: The original question
            response: Agent response with reasoning steps
            expected_answer: Optional expected answer for comparison
            question_complexity: "simple", "medium", or "complex"

        Returns:
            TrajectoryScore with detailed metrics
        """
        steps = response.reasoning_steps
        score = TrajectoryScore()

        if not steps:
            logger.warning("No reasoning steps to evaluate")
            return score

        # Evaluate thought relevance
        score.thought_relevance, score.step_scores = self._score_thought_relevance(
            question, steps
        )

        # Evaluate tool selection
        score.tool_selection, score.tool_scores = self._score_tool_selection(
            question, steps
        )

        # Evaluate reasoning coherence
        score.reasoning_coherence = self._score_coherence(question, steps)

        # Evaluate efficiency
        score.efficiency = self._score_efficiency(
            len(steps), question_complexity
        )

        # Calculate overall
        score.calculate_overall()

        return score

    def _score_thought_relevance(
        self,
        question: str,
        steps: List[ThoughtStep],
    ) -> Tuple[float, List[Dict[str, float]]]:
        """Score relevance of each thought to the question."""
        step_scores = []
        total_score = 0.0

        context_so_far = ""
        for i, step in enumerate(steps):
            if not step.thought:
                step_scores.append({"step": i + 1, "score": 0.5})
                total_score += 0.5
                continue

            if self.use_llm_eval:
                score = self._llm_score(
                    THOUGHT_RELEVANCE_PROMPT.format(
                        question=question,
                        thought=step.thought,
                        context=context_so_far[:500],
                    )
                )
            else:
                # Heuristic: keyword overlap
                q_words = set(question.lower().split())
                t_words = set(step.thought.lower().split())
                overlap = len(q_words & t_words) / len(q_words) if q_words else 0
                score = min(1.0, overlap * 2)  # Scale up

            step_scores.append({"step": i + 1, "score": score})
            total_score += score

            # Update context
            if step.observation:
                context_so_far += f"\n{step.observation[:200]}"

        avg_score = total_score / len(steps) if steps else 0.0
        return avg_score, step_scores

    def _score_tool_selection(
        self,
        question: str,
        steps: List[ThoughtStep],
    ) -> Tuple[float, List[Dict[str, float]]]:
        """Score appropriateness of tool selections."""
        tool_scores = []
        total_score = 0.0
        n_tools = 0

        for i, step in enumerate(steps):
            if not step.action or step.action.tool_name == "none":
                continue

            n_tools += 1

            if self.use_llm_eval:
                score = self._llm_score(
                    TOOL_SELECTION_PROMPT.format(
                        question=question,
                        thought=step.thought,
                        tool_name=step.action.tool_name,
                        tool_args=json.dumps(step.action.arguments),
                    )
                )
            else:
                # Heuristic scoring based on common patterns
                score = self._heuristic_tool_score(question, step)

            tool_scores.append({
                "step": i + 1,
                "tool": step.action.tool_name,
                "score": score,
            })
            total_score += score

        avg_score = total_score / n_tools if n_tools > 0 else 0.5
        return avg_score, tool_scores

    def _heuristic_tool_score(
        self,
        question: str,
        step: ThoughtStep,
    ) -> float:
        """Heuristic tool selection scoring."""
        if not step.action:
            return 0.5

        tool = step.action.tool_name
        q_lower = question.lower()

        # Good patterns
        if "what is" in q_lower and tool == "graph_lookup":
            return 0.9
        if "define" in q_lower and tool in ["graph_lookup", "wiki_search"]:
            return 0.9
        if "current" in q_lower or "latest" in q_lower:
            if tool == "web_search":
                return 0.9
            elif tool == "graph_lookup":
                return 0.5
        if "who" in q_lower and tool == "graph_lookup":
            return 0.8

        # Default reasonable score
        return 0.7

    def _score_coherence(
        self,
        question: str,
        steps: List[ThoughtStep],
    ) -> float:
        """Score coherence between consecutive steps."""
        if len(steps) < 2:
            return 1.0  # Single step is coherent by default

        coherence_scores = []

        for i in range(1, len(steps)):
            prev_step = steps[i - 1]
            curr_step = steps[i]

            if self.use_llm_eval:
                score = self._llm_score(
                    COHERENCE_PROMPT.format(
                        question=question,
                        prev_thought=prev_step.thought,
                        prev_action=f"{prev_step.action.tool_name}({prev_step.action.arguments})"
                            if prev_step.action else "none",
                        prev_observation=prev_step.observation[:300] if prev_step.observation else "none",
                        curr_thought=curr_step.thought,
                    )
                )
            else:
                # Heuristic: check if observation is referenced
                score = 0.7  # Default reasonable
                if prev_step.observation and curr_step.thought:
                    # Check for any keyword overlap with observation
                    obs_words = set(prev_step.observation.lower().split()[:20])
                    thought_words = set(curr_step.thought.lower().split())
                    if obs_words & thought_words:
                        score = 0.9

            coherence_scores.append(score)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0

    def _score_efficiency(
        self,
        n_steps: int,
        complexity: str,
    ) -> float:
        """Score efficiency based on number of steps vs optimal."""
        optimal = self.OPTIMAL_STEPS.get(complexity, 3)

        if n_steps <= optimal:
            return 1.0
        elif n_steps <= optimal + 2:
            return 0.8
        elif n_steps <= optimal + 4:
            return 0.5
        else:
            return 0.3

    def _llm_score(self, prompt: str) -> float:
        """Get a score from LLM."""
        try:
            response = self.llm.invoke([
                HumanMessage(content=prompt),
            ])
            content = response.content.strip()

            # Parse score
            try:
                score = float(content)
                return max(0.0, min(1.0, score))
            except ValueError:
                # Try to extract number from response
                import re
                numbers = re.findall(r"0?\.\d+|1\.0|0|1", content)
                if numbers:
                    return float(numbers[0])
                return 0.5

        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")
            return 0.5


def evaluate_trajectory_batch(
    evaluator: TrajectoryEvaluator,
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate trajectories for a batch of test cases.

    Args:
        evaluator: TrajectoryEvaluator instance
        test_cases: List of {"question": str, "response": QAResponse, ...}

    Returns:
        Aggregated results
    """
    scores = []

    for case in test_cases:
        score = evaluator.evaluate(
            question=case["question"],
            response=case["response"],
            expected_answer=case.get("expected_answer"),
            question_complexity=case.get("complexity", "medium"),
        )
        scores.append(score)

    # Aggregate
    n = len(scores)
    if n == 0:
        return {"error": "No test cases evaluated"}

    return {
        "n_cases": n,
        "avg_thought_relevance": sum(s.thought_relevance for s in scores) / n,
        "avg_tool_selection": sum(s.tool_selection for s in scores) / n,
        "avg_reasoning_coherence": sum(s.reasoning_coherence for s in scores) / n,
        "avg_efficiency": sum(s.efficiency for s in scores) / n,
        "avg_overall": sum(s.overall for s in scores) / n,
        "individual_scores": [s.to_dict() for s in scores],
    }


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trajectory Evaluator CLI")
    parser.add_argument("--demo", action="store_true", help="Run demo evaluation")
    parser.add_argument("--no-llm", action="store_true", help="Use heuristics only")
    args = parser.parse_args()

    if args.demo:
        # Create demo response
        demo_response = QAResponse(
            question="What is inflation?",
            answer="Inflation is the rate at which prices increase over time.",
            reasoning_steps=[
                ThoughtStep(
                    thought="I need to look up information about inflation in the knowledge graph.",
                    action=ToolCall(tool_name="graph_lookup", arguments={"entity_name": "inflation"}),
                    observation="Entity: Inflation\nDescription: A general increase in prices...",
                ),
                ThoughtStep(
                    thought="I found information about inflation. Let me synthesize an answer.",
                    action=None,
                    observation=None,
                ),
            ],
        )

        evaluator = TrajectoryEvaluator(use_llm_eval=not args.no_llm)
        score = evaluator.evaluate(
            question="What is inflation?",
            response=demo_response,
            question_complexity="simple",
        )

        print("\n=== Trajectory Evaluation Results ===")
        print(f"Thought Relevance: {score.thought_relevance:.2f}")
        print(f"Tool Selection: {score.tool_selection:.2f}")
        print(f"Reasoning Coherence: {score.reasoning_coherence:.2f}")
        print(f"Efficiency: {score.efficiency:.2f}")
        print(f"Overall: {score.overall:.2f}")

        if score.step_scores:
            print("\nStep Scores:")
            for ss in score.step_scores:
                print(f"  Step {ss['step']}: {ss['score']:.2f}")

        if score.tool_scores:
            print("\nTool Scores:")
            for ts in score.tool_scores:
                print(f"  Step {ts['step']} ({ts['tool']}): {ts['score']:.2f}")
    else:
        print("Use --demo to run a demo evaluation")
