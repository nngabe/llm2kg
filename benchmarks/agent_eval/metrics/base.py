"""
Base classes for evaluation metrics.

Provides:
- BaseMetric: Abstract base for all metrics
- LLMJudgeMetric: Base for LLM-as-judge metrics (uses GPT-5.2)
- TestCase: Data structure for golden dataset test cases
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from ..config import EvaluationLayer, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case for evaluation from the golden dataset."""

    id: str
    question: str

    # Ground truth for retrieval/generation
    expected_answer: Optional[str] = None
    expected_entities: List[str] = field(default_factory=list)
    expected_relationships: List[Dict[str, str]] = field(default_factory=list)
    ground_truth_context: List[str] = field(default_factory=list)

    # Ground truth for agentic evaluation
    optimal_tool_sequence: List[str] = field(default_factory=list)
    minimum_steps: Optional[int] = None

    # Rejection test (for unanswerable questions)
    should_reject: bool = False
    rejection_reason: Optional[str] = None

    # Metadata
    type: str = "all"  # retrieval, agentic, integrity, generation, rejection, all
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "generated"  # generated, human, hybrid
    reviewed: bool = False
    reviewer_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_layer(self, layer: EvaluationLayer) -> bool:
        """Check if this test case is relevant to a layer."""
        if self.type == "all":
            return True
        return self.type == layer.value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create TestCase from dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            question=data.get("question", ""),
            expected_answer=data.get("expected_answer"),
            expected_entities=data.get("expected_entities", []),
            expected_relationships=data.get("expected_relationships", []),
            ground_truth_context=data.get("ground_truth_context", []),
            optimal_tool_sequence=data.get("optimal_tool_sequence", []),
            minimum_steps=data.get("minimum_steps"),
            should_reject=data.get("should_reject", False),
            rejection_reason=data.get("rejection_reason"),
            type=data.get("type", "all"),
            difficulty=data.get("difficulty", "medium"),
            source=data.get("source", "generated"),
            reviewed=data.get("reviewed", False),
            reviewer_notes=data.get("reviewer_notes"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_entities": self.expected_entities,
            "expected_relationships": self.expected_relationships,
            "ground_truth_context": self.ground_truth_context,
            "optimal_tool_sequence": self.optimal_tool_sequence,
            "minimum_steps": self.minimum_steps,
            "should_reject": self.should_reject,
            "rejection_reason": self.rejection_reason,
            "type": self.type,
            "difficulty": self.difficulty,
            "source": self.source,
            "reviewed": self.reviewed,
            "reviewer_notes": self.reviewer_notes,
            "metadata": self.metadata,
        }


@dataclass
class AgentOutput:
    """Captured output from running the agent on a test case.

    This wraps the agent's internal state and final response for evaluation.
    """

    # From QAResponse
    question: str
    answer: str
    confidence: float
    citations: List[Dict[str, Any]] = field(default_factory=list)
    external_info_used: bool = False

    # From QAAgentState
    context_items: List[Dict[str, Any]] = field(default_factory=list)
    thought_history: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_plan: Optional[Dict[str, Any]] = None
    iteration_count: int = 0

    # Graph updates (for integrity metrics)
    new_nodes: List[Dict[str, Any]] = field(default_factory=list)
    new_edges: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_agent(cls, response, state=None) -> "AgentOutput":
        """Create AgentOutput from agent response and state."""
        output = cls(
            question=response.question,
            answer=response.answer,
            confidence=response.confidence,
            external_info_used=response.external_info_used,
        )

        # Extract citations
        if hasattr(response, 'citations'):
            output.citations = [
                {
                    "source_type": c.source_type,
                    "source_id": c.source_id,
                    "source_title": getattr(c, 'source_title', None),
                    "trust_level": c.trust_level,
                    "excerpt": c.excerpt,
                }
                for c in response.citations
            ]

        # Extract reasoning steps
        if hasattr(response, 'reasoning_steps'):
            output.thought_history = [
                {
                    "thought": step.thought,
                    "action": {
                        "tool_name": step.action.tool_name,
                        "arguments": step.action.arguments,
                    } if step.action else None,
                    "observation": step.observation,
                }
                for step in response.reasoning_steps
            ]

        # Extract state if provided
        if state:
            if hasattr(state, 'context'):
                output.context_items = [
                    {
                        "source_type": c.source_type,
                        "content": c.content,
                        "source_id": c.source_id,
                        "relevance_score": c.relevance_score,
                        "metadata": c.metadata,
                    }
                    for c in state.context
                ]
            if hasattr(state, 'retrieval_plan') and state.retrieval_plan:
                output.retrieval_plan = {
                    "reasoning": state.retrieval_plan.reasoning,
                    "entity_targets": state.retrieval_plan.entity_targets,
                    "relationship_queries": state.retrieval_plan.relationship_queries,
                    "information_needs": state.retrieval_plan.information_needs,
                }
            if hasattr(state, 'iteration_count'):
                output.iteration_count = state.iteration_count

        return output


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.

    Subclasses must implement:
    - measure(): The actual measurement logic
    - layer: The evaluation layer this metric belongs to
    - name: Human-readable name for the metric
    """

    layer: EvaluationLayer = EvaluationLayer.RETRIEVAL
    name: str = "BaseMetric"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abstractmethod
    def measure(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **kwargs: Any,
    ) -> MetricResult:
        """Measure this metric for a given test case and agent output.

        Args:
            test_case: The golden dataset test case
            agent_output: Captured agent response and state
            **kwargs: Additional context (e.g., neo4j_driver, embeddings)

        Returns:
            MetricResult with score, pass/fail status, and details
        """
        pass

    def _create_result(
        self,
        score: float,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: int = 0,
    ) -> MetricResult:
        """Helper to create a MetricResult."""
        return MetricResult(
            metric_name=self.name,
            layer=self.layer,
            score=score,
            passed=score >= self.threshold and error is None,
            threshold=self.threshold,
            details=details or {},
            error=error,
            latency_ms=latency_ms,
        )


class LLMJudgeMetric(BaseMetric):
    """Base class for metrics that use LLM-as-judge evaluation.

    Supports multiple providers (Google, OpenAI, Anthropic) with automatic fallback.
    Default: Gemini 2.5 Pro with GPT-5.2 fallback.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        judge_provider: str = "google",
        judge_model: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        fallback_provider: str = "openai",
        fallback_model: str = "gpt-5-mini",
    ):
        super().__init__(threshold)
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.temperature = temperature
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self._judge_llm = None
        self._using_fallback = False

    def _get_judge(self):
        """Lazy initialization of judge LLM with multi-provider support."""
        if self._judge_llm is None:
            provider = self.judge_provider
            model = self.judge_model

            try:
                if provider == "anthropic":
                    from langchain_anthropic import ChatAnthropic
                    self._judge_llm = ChatAnthropic(
                        model=model,
                        temperature=self.temperature,
                    )
                elif provider == "google":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self._judge_llm = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                    )
                elif provider == "openai":
                    from langchain_openai import ChatOpenAI
                    self._judge_llm = ChatOpenAI(
                        model=model,
                        temperature=self.temperature,
                    )
                else:
                    raise ValueError(f"Unknown judge provider: {provider}")

                logger.debug(f"Initialized judge LLM: {provider}/{model}")

            except Exception as e:
                logger.error(f"Failed to initialize judge LLM ({provider}/{model}): {e}")
                raise

        return self._judge_llm

    def _switch_to_fallback(self):
        """Switch to fallback provider."""
        logger.warning(f"Switching from {self.judge_provider}/{self.judge_model} "
                      f"to fallback {self.fallback_provider}/{self.fallback_model}")
        self._judge_llm = None
        self.judge_provider = self.fallback_provider
        self.judge_model = self.fallback_model
        self._using_fallback = True

    def _invoke_judge_internal(self, prompt: str) -> Dict[str, Any]:
        """Internal judge invocation (single attempt)."""
        from langchain_core.messages import HumanMessage

        judge = self._get_judge()
        response = judge.invoke([HumanMessage(content=prompt)])
        content = response.content

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # Find JSON object
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(content[start_idx:end_idx])

        logger.warning(f"Could not parse JSON from judge response: {content[:200]}")
        return {"error": "Failed to parse JSON"}

    def _invoke_judge(self, prompt: str) -> Dict[str, Any]:
        """Invoke the judge LLM with automatic fallback.

        Args:
            prompt: The evaluation prompt

        Returns:
            Parsed JSON response from the judge
        """
        start = time.time()

        try:
            result = self._invoke_judge_internal(prompt)
            result["_latency_ms"] = int((time.time() - start) * 1000)
            result["_provider"] = self.judge_provider
            return result

        except Exception as e:
            logger.warning(f"Primary judge ({self.judge_provider}) failed: {e}")

            # Try fallback if not already using it
            if not self._using_fallback:
                try:
                    self._switch_to_fallback()
                    result = self._invoke_judge_internal(prompt)
                    result["_latency_ms"] = int((time.time() - start) * 1000)
                    result["_provider"] = self.judge_provider
                    result["_fallback_used"] = True
                    return result
                except Exception as fallback_e:
                    logger.error(f"Fallback judge also failed: {fallback_e}")
                    return {
                        "error": f"Both primary and fallback failed: {e} / {fallback_e}",
                        "_latency_ms": int((time.time() - start) * 1000),
                    }

            return {"error": str(e), "_latency_ms": int((time.time() - start) * 1000)}
