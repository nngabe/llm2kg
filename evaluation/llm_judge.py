"""
LLM-as-a-Judge evaluation for Q&A quality.

Uses an LLM to evaluate answer quality across multiple dimensions.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """Result from LLM judge evaluation."""

    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0, description="Faithfulness to context")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance to question")
    reasoning_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality of reasoning")
    citation_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Accuracy of citations")
    external_source_usage: float = Field(default=0.0, ge=0.0, le=1.0, description="Appropriate use of external sources")
    overall: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall quality")
    reasoning: str = Field(default="", description="Judge's reasoning")
    feedback: str = Field(default="", description="Improvement suggestions")


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems.
Your task is to evaluate the quality of an answer based on multiple criteria.

Evaluate on a scale of 0.0 to 1.0 for each dimension:
1. Faithfulness (0-1): Is the answer faithful to the provided context? No hallucinations?
2. Relevance (0-1): Does the answer address the question directly and completely?
3. Reasoning Quality (0-1): Is the reasoning process logical and well-structured?
4. Citation Accuracy (0-1): Are citations accurate and properly attributed?
5. External Source Usage (0-1): Were external sources used appropriately and transparently?

Be strict but fair. A score of 1.0 means perfect, 0.5 is average, and 0.0 is poor."""


JUDGE_PROMPT = """Question: {question}

Context Provided:
{context}

Answer Given:
{answer}

Reasoning Steps (if any):
{reasoning_steps}

Citations:
{citations}

External Sources Used: {external_used}

Evaluate this answer and provide scores for each dimension.
Return your evaluation in the following JSON format:
{{
    "faithfulness": 0.0-1.0,
    "relevance": 0.0-1.0,
    "reasoning_quality": 0.0-1.0,
    "citation_accuracy": 0.0-1.0,
    "external_source_usage": 0.0-1.0,
    "overall": 0.0-1.0,
    "reasoning": "Your reasoning for these scores",
    "feedback": "Specific suggestions for improvement"
}}"""


class QALLMJudge:
    """
    LLM-as-a-Judge evaluator for Q&A systems.

    Uses a powerful LLM to evaluate answer quality across multiple dimensions.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ):
        """
        Initialize the judge.

        Args:
            model: LLM model to use as judge.
            temperature: Temperature for judge responses.
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str = "",
        reasoning_steps: Optional[List[str]] = None,
        citations: Optional[List[Dict]] = None,
        external_sources_used: bool = False,
    ) -> JudgeResult:
        """
        Evaluate an answer using LLM judge.

        Args:
            question: The original question.
            answer: The generated answer.
            context: Retrieved context.
            reasoning_steps: ReAct reasoning steps.
            citations: Answer citations.
            external_sources_used: Whether external sources were used.

        Returns:
            JudgeResult with scores and feedback.
        """
        # Format inputs
        steps_text = "\n".join(reasoning_steps) if reasoning_steps else "None"
        citations_text = json.dumps(citations, indent=2) if citations else "None"

        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context[:4000] if context else "None provided",
            answer=answer,
            reasoning_steps=steps_text,
            citations=citations_text,
            external_used="Yes" if external_sources_used else "No",
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            # Parse response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            return JudgeResult(
                faithfulness=data.get("faithfulness", 0.5),
                relevance=data.get("relevance", 0.5),
                reasoning_quality=data.get("reasoning_quality", 0.5),
                citation_accuracy=data.get("citation_accuracy", 0.5),
                external_source_usage=data.get("external_source_usage", 0.5),
                overall=data.get("overall", 0.5),
                reasoning=data.get("reasoning", ""),
                feedback=data.get("feedback", ""),
            )

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return JudgeResult(
                reasoning=f"Evaluation failed: {str(e)}",
            )

    def batch_evaluate(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[JudgeResult]:
        """
        Evaluate multiple samples.

        Args:
            samples: List of sample dicts.

        Returns:
            List of judge results.
        """
        results = []
        for sample in samples:
            result = self.evaluate(
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                context=sample.get("context", ""),
                reasoning_steps=sample.get("reasoning_steps"),
                citations=sample.get("citations"),
                external_sources_used=sample.get("external_sources_used", False),
            )
            results.append(result)

        return results

    def aggregate_results(self, results: List[JudgeResult]) -> JudgeResult:
        """Compute aggregate scores across results."""
        if not results:
            return JudgeResult()

        n = len(results)
        return JudgeResult(
            faithfulness=sum(r.faithfulness for r in results) / n,
            relevance=sum(r.relevance for r in results) / n,
            reasoning_quality=sum(r.reasoning_quality for r in results) / n,
            citation_accuracy=sum(r.citation_accuracy for r in results) / n,
            external_source_usage=sum(r.external_source_usage for r in results) / n,
            overall=sum(r.overall for r in results) / n,
            reasoning=f"Aggregated from {n} evaluations",
        )

    def compare_answers(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Compare two answers and determine which is better.

        Args:
            question: The question.
            answer_a: First answer.
            answer_b: Second answer.
            context: Retrieved context.

        Returns:
            Comparison result with winner and reasoning.
        """
        result_a = self.evaluate(question=question, answer=answer_a, context=context)
        result_b = self.evaluate(question=question, answer=answer_b, context=context)

        if result_a.overall > result_b.overall:
            winner = "A"
            margin = result_a.overall - result_b.overall
        elif result_b.overall > result_a.overall:
            winner = "B"
            margin = result_b.overall - result_a.overall
        else:
            winner = "TIE"
            margin = 0.0

        return {
            "winner": winner,
            "margin": margin,
            "score_a": result_a.overall,
            "score_b": result_b.overall,
            "result_a": result_a,
            "result_b": result_b,
        }
