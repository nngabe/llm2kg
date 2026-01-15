#!/usr/bin/env python3
"""
Hybrid RAG Benchmark Script.

Tests the ReActQAAgent with different configurations using realistic
grad-student level questions. Evaluates retrieval quality, answer accuracy,
and the effectiveness of various RAG features.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, "/app")

from agent_qa import ReActQAAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


# =============================================================================
# CONFIGURATIONS
# =============================================================================

CONFIGS = {
    # Baseline - all features enabled
    "baseline": {
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": True,
        "description": "All features ON (default)",
    },

    # Single feature ablations
    "no_planning": {
        "use_retrieval_planning": False,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": True,
        "description": "Disable retrieval planning",
    },
    "no_compression": {
        "use_retrieval_planning": True,
        "compression_enabled": False,
        "web_search_enabled": True,
        "auto_add_documents": True,
        "description": "Disable context compression",
    },
    "no_web": {
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": False,
        "auto_add_documents": False,  # No auto-ingest without web
        "description": "Disable web search",
    },
    "no_auto_ingest": {
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": False,
        "description": "Disable auto document ingestion",
    },

    # Minimal configuration
    "minimal": {
        "use_retrieval_planning": False,
        "compression_enabled": False,
        "web_search_enabled": False,
        "auto_add_documents": False,
        "description": "All features OFF (graph-only)",
    },

    # Combined configurations
    "planning_no_compress": {
        "use_retrieval_planning": True,
        "compression_enabled": False,
        "web_search_enabled": False,
        "auto_add_documents": False,
        "description": "Planning only, no compression or web",
    },
    "compress_no_planning": {
        "use_retrieval_planning": False,
        "compression_enabled": True,
        "web_search_enabled": False,
        "auto_add_documents": False,
        "description": "Compression only, no planning or web",
    },
    "web_only": {
        "use_retrieval_planning": False,
        "compression_enabled": False,
        "web_search_enabled": True,
        "auto_add_documents": False,
        "description": "Web search enabled, no planning/compression",
    },
    "graph_optimized": {
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": False,
        "auto_add_documents": False,
        "description": "Optimized for graph-only queries",
    },
    "web_optimized": {
        "use_retrieval_planning": False,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": True,
        "description": "Optimized for web-augmented queries",
    },
    "full_pipeline": {
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": True,
        "description": "Full pipeline (same as baseline)",
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question: str
    category: str
    domain: str
    requires_web: bool
    config_name: str
    answer: str
    confidence: float
    citations_count: int
    elapsed_seconds: float
    error: Optional[str] = None

    # Evaluation scores (filled later by judge)
    correctness: float = 0.0
    completeness: float = 0.0
    groundedness: float = 0.0


@dataclass
class ConfigResults:
    """Results for a single configuration."""
    config_name: str
    description: str
    question_results: List[QuestionResult] = field(default_factory=list)
    total_time_seconds: float = 0.0

    @property
    def avg_correctness(self) -> float:
        scores = [r.correctness for r in self.question_results if r.error is None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_completeness(self) -> float:
        scores = [r.completeness for r in self.question_results if r.error is None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_groundedness(self) -> float:
        scores = [r.groundedness for r in self.question_results if r.error is None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_confidence(self) -> float:
        scores = [r.confidence for r in self.question_results if r.error is None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def success_rate(self) -> float:
        if not self.question_results:
            return 0.0
        successes = sum(1 for r in self.question_results if r.error is None)
        return successes / len(self.question_results)

    @property
    def category_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get scores broken down by category."""
        categories: Dict[str, List[QuestionResult]] = {}
        for r in self.question_results:
            if r.error is None:
                if r.category not in categories:
                    categories[r.category] = []
                categories[r.category].append(r)

        breakdown = {}
        for cat, results in categories.items():
            breakdown[cat] = {
                "count": len(results),
                "avg_correctness": sum(r.correctness for r in results) / len(results),
                "avg_completeness": sum(r.completeness for r in results) / len(results),
            }
        return breakdown


# =============================================================================
# LLM JUDGE
# =============================================================================

def evaluate_answer(
    question: str,
    answer: str,
    ground_truth_facts: List[str],
    domain: str,
) -> Dict[str, float]:
    """
    Evaluate an answer using LLM-as-Judge.

    Returns:
        Dict with correctness, completeness, groundedness scores (0-1)
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt = f"""Evaluate this Q&A response for a {domain} question.

Question: {question}

Answer: {answer}

Expected key facts that should be covered:
{json.dumps(ground_truth_facts, indent=2)}

Rate each dimension 0-1:
1. Correctness: Is the information accurate and factually correct?
2. Completeness: Does it cover the expected key facts?
3. Groundedness: Are claims supported by evidence/citations?

Respond in JSON:
{{"correctness": 0.0-1.0, "completeness": 0.0-1.0, "groundedness": 0.0-1.0, "reasoning": "brief explanation"}}"""

        response = judge_llm.invoke([
            SystemMessage(content="You are an expert evaluator for knowledge graph Q&A systems."),
            HumanMessage(content=prompt),
        ])

        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content)
        return {
            "correctness": float(result.get("correctness", 0.5)),
            "completeness": float(result.get("completeness", 0.5)),
            "groundedness": float(result.get("groundedness", 0.5)),
        }

    except Exception as e:
        logger.warning(f"Judge evaluation failed: {e}")
        # Fallback: simple keyword matching
        answer_lower = answer.lower()
        matches = sum(1 for fact in ground_truth_facts if fact.lower() in answer_lower)
        score = matches / len(ground_truth_facts) if ground_truth_facts else 0.5
        return {
            "correctness": score,
            "completeness": score,
            "groundedness": 0.5,  # Can't evaluate without judge
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class HybridRAGBenchmark:
    """Benchmark runner for Hybrid RAG configurations."""

    def __init__(
        self,
        dataset_path: str = "/app/benchmarks/realistic_qa_dataset.json",
        output_dir: str = "/app/benchmarks/results",
        use_judge: bool = True,
    ):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_judge = use_judge

        # Load dataset
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.questions = self.dataset["questions"]
        logger.info(f"Loaded {len(self.questions)} questions from {dataset_path}")

        os.makedirs(output_dir, exist_ok=True)

    def run_config(
        self,
        config_name: str,
        questions: Optional[List[Dict]] = None,
    ) -> ConfigResults:
        """Run benchmark for a single configuration."""
        config = CONFIGS.get(config_name)
        if not config:
            raise ValueError(f"Unknown config: {config_name}")

        questions = questions or self.questions
        logger.info(f"\n{'='*60}")
        logger.info(f"Running config: {config_name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"{'='*60}")

        results = ConfigResults(
            config_name=config_name,
            description=config["description"],
        )

        # Initialize agent with config
        agent = ReActQAAgent(
            use_retrieval_planning=config["use_retrieval_planning"],
            compression_enabled=config["compression_enabled"],
            web_search_enabled=config["web_search_enabled"],
            auto_add_documents=config["auto_add_documents"],
        )

        config_start = time.time()

        for i, q in enumerate(questions):
            logger.info(f"\n[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")

            start_time = time.time()
            try:
                response = agent.answer_question(q["question"])
                elapsed = time.time() - start_time

                result = QuestionResult(
                    question_id=q["id"],
                    question=q["question"],
                    category=q["category"],
                    domain=q["domain"],
                    requires_web=q.get("requires_web", False),
                    config_name=config_name,
                    answer=response.answer,
                    confidence=response.confidence,
                    citations_count=len(response.citations),
                    elapsed_seconds=round(elapsed, 2),
                )

                # Evaluate with judge
                if self.use_judge and "ground_truth_key_facts" in q:
                    scores = evaluate_answer(
                        question=q["question"],
                        answer=response.answer,
                        ground_truth_facts=q["ground_truth_key_facts"],
                        domain=q["domain"],
                    )
                    result.correctness = scores["correctness"]
                    result.completeness = scores["completeness"]
                    result.groundedness = scores["groundedness"]

                logger.info(f"  Answer: {response.answer[:100]}...")
                logger.info(f"  Confidence: {response.confidence:.2f}, Time: {elapsed:.1f}s")
                if self.use_judge:
                    logger.info(f"  Scores: correctness={result.correctness:.2f}, completeness={result.completeness:.2f}")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"  Error: {e}")
                result = QuestionResult(
                    question_id=q["id"],
                    question=q["question"],
                    category=q["category"],
                    domain=q["domain"],
                    requires_web=q.get("requires_web", False),
                    config_name=config_name,
                    answer="",
                    confidence=0.0,
                    citations_count=0,
                    elapsed_seconds=round(elapsed, 2),
                    error=str(e),
                )

            results.question_results.append(result)

        results.total_time_seconds = time.time() - config_start
        agent.close()

        logger.info(f"\nConfig '{config_name}' complete:")
        logger.info(f"  Success rate: {results.success_rate*100:.1f}%")
        logger.info(f"  Avg correctness: {results.avg_correctness*100:.1f}%")
        logger.info(f"  Total time: {results.total_time_seconds:.1f}s")

        return results

    def run_all(
        self,
        config_names: Optional[List[str]] = None,
        quick: bool = False,
    ) -> Dict[str, ConfigResults]:
        """Run benchmark for all specified configurations."""
        config_names = config_names or list(CONFIGS.keys())

        # Quick mode: subset of questions
        questions = self.questions
        if quick:
            # Select 5 diverse questions for quick testing
            categories = {}
            for q in self.questions:
                cat = q["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(q)

            questions = []
            for cat, qs in categories.items():
                questions.append(qs[0])  # First from each category
            questions = questions[:5]
            logger.info(f"Quick mode: using {len(questions)} questions")

        all_results = {}
        total_start = time.time()

        for config_name in config_names:
            try:
                results = self.run_config(config_name, questions)
                all_results[config_name] = results
            except Exception as e:
                logger.error(f"Config '{config_name}' failed: {e}")

        total_time = time.time() - total_start

        # Save results
        self._save_results(all_results, total_time)

        # Print summary
        self._print_summary(all_results, total_time)

        return all_results

    def _save_results(
        self,
        results: Dict[str, ConfigResults],
        total_time: float,
    ):
        """Save benchmark results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"benchmark_{timestamp}.json")

        output = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "configs_run": list(results.keys()),
            "questions_per_config": len(self.questions),
            "results": {
                name: {
                    "config_name": r.config_name,
                    "description": r.description,
                    "success_rate": r.success_rate,
                    "avg_correctness": r.avg_correctness,
                    "avg_completeness": r.avg_completeness,
                    "avg_groundedness": r.avg_groundedness,
                    "avg_confidence": r.avg_confidence,
                    "total_time_seconds": r.total_time_seconds,
                    "category_breakdown": r.category_breakdown,
                    "question_results": [asdict(q) for q in r.question_results],
                }
                for name, r in results.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    def _print_summary(
        self,
        results: Dict[str, ConfigResults],
        total_time: float,
    ):
        """Print a summary table of results."""
        print("\n" + "="*80)
        print("HYBRID RAG BENCHMARK RESULTS")
        print("="*80)

        # Sort by avg_correctness descending
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].avg_correctness,
            reverse=True
        )

        print(f"\n{'Config':<20} {'Correct':>10} {'Complete':>10} {'Ground':>10} {'Success':>10} {'Time':>8}")
        print("-"*80)

        baseline_score = results.get("baseline", ConfigResults("", "")).avg_correctness

        for name, r in sorted_results:
            delta = r.avg_correctness - baseline_score
            delta_str = f"({delta:+.1%})" if name != "baseline" else ""
            print(
                f"{name:<20} "
                f"{r.avg_correctness:>9.1%} "
                f"{r.avg_completeness:>9.1%} "
                f"{r.avg_groundedness:>9.1%} "
                f"{r.success_rate:>9.1%} "
                f"{r.total_time_seconds:>7.0f}s "
                f"{delta_str}"
            )

        print("-"*80)

        # Category breakdown for best config
        if sorted_results:
            best_name, best_results = sorted_results[0]
            print(f"\nBest config: {best_name} ({best_results.avg_correctness*100:.1f}%)")
            print("\nCategory breakdown:")
            for cat, scores in best_results.category_breakdown.items():
                print(f"  {cat}: {scores['avg_correctness']*100:.1f}% ({scores['count']} questions)")

        print(f"\nTotal benchmark time: {total_time/60:.1f} minutes")
        print("="*80)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG Benchmark")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Run specific config (default: all)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Run multiple specific configs",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test with 5 questions",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM judge evaluation",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available configurations",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable configurations:")
        print("-"*60)
        for name, config in CONFIGS.items():
            print(f"  {name:<20} - {config['description']}")
        print()
        return

    benchmark = HybridRAGBenchmark(use_judge=not args.no_judge)

    if args.config:
        benchmark.run_config(args.config)
    elif args.configs:
        benchmark.run_all(config_names=args.configs, quick=args.quick)
    else:
        benchmark.run_all(quick=args.quick)


if __name__ == "__main__":
    main()
