#!/usr/bin/env python3
"""
Simple GE Vernova benchmark using text-based search (no vector index required).

Usage:
    python benchmarks/run_ge_vernova_simple.py [--graph test_kg]
"""

import json
import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_qa import ReActQAAgent, QAResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("planned_graphrag").setLevel(logging.WARNING)


def load_benchmark(filepath: str = "benchmarks/ge_vernova_questions.json") -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def run_benchmark(
    agent: ReActQAAgent,
    questions: list,
    output_file: str = None,
    verbose: bool = True,
) -> dict:
    """Run benchmark questions against the QA agent."""
    results = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected = q["expected_answer"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Q{i}/{len(questions)}: {question}")
            print(f"Expected: {expected}")
            print("-" * 60)

        try:
            response = agent.answer_question(question)

            result = {
                "id": q["id"],
                "question": question,
                "expected_answer": expected,
                "actual_answer": response.answer,
                "confidence": response.confidence,
                "external_info_used": response.external_info_used,
                "citations": [c.model_dump() for c in response.citations],
                "reasoning_steps": len(response.reasoning_steps),
                "category": q.get("category", "unknown"),
                "difficulty": q.get("difficulty", "unknown"),
                "status": "success",
            }

            if verbose:
                # Truncate answer for display
                answer_preview = response.answer[:300] + "..." if len(response.answer) > 300 else response.answer
                print(f"Answer: {answer_preview}")
                print(f"Confidence: {response.confidence:.2f}, Citations: {len(response.citations)}")

        except Exception as e:
            logger.error(f"Error on Q{i}: {e}")
            result = {
                "id": q["id"],
                "question": question,
                "expected_answer": expected,
                "actual_answer": None,
                "error": str(e),
                "status": "error",
                "category": q.get("category", "unknown"),
                "difficulty": q.get("difficulty", "unknown"),
            }

        results.append(result)

    # Calculate statistics
    successful = [r for r in results if r["status"] == "success"]
    stats = {
        "total_questions": len(questions),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_confidence": sum(r.get("confidence", 0) for r in successful) / len(successful) if successful else 0,
        "external_info_used_count": sum(1 for r in successful if r.get("external_info_used")),
    }

    output = {
        "benchmark_name": "GE Vernova KG QA Benchmark (Simple)",
        "run_timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "results": results,
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return output


def print_summary(output: dict):
    stats = output["statistics"]
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total: {stats['total_questions']} | Success: {stats['successful']} | Failed: {stats['failed']}")
    print(f"Avg Confidence: {stats['avg_confidence']:.2f}")
    print(f"External Info Used: {stats['external_info_used_count']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run GE Vernova QA benchmark (simple)")
    parser.add_argument("--graph", default="test_kg", help="FalkorDB graph name")
    parser.add_argument("--output", default=None, help="Output file for results")
    parser.add_argument("--questions", type=int, default=None, help="Number of questions")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("--difficulty", default=None, help="Filter by difficulty")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    benchmark = load_benchmark()
    questions = benchmark["questions"]

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.difficulty:
        questions = [q for q in questions if q["difficulty"] == args.difficulty]
    if args.questions:
        questions = questions[:args.questions]

    print(f"Running {len(questions)} questions on graph: {args.graph}")

    # Use simpler agent config - disable planning to use text search fallback
    agent = ReActQAAgent(
        backend="falkordb",
        graph_name=args.graph,
        web_search_enabled=True,
        wiki_search_enabled=True,
        use_retrieval_planning=False,  # Disable planning to use basic retrieval
        use_followup_planning=False,   # Disable follow-up planning
        use_improved_retrieval=False,  # Disable improved retrieval
        compression_enabled=False,
        skip_uncertainty=True,
        auto_add_documents=False,      # Don't try to add documents
    )

    output_file = args.output or f"benchmarks/results/simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = run_benchmark(
        agent=agent,
        questions=questions,
        output_file=output_file,
        verbose=not args.quiet,
    )

    print_summary(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
