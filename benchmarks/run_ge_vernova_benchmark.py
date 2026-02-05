#!/usr/bin/env python3
"""
Run GE Vernova benchmark questions against the ReAct QA agent.

Usage:
    python benchmarks/run_ge_vernova_benchmark.py [--backend falkordb] [--graph test_kg]
"""

import json
import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_qa import ReActQAAgent, QAResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_benchmark(filepath: str = "benchmarks/ge_vernova_questions.json") -> dict:
    """Load benchmark questions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_benchmark(
    agent: ReActQAAgent,
    questions: list,
    output_file: str = None,
    verbose: bool = True,
) -> dict:
    """
    Run benchmark questions against the QA agent.

    Args:
        agent: ReActQAAgent instance
        questions: List of question dictionaries
        output_file: Optional file to save results
        verbose: Print progress

    Returns:
        Dictionary with results and statistics
    """
    results = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected = q["expected_answer"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Question {i}/{len(questions)}: {question}")
            print(f"Expected: {expected}")
            print("-" * 60)

        try:
            # Run the agent
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

            # Add uncertainty scores if available
            if response.uncertainty:
                result["uncertainty"] = response.uncertainty.model_dump()

            if verbose:
                print(f"Answer: {response.answer[:500]}...")
                print(f"Confidence: {response.confidence:.2f}")
                print(f"Citations: {len(response.citations)}")
                print(f"External info: {response.external_info_used}")

        except Exception as e:
            logger.error(f"Error on question {i}: {e}")
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
        "avg_reasoning_steps": sum(r.get("reasoning_steps", 0) for r in successful) / len(successful) if successful else 0,
        "external_info_used_count": sum(1 for r in successful if r.get("external_info_used")),
        "by_category": {},
        "by_difficulty": {},
    }

    # Stats by category
    for cat in set(r["category"] for r in results):
        cat_results = [r for r in successful if r["category"] == cat]
        stats["by_category"][cat] = {
            "count": len(cat_results),
            "avg_confidence": sum(r.get("confidence", 0) for r in cat_results) / len(cat_results) if cat_results else 0,
        }

    # Stats by difficulty
    for diff in set(r["difficulty"] for r in results):
        diff_results = [r for r in successful if r["difficulty"] == diff]
        stats["by_difficulty"][diff] = {
            "count": len(diff_results),
            "avg_confidence": sum(r.get("confidence", 0) for r in diff_results) / len(diff_results) if diff_results else 0,
        }

    output = {
        "benchmark_name": "GE Vernova Knowledge Graph QA Benchmark",
        "run_timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "results": results,
    }

    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return output


def print_summary(output: dict):
    """Print benchmark summary."""
    stats = output["statistics"]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Questions: {stats['total_questions']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Reasoning Steps: {stats['avg_reasoning_steps']:.1f}")
    print(f"External Info Used: {stats['external_info_used_count']}")

    print("\nBy Category:")
    for cat, cat_stats in stats["by_category"].items():
        print(f"  {cat}: {cat_stats['count']} questions, avg confidence {cat_stats['avg_confidence']:.2f}")

    print("\nBy Difficulty:")
    for diff, diff_stats in stats["by_difficulty"].items():
        print(f"  {diff}: {diff_stats['count']} questions, avg confidence {diff_stats['avg_confidence']:.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run GE Vernova QA benchmark")
    parser.add_argument("--backend", default="falkordb", choices=["neo4j", "falkordb"],
                        help="Database backend")
    parser.add_argument("--graph", default="test_kg", help="FalkorDB graph name")
    parser.add_argument("--output", default=None, help="Output file for results")
    parser.add_argument("--questions", type=int, default=None,
                        help="Number of questions to run (default: all)")
    parser.add_argument("--category", default=None,
                        help="Filter by category (factual, technical, comparison, relationship)")
    parser.add_argument("--difficulty", default=None,
                        help="Filter by difficulty (easy, medium, hard)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    # Load benchmark
    benchmark = load_benchmark()
    questions = benchmark["questions"]

    # Filter questions
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.difficulty:
        questions = [q for q in questions if q["difficulty"] == args.difficulty]
    if args.questions:
        questions = questions[:args.questions]

    print(f"Running {len(questions)} questions...")
    print(f"Backend: {args.backend}, Graph: {args.graph}")

    # Initialize agent
    agent = ReActQAAgent(
        backend=args.backend,
        graph_name=args.graph,
        web_search_enabled=True,
        wiki_search_enabled=True,
        use_followup_planning=True,  # Use best retrieval strategy
        compression_enabled=True,
        skip_uncertainty=True,  # Skip for faster benchmarking
    )

    # Run benchmark
    output_file = args.output or f"benchmarks/results/ge_vernova_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output = run_benchmark(
        agent=agent,
        questions=questions,
        output_file=output_file,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
