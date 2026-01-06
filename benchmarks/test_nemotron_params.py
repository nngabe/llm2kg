#!/usr/bin/env python3
"""
A/B Test for Nemotron-3-Nano Parameter Changes.

Compares:
- OLD: temp=0, no thinking mode
- NEW: temp=0.6, top_p=0.95, thinking mode enabled

Tests on 2 questions to measure quality differences.
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, "/app")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
MODEL = "nemotron-3-nano:30b"

# Test prompts - one reasoning task, one tool-calling style task
TEST_PROMPTS = [
    {
        "id": "reasoning_task",
        "system": "You are an economic analyst. Answer questions about economics clearly and accurately.",
        "question": "Explain how inflation affects purchasing power and provide a concrete example.",
        "evaluate": ["purchasing power", "example", "clear explanation"],
    },
    {
        "id": "tool_selection_task",
        "system": """You have access to these tools:
1. graph_lookup(entity_name) - Look up an entity
2. web_search(query) - Search the web
3. cypher_query(query) - Query the database

Respond with JSON: {"thought": "your reasoning", "tool": "tool_name", "args": {...}}""",
        "question": "I need to find information about 'Aggregate Demand' and its effects. What tool should I use first?",
        "evaluate": ["graph_lookup", "Aggregate Demand", "JSON format"],
    },
]


def test_with_settings(temp: float, top_p: float, thinking_mode: bool, test_name: str):
    """Run tests with specific settings."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {test_name}")
    logger.info(f"Settings: temp={temp}, top_p={top_p}, thinking_mode={thinking_mode}")
    logger.info(f"{'='*60}")

    llm = ChatOllama(
        model=MODEL,
        base_url=OLLAMA_HOST,
        temperature=temp,
        top_p=top_p,
        num_ctx=4096,
    )

    results = []

    for test in TEST_PROMPTS:
        logger.info(f"\n--- Test: {test['id']} ---")

        # Build system prompt with or without thinking mode
        system_content = test["system"]
        if thinking_mode:
            system_content = "detailed thinking on\n\n" + system_content + "\n\nThink step by step before responding."

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=test["question"]),
        ]

        start_time = time.time()
        try:
            response = llm.invoke(messages)
            elapsed = time.time() - start_time
            content = response.content

            # Score based on evaluation criteria
            score = 0
            criteria_met = []
            for criterion in test["evaluate"]:
                if criterion.lower() in content.lower():
                    score += 1
                    criteria_met.append(criterion)

            total_criteria = len(test["evaluate"])
            score_pct = (score / total_criteria) * 100 if total_criteria > 0 else 0

            result = {
                "test_id": test["id"],
                "settings": test_name,
                "elapsed_seconds": round(elapsed, 2),
                "response_length": len(content),
                "criteria_met": criteria_met,
                "criteria_score": f"{score}/{total_criteria}",
                "score_pct": score_pct,
                "response_preview": content[:300].replace("\n", " "),
            }
            results.append(result)

            logger.info(f"  Time: {elapsed:.2f}s")
            logger.info(f"  Length: {len(content)} chars")
            logger.info(f"  Criteria met: {score}/{total_criteria} ({score_pct:.0f}%)")
            logger.info(f"  Met: {criteria_met}")
            logger.info(f"  Preview: {content[:150]}...")

        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                "test_id": test["id"],
                "settings": test_name,
                "error": str(e),
                "score_pct": 0,
            })

    return results


def main():
    logger.info("=" * 70)
    logger.info("NEMOTRON-3-NANO PARAMETER A/B TEST")
    logger.info("=" * 70)

    all_results = []

    # Test A: OLD settings (temp=0, no thinking mode)
    old_results = test_with_settings(
        temp=0.0,
        top_p=1.0,
        thinking_mode=False,
        test_name="OLD (temp=0, no thinking)"
    )
    all_results.extend(old_results)

    # Test B: NEW settings (temp=0.6, top_p=0.95, thinking mode)
    new_results = test_with_settings(
        temp=0.6,
        top_p=0.95,
        thinking_mode=True,
        test_name="NEW (temp=0.6, top_p=0.95, thinking)"
    )
    all_results.extend(new_results)

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Test':<25} {'Setting':<35} {'Score':<10} {'Time':<10}")
    print("-" * 80)

    for r in all_results:
        test_id = r.get("test_id", "unknown")
        settings = r.get("settings", "unknown")
        score = r.get("criteria_score", "N/A")
        elapsed = r.get("elapsed_seconds", "N/A")
        print(f"{test_id:<25} {settings:<35} {score:<10} {elapsed}s")

    # Calculate averages
    old_scores = [r["score_pct"] for r in old_results if "score_pct" in r]
    new_scores = [r["score_pct"] for r in new_results if "score_pct" in r]

    old_avg = sum(old_scores) / len(old_scores) if old_scores else 0
    new_avg = sum(new_scores) / len(new_scores) if new_scores else 0

    old_times = [r["elapsed_seconds"] for r in old_results if "elapsed_seconds" in r]
    new_times = [r["elapsed_seconds"] for r in new_results if "elapsed_seconds" in r]

    old_time_avg = sum(old_times) / len(old_times) if old_times else 0
    new_time_avg = sum(new_times) / len(new_times) if new_times else 0

    print("-" * 80)
    print(f"\n{'SUMMARY':^80}")
    print("-" * 80)
    print(f"{'Metric':<30} {'OLD Settings':<20} {'NEW Settings':<20}")
    print("-" * 80)
    print(f"{'Average Score':<30} {old_avg:>18.1f}% {new_avg:>18.1f}%")
    print(f"{'Average Response Time':<30} {old_time_avg:>17.1f}s {new_time_avg:>17.1f}s")
    print("-" * 80)

    improvement = new_avg - old_avg
    if improvement > 0:
        print(f"\n✓ NEW settings improved score by {improvement:.1f}%")
    elif improvement < 0:
        print(f"\n✗ NEW settings decreased score by {abs(improvement):.1f}%")
    else:
        print(f"\n= No change in score")

    time_diff = new_time_avg - old_time_avg
    if time_diff > 0:
        print(f"  (Response time increased by {time_diff:.1f}s due to thinking mode)")
    else:
        print(f"  (Response time decreased by {abs(time_diff):.1f}s)")

    # Save detailed results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "tests": TEST_PROMPTS,
        "results": all_results,
        "summary": {
            "old_avg_score": old_avg,
            "new_avg_score": new_avg,
            "old_avg_time": old_time_avg,
            "new_avg_time": new_time_avg,
            "improvement": improvement,
        }
    }

    output_path = "/app/benchmarks/nemotron_param_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
