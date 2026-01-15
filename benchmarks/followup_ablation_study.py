#!/usr/bin/env python3
"""
Ablation Study for Follow-Up Planning GraphRAG.

Study 1: Compare planning modes with different primary vec/hop values
- no_planning (baseline from previous best)
- old_planning (entity/relationship extraction)
- followup_planning (follow-up questions)
- followup_planning + reasoning (detailed thinking)

Primary search variations: v4_h4, v5_h5, v4_h6
Secondary search fixed at: v3_h2

Study 2: Best primary config, ablate secondary (follow-up) vec/hop values
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

# Add parent directory and benchmarks to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_qa import ReActQAAgent, QAResponse
from ablation_configs import AblationConfig, STUDY1_CONFIGS, generate_study2_configs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGS
# Imported from ablation_configs.py:
# - AblationConfig: Unified config dataclass
# - STUDY1_CONFIGS: 7 follow-up planning configs
# - generate_study2_configs: Function to generate Study 2 configs
# =============================================================================


# =============================================================================
# TEST QUESTIONS (subset for faster iteration)
# =============================================================================

TEST_QUESTIONS = [
    {
        "question": "How does expansionary monetary policy lead to inflation?",
        "domain": "economics",
        "expected_entities": ["Monetary Policy", "Inflation", "Aggregate Demand"],
        "difficulty": "medium",
    },
    {
        "question": "What is the relationship between supply and demand in determining market price?",
        "domain": "economics",
        "expected_entities": ["Supply", "Demand", "Price", "Market"],
        "difficulty": "easy",
    },
    {
        "question": "Explain Newton's second law of motion and its applications.",
        "domain": "physics",
        "expected_entities": ["Force", "Mass", "Acceleration"],
        "difficulty": "medium",
    },
    {
        "question": "How does contractual liability differ from tort liability?",
        "domain": "law",
        "expected_entities": ["Contract", "Tort", "Liability"],
        "difficulty": "medium",
    },
    {
        "question": "What factors cause shifts in the aggregate demand curve?",
        "domain": "economics",
        "expected_entities": ["Aggregate Demand", "GDP", "Government Spending"],
        "difficulty": "medium",
    },
    {
        "question": "How is momentum conserved in elastic collisions?",
        "domain": "physics",
        "expected_entities": ["Momentum", "Collision", "Energy"],
        "difficulty": "hard",
    },
]


def create_agent(config: AblationConfig) -> ReActQAAgent:
    """Create an agent with the given configuration."""
    return ReActQAAgent(
        web_search_enabled=config.web_search_enabled,
        auto_add_documents=False,
        use_retrieval_planning=config.use_retrieval_planning,
        compression_enabled=config.compression_enabled,
        use_improved_retrieval=not config.use_followup_planning and not config.use_retrieval_planning,
        use_followup_planning=config.use_followup_planning,
        planning_reasoning=config.planning_reasoning,
        max_hops=config.max_hops,
        vector_limit=config.vector_limit,
        primary_vector_limit=config.primary_vector_limit,
        primary_max_hops=config.primary_max_hops,
        secondary_vector_limit=config.secondary_vector_limit,
        secondary_max_hops=config.secondary_max_hops,
    )


def run_single_config(config: AblationConfig, questions: List[Dict]) -> Dict[str, Any]:
    """Run evaluation for a single configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"{'='*60}")

    agent = create_agent(config)

    results = {
        "config": asdict(config),
        "questions": [],
        "metrics": {
            "avg_confidence": 0.0,
            "avg_response_time": 0.0,
            "total_questions": len(questions),
            "successful_responses": 0,
        }
    }

    total_confidence = 0.0
    total_time = 0.0

    for i, q in enumerate(questions, 1):
        question = q["question"]
        logger.info(f"\n[{i}/{len(questions)}] {question[:60]}...")

        start_time = time.time()
        try:
            response = agent.answer_question(question)
            elapsed = time.time() - start_time

            results["questions"].append({
                "question": question,
                "domain": q.get("domain", "unknown"),
                "answer": response.answer[:500],
                "confidence": response.confidence,
                "response_time": elapsed,
                "iterations": getattr(response, 'iterations', 0),
                "external_info_used": getattr(response, 'external_info_used', False),
                "citations_count": len(response.citations) if response.citations else 0,
                "success": True,
            })

            total_confidence += response.confidence
            total_time += elapsed
            results["metrics"]["successful_responses"] += 1

            logger.info(f"  Confidence: {response.confidence:.2f}, Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  ERROR: {e}")
            results["questions"].append({
                "question": question,
                "domain": q.get("domain", "unknown"),
                "error": str(e),
                "response_time": elapsed,
                "success": False,
            })
            total_time += elapsed

    # Calculate averages
    successful = results["metrics"]["successful_responses"]
    if successful > 0:
        results["metrics"]["avg_confidence"] = total_confidence / successful
    results["metrics"]["avg_response_time"] = total_time / len(questions)

    return results


def run_study(study_name: str, configs: List[AblationConfig], questions: List[Dict]) -> Dict[str, Any]:
    """Run a complete ablation study."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study_results = {
        "study_name": study_name,
        "timestamp": timestamp,
        "total_configs": len(configs),
        "questions_per_config": len(questions),
        "results": [],
    }

    for i, config in enumerate(configs, 1):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Config {i}/{len(configs)}: {config.name}")
        logger.info(f"{'#'*70}")

        result = run_single_config(config, questions)
        study_results["results"].append(result)

        # Save intermediate results
        output_file = f"benchmarks/results/followup_ablation_{study_name}_{timestamp}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(study_results, f, indent=2)
        logger.info(f"Intermediate results saved to: {output_file}")

    return study_results


def print_study_summary(study_results: Dict[str, Any]):
    """Print a summary table of study results."""
    print("\n" + "="*100)
    print(f"STUDY SUMMARY: {study_results['study_name']}")
    print("="*100)

    # Sort by average confidence
    sorted_results = sorted(
        study_results["results"],
        key=lambda x: x["metrics"]["avg_confidence"],
        reverse=True
    )

    print(f"\n{'Config':<35} {'Confidence':>12} {'Avg Time':>12} {'Success':>10}")
    print("-"*70)

    for result in sorted_results:
        config = result["config"]
        metrics = result["metrics"]

        name = config["name"][:35]
        conf = f"{metrics['avg_confidence']*100:.1f}%"
        time_s = f"{metrics['avg_response_time']:.1f}s"
        success = f"{metrics['successful_responses']}/{metrics['total_questions']}"

        print(f"{name:<35} {conf:>12} {time_s:>12} {success:>10}")

    # Find best config
    best = sorted_results[0]
    print(f"\nBest config: {best['config']['name']}")
    print(f"  Avg confidence: {best['metrics']['avg_confidence']*100:.1f}%")

    return best


def main():
    """Run the ablation studies."""
    import argparse

    parser = argparse.ArgumentParser(description="Follow-up Planning Ablation Study")
    parser.add_argument("--study", type=int, choices=[1, 2], default=1,
                       help="Study to run: 1=planning modes, 2=secondary params")
    parser.add_argument("--best-primary-vec", type=int, default=5,
                       help="Best primary vector limit (for study 2)")
    parser.add_argument("--best-primary-hops", type=int, default=5,
                       help="Best primary max hops (for study 2)")
    parser.add_argument("--use-reasoning", action="store_true",
                       help="Use reasoning for study 2")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer questions")
    args = parser.parse_args()

    questions = TEST_QUESTIONS[:3] if args.quick else TEST_QUESTIONS

    if args.study == 1:
        logger.info("="*70)
        logger.info("STUDY 1: Planning Modes + Primary Vec/Hop Variations")
        logger.info("="*70)

        study_results = run_study("study1_planning_modes", STUDY1_CONFIGS, questions)
        best = print_study_summary(study_results)

        # Extract best primary config for study 2
        best_config = best["config"]
        print(f"\nFor Study 2, use:")
        print(f"  --best-primary-vec {best_config.get('primary_vector_limit', 5)}")
        print(f"  --best-primary-hops {best_config.get('primary_max_hops', 4)}")
        if best_config.get("planning_reasoning"):
            print(f"  --use-reasoning")

    elif args.study == 2:
        logger.info("="*70)
        logger.info("STUDY 2: Secondary (Follow-up) Vec/Hop Variations")
        logger.info(f"Using best primary: v{args.best_primary_vec}_h{args.best_primary_hops}")
        logger.info(f"Reasoning: {args.use_reasoning}")
        logger.info("="*70)

        configs = generate_study2_configs(
            args.best_primary_vec,
            args.best_primary_hops,
            args.use_reasoning
        )

        study_results = run_study("study2_secondary_params", configs, questions)
        print_study_summary(study_results)

    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
