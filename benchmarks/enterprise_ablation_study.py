#!/usr/bin/env python3
"""
Enterprise Ablation Study for agent_qa.py

Tests the impact of different agent features using the full 4-layer
enterprise evaluation framework with 14 metrics.

Ablation Configurations:
1. baseline       - All features ON (default)
2. no_planning    - use_retrieval_planning=False
3. no_compression - compression_enabled=False
4. no_web         - web_search_enabled=False
5. no_auto_ingest - auto_add_documents=False
6. minimal        - All features OFF

Usage:
    python benchmarks/enterprise_ablation_study.py
    python benchmarks/enterprise_ablation_study.py --configs baseline no_planning
    python benchmarks/enterprise_ablation_study.py --quick  # 4 test cases only
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/benchmarks")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


# ============================================================================
# ABLATION CONFIGURATIONS
# ============================================================================

ABLATION_CONFIGS = {
    "baseline": {
        "description": "All features ON (default)",
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": True,
    },
    "no_planning": {
        "description": "Disable retrieval planning",
        "use_retrieval_planning": False,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": True,
    },
    "no_compression": {
        "description": "Disable context compression",
        "use_retrieval_planning": True,
        "compression_enabled": False,
        "web_search_enabled": True,
        "auto_add_documents": True,
    },
    "no_web": {
        "description": "Disable web search",
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": False,
        "auto_add_documents": False,  # Also disable since it depends on web
    },
    "no_auto_ingest": {
        "description": "Disable auto document ingestion",
        "use_retrieval_planning": True,
        "compression_enabled": True,
        "web_search_enabled": True,
        "auto_add_documents": False,
    },
    "minimal": {
        "description": "All features OFF (graph lookup only)",
        "use_retrieval_planning": False,
        "compression_enabled": False,
        "web_search_enabled": False,
        "auto_add_documents": False,
    },
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LayerScore:
    """Score for a single evaluation layer."""
    layer_name: str
    score: float
    passed: bool
    metric_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Result from running one ablation configuration."""
    config_name: str
    config_description: str
    timestamp: str
    duration_ms: int
    test_case_count: int
    overall_score: float
    overall_passed: bool
    layer_scores: List[LayerScore] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_test_cases(path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif "test_cases" in data:
        return data["test_cases"]
    else:
        raise ValueError(f"Invalid test case format in {path}")


def _extract_step_data(step) -> Dict[str, Any]:
    """Extract thought step data from either dict or ThoughtStep object."""
    if isinstance(step, dict):
        thought = step.get("thought", "")
        action = step.get("action")
        observation = step.get("observation", "")
        if action:
            if isinstance(action, dict):
                action_data = {"tool_name": action.get("tool_name", ""), "arguments": action.get("arguments", {})}
            else:
                action_data = {"tool_name": getattr(action, 'tool_name', ''), "arguments": getattr(action, 'arguments', {})}
        else:
            action_data = None
    else:
        thought = getattr(step, 'thought', '')
        observation = getattr(step, 'observation', '')
        action = getattr(step, 'action', None)
        if action:
            action_data = {"tool_name": getattr(action, 'tool_name', ''), "arguments": getattr(action, 'arguments', {})}
        else:
            action_data = None
    return {"thought": thought, "action": action_data, "observation": observation}


def _extract_context_items(response) -> List[Dict[str, Any]]:
    """Extract context items from reasoning steps for Graph Traversal metric."""
    context_items = []

    if hasattr(response, 'reasoning_steps') and response.reasoning_steps:
        for step in response.reasoning_steps:
            step_data = _extract_step_data(step)
            action = step_data.get("action")
            observation = step_data.get("observation", "")

            if action and action.get("tool_name") == "graph_lookup":
                entity_name = action.get("arguments", {}).get("entity_name", "")
                if entity_name:
                    context_items.append({
                        "source_type": "entity",
                        "source_id": entity_name,
                        "content": observation,
                        "relevance_score": 1.0,
                        "metadata": {"entity_name": entity_name},
                    })
            elif action and action.get("tool_name") == "cypher_query":
                query = action.get("arguments", {}).get("query", "")
                context_items.append({
                    "source_type": "neo4j",
                    "source_id": "cypher_query",
                    "content": observation,
                    "relevance_score": 1.0,
                    "metadata": {"query": query},
                })

    if hasattr(response, 'citations') and response.citations:
        for c in response.citations:
            source_type = getattr(c, 'source_type', '') if hasattr(c, 'source_type') else c.get('source_type', '')
            if source_type == "graph":
                source_id = getattr(c, 'source_id', '') if hasattr(c, 'source_id') else c.get('source_id', '')
                context_items.append({
                    "source_type": "entity",
                    "source_id": source_id,
                    "content": getattr(c, 'excerpt', '') if hasattr(c, 'excerpt') else c.get('excerpt', ''),
                    "relevance_score": 1.0,
                    "metadata": {"entity_name": source_id},
                })

    return context_items


def run_agent_on_test_cases(config_name: str, config: Dict, test_cases: List[Dict]) -> Tuple[List[Any], List[Any], int]:
    """Run the agent with given config on all test cases."""
    from agent_qa import ReActQAAgent
    from enterprise_eval.metrics.base import TestCase, AgentOutput
    import time

    logger.info(f"  Initializing agent with config: {config_name}")

    start_time = time.time()

    agent = ReActQAAgent(
        use_retrieval_planning=config["use_retrieval_planning"],
        compression_enabled=config["compression_enabled"],
        web_search_enabled=config["web_search_enabled"],
        auto_add_documents=config["auto_add_documents"],
    )

    test_case_objects = []
    agent_outputs = []

    for i, tc_data in enumerate(test_cases):
        question = tc_data.get("question", "")
        logger.info(f"    [{i+1}/{len(test_cases)}] {question[:50]}...")

        # Create TestCase object
        test_case = TestCase(
            id=tc_data.get("id", f"tc_{i}"),
            question=question,
            expected_answer=tc_data.get("expected_answer"),
            expected_entities=tc_data.get("expected_entities", []),
            expected_relationships=tc_data.get("expected_relationships", []),
            ground_truth_context=tc_data.get("ground_truth_context", []),
            optimal_tool_sequence=tc_data.get("optimal_tool_sequence", []),
            minimum_steps=tc_data.get("minimum_steps"),
            type=tc_data.get("type", "all"),
            difficulty=tc_data.get("difficulty", "medium"),
            should_reject=tc_data.get("should_reject", False),
            rejection_reason=tc_data.get("rejection_reason"),
            metadata=tc_data.get("metadata", {}),
        )
        test_case_objects.append(test_case)

        # Run agent
        try:
            response = agent.answer_question(question)

            # Extract citations
            citations = []
            if hasattr(response, 'citations') and response.citations:
                for c in response.citations:
                    if isinstance(c, dict):
                        citations.append(c)
                    else:
                        citations.append({
                            "source_type": c.source_type,
                            "source_id": c.source_id,
                            "source_title": getattr(c, 'source_title', None),
                            "trust_level": c.trust_level,
                            "excerpt": c.excerpt,
                        })

            # Extract thought history
            thought_history = []
            if hasattr(response, 'reasoning_steps') and response.reasoning_steps:
                for step in response.reasoning_steps:
                    thought_history.append(_extract_step_data(step))

            # Extract context items for graph traversal metric
            context_items = _extract_context_items(response)

            agent_output = AgentOutput(
                question=question,
                answer=response.answer,
                confidence=response.confidence,
                citations=citations,
                thought_history=thought_history,
                context_items=context_items,
                external_info_used=response.external_info_used,
            )

            logger.info(f"      Confidence: {response.confidence:.2f}, Citations: {len(citations)}")

        except Exception as e:
            logger.error(f"      Error: {e}")
            agent_output = AgentOutput(
                question=question,
                answer=f"ERROR: {e}",
                confidence=0.0,
            )

        agent_outputs.append(agent_output)

    agent.close()

    duration_ms = int((time.time() - start_time) * 1000)
    return test_case_objects, agent_outputs, duration_ms


def run_evaluation(test_cases: List, agent_outputs: List) -> Dict[str, Any]:
    """Run the enterprise evaluation suite."""
    from enterprise_eval import EvalConfig
    from enterprise_eval.runner import create_default_runner

    config = EvalConfig()
    runner = create_default_runner(config)

    result = runner.run_evaluation(test_cases, agent_outputs)

    runner.close()
    return result


def extract_layer_scores(eval_result) -> List[LayerScore]:
    """Extract layer scores from evaluation result."""
    layer_scores = []

    # The result structure has 'layers' dict with layer names as keys
    layers_data = getattr(eval_result, 'layers', {})

    if isinstance(layers_data, dict):
        for layer_name, layer_data in layers_data.items():
            if hasattr(layer_data, 'score'):
                metric_scores = {}
                if hasattr(layer_data, 'metric_results'):
                    for m in layer_data.metric_results:
                        metric_scores[m.name] = m.score

                layer_scores.append(LayerScore(
                    layer_name=layer_name,
                    score=layer_data.score,
                    passed=layer_data.passed,
                    metric_scores=metric_scores,
                ))

    return layer_scores


# ============================================================================
# MAIN ABLATION RUNNER
# ============================================================================

def run_ablation_study(
    config_names: List[str] = None,
    test_cases_path: str = None,
    quick_mode: bool = False,
) -> List[AblationResult]:
    """Run the complete ablation study."""

    # Default to all configs
    if config_names is None:
        config_names = list(ABLATION_CONFIGS.keys())

    # Default test cases path
    if test_cases_path is None:
        test_cases_path = "/app/benchmarks/enterprise_eval/golden/datasets/comprehensive_v1.json"

    logger.info("=" * 70)
    logger.info("ENTERPRISE ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(f"Configs to test: {config_names}")
    logger.info(f"Test cases path: {test_cases_path}")
    logger.info(f"Quick mode: {quick_mode}")

    # Load test cases
    logger.info("\nLoading test cases...")
    all_test_cases = load_test_cases(test_cases_path)

    # In quick mode, take only 1 per layer type
    if quick_mode:
        layer_types = ["retrieval", "agentic", "integrity", "generation"]
        quick_cases = []
        for layer in layer_types:
            for tc in all_test_cases:
                if tc.get("type") == layer:
                    quick_cases.append(tc)
                    break
        all_test_cases = quick_cases
        logger.info(f"Quick mode: Using {len(all_test_cases)} test cases (1 per layer)")
    else:
        logger.info(f"Loaded {len(all_test_cases)} test cases")

    # Run each configuration
    all_results = []

    for config_name in config_names:
        if config_name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown config: {config_name}, skipping")
            continue

        config = ABLATION_CONFIGS[config_name]
        logger.info(f"\n{'='*70}")
        logger.info(f"Running config: {config_name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*70}")

        timestamp = datetime.now().isoformat()

        try:
            # Run agent on all test cases
            test_case_objects, agent_outputs, duration_ms = run_agent_on_test_cases(
                config_name, config, all_test_cases
            )

            # Run evaluation
            logger.info(f"  Running enterprise evaluation...")
            eval_result = run_evaluation(test_case_objects, agent_outputs)

            # Extract scores
            layer_scores = extract_layer_scores(eval_result)

            result = AblationResult(
                config_name=config_name,
                config_description=config["description"],
                timestamp=timestamp,
                duration_ms=duration_ms,
                test_case_count=len(all_test_cases),
                overall_score=eval_result.overall_score,
                overall_passed=eval_result.overall_passed,
                layer_scores=layer_scores,
            )

            logger.info(f"  Overall score: {eval_result.overall_score:.1%}")

        except Exception as e:
            logger.error(f"  Config {config_name} failed: {e}")
            result = AblationResult(
                config_name=config_name,
                config_description=config["description"],
                timestamp=timestamp,
                duration_ms=0,
                test_case_count=len(all_test_cases),
                overall_score=0.0,
                overall_passed=False,
                errors=[str(e)],
            )

        all_results.append(result)

    return all_results


def print_comparison_report(results: List[AblationResult]):
    """Print a comparison report of all ablation results."""

    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    # Get all layer names from first result with layer scores
    layer_names = []
    for r in results:
        if r.layer_scores:
            layer_names = [ls.layer_name for ls in r.layer_scores]
            break

    # Header
    header = f"{'Config':<15}"
    for layer in layer_names:
        header += f" {layer[:8]:>10}"
    header += f" {'Overall':>10} {'Status':>8}"
    print(header)
    print("-" * 80)

    # Rows
    for r in results:
        row = f"{r.config_name:<15}"

        layer_score_map = {ls.layer_name: ls.score for ls in r.layer_scores}

        for layer in layer_names:
            score = layer_score_map.get(layer, 0)
            row += f" {score:>9.1%}"

        status = "PASS" if r.overall_passed else "FAIL"
        row += f" {r.overall_score:>9.1%} {status:>8}"
        print(row)

    print("-" * 80)

    # Feature impact analysis
    print("\n" + "=" * 80)
    print("FEATURE IMPACT ANALYSIS (Delta vs Baseline)")
    print("=" * 80)

    baseline = None
    for r in results:
        if r.config_name == "baseline":
            baseline = r
            break

    if baseline:
        print(f"\n{'Config':<15} {'Delta':>10} {'Impact':<30}")
        print("-" * 60)

        for r in results:
            if r.config_name == "baseline":
                continue

            delta = r.overall_score - baseline.overall_score
            impact = "Positive" if delta > 0 else "Negative" if delta < 0 else "None"
            feature = r.config_name.replace("no_", "").replace("_", " ").title()

            print(f"{r.config_name:<15} {delta:>+9.1%} {feature} feature {impact.lower()}")

    # Per-metric breakdown for key configs
    print("\n" + "=" * 80)
    print("PER-METRIC BREAKDOWN")
    print("=" * 80)

    for r in results:
        print(f"\n{r.config_name} ({r.config_description}):")
        for ls in r.layer_scores:
            print(f"  {ls.layer_name}: {ls.score:.1%}")
            for metric_name, metric_score in ls.metric_scores.items():
                status = "+" if metric_score >= 0.7 else "-"
                print(f"    [{status}] {metric_name}: {metric_score:.1%}")

    print("\n" + "=" * 80)


def save_results(results: List[AblationResult], output_dir: str = "benchmarks/enterprise_eval/reports"):
    """Save ablation results to JSON."""
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/ablation_study_{timestamp}.json"

    # Convert to serializable format
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Enterprise Ablation Study for agent_qa.py")

    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific configs to run (default: all)",
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default=None,
        help="Path to test cases JSON",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only 4 test cases (1 per layer)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/enterprise_eval/reports",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Run ablation study
    results = run_ablation_study(
        config_names=args.configs,
        test_cases_path=args.test_cases,
        quick_mode=args.quick,
    )

    # Print comparison report
    print_comparison_report(results)

    # Save results
    save_results(results, args.output_dir)

    logger.info("\nAblation study complete!")


if __name__ == "__main__":
    main()
