#!/usr/bin/env python3
"""
Merged Ablation Study for Hybrid RAG with Keywords + Vector + Fixed Hop Sampling

Tests 7 configurations using:
- 20 realistic grad-student level questions
- 4-layer agent evaluation framework (14 metrics)
- ImprovedGraphRAG: Keywords + Vector → Fixed N-hop Sampling (no LLM Cypher)
- Gemini 2.5 Pro as LLM-as-judge (GPT-5.2 fallback)

Key findings from prior studies:
- Optimal vec: 5 (more vectors = worse quality due to noise)
- Optimal hop: 3 (deeper traversal = better reasoning)
- ALL vec/hop tests now run with ALL tools enabled for fair comparison

Configurations (7 total):

Feature Flag Tests (vec=5, hop=2, all tools ON):
  baseline       - All features ON (default)
  no_planning    - Disable retrieval planning
  no_compression - Disable context compression

Vec/Hop Grid Search (all tools ON):
  v5_h3          - Best performer: vec=5, hop=3
  v4_h4          - Deeper traversal: vec=4, hop=4
  v7_h3          - Wider deep: vec=7, hop=3

Edge Case:
  minimal        - All features OFF (graph only)

Usage:
    python benchmarks/improved_ablation_study.py --test-run  # 6 hard questions
    python benchmarks/improved_ablation_study.py --full      # 7 configs x 20 questions
    python benchmarks/improved_ablation_study.py --configs baseline v5_h3
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

from ablation_configs import AblationConfig, STUDY1_CONFIGS, IMPROVED_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


# ============================================================================
# CONFIGURATIONS
# Imported from ablation_configs.py:
# - IMPROVED_CONFIGS: Hybrid RAG variants (7 configs)
# - STUDY1_CONFIGS: Follow-up planning variants (7 configs)
# ============================================================================

# 6 hard test questions for initial validation
TEST_QUESTION_IDS = [
    "econ_multihop_01",   # Multi-hop: monetary policy → inflation causal chain
    "law_multihop_01",    # Multi-hop: fiduciary duty → derivative suits
    "physics_multihop_01",  # Multi-hop: entropy → spontaneity
    "econ_compare_01",    # Comparison: Keynesian vs Neoclassical
    "econ_current_01",    # Current events: US inflation rate
    "physics_compare_02", # Comparison: Special vs General relativity
]


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
    config_params: Dict[str, Any]
    timestamp: str
    duration_ms: int
    test_case_count: int
    overall_score: float
    overall_passed: bool
    layer_scores: List[LayerScore] = field(default_factory=list)
    per_category_scores: Dict[str, float] = field(default_factory=dict)
    cypher_errors: int = 0  # Track Cypher errors (should be 0 with improved)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_realistic_questions(path: str, test_ids: List[str] = None) -> List[Dict[str, Any]]:
    """Load questions from the realistic QA dataset."""
    with open(path, "r") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    # Filter to specific test IDs if provided
    if test_ids:
        questions = [q for q in questions if q.get("id") in test_ids]
        # Preserve order
        id_to_q = {q["id"]: q for q in questions}
        questions = [id_to_q[tid] for tid in test_ids if tid in id_to_q]

    # Convert to test case format
    test_cases = []
    for q in questions:
        test_case = {
            "id": q.get("id", f"q_{len(test_cases)}"),
            "question": q.get("question", ""),
            "expected_answer": None,  # Not using exact match
            "expected_entities": q.get("expected_entities", []),
            "expected_relationships": [],
            "ground_truth_context": q.get("ground_truth_key_facts", []),
            "optimal_tool_sequence": ["graph_lookup"],  # Simplified
            "minimum_steps": 3,
            "type": "all",  # Evaluate on all layers
            "difficulty": q.get("difficulty", "medium"),
            "should_reject": False,
            "rejection_reason": None,
            "metadata": {
                "domain": q.get("domain"),
                "category": q.get("category"),  # Preserve category for analysis
                "requires_web": q.get("requires_web", False),
            },
        }
        test_cases.append(test_case)

    return test_cases


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
    """Extract context items from reasoning steps."""
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


def run_agent_with_config(config: AblationConfig, test_cases: List[Dict]) -> Tuple[List, List, int, int]:
    """Run the agent with given config on all test cases."""
    from agent_qa import ReActQAAgent
    from agent_eval.metrics.base import TestCase, AgentOutput
    import time

    logger.info(f"  Initializing agent: {config.name}")
    logger.info(f"    use_followup_planning={config.use_followup_planning}")
    logger.info(f"    use_retrieval_planning={config.use_retrieval_planning}")
    if config.use_followup_planning:
        logger.info(f"    primary: vec={config.primary_vector_limit}, hops={config.primary_max_hops}")
        logger.info(f"    secondary: vec={config.secondary_vector_limit}, hops={config.secondary_max_hops}")
    else:
        logger.info(f"    vector_limit={config.vector_limit}, max_hops={config.max_hops}")

    start_time = time.time()
    cypher_errors = 0

    # Determine use_improved_retrieval based on planning mode
    # If using followup or old planning, don't use improved retrieval
    use_improved = config.use_improved_retrieval
    if config.use_followup_planning or config.use_retrieval_planning:
        use_improved = False

    agent = ReActQAAgent(
        use_retrieval_planning=config.use_retrieval_planning,
        use_followup_planning=config.use_followup_planning,
        planning_reasoning=config.planning_reasoning,
        compression_enabled=config.compression_enabled,
        web_search_enabled=config.web_search_enabled,
        auto_add_documents=config.auto_add_documents,
        use_improved_retrieval=use_improved,
        max_hops=config.max_hops,
        vector_limit=config.vector_limit,
        primary_vector_limit=config.primary_vector_limit,
        primary_max_hops=config.primary_max_hops,
        secondary_vector_limit=config.secondary_vector_limit,
        secondary_max_hops=config.secondary_max_hops,
    )

    test_case_objects = []
    agent_outputs = []

    for i, tc_data in enumerate(test_cases):
        question = tc_data.get("question", "")
        q_id = tc_data.get("id", f"q_{i}")
        logger.info(f"    [{i+1}/{len(test_cases)}] {q_id}: {question[:50]}...")

        # Create TestCase object
        test_case = TestCase(
            id=q_id,
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

            # Check for Cypher errors in reasoning steps
            if hasattr(response, 'reasoning_steps') and response.reasoning_steps:
                for step in response.reasoning_steps:
                    obs = getattr(step, 'observation', '') or ''
                    if isinstance(obs, str):
                        if 'SyntaxError' in obs or 'CypherSyntaxError' in obs:
                            cypher_errors += 1
                            logger.warning(f"      Cypher error detected!")

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

            # Extract context items
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

            logger.info(f"      Conf: {response.confidence:.2f}, Cites: {len(citations)}, Context: {len(context_items)}")

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
    return test_case_objects, agent_outputs, duration_ms, cypher_errors


def run_evaluation(test_cases: List, agent_outputs: List) -> Dict[str, Any]:
    """Run the agent evaluation suite."""
    from agent_eval import EvalConfig
    from agent_eval.runner import create_default_runner

    config = EvalConfig()
    runner = create_default_runner(config)

    result = runner.run_evaluation(test_cases, agent_outputs)

    runner.close()
    return result


def extract_layer_scores(eval_result) -> List[LayerScore]:
    """Extract layer scores from evaluation result."""
    layer_scores = []

    layers_data = getattr(eval_result, 'layers', {})

    if isinstance(layers_data, dict):
        for layer_key, layer_data in layers_data.items():
            # Handle both EvaluationLayer enum keys and string keys
            layer_name = layer_key.value if hasattr(layer_key, 'value') else str(layer_key)

            # LayerResult uses 'overall_score' not 'score'
            if hasattr(layer_data, 'overall_score'):
                metric_scores = {}
                # LayerResult uses 'metrics' not 'metric_results'
                if hasattr(layer_data, 'metrics'):
                    for m in layer_data.metrics:
                        metric_name = getattr(m, 'metric_name', getattr(m, 'name', 'unknown'))
                        metric_scores[metric_name] = m.score

                layer_scores.append(LayerScore(
                    layer_name=layer_name,
                    score=layer_data.overall_score,
                    passed=layer_data.passed,
                    metric_scores=metric_scores,
                ))

    return layer_scores


def calculate_category_scores(test_cases: List, agent_outputs: List) -> Dict[str, float]:
    """Calculate average confidence per question category."""
    from collections import defaultdict

    category_scores = defaultdict(list)

    for tc, ao in zip(test_cases, agent_outputs):
        category = getattr(tc, 'type', 'unknown')
        confidence = getattr(ao, 'confidence', 0.0)
        category_scores[category].append(confidence)

    return {cat: sum(scores) / len(scores) for cat, scores in category_scores.items() if scores}


# ============================================================================
# MAIN ABLATION RUNNER
# ============================================================================

def run_improved_ablation(
    configs: List[AblationConfig] = None,
    test_run: bool = False,
    questions_path: str = None,
    study_name: str = "improved",
) -> List[AblationResult]:
    """Run the improved ablation study.

    Args:
        configs: List of AblationConfig objects. Defaults to IMPROVED_CONFIGS.
        test_run: If True, use only 6 hard test questions.
        questions_path: Path to questions JSON file.
        study_name: Name for logging (e.g., "improved" or "study1").
    """

    # Default configs
    if configs is None:
        configs = IMPROVED_CONFIGS

    # Default questions path
    if questions_path is None:
        questions_path = "/app/benchmarks/realistic_qa_dataset.json"

    config_names = [c.name for c in configs]

    logger.info("=" * 80)
    logger.info(f"ABLATION STUDY: {study_name.upper()}")
    logger.info("4-Layer Evaluation (14 metrics)")
    logger.info("=" * 80)
    logger.info(f"Configs ({len(configs)}): {config_names}")
    logger.info(f"Test run: {test_run}")

    # Load questions
    if test_run:
        logger.info(f"\nTest run mode: Loading 6 hard questions")
        test_cases = load_realistic_questions(questions_path, TEST_QUESTION_IDS)
        logger.info(f"Loaded {len(test_cases)} test questions:")
        for tc in test_cases:
            logger.info(f"  - {tc['id']}: {tc['question'][:60]}...")
    else:
        logger.info(f"\nFull run mode: Loading all questions from {questions_path}")
        test_cases = load_realistic_questions(questions_path)
        logger.info(f"Loaded {len(test_cases)} questions")

    # Run each configuration
    all_results = []

    for config in configs:
        logger.info(f"\n{'='*80}")
        logger.info(f"CONFIG: {config.name}")
        logger.info(f"  {config.description}")
        if config.use_followup_planning:
            logger.info(f"  Mode: Follow-up Planning")
            logger.info(f"  Primary: vec={config.primary_vector_limit}, hops={config.primary_max_hops}")
            logger.info(f"  Secondary: vec={config.secondary_vector_limit}, hops={config.secondary_max_hops}")
        else:
            logger.info(f"  vector_limit={config.vector_limit}, max_hops={config.max_hops}")
        logger.info(f"  compression={config.compression_enabled}, web={config.web_search_enabled}")
        logger.info(f"{'='*80}")

        timestamp = datetime.now().isoformat()

        try:
            # Run agent
            test_case_objects, agent_outputs, duration_ms, cypher_errors = run_agent_with_config(
                config, test_cases
            )

            # Log Cypher error count
            if cypher_errors > 0:
                logger.warning(f"  Cypher errors: {cypher_errors} (improved retrieval should have 0)")
            else:
                logger.info(f"  Cypher errors: 0 (as expected)")

            # Run 4-layer evaluation
            logger.info(f"  Running 4-layer agent evaluation...")
            eval_result = run_evaluation(test_case_objects, agent_outputs)

            # Extract scores
            layer_scores = extract_layer_scores(eval_result)
            category_scores = calculate_category_scores(test_case_objects, agent_outputs)

            # Build config_params with all relevant settings
            config_params = {
                "vector_limit": config.vector_limit,
                "max_hops": config.max_hops,
                "compression_enabled": config.compression_enabled,
                "web_search_enabled": config.web_search_enabled,
                "use_retrieval_planning": config.use_retrieval_planning,
                "use_followup_planning": config.use_followup_planning,
            }
            if config.use_followup_planning:
                config_params.update({
                    "primary_vector_limit": config.primary_vector_limit,
                    "primary_max_hops": config.primary_max_hops,
                    "secondary_vector_limit": config.secondary_vector_limit,
                    "secondary_max_hops": config.secondary_max_hops,
                    "planning_reasoning": config.planning_reasoning,
                })

            result = AblationResult(
                config_name=config.name,
                config_description=config.description,
                config_params=config_params,
                timestamp=timestamp,
                duration_ms=duration_ms,
                test_case_count=len(test_cases),
                overall_score=eval_result.overall_score,
                overall_passed=eval_result.overall_passed,
                layer_scores=layer_scores,
                per_category_scores=category_scores,
                cypher_errors=cypher_errors,
            )

            # Log results
            logger.info(f"\n  Results for {config.name}:")
            logger.info(f"    Overall: {eval_result.overall_score:.1%} {'PASS' if eval_result.overall_passed else 'FAIL'}")
            for ls in layer_scores:
                logger.info(f"    {ls.layer_name}: {ls.score:.1%}")
            logger.info(f"    Duration: {duration_ms/1000:.1f}s")

        except Exception as e:
            import traceback
            logger.error(f"  Config {config.name} failed: {e}")
            logger.error(traceback.format_exc())
            result = AblationResult(
                config_name=config.name,
                config_description=config.description,
                config_params={},
                timestamp=timestamp,
                duration_ms=0,
                test_case_count=len(test_cases),
                overall_score=0.0,
                overall_passed=False,
                errors=[str(e)],
            )

        all_results.append(result)

    return all_results


def print_comparison_table(results: List[AblationResult]):
    """Print a detailed comparison table."""

    print("\n" + "=" * 120)
    print("ABLATION STUDY RESULTS (4-Layer Evaluation)")
    print("=" * 120)

    # Header - adapt to show primary/secondary for followup configs
    print(f"\n{'Config':<25} {'Pri-V':>5} {'Pri-H':>5} {'Sec-V':>5} {'Sec-H':>5} "
          f"{'Retriev':>8} {'Agentic':>8} {'Integr':>8} {'Gener':>8} {'Overall':>8}")
    print("-" * 120)

    for r in results:
        # For followup configs, show primary/secondary; for others, show vec/hop
        if r.config_params.get("use_followup_planning"):
            pri_v = r.config_params.get("primary_vector_limit", "-")
            pri_h = r.config_params.get("primary_max_hops", "-")
            sec_v = r.config_params.get("secondary_vector_limit", "-")
            sec_h = r.config_params.get("secondary_max_hops", "-")
        else:
            pri_v = r.config_params.get("vector_limit", "?")
            pri_h = r.config_params.get("max_hops", "?")
            sec_v = "-"
            sec_h = "-"

        layer_map = {ls.layer_name: ls.score for ls in r.layer_scores}

        print(f"{r.config_name:<25} {pri_v:>5} {pri_h:>5} {sec_v:>5} {sec_h:>5} "
              f"{layer_map.get('retrieval', 0):>7.1%} "
              f"{layer_map.get('agentic', 0):>7.1%} "
              f"{layer_map.get('integrity', 0):>7.1%} "
              f"{layer_map.get('generation', 0):>7.1%} "
              f"{r.overall_score:>7.1%}")

    print("-" * 120)

    # Category breakdown
    print("\n" + "=" * 100)
    print("PER-CATEGORY BREAKDOWN (Average Confidence)")
    print("=" * 100)

    categories = set()
    for r in results:
        categories.update(r.per_category_scores.keys())
    categories = sorted(categories)

    header = f"{'Config':<20}"
    for cat in categories:
        header += f" {cat[:12]:>12}"
    print(header)
    print("-" * 100)

    for r in results:
        row = f"{r.config_name:<20}"
        for cat in categories:
            score = r.per_category_scores.get(cat, 0)
            row += f" {score:>11.1%}"
        print(row)

    # Per-metric breakdown for top configs
    print("\n" + "=" * 100)
    print("METRIC DETAILS (Top 3 configs)")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)[:3]

    for r in sorted_results:
        print(f"\n{r.config_name} ({r.config_description}):")
        print(f"  Overall: {r.overall_score:.1%}")
        for ls in r.layer_scores:
            status = "PASS" if ls.passed else "FAIL"
            print(f"  {ls.layer_name}: {ls.score:.1%} [{status}]")
            for metric_name, metric_score in sorted(ls.metric_scores.items()):
                indicator = "+" if metric_score >= 0.7 else "-"
                print(f"    [{indicator}] {metric_name}: {metric_score:.1%}")

    print("\n" + "=" * 100)


def save_results(results: List[AblationResult], output_dir: str = "benchmarks/results"):
    """Save ablation results to JSON."""
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/improved_ablation_{timestamp}.json"

    # Convert to serializable format
    data = {
        "timestamp": datetime.now().isoformat(),
        "study_type": "improved_hybrid_rag",
        "description": "Keywords + Vector → Fixed N-hop Sampling ablation study",
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Improved Hybrid RAG Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/improved_ablation_study.py --test-run          # 6 hard questions, all configs
  python benchmarks/improved_ablation_study.py --full              # 20 questions, all configs
  python benchmarks/improved_ablation_study.py --configs baseline_improved vec_large
  python benchmarks/improved_ablation_study.py --test-run --configs baseline_improved shallow deep
        """
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run with 6 hard questions only",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full run with all 20 questions",
    )
    parser.add_argument(
        "--study1",
        action="store_true",
        help="Run STUDY1_CONFIGS (follow-up planning variants) instead of IMPROVED_CONFIGS",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific config names to run (filters from selected config set)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="/app/benchmarks/realistic_qa_dataset.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Determine mode
    test_run = args.test_run or not args.full

    if test_run:
        logger.info("Running in TEST mode (6 hard questions)")
    else:
        logger.info("Running in FULL mode (20 questions)")

    # Select config set
    if args.study1:
        base_configs = STUDY1_CONFIGS
        study_name = "study1_followup"
        logger.info("Using STUDY1_CONFIGS (follow-up planning variants)")
    else:
        base_configs = IMPROVED_CONFIGS
        study_name = "improved"
        logger.info("Using IMPROVED_CONFIGS (hybrid RAG variants)")

    # Filter to specific configs if requested
    if args.configs:
        configs = [c for c in base_configs if c.name in args.configs]
        if not configs:
            logger.error(f"No matching configs found for: {args.configs}")
            logger.info(f"Available configs: {[c.name for c in base_configs]}")
            return
    else:
        configs = base_configs

    # Run ablation study
    results = run_improved_ablation(
        configs=configs,
        test_run=test_run,
        questions_path=args.questions,
        study_name=study_name,
    )

    # Print comparison table
    print_comparison_table(results)

    # Save results
    save_results(results, args.output_dir)

    # Summary
    print("\n" + "=" * 100)
    print("STUDY COMPLETE")
    print("=" * 100)

    best = max(results, key=lambda r: r.overall_score)
    print(f"\nBest config: {best.config_name}")
    print(f"  Score: {best.overall_score:.1%}")
    print(f"  Vector limit: {best.config_params.get('vector_limit')}")
    print(f"  Max hops: {best.config_params.get('max_hops')}")
    print(f"  Cypher errors: {best.cypher_errors}")

    total_cypher_errors = sum(r.cypher_errors for r in results)
    if total_cypher_errors == 0:
        print("\nNo Cypher errors across all configs - improved retrieval working!")
    else:
        print(f"\nTotal Cypher errors: {total_cypher_errors} - investigate!")


if __name__ == "__main__":
    main()
