#!/usr/bin/env python3
"""
Enterprise Evaluation Suite CLI.

Usage:
    # Full evaluation
    python benchmarks/enterprise_eval.py

    # Specific layers only
    python benchmarks/enterprise_eval.py --layers retrieval generation

    # CI/CD mode with failure on threshold
    python benchmarks/enterprise_eval.py --ci --fail-on-threshold

    # Generate golden dataset
    python benchmarks/enterprise_eval.py --generate-golden

    # Review generated test cases
    python benchmarks/enterprise_eval.py --review-golden

    # Run with custom config
    python benchmarks/enterprise_eval.py --config config.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enterprise_eval import (
    EvalConfig,
    EvaluationLayer,
    EnterpriseEvaluationRunner,
)
from enterprise_eval.runner import create_default_runner
from enterprise_eval.metrics.base import TestCase, AgentOutput
from enterprise_eval.golden import GoldenDataset, GoldenDatasetGenerator
from enterprise_eval.golden.reviewer import run_review_cli
from enterprise_eval.reporting import JSONReporter, HumanReporter, CIReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from JSON file.

    Args:
        path: Path to test cases JSON

    Returns:
        List of TestCase objects
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Handle both list format and dataset format
    if isinstance(data, list):
        return [TestCase.from_dict(tc) for tc in data]
    elif "test_cases" in data:
        return [TestCase.from_dict(tc) for tc in data["test_cases"]]
    else:
        raise ValueError(f"Invalid test case format in {path}")


def run_agent_on_test_cases(test_cases: List[TestCase]) -> List[AgentOutput]:
    """Run the QA agent on test cases and capture outputs.

    Args:
        test_cases: Test cases to run

    Returns:
        List of AgentOutput objects
    """
    try:
        from agent_qa import QAAgent, QAAgentConfig
    except ImportError:
        logger.error("Could not import QAAgent. Make sure agent_qa.py is available.")
        raise

    # Initialize agent
    config = QAAgentConfig()
    agent = QAAgent(config)

    outputs = []
    for i, tc in enumerate(test_cases):
        logger.info(f"Running test case {i+1}/{len(test_cases)}: {tc.id}")

        try:
            # Run agent
            response, state = agent.answer_question(tc.question, return_state=True)

            # Capture output
            output = AgentOutput.from_agent(response, state)
            outputs.append(output)

        except Exception as e:
            logger.error(f"Agent failed on test case {tc.id}: {e}")
            outputs.append(AgentOutput(
                question=tc.question,
                answer=f"ERROR: {e}",
                confidence=0.0,
            ))

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Evaluation Suite for GraphRAG + ReAct Agent"
    )

    # Mode selection
    parser.add_argument(
        "--generate-golden",
        action="store_true",
        help="Generate golden dataset using LLM",
    )
    parser.add_argument(
        "--review-golden",
        action="store_true",
        help="Review golden dataset interactively",
    )

    # Evaluation options
    parser.add_argument(
        "--test-cases",
        type=str,
        default="benchmarks/enterprise_eval/golden/datasets/default_v1.json",
        help="Path to test cases JSON file",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=["retrieval", "agentic", "integrity", "generation"],
        help="Specific layers to evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config JSON",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/enterprise_eval/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["human", "json", "markdown", "junit"],
        default=["human"],
        help="Output formats",
    )

    # CI/CD options
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI/CD mode with GitHub Actions integration",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Return non-zero exit code if any metric fails threshold",
    )

    # Golden dataset options
    parser.add_argument(
        "--golden-name",
        type=str,
        default="generated_v1",
        help="Name for generated golden dataset",
    )
    parser.add_argument(
        "--reviewer-name",
        type=str,
        default="anonymous",
        help="Reviewer name for review mode",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle generate golden mode
    if args.generate_golden:
        logger.info("Generating golden dataset...")

        generator = GoldenDatasetGenerator(
            model="gpt-4o",
            temperature=0.7,
        )

        dataset = generator.generate_dataset(name=args.golden_name)

        output_path = Path(f"benchmarks/enterprise_eval/golden/datasets/{args.golden_name}.json")
        dataset.save(output_path)

        stats = dataset.get_statistics()
        logger.info(f"Generated {stats['total']} test cases")
        logger.info(f"Saved to {output_path}")

        generator.close()
        return 0

    # Handle review golden mode
    if args.review_golden:
        dataset_path = args.test_cases
        run_review_cli(dataset_path, args.reviewer_name)
        return 0

    # Regular evaluation mode
    logger.info("Starting evaluation...")

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config_data = json.load(f)
        config = EvalConfig(**config_data)
    else:
        config = EvalConfig()

    # Filter layers if specified
    if args.layers:
        config.enabled_layers = [
            EvaluationLayer(layer) for layer in args.layers
        ]

    # Create runner with all metrics
    runner = create_default_runner(config)

    # Load test cases
    try:
        test_cases = load_test_cases(args.test_cases)
        logger.info(f"Loaded {len(test_cases)} test cases from {args.test_cases}")
    except FileNotFoundError:
        logger.warning(f"Test cases file not found: {args.test_cases}")
        logger.info("Creating sample test cases...")

        # Create sample test cases for demo
        test_cases = [
            TestCase(
                id="sample_1",
                question="What is the relationship between AI and machine learning?",
                expected_answer="Machine learning is a subset of AI.",
                expected_entities=["AI", "Machine Learning"],
                type="retrieval",
                difficulty="easy",
            ),
            TestCase(
                id="sample_2",
                question="Who invented the transformer architecture?",
                expected_answer="The transformer was introduced by Vaswani et al. in 2017.",
                expected_entities=["Transformer", "Vaswani"],
                optimal_tool_sequence=["graph_lookup", "web_search"],
                type="agentic",
                difficulty="medium",
            ),
        ]
        logger.info(f"Created {len(test_cases)} sample test cases")

    # Run agent on test cases
    logger.info("Running agent on test cases...")
    agent_outputs = run_agent_on_test_cases(test_cases)

    # Run evaluation
    result = runner.run_evaluation(test_cases, agent_outputs)

    # Generate reports
    output_dir = args.output_dir

    if "human" in args.format:
        reporter = HumanReporter()
        reporter.print_report(result)

    if "json" in args.format:
        reporter = JSONReporter(output_dir)
        reporter.generate(result)
        reporter.generate_summary(result)

    if "markdown" in args.format:
        reporter = HumanReporter()
        reporter.generate_markdown(result, output_dir)

    if "junit" in args.format or args.ci:
        reporter = CIReporter()
        reporter.write_junit_xml(result, f"{output_dir}/test-results.xml")
        reporter.write_badge_json(result, f"{output_dir}/evaluation-badge.json")

    # CI mode output
    if args.ci:
        ci_reporter = CIReporter()
        exit_code = ci_reporter.report(result, fail_on_threshold=args.fail_on_threshold)
        return exit_code

    # Check threshold
    if args.fail_on_threshold and not result.overall_passed:
        logger.warning("Evaluation failed threshold checks!")
        return 1

    runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
