#!/usr/bin/env python3
"""
Evaluate fine-tuned models against teacher labels.

Compares baseline vs fine-tuned models on parse success rate,
entity overlap, edge overlap, and semantic F1.

Usage:
    python -m finetuning.scripts.04_evaluate \
        --models models/finetuned/qlora/gemma-3-12b-it \
        --eval-data data/finetuning/eval/formatted_eval.jsonl \
        --base-model google/gemma-3-12b-it
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetuning.config import EvaluationConfig
from finetuning.evaluation.evaluator import FinetuningEvaluator, EvaluationMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to fine-tuned models or adapters",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Path to evaluation data JSONL",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID (required for LoRA adapters)",
    )
    parser.add_argument(
        "--is-adapter",
        action="store_true",
        help="Whether models are LoRA adapters",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also evaluate the unmodified base model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.85,
        help="Threshold for semantic similarity matching",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for semantic similarity",
    )
    return parser.parse_args()


def load_eval_data(eval_path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSONL file."""
    eval_data = []

    with open(eval_path, "r") as f:
        for line in f:
            sample = json.loads(line)

            # Handle different formats
            if "messages" in sample:
                # Extract from messages format
                kg_text = sample["messages"][-1]["content"]
                eval_data.append({
                    "id": sample.get("id", f"sample_{len(eval_data)}"),
                    "input_text": sample.get("input_text", sample["messages"][1]["content"].replace("Text: ", "")),
                    "knowledge_graph": json.loads(kg_text),
                })
            elif "knowledge_graph" in sample:
                eval_data.append(sample)
            else:
                logger.warning(f"Skipping sample with unknown format: {list(sample.keys())}")

    return eval_data


def main():
    args = parse_args()

    # Create config
    config = EvaluationConfig(
        max_new_tokens=args.max_new_tokens,
        semantic_similarity_threshold=args.semantic_threshold,
        embedding_model=args.embedding_model,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Models to evaluate: {args.models}")
    logger.info(f"Eval data: {args.eval_data}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Is adapter: {args.is_adapter}")
    logger.info(f"Include baseline: {args.include_baseline}")
    logger.info("=" * 60)

    # Load eval data
    logger.info("\nLoading evaluation data...")
    eval_data = load_eval_data(args.eval_data)
    logger.info(f"Loaded {len(eval_data)} evaluation samples")

    # Initialize evaluator
    evaluator = FinetuningEvaluator(config)

    # Build model list
    models_to_evaluate = []

    # Add baseline if requested
    if args.include_baseline and args.base_model:
        models_to_evaluate.append({
            "name": f"{args.base_model.split('/')[-1]} (baseline)",
            "path": args.base_model,
            "base_model_id": None,
            "is_adapter": False,
        })

    # Add fine-tuned models
    for model_path in args.models:
        model_name = os.path.basename(model_path)

        # Detect if it's an adapter
        is_adapter = args.is_adapter
        if not is_adapter:
            # Check for adapter_config.json
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            is_adapter = os.path.exists(adapter_config_path)

        models_to_evaluate.append({
            "name": model_name,
            "path": model_path,
            "base_model_id": args.base_model if is_adapter else None,
            "is_adapter": is_adapter,
        })

    # Evaluate all models
    logger.info(f"\nEvaluating {len(models_to_evaluate)} models...")
    all_metrics = evaluator.compare_models(models_to_evaluate, eval_data)

    # Generate report
    report = evaluator.generate_report(
        all_metrics,
        output_path=os.path.join(args.output_dir, "evaluation_report.txt"),
    )

    # Print report
    print("\n" + report)

    # Save detailed metrics
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {name: metrics.to_dict() for name, metrics in all_metrics.items()},
            f,
            indent=2,
        )
    logger.info(f"\nDetailed metrics saved to: {metrics_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    # Find best model
    best_model = max(all_metrics.items(), key=lambda x: x[1].semantic_f1)
    logger.info(f"Best model by Semantic F1: {best_model[0]} ({best_model[1].semantic_f1:.2%})")

    # Improvement summary
    if args.include_baseline and len(all_metrics) > 1:
        baseline_name = f"{args.base_model.split('/')[-1]} (baseline)"
        if baseline_name in all_metrics:
            baseline_f1 = all_metrics[baseline_name].semantic_f1
            for name, metrics in all_metrics.items():
                if name != baseline_name:
                    improvement = metrics.semantic_f1 - baseline_f1
                    logger.info(f"  {name}: {improvement:+.2%} vs baseline")


if __name__ == "__main__":
    main()
