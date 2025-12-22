#!/usr/bin/env python3
"""
Generate DPO training data for agent faithfulness.

This script takes tool-use conversations (from SFT data generation) and creates
preference pairs with faithful (chosen) and hallucinated (rejected) responses.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Import teacher LLM for hallucination generation
try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LangChain LLMs not available. Will use perturbation-only mode.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DPO training data for agent faithfulness")
    parser.add_argument("--input_dir", type=str, default="data/agent_sft", help="Directory with SFT conversation data")
    parser.add_argument("--output_dir", type=str, default="data/agent_dpo", help="Output directory for DPO data")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of DPO samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_llm_hallucinations", action="store_true", help="Use LLM to generate hallucinated responses")
    parser.add_argument("--llm_ratio", type=float, default=0.3, help="Ratio of samples to generate with LLM hallucinations")
    parser.add_argument("--teacher_model", type=str, default="gpt-4o-mini", help="Teacher model for hallucination generation")
    parser.add_argument("--teacher_provider", type=str, default="openai", choices=["openai", "google"], help="LLM provider")
    return parser.parse_args()


def load_conversations(input_dir: str) -> List[Dict[str, Any]]:
    """Load SFT conversations from JSONL files."""
    conversations = []
    input_path = Path(input_dir)

    # Try to load from train.jsonl first
    train_path = input_path / "train.jsonl"
    if train_path.exists():
        with open(train_path, "r") as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
        logger.info(f"Loaded {len(conversations)} conversations from {train_path}")

    # Also check for other JSONL files
    for jsonl_file in input_path.glob("*.jsonl"):
        if jsonl_file.name != "train.jsonl" and jsonl_file.name != "eval.jsonl":
            with open(jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        conversations.append(json.loads(line))
            logger.info(f"Loaded additional conversations from {jsonl_file}")

    return conversations


def get_teacher_llm(provider: str, model: str):
    """Initialize teacher LLM for hallucination generation."""
    if not LLM_AVAILABLE:
        return None

    try:
        if provider == "openai":
            return ChatOpenAI(model=model, temperature=0.7)
        elif provider == "google":
            return ChatGoogleGenerativeAI(model=model, temperature=0.7)
        else:
            logger.warning(f"Unknown provider: {provider}")
            return None
    except Exception as e:
        logger.warning(f"Failed to initialize teacher LLM: {e}")
        return None


def save_dpo_samples(
    samples: List[Any],
    output_dir: str,
) -> Dict[str, str]:
    """
    Save DPO samples to disk.

    Returns:
        Dict mapping format name to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # Convert samples to dicts
    sample_dicts = [s.to_dict() if hasattr(s, 'to_dict') else s for s in samples]

    # Save as JSONL
    jsonl_path = output_path / f"dpo_samples_{timestamp}.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in sample_dicts:
            f.write(json.dumps(sample) + "\n")
    saved_files["jsonl"] = str(jsonl_path)
    logger.info(f"Saved {len(samples)} DPO samples to {jsonl_path}")

    # Save as JSON for inspection
    json_path = output_path / f"dpo_samples_{timestamp}.json"
    with open(json_path, "w") as f:
        # Count perturbation types
        type_counts = {}
        for s in sample_dicts:
            pt = s.get("perturbation_type", "unknown")
            type_counts[pt] = type_counts.get(pt, 0) + 1

        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "num_samples": len(samples),
                "perturbation_distribution": type_counts,
            },
            "samples": sample_dicts,
        }, f, indent=2)
    saved_files["json"] = str(json_path)

    # Save train/eval split
    split_idx = int(len(sample_dicts) * 0.9)
    train_samples = sample_dicts[:split_idx]
    eval_samples = sample_dicts[split_idx:]

    train_path = output_path / "train.jsonl"
    eval_path = output_path / "eval.jsonl"

    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    saved_files["train"] = str(train_path)

    with open(eval_path, "w") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    saved_files["eval"] = str(eval_path)

    logger.info(f"Split: {len(train_samples)} train, {len(eval_samples)} eval")

    return saved_files


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DPO Data Generation for Agent Faithfulness")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target samples: {args.num_samples}")
    logger.info(f"Use LLM hallucinations: {args.use_llm_hallucinations}")
    if args.use_llm_hallucinations:
        logger.info(f"LLM ratio: {args.llm_ratio}")
        logger.info(f"Teacher model: {args.teacher_provider}/{args.teacher_model}")

    # Load conversations
    conversations = load_conversations(args.input_dir)

    if not conversations:
        logger.error(f"No conversations found in {args.input_dir}")
        logger.info("Please run 01_generate_tool_data.py first to generate SFT data.")
        return

    logger.info(f"Loaded {len(conversations)} source conversations")

    # Initialize teacher LLM if requested
    teacher_llm = None
    if args.use_llm_hallucinations:
        teacher_llm = get_teacher_llm(args.teacher_provider, args.teacher_model)
        if teacher_llm:
            logger.info("Teacher LLM initialized for hallucination generation")
        else:
            logger.warning("Teacher LLM not available, using perturbation-only mode")

    # Import DPO generator
    from finetuning.agent.data.dpo_data_generator import DPODataGenerator

    # Initialize generator
    generator = DPODataGenerator(
        teacher_llm=teacher_llm,
        seed=args.seed,
    )

    # Limit conversations if we have more than needed
    import random
    random.seed(args.seed)
    if len(conversations) > args.num_samples:
        conversations = random.sample(conversations, args.num_samples)
        logger.info(f"Sampled {len(conversations)} conversations for DPO generation")

    # Generate DPO samples
    logger.info("\nGenerating DPO samples...")
    samples = generator.generate_dataset(
        conversations=conversations,
        use_llm_hallucinations=args.use_llm_hallucinations and teacher_llm is not None,
        llm_ratio=args.llm_ratio,
    )

    if not samples:
        logger.error("No DPO samples generated!")
        return

    # Save to disk
    logger.info("\nSaving DPO data...")
    saved_files = save_dpo_samples(samples, args.output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DPO Data Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(samples)}")

    # Count by perturbation type
    type_counts = {}
    for s in samples:
        pt = s.perturbation_type if hasattr(s, 'perturbation_type') else s.get("perturbation_type", "unknown")
        type_counts[pt] = type_counts.get(pt, 0) + 1
    logger.info(f"Perturbation distribution: {type_counts}")

    logger.info(f"\nSaved files:")
    for fmt, path in saved_files.items():
        logger.info(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
