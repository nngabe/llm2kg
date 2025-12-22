#!/usr/bin/env python3
"""
Generate synthetic tool-use training data for agent SFT.

This script samples entities from the Neo4j knowledge graph and generates
multi-turn conversations demonstrating proper tool use patterns.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

from finetuning.agent.data.tool_data_generator import (
    ToolDataGenerator,
    Neo4jSampler,
    ToolConversation,
)
from finetuning.agent.tools import get_tools_for_qwen3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate tool-use training data for agent SFT")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--output_dir", type=str, default="data/agent_sft", help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--neo4j_uri", type=str, default=None, help="Neo4j URI (default: from env)")
    parser.add_argument("--neo4j_user", type=str, default=None, help="Neo4j username (default: from env)")
    parser.add_argument("--neo4j_password", type=str, default=None, help="Neo4j password (default: from env)")
    parser.add_argument("--simple_lookup_ratio", type=float, default=0.30, help="Ratio of simple lookup scenarios")
    parser.add_argument("--multi_hop_ratio", type=float, default=0.20, help="Ratio of multi-hop scenarios")
    parser.add_argument("--disambiguation_ratio", type=float, default=0.15, help="Ratio of disambiguation scenarios")
    parser.add_argument("--complex_query_ratio", type=float, default=0.15, help="Ratio of complex query scenarios")
    parser.add_argument("--web_fallback_ratio", type=float, default=0.10, help="Ratio of web fallback scenarios")
    parser.add_argument("--combined_ratio", type=float, default=0.10, help="Ratio of combined tool scenarios")
    return parser.parse_args()


def save_conversations(
    conversations: list[ToolConversation],
    output_dir: str,
    include_tools: bool = True,
) -> dict[str, str]:
    """
    Save conversations to disk in multiple formats.

    Returns:
        Dict mapping format name to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # Save as JSONL (one conversation per line)
    jsonl_path = output_path / f"tool_conversations_{timestamp}.jsonl"
    with open(jsonl_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_dict()) + "\n")
    saved_files["jsonl"] = str(jsonl_path)
    logger.info(f"Saved {len(conversations)} conversations to {jsonl_path}")

    # Save as JSON array (for easier inspection)
    json_path = output_path / f"tool_conversations_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "num_conversations": len(conversations),
                    "scenario_distribution": _count_scenarios(conversations),
                },
                "tools": get_tools_for_qwen3() if include_tools else None,
                "conversations": [conv.to_dict() for conv in conversations],
            },
            f,
            indent=2,
        )
    saved_files["json"] = str(json_path)
    logger.info(f"Saved JSON format to {json_path}")

    # Save train/eval split
    split_idx = int(len(conversations) * 0.9)
    train_convs = conversations[:split_idx]
    eval_convs = conversations[split_idx:]

    train_path = output_path / "train.jsonl"
    eval_path = output_path / "eval.jsonl"

    with open(train_path, "w") as f:
        for conv in train_convs:
            f.write(json.dumps(conv.to_dict()) + "\n")
    saved_files["train"] = str(train_path)

    with open(eval_path, "w") as f:
        for conv in eval_convs:
            f.write(json.dumps(conv.to_dict()) + "\n")
    saved_files["eval"] = str(eval_path)

    logger.info(f"Split: {len(train_convs)} train, {len(eval_convs)} eval")

    # Save tool definitions
    tools_path = output_path / "tools.json"
    with open(tools_path, "w") as f:
        json.dump(get_tools_for_qwen3(), f, indent=2)
    saved_files["tools"] = str(tools_path)

    return saved_files


def _count_scenarios(conversations: list[ToolConversation]) -> dict[str, int]:
    """Count conversations by scenario type."""
    counts = {}
    for conv in conversations:
        scenario = conv.scenario_type
        counts[scenario] = counts.get(scenario, 0) + 1
    return counts


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Agent Tool Data Generation")
    logger.info("=" * 60)
    logger.info(f"Target samples: {args.num_samples}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")

    # Build scenario distribution
    scenario_distribution = {
        "simple_lookup": args.simple_lookup_ratio,
        "multi_hop": args.multi_hop_ratio,
        "disambiguation": args.disambiguation_ratio,
        "complex_query": args.complex_query_ratio,
        "web_fallback": args.web_fallback_ratio,
        "combined": args.combined_ratio,
    }
    logger.info(f"Scenario distribution: {scenario_distribution}")

    # Initialize Neo4j sampler
    neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = args.neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

    logger.info(f"Connecting to Neo4j at {neo4j_uri}")

    try:
        sampler = Neo4jSampler(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

        # Check graph stats
        stats = sampler.get_graph_stats()
        logger.info(f"Graph stats: {stats.get('total_entities', 0)} entities, "
                   f"{stats.get('total_relationships', 0)} relationships")

        if stats.get('total_entities', 0) == 0:
            logger.error("Knowledge graph is empty! Please populate it first.")
            return

        # Initialize generator
        generator = ToolDataGenerator(
            neo4j_sampler=sampler,
            seed=args.seed,
        )

        # Generate conversations
        logger.info("\nGenerating conversations...")
        conversations = generator.generate_dataset(
            num_samples=args.num_samples,
            scenario_distribution=scenario_distribution,
        )

        if not conversations:
            logger.error("No conversations generated!")
            return

        # Save to disk
        logger.info("\nSaving data...")
        saved_files = save_conversations(conversations, args.output_dir)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Generation Complete")
        logger.info("=" * 60)
        logger.info(f"Total conversations: {len(conversations)}")
        logger.info(f"Scenario breakdown: {_count_scenarios(conversations)}")
        logger.info(f"Saved files:")
        for fmt, path in saved_files.items():
            logger.info(f"  {fmt}: {path}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

    finally:
        if 'generator' in dir():
            generator.close()


if __name__ == "__main__":
    main()
