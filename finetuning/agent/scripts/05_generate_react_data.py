#!/usr/bin/env python3
"""
Generate ReAct DPO training data for faithful reasoning.

Creates preference pairs with:
- Chosen: Complete Thought→Action→Observation chains, grounded answers
- Rejected: Unfaithful traces (premature conclusions, hallucinations, etc.)

Perturbation types:
- skip_verification: Answer without using tools
- ignore_observation: Contradict tool output
- wrong_tool_selection: Use inappropriate tool
- premature_conclusion: Stop reasoning too early
- hallucinated_reasoning: Non-sequitur thoughts
- missing_thought: Skip reasoning steps
- wrong_entity_reference: Confuse entities
"""

import os
import argparse
import logging
import json
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ReAct DPO training data")

    # Input/output
    parser.add_argument(
        "--input_conversations",
        type=str,
        default="data/react_conversations.jsonl",
        help="Input file with recorded ReAct conversations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/react_dpo",
        help="Output directory for generated data",
    )

    # Generation settings
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # LLM settings (for generating unfaithful traces)
    parser.add_argument("--use_llm", action="store_true", help="Use LLM to generate some rejected traces")
    parser.add_argument("--llm_ratio", type=float, default=0.3, help="Ratio of LLM-generated rejected traces")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM model for generation")

    # Synthetic data generation
    parser.add_argument(
        "--generate_synthetic",
        action="store_true",
        help="Generate synthetic conversations if input doesn't exist",
    )
    parser.add_argument("--num_synthetic", type=int, default=500, help="Number of synthetic samples to generate")

    return parser.parse_args()


def generate_synthetic_conversations(num_samples: int, seed: int = 42) -> list:
    """
    Generate synthetic ReAct conversations for training.

    Creates diverse question types with proper ReAct traces.
    """
    import random
    rng = random.Random(seed)

    # Sample questions and expected tool use
    question_templates = [
        {
            "template": "What is the relationship between {entity1} and {entity2}?",
            "tools": ["entity_resolve", "graph_lookup", "get_entity_relationships"],
            "category": "relationship",
        },
        {
            "template": "Tell me about {entity1}.",
            "tools": ["entity_resolve", "graph_lookup"],
            "category": "entity_info",
        },
        {
            "template": "What are the key contributions of {entity1}?",
            "tools": ["entity_resolve", "graph_lookup", "get_entity_relationships"],
            "category": "contributions",
        },
        {
            "template": "How is {entity1} connected to {field}?",
            "tools": ["entity_resolve", "graph_lookup", "cypher_query"],
            "category": "connection",
        },
        {
            "template": "What recent developments involve {entity1}?",
            "tools": ["graph_lookup", "web_search"],
            "category": "recent",
        },
    ]

    # Sample entities and fields
    entities = [
        "Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin",
        "Nikola Tesla", "Ada Lovelace", "Alan Turing", "Richard Feynman",
        "Stephen Hawking", "Rosalind Franklin", "Niels Bohr", "Max Planck",
        "Claude Shannon", "John von Neumann", "Grace Hopper", "Linus Torvalds",
    ]

    fields = [
        "physics", "mathematics", "computer science", "biology", "chemistry",
        "astronomy", "quantum mechanics", "machine learning", "genetics",
    ]

    conversations = []

    for i in range(num_samples):
        template = rng.choice(question_templates)
        entity1 = rng.choice(entities)
        entity2 = rng.choice([e for e in entities if e != entity1])
        field = rng.choice(fields)

        question = template["template"].format(
            entity1=entity1, entity2=entity2, field=field
        )

        # Generate a faithful ReAct trace
        trace_parts = []

        # Thought 1
        trace_parts.append(f"Thought: I need to find information about {entity1}. Let me first resolve the entity name.")

        # Action 1 - Entity resolve
        if "entity_resolve" in template["tools"]:
            trace_parts.append("Action: entity_resolve")
            trace_parts.append(f'Action Input: {{"query": "{entity1}"}}')
            trace_parts.append(f'Observation: Resolved "{entity1}" to canonical entity ID: {entity1.lower().replace(" ", "_")}')

        # Thought 2
        trace_parts.append(f"Thought: Now I should look up {entity1} in the knowledge graph.")

        # Action 2 - Graph lookup
        if "graph_lookup" in template["tools"]:
            trace_parts.append("Action: graph_lookup")
            trace_parts.append(f'Action Input: {{"entity_id": "{entity1.lower().replace(" ", "_")}"}}')
            trace_parts.append(f'Observation: Found entity {entity1}. Type: Person. Description: Notable figure in {rng.choice(fields)}. Related entities: {rng.choice(entities)}, {rng.choice(entities)}.')

        # Thought 3
        trace_parts.append(f"Thought: I found basic information. Let me get the relationships for more detail.")

        # Action 3 - Get relationships
        if "get_entity_relationships" in template["tools"]:
            trace_parts.append("Action: get_entity_relationships")
            trace_parts.append(f'Action Input: {{"entity_id": "{entity1.lower().replace(" ", "_")}"}}')
            related = rng.choice(entities)
            trace_parts.append(f'Observation: Relationships: COLLABORATED_WITH: [{related}], INFLUENCED: [{rng.choice(entities)}], WORKED_IN: [{rng.choice(fields)}]')

        # Final thought and answer
        trace_parts.append("Thought: I now have enough information to answer the question based on my observations.")
        trace_parts.append(f"Final Answer: Based on the knowledge graph, {entity1} is a notable figure who made significant contributions. They collaborated with {related} and worked primarily in {rng.choice(fields)}.")

        trace = "\n\n".join(trace_parts)

        conversations.append({
            "question": question,
            "trace": trace,
            "context": f"Knowledge graph contains information about {entity1}.",
            "entities": [
                {"name": entity1, "type": "Person"},
                {"name": entity2, "type": "Person"},
            ],
            "category": template["category"],
        })

    return conversations


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ReAct DPO Data Generation")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input_conversations}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Use LLM: {args.use_llm}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate conversations
    if os.path.exists(args.input_conversations):
        logger.info(f"Loading conversations from {args.input_conversations}")
        conversations = []
        with open(args.input_conversations, "r") as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
        logger.info(f"Loaded {len(conversations)} conversations")
    elif args.generate_synthetic:
        logger.info(f"Generating {args.num_synthetic} synthetic conversations...")
        conversations = generate_synthetic_conversations(args.num_synthetic, args.seed)
        logger.info(f"Generated {len(conversations)} synthetic conversations")

        # Save synthetic conversations
        synthetic_path = os.path.join(args.output_dir, "synthetic_conversations.jsonl")
        with open(synthetic_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")
        logger.info(f"Saved synthetic conversations to {synthetic_path}")
    else:
        logger.error(f"Input file not found: {args.input_conversations}")
        logger.info("Use --generate_synthetic to create synthetic training data")
        return

    # Initialize data generator
    from finetuning.agent.data.react_data_generator import ReActDPODataGenerator

    teacher_llm = None
    if args.use_llm:
        try:
            from langchain_openai import ChatOpenAI
            teacher_llm = ChatOpenAI(model=args.llm_model, temperature=0.7)
            logger.info(f"Initialized teacher LLM: {args.llm_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            logger.info("Falling back to perturbation-only generation")

    generator = ReActDPODataGenerator(teacher_llm=teacher_llm, seed=args.seed)

    # Generate samples
    logger.info("\nGenerating DPO samples...")
    samples = generator.generate_dataset(
        conversations=conversations,
        use_llm=args.use_llm and teacher_llm is not None,
        llm_ratio=args.llm_ratio,
    )

    logger.info(f"Generated {len(samples)} DPO samples")

    if not samples:
        logger.error("No samples generated. Check input data format.")
        return

    # Split train/eval
    import random
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    logger.info(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # Save
    train_path = os.path.join(args.output_dir, "train.jsonl")
    eval_path = os.path.join(args.output_dir, "eval.jsonl")

    generator.save_dataset(train_samples, train_path)
    generator.save_dataset(eval_samples, eval_path)

    logger.info(f"\nSaved training data to: {train_path}")
    logger.info(f"Saved eval data to: {eval_path}")

    # Print perturbation type distribution
    type_counts = {}
    for s in samples:
        pt = s.perturbation_type.value if hasattr(s.perturbation_type, 'value') else str(s.perturbation_type)
        type_counts[pt] = type_counts.get(pt, 0) + 1

    logger.info("\nPerturbation type distribution:")
    for pt, count in sorted(type_counts.items()):
        pct = count / len(samples) * 100
        logger.info(f"  {pt}: {count} ({pct:.1f}%)")

    # Print sample
    logger.info("\n" + "=" * 60)
    logger.info("Sample DPO pair:")
    logger.info("=" * 60)
    sample = samples[0]
    logger.info(f"Prompt: {sample.prompt[:200]}...")
    logger.info(f"\nChosen (first 300 chars):\n{sample.chosen[:300]}...")
    logger.info(f"\nRejected (first 300 chars):\n{sample.rejected[:300]}...")
    logger.info(f"\nPerturbation type: {sample.perturbation_type}")


if __name__ == "__main__":
    main()
