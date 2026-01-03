#!/usr/bin/env python3
"""
Train ReAct agent for faithful reasoning with DPO.

Fine-tunes the model using Direct Preference Optimization to prefer:
- Complete Thought→Action→Observation chains
- Grounded answers based on tool observations
- Appropriate tool selection
- No premature conclusions or hallucinated reasoning
"""

import os
import argparse
import logging

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ReAct agent for faithful reasoning (DPO)")

    # Model settings
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B", help="Base HuggingFace model ID")
    parser.add_argument("--sft_checkpoint", type=str, default="", help="Path to SFT checkpoint")
    parser.add_argument("--output_dir", type=str, default="models/finetuned/react_dpo", help="Output directory")

    # Data settings
    parser.add_argument("--train_data", type=str, default="data/react_dpo/train.jsonl", help="Training data path")
    parser.add_argument("--eval_data", type=str, default="data/react_dpo/eval.jsonl", help="Evaluation data path")

    # DPO settings
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty coefficient)")

    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length (longer for ReAct traces)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    # Optimization
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=20, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llm2kg-react", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    # Analysis
    parser.add_argument("--analyze_perturbations", action="store_true", help="Analyze performance by perturbation type after training")

    return parser.parse_args()


def find_sft_checkpoint(default_paths: list = None) -> str:
    """Try to find an SFT checkpoint."""
    if default_paths is None:
        default_paths = [
            "models/finetuned/agent_sft",
            "models/finetuned/react_sft",
        ]

    for default_path in default_paths:
        if not os.path.exists(default_path):
            continue

        # Look for model subdirectories
        for item in os.listdir(default_path):
            item_path = os.path.join(default_path, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "adapter_config.json")) or \
                   os.path.exists(os.path.join(item_path, "config.json")):
                    return item_path

        # Check if default_path itself is a checkpoint
        if os.path.exists(os.path.join(default_path, "adapter_config.json")) or \
           os.path.exists(os.path.join(default_path, "config.json")):
            return default_path

    return ""


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ReAct DPO Training (Faithful Reasoning)")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.model}")
    logger.info(f"SFT checkpoint: {args.sft_checkpoint or '(none specified)'}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Eval data: {args.eval_data}")
    logger.info(f"DPO beta: {args.beta}")
    logger.info(f"Max sequence length: {args.max_seq_length}")

    # Check data exists
    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        logger.info("Please run 05_generate_react_data.py first to generate ReAct DPO training data.")
        return

    if not os.path.exists(args.eval_data):
        logger.error(f"Eval data not found: {args.eval_data}")
        return

    # Try to find SFT checkpoint if not specified
    sft_checkpoint = args.sft_checkpoint
    if not sft_checkpoint:
        sft_checkpoint = find_sft_checkpoint()
        if sft_checkpoint:
            logger.info(f"Found SFT checkpoint: {sft_checkpoint}")
        else:
            logger.warning("No SFT checkpoint found. Training from base model.")
            logger.info("For best results, run agent SFT training first.")

    # Initialize W&B
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"react-dpo-{args.model.split('/')[-1]}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "sft_checkpoint": sft_checkpoint,
                "beta": args.beta,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
            }
        )

    # Import trainer
    from finetuning.agent.training.react_dpo_trainer import ReActDPOTrainer, ReActDPOConfig

    # Create config
    config = ReActDPOConfig(
        model_id=args.model,
        sft_checkpoint=sft_checkpoint,
        output_dir=args.output_dir,

        # DPO
        beta=args.beta,

        # LoRA
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        # Training
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,

        # Optimization
        gradient_checkpointing=not args.no_gradient_checkpointing,

        # Logging
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,

        # Data paths
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
    )

    logger.info("\nConfiguration:")
    logger.info(f"  DPO beta: {config.beta}")
    logger.info(f"  LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Max sequence length: {config.max_seq_length}")
    logger.info(f"  Gradient checkpointing: {config.gradient_checkpointing}")

    # Initialize trainer
    trainer = ReActDPOTrainer(config)

    # Run training
    logger.info("\nStarting ReAct DPO training...")
    try:
        result = trainer.train(
            train_path=args.train_data,
            eval_path=args.eval_data,
            output_dir=args.output_dir,
        )

        logger.info("\n" + "=" * 60)
        logger.info("ReAct DPO Training Complete")
        logger.info("=" * 60)
        logger.info(f"Output directory: {result['output_dir']}")
        logger.info(f"Train samples: {result['train_samples']}")
        logger.info(f"Eval samples: {result['eval_samples']}")
        logger.info(f"Final metrics: {result['metrics']}")

        # Analyze by perturbation type if requested
        if args.analyze_perturbations:
            logger.info("\nAnalyzing performance by perturbation type...")
            perturbation_results = trainer.analyze_perturbation_performance(args.eval_data)

            logger.info("\nPerformance by perturbation type:")
            for pt, pt_results in sorted(perturbation_results.items()):
                logger.info(f"  {pt}:")
                logger.info(f"    Count: {pt_results['count']}")
                logger.info(f"    Eval Loss: {pt_results['eval_loss']:.4f}")

        # Save adapter separately
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_adapter(adapter_dir)
        logger.info(f"\nReAct DPO adapter saved to: {adapter_dir}")

        logger.info("\n" + "=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info("The model has been trained to prefer faithful ReAct reasoning:")
        logger.info("  ✓ Complete Thought→Action→Observation chains")
        logger.info("  ✓ Grounded answers based on tool observations")
        logger.info("  ✓ Appropriate tool selection for the task")
        logger.info("  ✓ No premature conclusions or hallucinated reasoning")

    except Exception as e:
        logger.error(f"ReAct DPO training failed: {e}")
        raise

    finally:
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
