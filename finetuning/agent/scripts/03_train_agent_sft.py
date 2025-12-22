#!/usr/bin/env python3
"""
Train agent model for tool use with SFT.

Fine-tunes Qwen3-30B-A3B (or other models) on multi-turn tool-calling
conversations using QLoRA.
"""

import os
import argparse
import logging

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune agent model for tool use (SFT)")

    # Model settings
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B", help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="models/finetuned/agent_sft", help="Output directory")

    # Data settings
    parser.add_argument("--train_data", type=str, default="data/agent_sft/train.jsonl", help="Training data path")
    parser.add_argument("--eval_data", type=str, default="data/agent_sft/eval.jsonl", help="Evaluation data path")

    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    # Optimization
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer (adamw_8bit, paged_adamw_8bit, adamw_torch_fused)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps to console")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llm2kg-agent", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    # Tool format
    parser.add_argument("--tool_format", type=str, default="qwen3", choices=["qwen3", "chatml"], help="Tool-calling format")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Agent SFT Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Eval data: {args.eval_data}")

    # Check data exists
    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        logger.info("Please run 01_generate_tool_data.py first to generate training data.")
        return

    if not os.path.exists(args.eval_data):
        logger.error(f"Eval data not found: {args.eval_data}")
        return

    # Initialize W&B
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"agent-sft-{args.model.split('/')[-1]}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "tool_format": args.tool_format,
            }
        )

    # Import config and trainer
    from finetuning.config import AgentSFTConfig
    from finetuning.agent.training.agent_sft_trainer import AgentSFTTrainer

    # Create config
    config = AgentSFTConfig(
        model_id=args.model,
        output_dir=args.output_dir,
        tool_format=args.tool_format,

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
        optim=args.optim,
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
    logger.info(f"  LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Max sequence length: {config.max_seq_length}")
    logger.info(f"  Optimizer: {config.optim}")
    logger.info(f"  Gradient checkpointing: {config.gradient_checkpointing}")

    # Initialize trainer
    trainer = AgentSFTTrainer(config)

    # Run training
    logger.info("\nStarting training...")
    try:
        result = trainer.train(
            train_path=args.train_data,
            eval_path=args.eval_data,
            output_dir=args.output_dir,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        logger.info(f"Output directory: {result['output_dir']}")
        logger.info(f"Train samples: {result['train_samples']}")
        logger.info(f"Eval samples: {result['eval_samples']}")
        logger.info(f"Final metrics: {result['metrics']}")

        # Save adapter separately
        adapter_dir = os.path.join(args.output_dir, "adapter")
        trainer.save_adapter(adapter_dir)
        logger.info(f"Adapter saved to: {adapter_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
