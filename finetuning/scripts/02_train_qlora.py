#!/usr/bin/env python3
"""
Fine-tune models using QLoRA.

Trains student models (Gemma3 12B, Qwen3 30B) on teacher-generated labels
using 4-bit quantization and LoRA adapters.

Usage:
    python -m finetuning.scripts.02_train_qlora \
        --model google/gemma-3-12b-it \
        --data-dir data/finetuning \
        --output-dir models/finetuned/qlora
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetuning.config import TrainingConfig, StorageConfig, DataConfig
from finetuning.data.dataset_builder import DatasetBuilder
from finetuning.data.data_formatter import DataFormatter
from finetuning.data.teacher_labeler import LabeledSample
from finetuning.training.qlora_trainer import QLoRATrainer

import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_teacher_labels(data_dir: str) -> tuple[str, str]:
    """
    Format teacher labels into training format if not already done.

    Returns paths to formatted train and eval files.
    """
    labels_dir = os.path.join(data_dir, "teacher_labels")
    train_dir = os.path.join(data_dir, "train")
    eval_dir = os.path.join(data_dir, "eval")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    train_path = os.path.join(train_dir, "all_domains_messages.jsonl")
    eval_path = os.path.join(eval_dir, "all_domains_messages.jsonl")

    # Check if already formatted
    if os.path.exists(train_path) and os.path.exists(eval_path):
        return train_path, eval_path

    logger.info("Formatting teacher labels for training...")

    data_config = DataConfig(output_dir=data_dir)
    formatter = DataFormatter(data_config)

    train_samples = []
    eval_samples = []

    for domain in ["economics", "law", "physics"]:
        for split, samples_list in [("train", train_samples), ("eval", eval_samples)]:
            label_file = os.path.join(labels_dir, f"{domain}_{split}_labels.jsonl")
            if os.path.exists(label_file):
                count = 0
                with open(label_file, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        sample = LabeledSample.from_dict(data)
                        formatted = formatter.format_sample(sample)
                        samples_list.append(formatted)
                        count += 1
                logger.info(f"  Loaded {domain}/{split}: {count} samples")

    formatter.save_formatted_data(train_samples, train_path, "messages")
    formatter.save_formatted_data(eval_samples, eval_path, "messages")

    logger.info(f"Formatted {len(train_samples)} train, {len(eval_samples)} eval samples")

    return train_path, eval_path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune models using QLoRA")
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it-qat-q4_0-unquantized" , help="HuggingFace model ID")
    parser.add_argument("--data_dir", type=str, default="data/finetuning", help="Training data directory")
    parser.add_argument("--output_dir", type=str, default="models/finetuned/qlora", help="Output directory")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer (adamw_8bit, paged_adamw_8bit, adamw_torch_fused)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing for faster training")
    parser.add_argument("--wandb_project", type=str, default="kg-finetuning", help="W&B project name")
    parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name")
    parser.add_argument("--upload_hf", action="store_true", help="Upload adapter to HuggingFace Hub")
    parser.add_argument("--hf_username", type=str, default=None, help="HuggingFace username")
    parser.add_argument("--upload_gdrive", action="store_true", help="Sync checkpoints to Google Drive")
    parser.add_argument("--gdrive_folder_id", type=str, default=None, help="Google Drive folder ID")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create training config
    model_name = args.model.split("/")[-1]
    run_name = args.wandb_run or f"{model_name}-qlora"

    config = TrainingConfig(
        model_id=args.model,
        output_dir=os.path.join(args.output_dir, model_name),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        optim=args.optim,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
    )

    logger.info("=" * 60)
    logger.info("QLORA FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"Optimizer: {args.optim}, gradient_checkpointing: {not args.no_gradient_checkpointing}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)

    # Prepare datasets
    logger.info("\nLoading datasets...")

    # Format teacher labels if needed
    train_path, eval_path = format_teacher_labels(args.data_dir)

    builder = DatasetBuilder(config)
    dataset = builder.build_dataset(train_path, eval_path)

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Eval samples: {len(dataset['eval'])}")

    # Train
    logger.info("\nStarting training...")
    trainer = QLoRATrainer(config)
    result = trainer.train(dataset=dataset)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final metrics: {result.get('metrics', {})}")
    logger.info(f"Adapter saved to: {result.get('output_dir', config.output_dir)}")

    # Upload to HuggingFace Hub
    if args.upload_hf:
        logger.info("\nUploading to HuggingFace Hub...")
        from finetuning.storage.hf_upload import HFUploader

        storage_config = StorageConfig(
            hf_username=args.hf_username,
            upload_adapters=True,
        )
        uploader = HFUploader(storage_config)

        url = uploader.upload_adapter(
            adapter_path=result.get("output_dir", config.output_dir),
            training_config=config,
            metrics=result.get("metrics", {}),
            method="qlora",
        )
        logger.info(f"Uploaded to: {url}")

    # Sync to Google Drive
    if args.upload_gdrive:
        logger.info("\nSyncing to Google Drive...")
        from finetuning.storage.gdrive_sync import GDriveSync, PYDRIVE_AVAILABLE

        if PYDRIVE_AVAILABLE:
            storage_config = StorageConfig(
                gdrive_folder_id=args.gdrive_folder_id,
            )
            gdrive = GDriveSync(storage_config)
            gdrive.sync_checkpoints(config.output_dir)
            logger.info("Checkpoints synced to Google Drive")
        else:
            logger.warning("pydrive2 not installed. Skipping Google Drive sync.")

    logger.info("\nNext step: Run 04_evaluate.py to evaluate the fine-tuned model")


if __name__ == "__main__":
    main()
