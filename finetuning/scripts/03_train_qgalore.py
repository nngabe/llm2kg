#!/usr/bin/env python3
"""
Fine-tune models using Q-GaLore.

Trains student models (Gemma3 12B, Qwen3 30B) on teacher-generated labels
using gradient low-rank projection with INT4 quantization.

Usage:
    python -m finetuning.scripts.03_train_qgalore \
        --model google/gemma-3-12b-it \
        --data-dir data/finetuning \
        --output-dir models/finetuned/qgalore
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetuning.config import QGaloreConfig, StorageConfig, DataConfig
from finetuning.data.dataset_builder import DatasetBuilder
from finetuning.data.data_formatter import DataFormatter
from finetuning.data.teacher_labeler import LabeledSample
from finetuning.training.qgalore_trainer import QGaloreTrainer

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
    parser = argparse.ArgumentParser(
        description="Fine-tune models using Q-GaLore"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-12b-it",
        help="HuggingFace model ID to fine-tune",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/finetuning",
        help="Directory containing formatted training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned/qgalore",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--galore-rank",
        type=int,
        default=128,
        help="GaLore projection rank",
    )
    parser.add_argument(
        "--update-proj-gap",
        type=int,
        default=200,
        help="Steps between projection updates",
    )
    parser.add_argument(
        "--galore-scale",
        type=float,
        default=0.25,
        help="GaLore gradient scale",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="kg-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default=None,
        help="HuggingFace username for upload",
    )
    parser.add_argument(
        "--upload-gdrive",
        action="store_true",
        help="Sync checkpoints to Google Drive",
    )
    parser.add_argument(
        "--gdrive-folder-id",
        type=str,
        default=None,
        help="Google Drive folder ID for checkpoint sync",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create training config
    model_name = args.model.split("/")[-1]
    run_name = args.wandb_run or f"{model_name}-qgalore"

    config = QGaloreConfig(
        model_id=args.model,
        output_dir=os.path.join(args.output_dir, model_name),
        galore_rank=args.galore_rank,
        galore_update_proj_gap=args.update_proj_gap,
        galore_scale=args.galore_scale,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
    )

    logger.info("=" * 60)
    logger.info("Q-GALORE FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"GaLore rank: {args.galore_rank}")
    logger.info(f"Update projection gap: {args.update_proj_gap}")
    logger.info(f"GaLore scale: {args.galore_scale}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
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
    trainer = QGaloreTrainer(config)
    result = trainer.train(dataset=dataset)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final metrics: {result.get('metrics', {})}")
    logger.info(f"Model saved to: {result.get('output_dir', config.output_dir)}")

    # Upload to HuggingFace Hub
    if args.upload_hf:
        logger.info("\nUploading to HuggingFace Hub...")
        from finetuning.storage.hf_upload import HFUploader

        storage_config = StorageConfig(
            hf_username=args.hf_username,
            upload_merged=True,  # Q-GaLore produces full model, not adapter
        )
        uploader = HFUploader(storage_config)

        url = uploader.upload_merged_model(
            model_path=result.get("output_dir", config.output_dir),
            training_config=config,
            metrics=result.get("metrics", {}),
            method="qgalore",
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
