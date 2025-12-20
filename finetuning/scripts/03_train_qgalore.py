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

from finetuning.config import QGaloreConfig, StorageConfig
from finetuning.data.dataset_builder import prepare_datasets
from finetuning.training.qgalore_trainer import QGaloreTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        run_name=run_name,
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
    train_path = os.path.join(args.data_dir, "train", "formatted_train.jsonl")
    eval_path = os.path.join(args.data_dir, "eval", "formatted_eval.jsonl")

    if not os.path.exists(train_path):
        # Try alternative path structure
        train_path = os.path.join(args.data_dir, "formatted", "train.jsonl")
        eval_path = os.path.join(args.data_dir, "formatted", "eval.jsonl")

    train_dataset, eval_dataset = prepare_datasets(
        train_path=train_path,
        eval_path=eval_path,
        model_id=args.model,
        max_seq_length=args.max_seq_length,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # Initialize trainer
    trainer = QGaloreTrainer(config)
    trainer.setup()

    # Train
    logger.info("\nStarting training...")
    metrics = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume_from,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final metrics: {metrics}")
    logger.info(f"Model saved to: {config.output_dir}")

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
            model_path=config.output_dir,
            training_config=config,
            metrics=metrics,
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
