#!/usr/bin/env python3
"""
Generate teacher labels for fine-tuning.

Uses Gemini 2.5 Flash (primary) and GPT-5.2 (fallback) to generate
knowledge graph labels for training data.

Usage:
    python -m finetuning.scripts.01_generate_labels \
        --domains economics law physics \
        --train-samples 100 \
        --eval-samples 20 \
        --output-dir data/finetuning
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetuning.config import DataConfig
from finetuning.data.teacher_labeler import TeacherLabeler, generate_labels_for_domain
from finetuning.storage.gdrive_sync import GDriveSync, PYDRIVE_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate teacher labels for fine-tuning"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["economics", "law", "physics"],
        help="Domains to generate labels for",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=100,
        help="Number of training samples per domain",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=20,
        help="Number of evaluation samples per domain",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/finetuning",
        help="Output directory for generated labels",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Target chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters",
    )
    parser.add_argument(
        "--primary-teacher",
        type=str,
        default="gemini-2.5-flash",
        choices=["gemini-2.5-flash", "gpt-5.2"],
        help="Primary teacher model",
    )
    parser.add_argument(
        "--fallback-teacher",
        type=str,
        default="gpt-5.2",
        choices=["gemini-2.5-flash", "gpt-5.2"],
        help="Fallback teacher model (used when primary hits rate limits)",
    )
    parser.add_argument(
        "--upload-gdrive",
        action="store_true",
        help="Upload generated labels to Google Drive",
    )
    parser.add_argument(
        "--gdrive-folder-id",
        type=str,
        default=None,
        help="Google Drive folder ID for upload",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create config
    config = DataConfig(
        output_dir=args.output_dir,
        train_samples_per_domain=args.train_samples,
        eval_samples_per_domain=args.eval_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        primary_teacher=args.primary_teacher,
        fallback_teacher=args.fallback_teacher,
    )

    # Create output directories
    os.makedirs(config.labels_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TEACHER LABEL GENERATION")
    logger.info("=" * 60)
    logger.info(f"Domains: {args.domains}")
    logger.info(f"Train samples per domain: {args.train_samples}")
    logger.info(f"Eval samples per domain: {args.eval_samples}")
    logger.info(f"Primary teacher: {args.primary_teacher}")
    logger.info(f"Fallback teacher: {args.fallback_teacher}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    # Generate labels for each domain
    all_train_samples = []
    all_eval_samples = []

    for domain in args.domains:
        logger.info(f"\nProcessing domain: {domain}")

        train_samples, eval_samples = generate_labels_for_domain(
            domain=domain,
            config=config,
        )

        all_train_samples.extend(train_samples)
        all_eval_samples.extend(eval_samples)

        logger.info(f"  Generated {len(train_samples)} train, {len(eval_samples)} eval samples")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total training samples: {len(all_train_samples)}")
    logger.info(f"Total evaluation samples: {len(all_eval_samples)}")
    logger.info(f"Labels saved to: {config.labels_dir}")

    # Upload to Google Drive if requested
    if args.upload_gdrive:
        if not PYDRIVE_AVAILABLE:
            logger.warning("pydrive2 not installed. Skipping Google Drive upload.")
        else:
            logger.info("\nUploading to Google Drive...")
            from finetuning.config import StorageConfig

            storage_config = StorageConfig(
                gdrive_folder_id=args.gdrive_folder_id,
            )
            gdrive = GDriveSync(storage_config)

            folder_id = gdrive.upload_dataset(
                local_path=config.labels_dir,
                dataset_name="teacher_labels",
            )
            logger.info(f"Uploaded to Google Drive folder: {folder_id}")

    logger.info("\nNext step: Run 02_train_qlora.py or 03_train_qgalore.py")


if __name__ == "__main__":
    main()
