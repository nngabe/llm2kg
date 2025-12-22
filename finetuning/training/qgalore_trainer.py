"""
Q-GaLore fine-tuning for Knowledge Graph extraction.

Uses Gradient Low-Rank Projection for memory-efficient full-parameter training.
"""

import os
import logging
from typing import Optional, Dict, Any, List

import torch
import transformers
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

try:
    from galore_torch import GaLoreAdamW8bit
    GALORE_AVAILABLE = True
except ImportError:
    GALORE_AVAILABLE = False
    logging.warning("galore_torch not installed. Install with: pip install galore-torch")

from ..config import QGaloreConfig
from .callbacks import MetricsCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QGaloreTrainer:
    """
    Q-GaLore fine-tuning for KG extraction.

    Unlike QLoRA which freezes base weights and trains adapters,
    GaLore trains all parameters but projects gradients into a
    low-rank subspace for memory efficiency.

    Features:
    - Full parameter training (no frozen weights)
    - Low-rank gradient projection
    - INT4 quantized projection matrices
    - Layer-adaptive rank selection
    """

    def __init__(self, config: QGaloreConfig):
        if not GALORE_AVAILABLE:
            raise ImportError(
                "galore_torch is required for Q-GaLore training. "
                "Install with: pip install galore-torch"
            )

        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self) -> None:
        """Load base model in bfloat16 (no initial quantization)."""
        logger.info(f"Loading model: {self.config.model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model in bf16 (GaLore handles memory efficiency differently)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _get_galore_params(self) -> tuple[List, List]:
        """
        Separate parameters into GaLore and regular groups.

        GaLore is applied to large matrix layers (attention, MLP).
        Regular optimization for smaller parameters (embeddings, norms).
        """
        galore_params = []
        regular_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Apply GaLore to target modules
            if any(target in name for target in self.config.target_modules):
                galore_params.append(param)
            else:
                regular_params.append(param)

        logger.info(f"GaLore params: {len(galore_params)}, Regular params: {len(regular_params)}")

        return galore_params, regular_params

    def _create_optimizer(self) -> GaLoreAdamW8bit:
        """Create GaLore optimizer with layer-wise configuration."""
        galore_params, regular_params = self._get_galore_params()

        optimizer = GaLoreAdamW8bit(
            [
                {
                    "params": galore_params,
                    "rank": self.config.galore_rank,
                    "update_proj_gap": self.config.galore_update_proj_gap,
                    "scale": self.config.galore_scale,
                    "proj_quant": self.config.proj_quant,
                },
                {
                    "params": regular_params,
                },
            ],
            lr=self.config.learning_rate,
        )

        return optimizer

    def _get_training_args(self, output_dir: str) -> TrainingArguments:
        """Create training arguments."""
        run_name = self.config.wandb_run_name or f"qgalore-{self.config.model_id.split('/')[-1]}"

        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            warmup_ratio=self.config.warmup_ratio,

            # Note: learning_rate is set in optimizer, not here
            # But we still need it for scheduler
            learning_rate=self.config.learning_rate,

            # Optimization
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,

            # Logging - log every step to wandb, suppress console dict output
            logging_steps=1,
            logging_first_step=True,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=run_name,
            log_level="warning",

            # Evaluation
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            eval_on_start=True,

            # Saving
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,

            # Max sequence length
            max_grad_norm=1.0,
        )

    def _preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Tokenize the dataset for causal LM training."""
        def tokenize_function(examples):
            # Tokenize the 'text' field
            outputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        return tokenized

    def train(
        self,
        dataset: DatasetDict,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run Q-GaLore fine-tuning.

        Args:
            dataset: DatasetDict with 'train' and 'eval' splits
            output_dir: Override default output directory

        Returns:
            Training results dict
        """
        if self.model is None:
            self.load_model()

        if output_dir is None:
            output_dir = self.config.get_output_dir()

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Starting Q-GaLore training. Output: {output_dir}")
        logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['eval'])}")

        # Preprocess dataset
        tokenized_dataset = self._preprocess_dataset(dataset)

        # Create optimizer
        optimizer = self._create_optimizer()

        # Create training arguments
        training_args = self._get_training_args(output_dir)

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked
        )

        # Create trainer with custom optimizer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            data_collator=data_collator,
            optimizers=(optimizer, None),  # Custom optimizer, default scheduler
            callbacks=[MetricsCallback(print_steps=self.config.logging_steps)],
        )

        # Remove default PrinterCallback to avoid duplicate console output
        self.trainer.remove_callback(transformers.PrinterCallback)

        # Suppress transformers logging to console (wandb still gets logs)
        transformers.logging.set_verbosity_warning()

        # Train
        train_result = self.trainer.train()

        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        logger.info(f"Training complete. Model saved to {output_dir}")

        return {
            "output_dir": output_dir,
            "metrics": metrics,
            "train_samples": len(dataset["train"]),
            "eval_samples": len(dataset["eval"]),
        }

    def evaluate(self, dataset: DatasetDict) -> Dict[str, float]:
        """Evaluate the model on the eval set."""
        if self.trainer is None:
            raise RuntimeError("Must train before evaluating")

        tokenized_dataset = self._preprocess_dataset(dataset)
        metrics = self.trainer.evaluate(tokenized_dataset["eval"])
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

    def save_model(self, output_dir: Optional[str] = None):
        """Save the full model weights."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if output_dir is None:
            output_dir = self.config.get_output_dir()

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


def train_qgalore(
    config: QGaloreConfig,
    dataset: DatasetDict,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run Q-GaLore training.

    Args:
        config: Q-GaLore configuration
        dataset: Prepared dataset
        output_dir: Optional output directory override

    Returns:
        Training results
    """
    trainer = QGaloreTrainer(config)
    return trainer.train(dataset, output_dir)
