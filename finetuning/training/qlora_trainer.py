"""
QLoRA fine-tuning for Knowledge Graph extraction.

Uses 4-bit quantization with LoRA adapters for memory-efficient training.
"""

import os
import logging
from typing import Optional, Dict, Any

import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

from ..config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QLoRATrainer:
    """
    QLoRA fine-tuning for KG extraction.

    Features:
    - 4-bit NF4 quantization via BitsAndBytes
    - LoRA adapters on attention + MLP layers
    - Gradient checkpointing for memory efficiency
    - Integration with Weights & Biases for logging
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytes 4-bit quantization config."""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

    def _get_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def load_model(self) -> None:
        """Load base model with 4-bit quantization and apply LoRA."""
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

        # Load model with quantization
        bnb_config = self._get_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Use Flash Attention if available
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Apply LoRA
        lora_config = self._get_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("Model loaded successfully with QLoRA")

    def _get_training_args(self, output_dir: str) -> SFTConfig:
        """Create training arguments."""
        run_name = self.config.wandb_run_name or f"qlora-{self.config.model_id.split('/')[-1]}"

        return SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            warmup_ratio=self.config.warmup_ratio,
            max_length=self.config.max_seq_length,

            # Optimization
            optim=self.config.optim,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,

            # Logging
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=run_name,

            # Evaluation
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,

            # Saving
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,

            # Dataset
            dataset_text_field="text",
            packing=False,  # Don't pack samples for structured output
        )

    def train(
        self,
        dataset: DatasetDict,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run QLoRA fine-tuning.

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

        logger.info(f"Starting QLoRA training. Output: {output_dir}")
        logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['eval'])}")

        # Create trainer
        training_args = self._get_training_args(output_dir)

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            processing_class=self.tokenizer,
        )

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

        metrics = self.trainer.evaluate(dataset["eval"])
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

    def save_adapter(self, output_dir: Optional[str] = None):
        """Save only the LoRA adapter weights."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if output_dir is None:
            output_dir = os.path.join(self.config.get_output_dir(), "adapter")

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        logger.info(f"Adapter saved to {output_dir}")

    def merge_and_save(self, output_dir: str):
        """Merge LoRA weights into base model and save."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        logger.info("Merging LoRA weights into base model...")

        # Merge weights
        merged_model = self.model.merge_and_unload()

        # Save merged model
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Merged model saved to {output_dir}")


def train_qlora(
    config: TrainingConfig,
    dataset: DatasetDict,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run QLoRA training.

    Args:
        config: Training configuration
        dataset: Prepared dataset
        output_dir: Optional output directory override

    Returns:
        Training results
    """
    trainer = QLoRATrainer(config)
    return trainer.train(dataset, output_dir)
