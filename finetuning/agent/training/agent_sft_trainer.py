"""
Agent SFT Trainer for tool-use fine-tuning.

Extends QLoRA training for multi-turn tool-calling conversations.
Designed for Qwen3 models with native tool-calling support.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List

import torch
import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

from ...config import AgentSFTConfig
from ...training.callbacks import MetricsCallback
from ..data.agent_formatter import AgentFormatter
from ..tools import get_tools_for_qwen3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSFTTrainer:
    """
    Agent SFT Trainer for tool-use fine-tuning.

    Features:
    - QLoRA-based training for memory efficiency
    - Multi-turn conversation handling
    - Qwen3 native tool-calling format
    - Custom data collation for tool responses
    """

    def __init__(self, config: AgentSFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.formatter = AgentFormatter(
            format_type=config.tool_format,
            include_system_prompt=True,
        )

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
            attn_implementation="flash_attention_2",
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

        logger.info("Model loaded successfully with QLoRA for agent training")

    def _load_conversations(self, data_path: str) -> List[Dict[str, Any]]:
        """Load conversations from JSONL file."""
        conversations = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
        return conversations

    def _prepare_dataset(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
    ) -> DatasetDict:
        """
        Prepare dataset from conversation files.

        Args:
            train_path: Path to training JSONL
            eval_path: Path to evaluation JSONL

        Returns:
            DatasetDict with train and eval splits
        """
        train_path = train_path or self.config.train_data_path
        eval_path = eval_path or self.config.eval_data_path

        # Load and format conversations
        train_convs = self._load_conversations(train_path)
        eval_convs = self._load_conversations(eval_path)

        logger.info(f"Loaded {len(train_convs)} train, {len(eval_convs)} eval conversations")

        # Format for training
        train_formatted = self.formatter.format_dataset(train_convs)
        eval_formatted = self.formatter.format_dataset(eval_convs)

        # Create datasets
        train_dataset = Dataset.from_list(train_formatted)
        eval_dataset = Dataset.from_list(eval_formatted)

        return DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset,
        })

    def _get_training_args(self, output_dir: str) -> SFTConfig:
        """Create training arguments."""
        run_name = self.config.wandb_run_name or f"agent-sft-{self.config.model_id.split('/')[-1]}"

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
            logging_steps=1,
            logging_first_step=True,
            report_to=["wandb"] if self.config.use_wandb else [],
            run_name=run_name,
            log_level="error",
            disable_tqdm=False,

            # Evaluation
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            eval_on_start=True,

            # Saving
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,

            # Dataset
            dataset_text_field="text",
            packing=False,  # Don't pack for tool-calling (structure matters)
        )

    def train(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run agent SFT training.

        Args:
            train_path: Path to training data (JSONL)
            eval_path: Path to eval data (JSONL)
            output_dir: Override default output directory

        Returns:
            Training results dict
        """
        if self.model is None:
            self.load_model()

        if output_dir is None:
            output_dir = self.config.get_output_dir()

        os.makedirs(output_dir, exist_ok=True)

        # Prepare dataset
        dataset = self._prepare_dataset(train_path, eval_path)

        logger.info(f"Starting Agent SFT training. Output: {output_dir}")
        logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['eval'])}")

        # Create trainer
        training_args = self._get_training_args(output_dir)

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            processing_class=self.tokenizer,
            callbacks=[MetricsCallback(print_steps=self.config.logging_steps)],
        )

        # Remove default callbacks that print to console
        from transformers.trainer_callback import PrinterCallback
        self.trainer.remove_callback(PrinterCallback)

        # Suppress transformers logging to console
        transformers.logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("trl").setLevel(logging.ERROR)

        # Train
        train_result = self.trainer.train()

        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Save tool definitions with the model
        tools_path = os.path.join(output_dir, "tools.json")
        with open(tools_path, "w") as f:
            json.dump(get_tools_for_qwen3(), f, indent=2)

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

    def evaluate(self, eval_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the model on the eval set."""
        if self.trainer is None:
            raise RuntimeError("Must train before evaluating")

        if eval_path:
            # Load and format new eval data
            eval_convs = self._load_conversations(eval_path)
            eval_formatted = self.formatter.format_dataset(eval_convs)
            eval_dataset = Dataset.from_list(eval_formatted)
            metrics = self.trainer.evaluate(eval_dataset)
        else:
            metrics = self.trainer.evaluate()

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

        # Also save tools
        tools_path = os.path.join(output_dir, "tools.json")
        with open(tools_path, "w") as f:
            json.dump(get_tools_for_qwen3(), f, indent=2)

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

        # Save tools
        tools_path = os.path.join(output_dir, "tools.json")
        with open(tools_path, "w") as f:
            json.dump(get_tools_for_qwen3(), f, indent=2)

        logger.info(f"Merged model saved to {output_dir}")


def train_agent_sft(
    config: AgentSFTConfig,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run Agent SFT training.

    Args:
        config: Agent SFT configuration
        train_path: Optional training data path
        eval_path: Optional eval data path
        output_dir: Optional output directory override

    Returns:
        Training results
    """
    trainer = AgentSFTTrainer(config)
    return trainer.train(train_path, eval_path, output_dir)
