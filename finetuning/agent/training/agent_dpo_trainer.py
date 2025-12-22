"""
Agent DPO Trainer for faithfulness fine-tuning.

Uses Direct Preference Optimization to train the model to prefer
faithful, grounded responses over hallucinated ones.
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
    PeftModel,
    TaskType,
)
from trl import DPOTrainer, DPOConfig

from ...config import AgentDPOConfig
from ...training.callbacks import MetricsCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentDPOTrainer:
    """
    Agent DPO Trainer for faithfulness fine-tuning.

    Features:
    - Loads SFT checkpoint as base model
    - Applies LoRA for memory-efficient DPO training
    - Supports both perturbation-based and LLM-generated rejected responses
    """

    def __init__(self, config: AgentDPOConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytes 4-bit quantization config."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
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
        """Load SFT checkpoint and prepare for DPO training."""
        # Determine model path
        if self.config.sft_checkpoint and os.path.exists(self.config.sft_checkpoint):
            model_path = self.config.sft_checkpoint
            logger.info(f"Loading SFT checkpoint: {model_path}")
        else:
            model_path = self.config.model_id
            logger.info(f"No SFT checkpoint, loading base model: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with quantization
        bnb_config = self._get_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        # Check if model already has LoRA adapters (from SFT)
        if hasattr(self.model, 'peft_config'):
            logger.info("Model has existing LoRA adapters, merging before DPO training")
            self.model = self.model.merge_and_unload()

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Apply fresh LoRA for DPO
        lora_config = self._get_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # Reference model for DPO (frozen copy)
        # For memory efficiency, we use the same model with LoRA disabled
        # TRL DPOTrainer handles this internally
        self.ref_model = None

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("Model loaded successfully for DPO training")

    def _load_dpo_samples(self, data_path: str) -> List[Dict[str, Any]]:
        """Load DPO samples from JSONL file."""
        samples = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _prepare_dataset(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
    ) -> DatasetDict:
        """
        Prepare DPO dataset.

        Expects JSONL files with:
        - prompt: The context + question
        - chosen: Faithful response
        - rejected: Hallucinated response

        Returns:
            DatasetDict with train and eval splits
        """
        train_path = train_path or self.config.train_data_path
        eval_path = eval_path or self.config.eval_data_path

        # Load samples
        train_samples = self._load_dpo_samples(train_path)
        eval_samples = self._load_dpo_samples(eval_path)

        logger.info(f"Loaded {len(train_samples)} train, {len(eval_samples)} eval DPO samples")

        # Convert to datasets format
        def convert_samples(samples: List[Dict]) -> List[Dict]:
            """Convert to TRL DPO format."""
            converted = []
            for s in samples:
                converted.append({
                    "prompt": s.get("prompt", ""),
                    "chosen": s.get("chosen", ""),
                    "rejected": s.get("rejected", ""),
                })
            return converted

        train_data = convert_samples(train_samples)
        eval_data = convert_samples(eval_samples)

        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)

        return DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset,
        })

    def _get_training_args(self, output_dir: str) -> DPOConfig:
        """Create DPO training arguments."""
        run_name = self.config.wandb_run_name or f"agent-dpo-{self.config.model_id.split('/')[-1]}"

        return DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            warmup_ratio=self.config.warmup_ratio,
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_seq_length // 2,

            # DPO-specific
            beta=self.config.beta,
            loss_type="sigmoid",  # Standard DPO loss

            # Optimization
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

            # Remove default packing
            remove_unused_columns=False,
        )

    def train(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run DPO training.

        Args:
            train_path: Path to training data (JSONL with prompt/chosen/rejected)
            eval_path: Path to eval data
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

        logger.info(f"Starting DPO training. Output: {output_dir}")
        logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['eval'])}")
        logger.info(f"DPO beta: {self.config.beta}")

        # Create trainer
        training_args = self._get_training_args(output_dir)

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,  # None = use implicit reference
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

        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # Save DPO config
        config_path = os.path.join(output_dir, "dpo_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "beta": self.config.beta,
                "sft_checkpoint": self.config.sft_checkpoint,
                "learning_rate": self.config.learning_rate,
            }, f, indent=2)

        logger.info(f"DPO training complete. Model saved to {output_dir}")

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
            eval_samples = self._load_dpo_samples(eval_path)
            eval_data = [{"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]} for s in eval_samples]
            eval_dataset = Dataset.from_list(eval_data)
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

        logger.info(f"DPO adapter saved to {output_dir}")

    def merge_and_save(self, output_dir: str):
        """Merge LoRA weights into base model and save."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        logger.info("Merging DPO LoRA weights into base model...")

        # Merge weights
        merged_model = self.model.merge_and_unload()

        # Save merged model
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Merged DPO model saved to {output_dir}")


def train_agent_dpo(
    config: AgentDPOConfig,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run Agent DPO training.

    Args:
        config: Agent DPO configuration
        train_path: Optional training data path
        eval_path: Optional eval data path
        output_dir: Optional output directory override

    Returns:
        Training results
    """
    trainer = AgentDPOTrainer(config)
    return trainer.train(train_path, eval_path, output_dir)
