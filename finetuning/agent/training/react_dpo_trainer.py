"""
ReAct DPO Trainer for faithful reasoning training.

Uses Direct Preference Optimization to train the model to prefer
faithful ReAct reasoning traces over unfaithful ones.

Key training objectives:
1. Complete Thought→Action→Observation chains
2. Grounded answers that follow from observations
3. Appropriate tool selection
4. No premature conclusions or hallucinated reasoning
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

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
from trl import DPOTrainer, DPOConfig

from ...config import AgentDPOConfig, MODELS_DIR
from ...training.callbacks import MetricsCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReActDPOConfig(AgentDPOConfig):
    """Configuration for ReAct DPO training."""

    # Output directory
    output_dir: str = os.path.join(MODELS_DIR, "react_dpo")

    # ReAct-specific settings
    max_react_steps: int = 10
    include_tool_calls: bool = True

    # Training hyperparameters
    beta: float = 0.1  # KL penalty coefficient
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    warmup_ratio: float = 0.1
    max_seq_length: int = 8192  # Longer for ReAct traces

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Data paths
    train_data_path: str = "data/react_dpo/train.jsonl"
    eval_data_path: str = "data/react_dpo/eval.jsonl"

    # Logging
    wandb_project: str = "llm2kg-react"
    wandb_run_name: Optional[str] = None


class ReActDPOTrainer:
    """
    ReAct DPO Trainer for faithful reasoning training.

    Features:
    - Specialized for ReAct reasoning traces
    - Supports multi-turn tool-use conversations
    - Penalizes unfaithful reasoning patterns
    """

    # System prompt for ReAct agent
    REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that uses tools to answer questions accurately.

When answering questions, you MUST follow the ReAct (Reasoning and Acting) pattern:
1. Thought: Reason about what you need to find out and which tool to use
2. Action: Choose the appropriate tool
3. Action Input: Provide the tool input
4. Observation: Analyze the tool's response
5. Repeat steps 1-4 as needed
6. Final Answer: Provide a grounded answer based on your observations

Available tools:
- graph_lookup: Look up entities in the knowledge graph
- entity_resolve: Resolve entity names to canonical forms
- cypher_query: Run custom Cypher queries on the graph
- web_search: Search the web for additional information
- get_entity_relationships: Get relationships for an entity

IMPORTANT:
- Always use tools to verify information before answering
- Never skip the Thought step - explain your reasoning
- Base your answer on tool observations, not assumptions
- If you're unsure, gather more information before concluding
"""

    def __init__(self, config: ReActDPOConfig):
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
        """Load model and prepare for DPO training."""
        # Determine model path
        if self.config.sft_checkpoint and os.path.exists(self.config.sft_checkpoint):
            model_path = self.config.sft_checkpoint
            logger.info(f"Loading SFT checkpoint: {model_path}")
        else:
            model_path = self.config.model_id
            logger.info(f"Loading base model: {model_path}")

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

        # Check if model already has LoRA adapters
        if hasattr(self.model, 'peft_config'):
            logger.info("Merging existing LoRA adapters")
            self.model = self.model.merge_and_unload()

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Apply LoRA
        lora_config = self._get_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # Reference model (None = use implicit reference in DPOTrainer)
        self.ref_model = None

        self.model.print_trainable_parameters()
        logger.info("Model loaded successfully for ReAct DPO training")

    def _load_samples(self, data_path: str) -> List[Dict[str, Any]]:
        """Load ReAct DPO samples from JSONL file."""
        samples = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _format_prompt(self, question: str, context: str = "") -> str:
        """Format prompt with ReAct system message."""
        parts = [self.REACT_SYSTEM_PROMPT]

        if context:
            parts.append(f"\nContext:\n{context}")

        parts.append(f"\nQuestion: {question}")
        parts.append("\nProvide your answer using the ReAct format:")

        return "\n".join(parts)

    def _prepare_dataset(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
    ) -> DatasetDict:
        """
        Prepare ReAct DPO dataset.

        Expects JSONL files with:
        - prompt: Question (+ optional context)
        - chosen: Faithful ReAct trace
        - rejected: Unfaithful ReAct trace
        - perturbation_type: Type of unfaithfulness (for analysis)

        Returns:
            DatasetDict with train and eval splits
        """
        train_path = train_path or self.config.train_data_path
        eval_path = eval_path or self.config.eval_data_path

        train_samples = self._load_samples(train_path)
        eval_samples = self._load_samples(eval_path)

        logger.info(f"Loaded {len(train_samples)} train, {len(eval_samples)} eval samples")

        # Log perturbation type distribution
        train_types = {}
        for s in train_samples:
            pt = s.get("perturbation_type", "unknown")
            train_types[pt] = train_types.get(pt, 0) + 1
        logger.info(f"Training perturbation types: {train_types}")

        def convert_samples(samples: List[Dict]) -> List[Dict]:
            """Convert to TRL DPO format with system prompt."""
            converted = []
            for s in samples:
                # Extract question from prompt
                prompt = s.get("prompt", "")

                # Add system prompt if not already included
                if "ReAct" not in prompt and "Thought:" not in prompt:
                    # Parse question from prompt
                    if "Question:" in prompt:
                        parts = prompt.split("Question:")
                        context = parts[0].strip() if parts[0].strip() else ""
                        question = parts[1].strip() if len(parts) > 1 else prompt
                    else:
                        context = ""
                        question = prompt

                    formatted_prompt = self._format_prompt(question, context)
                else:
                    formatted_prompt = prompt

                converted.append({
                    "prompt": formatted_prompt,
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
        run_name = self.config.wandb_run_name or f"react-dpo-{self.config.model_id.split('/')[-1]}"

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
            loss_type="sigmoid",

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

            remove_unused_columns=False,
        )

    def train(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run ReAct DPO training.

        Args:
            train_path: Path to training data (JSONL)
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

        logger.info(f"Starting ReAct DPO training. Output: {output_dir}")
        logger.info(f"Train: {len(dataset['train'])}, Eval: {len(dataset['eval'])} samples")
        logger.info(f"DPO beta: {self.config.beta}, LR: {self.config.learning_rate}")

        # Create trainer
        training_args = self._get_training_args(output_dir)

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            processing_class=self.tokenizer,
            callbacks=[MetricsCallback(print_steps=self.config.logging_steps)],
        )

        # Remove noisy callbacks
        from transformers.trainer_callback import PrinterCallback
        self.trainer.remove_callback(PrinterCallback)

        # Suppress logging
        transformers.logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("trl").setLevel(logging.ERROR)

        # Train
        train_result = self.trainer.train()

        # Save
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # Save config
        config_path = os.path.join(output_dir, "react_dpo_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "beta": self.config.beta,
                "sft_checkpoint": self.config.sft_checkpoint,
                "learning_rate": self.config.learning_rate,
                "max_react_steps": self.config.max_react_steps,
                "max_seq_length": self.config.max_seq_length,
            }, f, indent=2)

        logger.info(f"ReAct DPO training complete. Model saved to {output_dir}")

        return {
            "output_dir": output_dir,
            "metrics": metrics,
            "train_samples": len(dataset["train"]),
            "eval_samples": len(dataset["eval"]),
        }

    def evaluate(
        self,
        eval_path: Optional[str] = None,
        analyze_by_perturbation: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on eval set.

        Args:
            eval_path: Optional path to eval data
            analyze_by_perturbation: Whether to analyze results by perturbation type

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Must train before evaluating")

        if eval_path:
            eval_samples = self._load_samples(eval_path)
            eval_data = [{
                "prompt": s["prompt"],
                "chosen": s["chosen"],
                "rejected": s["rejected"],
            } for s in eval_samples]
            eval_dataset = Dataset.from_list(eval_data)
            metrics = self.trainer.evaluate(eval_dataset)
        else:
            metrics = self.trainer.evaluate()

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return {"metrics": metrics}

    def analyze_perturbation_performance(
        self,
        eval_path: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance by perturbation type.

        Args:
            eval_path: Path to eval data with perturbation_type field

        Returns:
            Dict mapping perturbation type to metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        samples = self._load_samples(eval_path)

        # Group by perturbation type
        by_type = {}
        for s in samples:
            pt = s.get("perturbation_type", "unknown")
            if pt not in by_type:
                by_type[pt] = []
            by_type[pt].append(s)

        results = {}
        for pt, pt_samples in by_type.items():
            pt_data = [{
                "prompt": s["prompt"],
                "chosen": s["chosen"],
                "rejected": s["rejected"],
            } for s in pt_samples]
            pt_dataset = Dataset.from_list(pt_data)

            # Evaluate subset
            pt_metrics = self.trainer.evaluate(pt_dataset)
            results[pt] = {
                "count": len(pt_samples),
                "eval_loss": pt_metrics.get("eval_loss", 0),
            }

        logger.info(f"Performance by perturbation type:")
        for pt, pt_results in results.items():
            logger.info(f"  {pt}: {pt_results}")

        return results

    def save_adapter(self, output_dir: Optional[str] = None):
        """Save only the LoRA adapter weights."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if output_dir is None:
            output_dir = os.path.join(self.config.get_output_dir(), "adapter")

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)

        logger.info(f"ReAct DPO adapter saved to {output_dir}")

    def merge_and_save(self, output_dir: str):
        """Merge LoRA weights into base model and save."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        logger.info("Merging ReAct DPO LoRA weights...")

        merged_model = self.model.merge_and_unload()

        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Merged model saved to {output_dir}")


def train_react_dpo(
    config: ReActDPOConfig,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run ReAct DPO training.

    Args:
        config: ReAct DPO configuration
        train_path: Optional training data path
        eval_path: Optional eval data path
        output_dir: Optional output directory override

    Returns:
        Training results
    """
    trainer = ReActDPOTrainer(config)
    return trainer.train(train_path, eval_path, output_dir)
