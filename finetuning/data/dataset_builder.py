"""
Dataset builder for creating HuggingFace datasets from formatted data.

Handles tokenization and prepares data for the Trainer.
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..config import DataConfig, TrainingConfig
from .data_formatter import FormattedSample


class DatasetBuilder:
    """
    Builds HuggingFace datasets for training.

    Handles:
    - Loading formatted data
    - Applying chat templates
    - Tokenization with proper padding/truncation
    - Creating train/eval splits
    """

    def __init__(
        self,
        config: TrainingConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        self.config = config

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        else:
            self.tokenizer = tokenizer

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def load_formatted_samples(self, file_path: str) -> List[Dict[str, Any]]:
        """Load formatted samples from JSONL file."""
        samples = []
        with open(file_path, "r") as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def _format_for_training(self, sample: Dict[str, Any]) -> str:
        """
        Format a sample for training using the tokenizer's chat template.

        Returns the full conversation including the assistant response.
        """
        messages = sample["messages"]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _tokenize_function(
        self,
        examples: Dict[str, List],
        max_length: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Tokenize examples for causal LM training.

        For SFT, we need to:
        1. Tokenize the full conversation
        2. Create labels that mask the prompt (only predict assistant response)
        """
        if max_length is None:
            max_length = self.config.max_seq_length

        # Format all samples
        formatted_texts = []
        for messages in examples["messages"]:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted_texts.append(text)

        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels = input_ids (shifted internally by the model)
        # We could mask prompt tokens, but TRL's SFTTrainer handles this
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def _create_sft_dataset(
        self,
        samples: List[Dict[str, Any]],
        max_length: Optional[int] = None
    ) -> Dataset:
        """
        Create a dataset ready for SFTTrainer.

        The dataset will have 'text' column with formatted conversations.
        """
        # Format each sample
        formatted_data = []
        for sample in samples:
            text = self._format_for_training(sample)
            formatted_data.append({
                "id": sample["id"],
                "domain": sample["domain"],
                "text": text,
                "messages": sample["messages"],
            })

        return Dataset.from_list(formatted_data)

    def build_dataset(
        self,
        train_path: str,
        eval_path: str,
        use_sft_format: bool = True
    ) -> DatasetDict:
        """
        Build complete dataset dictionary with train and eval splits.

        Args:
            train_path: Path to training data JSONL
            eval_path: Path to evaluation data JSONL
            use_sft_format: If True, create 'text' column for SFTTrainer

        Returns:
            DatasetDict with 'train' and 'eval' splits
        """
        train_samples = self.load_formatted_samples(train_path)
        eval_samples = self.load_formatted_samples(eval_path)

        if use_sft_format:
            train_dataset = self._create_sft_dataset(train_samples)
            eval_dataset = self._create_sft_dataset(eval_samples)
        else:
            # Tokenized format for standard Trainer
            train_dataset = Dataset.from_list(train_samples)
            eval_dataset = Dataset.from_list(eval_samples)

            train_dataset = train_dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=["id", "domain", "input_text", "output_text"],
            )
            eval_dataset = eval_dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=["id", "domain", "input_text", "output_text"],
            )

        return DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset,
        })

    def build_from_config(self, data_config: DataConfig) -> DatasetDict:
        """
        Build dataset from data config paths.

        Expects formatted data at:
        - {data_config.train_dir}/all_domains_messages.jsonl
        - {data_config.eval_dir}/all_domains_messages.jsonl
        """
        train_path = os.path.join(data_config.train_dir, "all_domains_messages.jsonl")
        eval_path = os.path.join(data_config.eval_dir, "all_domains_messages.jsonl")

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Run teacher labeling and formatting first."
            )

        if not os.path.exists(eval_path):
            raise FileNotFoundError(
                f"Evaluation data not found at {eval_path}. "
                "Run teacher labeling and formatting first."
            )

        return self.build_dataset(train_path, eval_path)

    def get_data_stats(self, dataset: DatasetDict) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {}

        for split_name, split_data in dataset.items():
            # Count by domain
            domain_counts = {}
            for item in split_data:
                domain = item.get("domain", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            # Average text length
            if "text" in split_data.column_names:
                avg_len = sum(len(t) for t in split_data["text"]) / len(split_data)
            else:
                avg_len = 0

            stats[split_name] = {
                "total_samples": len(split_data),
                "domains": domain_counts,
                "avg_text_length": round(avg_len, 0),
            }

        return stats


def prepare_datasets(
    data_config: DataConfig,
    training_config: TrainingConfig
) -> DatasetDict:
    """
    Convenience function to prepare datasets from configs.

    Returns:
        DatasetDict ready for training
    """
    builder = DatasetBuilder(training_config)
    return builder.build_from_config(data_config)
