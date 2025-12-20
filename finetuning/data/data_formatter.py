"""
Data formatter for converting teacher labels to training format.

Converts labeled samples into instruction-tuning format with model-specific
chat templates.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from transformers import AutoTokenizer

from ..config import DataConfig, SYSTEM_PROMPT
from .teacher_labeler import LabeledSample


@dataclass
class FormattedSample:
    """A formatted sample ready for training."""
    id: str
    domain: str
    messages: List[Dict[str, str]]
    input_text: str
    output_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "messages": self.messages,
            "input_text": self.input_text,
            "output_text": self.output_text,
        }


class DataFormatter:
    """
    Formats teacher labels into instruction-tuning format.

    Supports multiple chat template formats for different models.
    """

    def __init__(self, config: DataConfig, model_id: Optional[str] = None):
        self.config = config
        self.model_id = model_id
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load tokenizer if model_id is set."""
        if self._tokenizer is None and self.model_id:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def _format_kg_output(self, kg: Dict[str, Any]) -> str:
        """Format knowledge graph as JSON string."""
        return json.dumps(kg, indent=2)

    def format_sample(self, sample: LabeledSample) -> FormattedSample:
        """
        Format a single labeled sample for training.

        Uses chat message format compatible with most instruction-tuned models.
        """
        output_text = self._format_kg_output(sample.knowledge_graph)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text: {sample.input_text}"},
            {"role": "assistant", "content": output_text},
        ]

        return FormattedSample(
            id=sample.id,
            domain=sample.domain,
            messages=messages,
            input_text=sample.input_text,
            output_text=output_text,
        )

    def format_samples(self, samples: List[LabeledSample]) -> List[FormattedSample]:
        """Format multiple samples."""
        return [self.format_sample(s) for s in samples]

    def apply_chat_template(
        self,
        sample: FormattedSample,
        add_generation_prompt: bool = False
    ) -> str:
        """
        Apply model-specific chat template.

        Args:
            sample: Formatted sample with messages
            add_generation_prompt: If True, only include system + user (for inference)

        Returns:
            Formatted string ready for tokenization
        """
        if self.tokenizer is None:
            raise ValueError("model_id must be set to apply chat template")

        messages = sample.messages
        if add_generation_prompt:
            # For inference: exclude assistant response
            messages = messages[:-1]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def to_alpaca_format(self, sample: FormattedSample) -> Dict[str, str]:
        """
        Convert to Alpaca format for LLaMA-Factory compatibility.

        Returns:
            Dict with 'instruction', 'input', 'output' keys
        """
        return {
            "instruction": SYSTEM_PROMPT,
            "input": f"Text: {sample.input_text}",
            "output": sample.output_text,
        }

    def to_sharegpt_format(self, sample: FormattedSample) -> Dict[str, Any]:
        """
        Convert to ShareGPT format.

        Returns:
            Dict with 'conversations' key containing list of turns
        """
        conversations = []
        for msg in sample.messages:
            role = "human" if msg["role"] == "user" else msg["role"]
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "gpt"
            conversations.append({
                "from": role,
                "value": msg["content"],
            })

        return {
            "id": sample.id,
            "conversations": conversations,
        }

    def save_formatted_data(
        self,
        samples: List[FormattedSample],
        output_path: str,
        format_type: str = "messages"
    ):
        """
        Save formatted samples to file.

        Args:
            samples: List of formatted samples
            output_path: Output file path (.jsonl)
            format_type: One of 'messages', 'alpaca', 'sharegpt'
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            for sample in samples:
                if format_type == "alpaca":
                    data = self.to_alpaca_format(sample)
                elif format_type == "sharegpt":
                    data = self.to_sharegpt_format(sample)
                else:  # messages
                    data = sample.to_dict()

                f.write(json.dumps(data) + "\n")

    def load_formatted_data(self, input_path: str) -> List[FormattedSample]:
        """Load formatted samples from file."""
        samples = []
        with open(input_path, "r") as f:
            for line in f:
                data = json.loads(line)
                samples.append(FormattedSample(
                    id=data["id"],
                    domain=data["domain"],
                    messages=data["messages"],
                    input_text=data["input_text"],
                    output_text=data["output_text"],
                ))
        return samples


def format_all_data(
    config: DataConfig,
    labeled_data: Dict[str, Dict[str, List[LabeledSample]]],
    format_type: str = "messages"
) -> Dict[str, str]:
    """
    Format all labeled data and save to files.

    Args:
        config: Data configuration
        labeled_data: Nested dict from TeacherLabeler.generate_all_labels()
        format_type: Output format type

    Returns:
        Dict mapping split names to file paths
    """
    formatter = DataFormatter(config)
    output_paths = {}

    for domain in labeled_data:
        for split in labeled_data[domain]:
            samples = labeled_data[domain][split]
            formatted = formatter.format_samples(samples)

            output_path = os.path.join(
                config.train_dir if split == "train" else config.eval_dir,
                f"{domain}_{format_type}.jsonl"
            )

            formatter.save_formatted_data(formatted, output_path, format_type)
            output_paths[f"{domain}_{split}"] = output_path

    # Also create combined files
    all_train = []
    all_eval = []

    for domain in labeled_data:
        all_train.extend(formatter.format_samples(labeled_data[domain]["train"]))
        all_eval.extend(formatter.format_samples(labeled_data[domain]["eval"]))

    train_path = os.path.join(config.train_dir, f"all_domains_{format_type}.jsonl")
    eval_path = os.path.join(config.eval_dir, f"all_domains_{format_type}.jsonl")

    formatter.save_formatted_data(all_train, train_path, format_type)
    formatter.save_formatted_data(all_eval, eval_path, format_type)

    output_paths["all_train"] = train_path
    output_paths["all_eval"] = eval_path

    return output_paths
