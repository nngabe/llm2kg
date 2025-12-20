"""
HuggingFace Hub integration for uploading fine-tuned models.

Supports uploading LoRA adapters and merged models with model cards.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    ModelCard,
    ModelCardData,
)

from ..config import StorageConfig, TrainingConfig, QGaloreConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFUploader:
    """
    Uploads fine-tuned models to HuggingFace Hub.

    Supports:
    - LoRA adapters (small, quick uploads)
    - Merged full models (large, optional)
    - Auto-generated model cards with training details
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self._api = None

    @property
    def api(self) -> HfApi:
        """Lazy-load HfApi with authentication."""
        if self._api is None:
            token = self.config.hf_token or os.environ.get("HF_TOKEN")
            self._api = HfApi(token=token)
        return self._api

    def _create_model_card(
        self,
        training_config: TrainingConfig,
        metrics: Optional[Dict[str, float]] = None,
        method: str = "qlora",
    ) -> str:
        """Generate model card content."""
        base_model = training_config.model_id
        model_name = base_model.split("/")[-1]

        # Determine method-specific details
        if method == "qlora":
            method_desc = "QLoRA (Quantized Low-Rank Adaptation)"
            training_details = f"""
- **LoRA Rank (r):** {training_config.lora_r}
- **LoRA Alpha:** {training_config.lora_alpha}
- **LoRA Dropout:** {training_config.lora_dropout}
- **Target Modules:** {', '.join(training_config.target_modules)}
- **Quantization:** 4-bit NF4 with double quantization
"""
        else:  # qgalore
            config = training_config  # Should be QGaloreConfig
            method_desc = "Q-GaLore (Quantized Gradient Low-Rank Projection)"
            if hasattr(config, 'galore_rank'):
                training_details = f"""
- **GaLore Rank:** {config.galore_rank}
- **Update Projection Gap:** {config.galore_update_proj_gap}
- **Scale:** {config.galore_scale}
- **Projection Quantization:** INT4
- **Target Modules:** {', '.join(config.target_modules)}
"""
            else:
                training_details = "- See training config for details"

        # Build metrics section
        metrics_section = ""
        if metrics:
            metrics_section = "\n## Evaluation Results\n\n"
            metrics_section += "| Metric | Value |\n|--------|-------|\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_section += f"| {key} | {value:.4f} |\n"
                else:
                    metrics_section += f"| {key} | {value} |\n"

        card_content = f"""---
language:
- en
license: apache-2.0
base_model: {base_model}
tags:
- knowledge-graph
- extraction
- {method}
- fine-tuned
datasets:
- cais/wmdp-mmlu-auxiliary-corpora
---

# {model_name} - Knowledge Graph Extraction ({method.upper()})

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) for knowledge graph extraction from text.

## Model Description

This model was fine-tuned using **{method_desc}** to extract structured knowledge graphs from unstructured text. Given a text passage, it outputs a JSON representation of entities (nodes) and relationships (edges).

## Training Details

**Base Model:** {base_model}

**Fine-tuning Method:** {method_desc}

**Training Configuration:**
{training_details}

**Hyperparameters:**
- **Learning Rate:** {training_config.learning_rate}
- **Batch Size:** {training_config.per_device_train_batch_size} × {training_config.gradient_accumulation_steps} (effective)
- **Epochs:** {training_config.num_train_epochs}
- **Max Sequence Length:** {training_config.max_seq_length}
- **Optimizer:** {training_config.optim}
- **Precision:** BF16
{metrics_section}

## Intended Use

This model is designed for extracting knowledge graphs from text in domains such as:
- Economics
- Law
- Physics

### Input Format

```
Text: [Your input text here]
```

### Output Format

```json
{{
  "nodes": [
    {{"id": "Entity Name", "type": "Category", "description": "Brief description"}}
  ],
  "edges": [
    {{"source": "Entity A", "target": "Entity B", "relation": "relationship_type", "description": "Context"}}
  ]
}}
```

## Usage

### With PEFT (for LoRA adapters)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/THIS_MODEL")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Generate
messages = [
    {{"role": "system", "content": "You are a Knowledge Graph expert..."}},
    {{"role": "user", "content": "Text: Your text here"}}
]
# ... generate response
```

## Training Data

The model was fine-tuned on teacher labels generated by Gemini 2.5 Flash and GPT-5.2 from the following datasets:
- Economics corpus from `cais/wmdp-mmlu-auxiliary-corpora`
- Law corpus from `cais/wmdp-mmlu-auxiliary-corpora`
- Physics corpus from `cais/wmdp-mmlu-auxiliary-corpora`

## Limitations

- Best performance on English text
- Optimized for the domains used in training (economics, law, physics)
- May require prompt engineering for other domains

## Citation

If you use this model, please cite:

```bibtex
@misc{{kg-extraction-{method},
  author = {{Your Name}},
  title = {{{model_name} Fine-tuned for Knowledge Graph Extraction}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/YOUR_USERNAME/THIS_MODEL}}
}}
```
"""
        return card_content

    def create_repo(
        self,
        repo_id: str,
        private: bool = False,
    ) -> str:
        """Create a new repository on HuggingFace Hub."""
        try:
            create_repo(
                repo_id=repo_id,
                token=self.api.token,
                private=private,
                exist_ok=True,
            )
            logger.info(f"Repository created/verified: {repo_id}")
            return f"https://huggingface.co/{repo_id}"
        except Exception as e:
            logger.error(f"Failed to create repo {repo_id}: {e}")
            raise

    def upload_adapter(
        self,
        adapter_path: str,
        training_config: TrainingConfig,
        metrics: Optional[Dict[str, float]] = None,
        method: str = "qlora",
        private: bool = False,
    ) -> str:
        """
        Upload LoRA adapter to HuggingFace Hub.

        Args:
            adapter_path: Path to adapter directory
            training_config: Training configuration used
            metrics: Evaluation metrics to include in model card
            method: Fine-tuning method (qlora or qgalore)
            private: Whether to make repo private

        Returns:
            URL of uploaded model
        """
        # Generate repo ID
        model_name = training_config.model_id.split("/")[-1]
        repo_id = self.config.get_repo_id(model_name, method)

        logger.info(f"Uploading adapter to {repo_id}")

        # Create repo
        self.create_repo(repo_id, private)

        # Generate and save model card
        card_content = self._create_model_card(training_config, metrics, method)
        card_path = os.path.join(adapter_path, "README.md")
        with open(card_path, "w") as f:
            f.write(card_content)

        # Upload folder
        upload_folder(
            folder_path=adapter_path,
            repo_id=repo_id,
            token=self.api.token,
            commit_message=f"Upload {method} adapter",
        )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Adapter uploaded successfully: {url}")
        return url

    def upload_merged_model(
        self,
        model_path: str,
        training_config: TrainingConfig,
        metrics: Optional[Dict[str, float]] = None,
        method: str = "qlora",
        private: bool = False,
    ) -> str:
        """
        Upload merged full model to HuggingFace Hub.

        Args:
            model_path: Path to merged model directory
            training_config: Training configuration used
            metrics: Evaluation metrics to include in model card
            method: Fine-tuning method (qlora or qgalore)
            private: Whether to make repo private

        Returns:
            URL of uploaded model
        """
        # Generate repo ID with -merged suffix
        model_name = training_config.model_id.split("/")[-1]
        repo_id = self.config.get_repo_id(model_name, f"{method}-merged")

        logger.info(f"Uploading merged model to {repo_id}")

        # Create repo
        self.create_repo(repo_id, private)

        # Generate and save model card
        card_content = self._create_model_card(training_config, metrics, method)
        card_path = os.path.join(model_path, "README.md")
        with open(card_path, "w") as f:
            f.write(card_content)

        # Upload folder (this can take a while for large models)
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=self.api.token,
            commit_message=f"Upload merged {method} model",
        )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Merged model uploaded successfully: {url}")
        return url

    def upload_training_artifacts(
        self,
        output_dir: str,
        training_config: TrainingConfig,
        metrics: Optional[Dict[str, float]] = None,
        method: str = "qlora",
        private: bool = False,
    ) -> Dict[str, str]:
        """
        Upload all training artifacts based on config settings.

        Args:
            output_dir: Training output directory
            training_config: Training configuration
            metrics: Evaluation metrics
            method: Fine-tuning method
            private: Whether to make repos private

        Returns:
            Dict with URLs of uploaded artifacts
        """
        urls = {}

        # Upload adapter
        if self.config.upload_adapters:
            adapter_path = os.path.join(output_dir, "adapter")
            if os.path.exists(adapter_path):
                urls["adapter"] = self.upload_adapter(
                    adapter_path, training_config, metrics, method, private
                )
            else:
                # Adapter is in main output dir
                urls["adapter"] = self.upload_adapter(
                    output_dir, training_config, metrics, method, private
                )

        # Upload merged model
        if self.config.upload_merged:
            merged_path = os.path.join(output_dir, "merged")
            if os.path.exists(merged_path):
                urls["merged"] = self.upload_merged_model(
                    merged_path, training_config, metrics, method, private
                )

        return urls
