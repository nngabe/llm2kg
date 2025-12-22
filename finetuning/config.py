"""
Configuration dataclasses for fine-tuning pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class TeacherModel(Enum):
    """Supported teacher models for label generation."""
    GEMINI_FLASH = "gemini-2.5-flash"
    GPT_52 = "gpt-5.2"


class StudentModel(Enum):
    """Supported student models for fine-tuning."""
    GEMMA3_12B = "google/gemma-3-12b-it"
    QWEN3_30B_A3B = "Qwen/Qwen3-30B-A3B"


class Domain(Enum):
    """Supported dataset domains."""
    ECONOMICS = "economics"
    LAW = "law"
    PHYSICS = "physics"


# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetuning")
MODELS_DIR = os.path.join(BASE_DIR, "models", "finetuned")


@dataclass
class DataConfig:
    """Configuration for dataset preparation."""

    # HuggingFace dataset
    dataset_name: str = "cais/wmdp-mmlu-auxiliary-corpora"
    domains: List[str] = field(default_factory=lambda: ["economics", "law", "physics"])

    # Data split sizes
    train_samples_per_domain: int = 100
    eval_samples_per_domain: int = 20

    # Text chunking
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_chunks_per_doc: int = 3  # Limit chunks to avoid very long documents

    # Base output directory (paths derived in __post_init__)
    output_dir: Optional[str] = None
    teacher_labels_dir: Optional[str] = None
    train_dir: Optional[str] = None
    eval_dir: Optional[str] = None

    # Teacher model settings
    primary_teacher: str = TeacherModel.GEMINI_FLASH.value
    fallback_teacher: str = TeacherModel.GPT_52.value
    teacher_temperature: float = 0.0

    # Rate limiting
    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds, will use exponential backoff

    def __post_init__(self):
        """Set up paths and create directories."""
        # Use provided output_dir or default
        base_dir = self.output_dir if self.output_dir else DATA_DIR

        # Set derived paths if not explicitly provided
        if self.teacher_labels_dir is None:
            self.teacher_labels_dir = os.path.join(base_dir, "teacher_labels")
        if self.train_dir is None:
            self.train_dir = os.path.join(base_dir, "train")
        if self.eval_dir is None:
            self.eval_dir = os.path.join(base_dir, "eval")

        # Alias for compatibility
        self.labels_dir = self.teacher_labels_dir

        # Create directories
        for dir_path in [self.teacher_labels_dir, self.train_dir, self.eval_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class TrainingConfig:
    """Configuration for QLoRA training."""

    # Model
    model_id: str = StudentModel.GEMMA3_12B.value

    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training hyperparameters
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096

    # Optimization
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging and saving
    logging_steps: int = 5
    eval_steps: int = 10
    save_steps: int = 100
    output_dir: str = os.path.join(MODELS_DIR, "qlora")

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "llm2kg-finetuning"
    wandb_run_name: Optional[str] = None

    def get_output_dir(self) -> str:
        """Get model-specific output directory."""
        model_name = self.model_id.split("/")[-1]
        return os.path.join(self.output_dir, model_name)


@dataclass
class QGaloreConfig(TrainingConfig):
    """Configuration for QGalore training (extends TrainingConfig)."""

    # Override base settings
    load_in_4bit: bool = False  # QGalore uses bf16 base, not 4-bit
    output_dir: str = os.path.join(MODELS_DIR, "qgalore")

    # GaLore-specific settings
    galore_rank: int = 128
    galore_update_proj_gap: int = 200
    galore_scale: float = 0.25
    proj_quant: bool = True  # Quantize projection matrices to INT4

    # Memory optimization for larger models
    per_device_train_batch_size: int = 2  # May need smaller batch for 30B
    gradient_accumulation_steps: int = 8


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Evaluation settings
    eval_batch_size: int = 8
    max_new_tokens: int = 2048
    temperature: float = 0.0

    # Semantic similarity threshold for matching
    semantic_similarity_threshold: float = 0.85

    # Embedding model for semantic matching
    embedding_model: str = "all-MiniLM-L6-v2"

    # Output
    results_dir: str = os.path.join(DATA_DIR, "evaluation_results")

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)


@dataclass
class StorageConfig:
    """Configuration for cloud storage integration."""

    # Google Drive
    gdrive_folder_id: Optional[str] = None  # Set via environment or CLI
    gdrive_credentials_path: str = "credentials.json"
    sync_checkpoints: bool = True
    checkpoint_sync_interval: int = 5  # Sync every N checkpoints

    # HuggingFace Hub
    hf_username: Optional[str] = None  # Set via environment or CLI
    hf_token: Optional[str] = None  # Set via HF_TOKEN env var
    upload_adapters: bool = True
    upload_merged: bool = False  # Full merged models are large
    create_model_card: bool = True

    # Repository naming
    repo_prefix: str = "kg-extraction"

    def get_repo_id(self, model_name: str, method: str) -> str:
        """Generate HuggingFace repo ID."""
        if not self.hf_username:
            raise ValueError("hf_username must be set for HuggingFace upload")
        # e.g., "username/gemma3-12b-kg-extraction-qlora"
        clean_name = model_name.lower().replace("/", "-").replace("_", "-")
        return f"{self.hf_username}/{clean_name}-{self.repo_prefix}-{method}"


# System prompt for KG extraction (used in training data)
SYSTEM_PROMPT = """You are a Knowledge Graph expert. Extract a semi-structured graph from the text.

1. Identify Entities (Nodes): Include a 'description' summarizing who/what the entity is.
2. Identify Relationships (Edges): Include a 'description' explaining the context of the link.
3. Use consistent IDs.

Output valid JSON matching this schema:
{{
  "nodes": [
    {{"id": "Entity Name", "type": "Category", "description": "Brief summary"}}
  ],
  "edges": [
    {{"source": "Source Entity", "target": "Target Entity", "relation": "relationship_type", "description": "Context"}}
  ]
}}"""
