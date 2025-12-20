"""
Evaluation pipeline for comparing fine-tuned models.

Computes metrics like parse success, entity overlap, edge overlap,
and semantic F1 against teacher labels.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from ..config import EvaluationConfig, TrainingConfig, SYSTEM_PROMPT
from ..data.teacher_labeler import KnowledgeGraph, normalize_entity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for a single model evaluation."""
    model_name: str
    total_samples: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    parse_success_rate: float = 0.0

    # Entity metrics
    total_predicted_nodes: int = 0
    total_gold_nodes: int = 0
    node_overlap_count: int = 0
    node_overlap_ratio: float = 0.0
    avg_nodes_per_sample: float = 0.0

    # Edge metrics
    total_predicted_edges: int = 0
    total_gold_edges: int = 0
    edge_overlap_count: int = 0
    edge_overlap_ratio: float = 0.0
    avg_edges_per_sample: float = 0.0

    # Semantic metrics
    semantic_precision: float = 0.0
    semantic_recall: float = 0.0
    semantic_f1: float = 0.0

    # Performance
    avg_latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""
    sample_id: str
    parse_success: bool
    predicted_kg: Optional[Dict[str, Any]] = None
    gold_kg: Optional[Dict[str, Any]] = None
    node_overlap: float = 0.0
    edge_overlap: float = 0.0
    semantic_f1: float = 0.0
    latency_seconds: float = 0.0
    error_message: Optional[str] = None


class FinetuningEvaluator:
    """
    Evaluates fine-tuned models against teacher labels.

    Supports:
    - HuggingFace models (full or LoRA)
    - Ollama models (for baseline comparison)
    - Multiple metrics including semantic similarity
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._embedder = None

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load embedding model for semantic similarity."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def _load_hf_model(
        self,
        model_path: str,
        base_model_id: Optional[str] = None,
        is_adapter: bool = False,
    ):
        """Load a HuggingFace model (full or adapter)."""
        if is_adapter and base_model_id:
            # Load base model and apply adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load full model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id or model_path,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _generate_prediction(
        self,
        model,
        tokenizer,
        input_text: str,
    ) -> tuple[Optional[Dict], float, Optional[str]]:
        """
        Generate KG prediction from model.

        Returns:
            tuple: (knowledge_graph_dict, latency_seconds, error_message)
        """
        # Format prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text: {input_text}"},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.time() - start_time

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse JSON
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            kg_dict = json.loads(response)

            # Validate structure
            KnowledgeGraph.model_validate(kg_dict)

            # Normalize entities
            kg_dict["nodes"] = [
                {**n, "id": normalize_entity(n["id"])}
                for n in kg_dict["nodes"]
            ]
            kg_dict["edges"] = [
                {
                    **e,
                    "source": normalize_entity(e["source"]),
                    "target": normalize_entity(e["target"]),
                }
                for e in kg_dict["edges"]
            ]

            return kg_dict, latency, None

        except Exception as e:
            return None, latency, str(e)

    def _compute_exact_overlap(
        self,
        pred_items: List[str],
        gold_items: List[str],
    ) -> tuple[int, float]:
        """Compute exact string overlap between predicted and gold items."""
        pred_set = set(item.lower().strip() for item in pred_items)
        gold_set = set(item.lower().strip() for item in gold_items)

        overlap_count = len(pred_set & gold_set)
        overlap_ratio = overlap_count / len(gold_set) if gold_set else 0.0

        return overlap_count, overlap_ratio

    def _compute_semantic_f1(
        self,
        pred_items: List[str],
        gold_items: List[str],
        threshold: float = None,
    ) -> Dict[str, float]:
        """
        Compute F1 using semantic similarity matching.

        A prediction matches a gold item if cosine similarity > threshold.
        """
        if threshold is None:
            threshold = self.config.semantic_similarity_threshold

        if not pred_items or not gold_items:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Get embeddings
        pred_embs = self.embedder.encode(pred_items, convert_to_tensor=True)
        gold_embs = self.embedder.encode(gold_items, convert_to_tensor=True)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            pred_embs.unsqueeze(1),
            gold_embs.unsqueeze(0),
            dim=2,
        )

        # Precision: How many predictions match gold
        pred_max_sims = similarity_matrix.max(dim=1).values
        matched_preds = (pred_max_sims >= threshold).sum().item()
        precision = matched_preds / len(pred_items)

        # Recall: How many gold items were found
        gold_max_sims = similarity_matrix.max(dim=0).values
        matched_gold = (gold_max_sims >= threshold).sum().item()
        recall = matched_gold / len(gold_items)

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _evaluate_sample(
        self,
        model,
        tokenizer,
        sample: Dict[str, Any],
    ) -> SampleResult:
        """Evaluate a single sample."""
        gold_kg = sample["knowledge_graph"]
        input_text = sample["input_text"]
        sample_id = sample["id"]

        # Generate prediction
        pred_kg, latency, error = self._generate_prediction(model, tokenizer, input_text)

        if error:
            return SampleResult(
                sample_id=sample_id,
                parse_success=False,
                gold_kg=gold_kg,
                latency_seconds=latency,
                error_message=error,
            )

        # Extract node IDs
        pred_node_ids = [n["id"] for n in pred_kg["nodes"]]
        gold_node_ids = [n["id"] for n in gold_kg["nodes"]]

        # Extract edge tuples
        pred_edges = [
            f"{e['source']}|{e['relation']}|{e['target']}"
            for e in pred_kg["edges"]
        ]
        gold_edges = [
            f"{e['source']}|{e['relation']}|{e['target']}"
            for e in gold_kg["edges"]
        ]

        # Compute metrics
        _, node_overlap = self._compute_exact_overlap(pred_node_ids, gold_node_ids)
        _, edge_overlap = self._compute_exact_overlap(pred_edges, gold_edges)
        semantic_metrics = self._compute_semantic_f1(pred_node_ids, gold_node_ids)

        return SampleResult(
            sample_id=sample_id,
            parse_success=True,
            predicted_kg=pred_kg,
            gold_kg=gold_kg,
            node_overlap=node_overlap,
            edge_overlap=edge_overlap,
            semantic_f1=semantic_metrics["f1"],
            latency_seconds=latency,
        )

    def evaluate(
        self,
        model_path: str,
        eval_data: List[Dict[str, Any]],
        model_name: str = "model",
        base_model_id: Optional[str] = None,
        is_adapter: bool = False,
    ) -> EvaluationMetrics:
        """
        Evaluate a model on the evaluation set.

        Args:
            model_path: Path to model or adapter
            eval_data: List of evaluation samples with 'input_text' and 'knowledge_graph'
            model_name: Name for logging
            base_model_id: Base model ID if loading adapter
            is_adapter: Whether model_path is a LoRA adapter

        Returns:
            EvaluationMetrics with all computed metrics
        """
        logger.info(f"Evaluating {model_name} on {len(eval_data)} samples")

        # Load model
        model, tokenizer = self._load_hf_model(model_path, base_model_id, is_adapter)

        # Evaluate each sample
        results = []
        for sample in tqdm(eval_data, desc=f"Evaluating {model_name}"):
            result = self._evaluate_sample(model, tokenizer, sample)
            results.append(result)

        # Aggregate metrics
        metrics = EvaluationMetrics(model_name=model_name)
        metrics.total_samples = len(results)
        metrics.successful_parses = sum(1 for r in results if r.parse_success)
        metrics.failed_parses = metrics.total_samples - metrics.successful_parses
        metrics.parse_success_rate = metrics.successful_parses / metrics.total_samples

        successful_results = [r for r in results if r.parse_success]

        if successful_results:
            # Node metrics
            metrics.total_predicted_nodes = sum(
                len(r.predicted_kg["nodes"]) for r in successful_results
            )
            metrics.total_gold_nodes = sum(
                len(r.gold_kg["nodes"]) for r in successful_results
            )
            metrics.avg_nodes_per_sample = metrics.total_predicted_nodes / len(successful_results)
            metrics.node_overlap_ratio = sum(r.node_overlap for r in successful_results) / len(successful_results)

            # Edge metrics
            metrics.total_predicted_edges = sum(
                len(r.predicted_kg["edges"]) for r in successful_results
            )
            metrics.total_gold_edges = sum(
                len(r.gold_kg["edges"]) for r in successful_results
            )
            metrics.avg_edges_per_sample = metrics.total_predicted_edges / len(successful_results)
            metrics.edge_overlap_ratio = sum(r.edge_overlap for r in successful_results) / len(successful_results)

            # Semantic metrics
            metrics.semantic_f1 = sum(r.semantic_f1 for r in successful_results) / len(successful_results)

        # Latency (all samples)
        metrics.avg_latency_seconds = sum(r.latency_seconds for r in results) / len(results)

        # Clean up
        del model
        torch.cuda.empty_cache()

        return metrics

    def compare_models(
        self,
        models: List[Dict[str, Any]],
        eval_data: List[Dict[str, Any]],
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple models on the same evaluation set.

        Args:
            models: List of dicts with 'name', 'path', 'base_model_id', 'is_adapter'
            eval_data: Evaluation samples

        Returns:
            Dict mapping model names to metrics
        """
        all_metrics = {}

        for model_info in models:
            metrics = self.evaluate(
                model_path=model_info["path"],
                eval_data=eval_data,
                model_name=model_info["name"],
                base_model_id=model_info.get("base_model_id"),
                is_adapter=model_info.get("is_adapter", False),
            )
            all_metrics[model_info["name"]] = metrics

        return all_metrics

    def generate_report(
        self,
        metrics: Dict[str, EvaluationMetrics],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a comparison report."""
        lines = [
            "=" * 80,
            "FINE-TUNING EVALUATION REPORT",
            "=" * 80,
            "",
        ]

        # Summary table
        headers = ["Model", "Parse%", "Nodes/S", "Edges/S", "Node Ovlp", "Edge Ovlp", "Sem F1", "Latency"]
        rows = []

        for name, m in metrics.items():
            rows.append([
                name,
                f"{m.parse_success_rate * 100:.1f}%",
                f"{m.avg_nodes_per_sample:.1f}",
                f"{m.avg_edges_per_sample:.1f}",
                f"{m.node_overlap_ratio * 100:.1f}%",
                f"{m.edge_overlap_ratio * 100:.1f}%",
                f"{m.semantic_f1 * 100:.1f}%",
                f"{m.avg_latency_seconds:.2f}s",
            ])

        # Format table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)

        lines.append(header_line)
        lines.append(separator)
        for row in rows:
            lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


def evaluate_model(
    model_path: str,
    eval_data_path: str,
    model_name: str = "model",
    base_model_id: Optional[str] = None,
    is_adapter: bool = False,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationMetrics:
    """
    Convenience function to evaluate a single model.

    Args:
        model_path: Path to model or adapter
        eval_data_path: Path to evaluation data JSONL
        model_name: Name for logging
        base_model_id: Base model ID if loading adapter
        is_adapter: Whether model_path is a LoRA adapter
        config: Evaluation configuration

    Returns:
        EvaluationMetrics
    """
    if config is None:
        config = EvaluationConfig()

    evaluator = FinetuningEvaluator(config)

    # Load eval data
    eval_data = []
    with open(eval_data_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            # Need to reconstruct from formatted data
            if "messages" in sample:
                # Extract from messages format
                kg_text = sample["messages"][-1]["content"]  # Assistant response
                eval_data.append({
                    "id": sample["id"],
                    "input_text": sample["input_text"],
                    "knowledge_graph": json.loads(kg_text),
                })
            else:
                eval_data.append(sample)

    return evaluator.evaluate(
        model_path=model_path,
        eval_data=eval_data,
        model_name=model_name,
        base_model_id=base_model_id,
        is_adapter=is_adapter,
    )
