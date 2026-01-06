"""
Prompt Comparison Benchmark for Knowledge Graph Extraction.

This script benchmarks multiple models with different prompts (baseline vs advanced)
and uses LLM as a Judge to evaluate extraction quality.

Features:
- Runs local models one at a time with ollama stop between runs to prevent OOM
- Caches all input/output pairs for reproducibility
- Uses GPT-5.2 and Gemini-2.5 as gold standards
- LLM as a Judge evaluation for comparing outputs
"""

import os
import sys
import json
import time
import pickle
import argparse
import logging
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

import re
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tabulate import tabulate
from json_repair import repair_json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.kg_extraction_baseline import SYSTEM_PROMPT_BASELINE
from prompts.kg_extraction_advanced import SYSTEM_PROMPT_ADVANCED
from prompts.agent_skb_prompts import (
    KG_EXTRACTION_PROMPT_MINIMAL,
    KG_EXTRACTION_PROMPT_COMPACT,
    KG_EXTRACTION_PROMPT,
)

# All available prompts for testing
ALL_PROMPTS = {
    "minimal": KG_EXTRACTION_PROMPT_MINIMAL,
    "compact": KG_EXTRACTION_PROMPT_COMPACT,
    "standard": KG_EXTRACTION_PROMPT,
    "baseline": SYSTEM_PROMPT_BASELINE,
    "advanced": SYSTEM_PROMPT_ADVANCED,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "prompt_comparison_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# --- Pydantic Models for Structured Output ---
class NodeBaseline(BaseModel):
    id: str = Field(description="Unique identifier")
    label: str = Field(description="Category label")
    description: str = Field(description="Summary")


class RelationshipBaseline(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    type: str = Field(description="Relationship type")
    description: str = Field(description="Context")


class KGBaseline(BaseModel):
    nodes: List[NodeBaseline] = Field(default_factory=list)
    relationships: List[RelationshipBaseline] = Field(default_factory=list)


class NodePropertiesAdvanced(BaseModel):
    description: str = Field(description="10-15 word summary")


class NodeAdvanced(BaseModel):
    id: str = Field(description="Canonical unique identifier")
    label: str = Field(description="PascalCase label")
    properties: NodePropertiesAdvanced = Field(default_factory=NodePropertiesAdvanced)


class RelationshipPropertiesAdvanced(BaseModel):
    description: str = Field(description="Sentence describing the link")


class RelationshipAdvanced(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    type: str = Field(description="SCREAMING_SNAKE_CASE type")
    properties: RelationshipPropertiesAdvanced = Field(default_factory=RelationshipPropertiesAdvanced)


class KGAdvanced(BaseModel):
    reasoning: str = Field(default="", alias="_reasoning", description="Extraction logic")
    nodes: List[NodeAdvanced] = Field(default_factory=list)
    relationships: List[RelationshipAdvanced] = Field(default_factory=list)


# --- Pydantic Models for Agent SKB Schema (minimal/compact/standard) ---
class NodePropertiesSKB(BaseModel):
    description: str = Field(default="", description="Detailed summary")


class NodeSKB(BaseModel):
    id: str = Field(description="Unique identifier")
    labels: List[str] = Field(default_factory=list, description="Type labels")
    properties: NodePropertiesSKB = Field(default_factory=NodePropertiesSKB)


class RelationshipPropertiesSKB(BaseModel):
    description: str = Field(default="", description="Context explanation")


class RelationshipSKB(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    type: str = Field(description="SCREAMING_SNAKE_CASE type")
    properties: RelationshipPropertiesSKB = Field(default_factory=RelationshipPropertiesSKB)


class KGSKB(BaseModel):
    nodes: List[NodeSKB] = Field(default_factory=list)
    relationships: List[RelationshipSKB] = Field(default_factory=list)


# --- LLM Judge Models ---
class JudgeScores(BaseModel):
    """Structured output for LLM judge evaluation."""
    entity_completeness: float = Field(description="Score 0-1: Are all important entities from the text captured?")
    entity_accuracy: float = Field(description="Score 0-1: Are extracted entities correct and well-named?")
    relationship_completeness: float = Field(description="Score 0-1: Are important relationships captured?")
    relationship_accuracy: float = Field(description="Score 0-1: Are relationships factually correct?")
    schema_compliance: float = Field(description="Score 0-1: Does output follow the required schema?")
    overall_quality: float = Field(description="Score 0-1: Overall extraction quality")
    strengths: List[str] = Field(description="Key strengths of this extraction")
    weaknesses: List[str] = Field(description="Key weaknesses or issues")
    recommendation: str = Field(description="Which prompt/model is better for this text and why")


# --- Data Classes ---
@dataclass
class ExtractionResult:
    model_name: str
    prompt_type: str  # 'baseline' or 'advanced'
    chunk_id: str
    input_text: str
    output_raw: str
    nodes: List[Dict]
    edges: List[Dict]
    parse_success: bool
    error_message: Optional[str] = None
    latency_seconds: float = 0.0
    timestamp: str = ""


@dataclass
class JudgeResult:
    model_name: str
    prompt_type: str
    chunk_id: str
    scores: Dict[str, Any]
    timestamp: str = ""


# --- Ollama Management ---
def stop_ollama_model(model_name: str):
    """Stop a specific ollama model to free VRAM."""
    try:
        # Use ollama stop command via SSH/direct if available
        result = subprocess.run(
            ["ssh", "host.docker.internal", f"ollama stop {model_name}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info(f"Stopped ollama model: {model_name}")
        else:
            # Try alternative method - just let it timeout naturally
            logger.info(f"Could not stop {model_name} via SSH, may need manual stop")
    except Exception as e:
        logger.warning(f"Failed to stop ollama model {model_name}: {e}")


def stop_all_ollama_models():
    """Stop all running ollama models."""
    try:
        # Get list of running models
        result = subprocess.run(
            ["curl", "-s", "http://host.docker.internal:11434/api/ps"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout) if result.stdout else {}
            models = data.get("models", [])
            for model in models:
                model_name = model.get("name", "")
                if model_name:
                    stop_ollama_model(model_name)
        logger.info("Attempted to stop all ollama models")
    except Exception as e:
        logger.warning(f"Failed to stop ollama models: {e}")


# --- Cache Management ---
def get_cache_path(model_name: str, prompt_type: str) -> str:
    """Get cache file path for a model/prompt combination."""
    safe_name = model_name.replace(" ", "_").replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_{prompt_type}.pkl")


def load_model_cache(model_name: str, prompt_type: str) -> Dict[str, ExtractionResult]:
    """Load cached results for a model/prompt combination."""
    cache_path = get_cache_path(model_name, prompt_type)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_model_cache(model_name: str, prompt_type: str, results: Dict[str, ExtractionResult]):
    """Save results to cache."""
    cache_path = get_cache_path(model_name, prompt_type)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)


def load_judge_cache() -> Dict[str, JudgeResult]:
    """Load judge evaluation cache."""
    cache_path = os.path.join(CACHE_DIR, "judge_results.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_judge_cache(results: Dict[str, JudgeResult]):
    """Save judge results to cache."""
    cache_path = os.path.join(CACHE_DIR, "judge_results.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)


def save_all_io_pairs(all_results: Dict[str, Dict[str, Dict[str, ExtractionResult]]]):
    """Save all input/output pairs to JSON for inspection."""
    output_path = os.path.join(CACHE_DIR, "all_io_pairs.json")

    serializable = {}
    for model_name, prompt_results in all_results.items():
        serializable[model_name] = {}
        for prompt_type, chunk_results in prompt_results.items():
            serializable[model_name][prompt_type] = {}
            for chunk_id, result in chunk_results.items():
                serializable[model_name][prompt_type][chunk_id] = {
                    "input_text": result.input_text,
                    "output_raw": result.output_raw,
                    "nodes": result.nodes,
                    "edges": result.edges,
                    "parse_success": result.parse_success,
                    "error_message": result.error_message,
                    "latency_seconds": result.latency_seconds,
                }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Saved all I/O pairs to {output_path}")


# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    provider: str  # 'ollama', 'openai', 'google'
    model_id: str
    is_gold_standard: bool = False
    temperature: float = 0.0
    base_url: Optional[str] = None


# Models to benchmark
LOCAL_MODELS = [
    ModelConfig(
        name="Ministral-3 14B",
        provider="ollama",
        model_id="ministral-3:14b",
        base_url="http://host.docker.internal:11434",
    ),
    ModelConfig(
        name="Nemotron-3 Nano 30B",
        provider="ollama",
        model_id="nemotron-3-nano:30b",
        base_url="http://host.docker.internal:11434",
    ),
    ModelConfig(
        name="Qwen3 30B A3B",
        provider="ollama",
        model_id="qwen3:30b-a3b",
        base_url="http://host.docker.internal:11434",
    ),
    ModelConfig(
        name="Gemma3 27B IT QAT",
        provider="ollama",
        model_id="gemma3:27b-it-qat",
        base_url="http://host.docker.internal:11434",
    ),
]

API_MODELS = [
    ModelConfig(
        name="GPT-5.2",
        provider="openai",
        model_id="gpt-5.2",
        is_gold_standard=True,
    ),
    ModelConfig(
        name="Gemini-2.5",
        provider="google",
        model_id="gemini-2.5-flash",
        is_gold_standard=True,
    ),
]


class PromptComparisonBenchmark:
    """Benchmarks models with different prompts and evaluates with LLM judge."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        # Results: model_name -> prompt_type -> chunk_id -> ExtractionResult
        self.results: Dict[str, Dict[str, Dict[str, ExtractionResult]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.judge_results: Dict[str, JudgeResult] = {}

        if use_cache:
            self.judge_results = load_judge_cache()

    def _create_llm(self, config: ModelConfig):
        """Create LLM instance for a model config."""
        if config.provider == "openai":
            return ChatOpenAI(model=config.model_id, temperature=config.temperature)
        elif config.provider == "google":
            return ChatGoogleGenerativeAI(
                model=config.model_id, temperature=config.temperature
            )
        elif config.provider == "ollama":
            return ChatOllama(
                model=config.model_id,
                temperature=config.temperature,
                base_url=config.base_url,
                num_ctx=16384,
                num_predict=4096,
                format="json",
            )
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    def _parse_baseline_json(self, raw_json: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse baseline JSON output, handling various formats."""
        # Try to repair and parse JSON
        repaired = repair_json(raw_json, return_objects=True)

        if isinstance(repaired, dict):
            data = repaired
        else:
            # Try to find JSON object in string
            json_match = re.search(r'\{[\s\S]*\}', str(repaired))
            if json_match:
                data = json.loads(json_match.group())
            else:
                return [], []

        nodes = []
        for n in data.get("nodes", []):
            if isinstance(n, dict):
                nodes.append({
                    "id": n.get("id", ""),
                    "label": n.get("label", ""),
                    "description": n.get("description", ""),
                })

        edges = []
        for r in data.get("relationships", []):
            if isinstance(r, dict):
                edges.append({
                    "source": r.get("source", ""),
                    "target": r.get("target", ""),
                    "type": r.get("type", ""),
                    "description": r.get("description", ""),
                })

        return nodes, edges

    def _parse_advanced_json(self, raw_json: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse advanced JSON output, handling various formats."""
        # Try to repair and parse JSON
        repaired = repair_json(raw_json, return_objects=True)

        if isinstance(repaired, dict):
            data = repaired
        else:
            # Try to find JSON object in string
            json_match = re.search(r'\{[\s\S]*\}', str(repaired))
            if json_match:
                data = json.loads(json_match.group())
            else:
                return [], []

        nodes = []
        for n in data.get("nodes", []):
            if isinstance(n, dict):
                props = n.get("properties", {})
                if isinstance(props, dict):
                    desc = props.get("description", "")
                else:
                    desc = str(props)
                nodes.append({
                    "id": n.get("id", ""),
                    "label": n.get("label", ""),
                    "description": desc,
                })

        edges = []
        for r in data.get("relationships", []):
            if isinstance(r, dict):
                props = r.get("properties", {})
                if isinstance(props, dict):
                    desc = props.get("description", "")
                else:
                    desc = str(props)
                edges.append({
                    "source": r.get("source", ""),
                    "target": r.get("target", ""),
                    "type": r.get("type", ""),
                    "description": desc,
                })

        return nodes, edges

    def _parse_skb_json(self, raw_json: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse SKB-style JSON output (minimal/compact/standard prompts).

        SKB format uses 'labels' (list) and nested properties, not 'label' (string).
        """
        # Try to repair and parse JSON
        repaired = repair_json(raw_json, return_objects=True)

        if isinstance(repaired, dict):
            data = repaired
        else:
            # Try to find JSON object in string
            json_match = re.search(r'\{[\s\S]*\}', str(repaired))
            if json_match:
                data = json.loads(json_match.group())
            else:
                return [], []

        nodes = []
        for n in data.get("nodes", []):
            if isinstance(n, dict):
                # Handle nested properties structure
                props = n.get("properties", {})
                if isinstance(props, dict):
                    desc = props.get("description", "")
                else:
                    desc = str(props)

                # Handle labels as list or single label
                labels = n.get("labels", [])
                if isinstance(labels, list):
                    label = labels[0] if labels else ""
                else:
                    label = str(labels)

                # Also check for 'type' field as alternative to 'labels'
                if not label and "type" in n:
                    label = n.get("type", "")

                nodes.append({
                    "id": n.get("id", ""),
                    "label": label,
                    "description": desc,
                })

        edges = []
        # SKB uses 'edges' key, but may also use 'relationships'
        edge_list = data.get("edges", data.get("relationships", []))
        for r in edge_list:
            if isinstance(r, dict):
                props = r.get("properties", {})
                if isinstance(props, dict):
                    desc = props.get("description", "")
                else:
                    desc = str(props)
                edges.append({
                    "source": r.get("source", ""),
                    "target": r.get("target", ""),
                    "type": r.get("type", ""),
                    "description": desc,
                })

        return nodes, edges

    def _extract_skb(self, llm, text: str, prompt_template: str) -> Tuple[str, List[Dict], List[Dict], bool, Optional[str]]:
        """Extract KG using SKB-style prompts (minimal/compact/standard)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "{text}"),
        ])

        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(KGSKB)
            chain = prompt | structured_llm
            result = chain.invoke({"text": text})

            nodes = [{
                "id": n.id,
                "label": n.labels[0] if n.labels else "",
                "description": n.properties.description
            } for n in result.nodes]
            edges = [{
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "description": r.properties.description
            } for r in result.relationships]

            return json.dumps(result.model_dump(), indent=2), nodes, edges, True, None
        except Exception as e:
            logger.debug(f"Structured output failed, trying raw JSON: {e}")

        # Fallback: raw text extraction with json_repair
        try:
            chain = prompt | llm
            response = chain.invoke({"text": text})

            raw_output = response.content if hasattr(response, 'content') else str(response)
            nodes, edges = self._parse_skb_json(raw_output)

            if nodes or edges:
                return raw_output, nodes, edges, True, None
            else:
                return raw_output, [], [], False, "No nodes or edges extracted"
        except Exception as e:
            return str(e), [], [], False, str(e)

    def _extract_baseline(self, llm, text: str) -> Tuple[str, List[Dict], List[Dict], bool, Optional[str]]:
        """Extract KG using baseline prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_BASELINE),
            ("human", "{text}"),
        ])

        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(KGBaseline)
            chain = prompt | structured_llm
            result = chain.invoke({"text": text})

            nodes = [{"id": n.id, "label": n.label, "description": n.description} for n in result.nodes]
            edges = [{"source": r.source, "target": r.target, "type": r.type, "description": r.description}
                    for r in result.relationships]

            return json.dumps(asdict(result), indent=2), nodes, edges, True, None
        except Exception as e:
            logger.debug(f"Structured output failed, trying raw JSON: {e}")

        # Fallback: raw text extraction with json_repair
        try:
            chain = prompt | llm
            response = chain.invoke({"text": text})

            raw_output = response.content if hasattr(response, 'content') else str(response)
            nodes, edges = self._parse_baseline_json(raw_output)

            if nodes or edges:
                return raw_output, nodes, edges, True, None
            else:
                return raw_output, [], [], False, "No nodes or edges extracted"
        except Exception as e:
            return str(e), [], [], False, str(e)

    def _extract_advanced(self, llm, text: str) -> Tuple[str, List[Dict], List[Dict], bool, Optional[str]]:
        """Extract KG using advanced prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_ADVANCED),
            ("human", "{text}"),
        ])

        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(KGAdvanced)
            chain = prompt | structured_llm
            result = chain.invoke({"text": text})

            nodes = [{"id": n.id, "label": n.label, "description": n.properties.description}
                    for n in result.nodes]
            edges = [{"source": r.source, "target": r.target, "type": r.type,
                     "description": r.properties.description}
                    for r in result.relationships]

            return json.dumps(asdict(result), indent=2), nodes, edges, True, None
        except Exception as e:
            logger.debug(f"Structured output failed, trying raw JSON: {e}")

        # Fallback: raw text extraction with json_repair
        try:
            chain = prompt | llm
            response = chain.invoke({"text": text})

            raw_output = response.content if hasattr(response, 'content') else str(response)
            nodes, edges = self._parse_advanced_json(raw_output)

            if nodes or edges:
                return raw_output, nodes, edges, True, None
            else:
                return raw_output, [], [], False, "No nodes or edges extracted"
        except Exception as e:
            return str(e), [], [], False, str(e)

    def run_model(self, config: ModelConfig, chunks: List[Dict], prompt_type: str) -> Dict[str, ExtractionResult]:
        """Run extraction for a single model with a specific prompt."""
        # Load cache
        cached = load_model_cache(config.name, prompt_type) if self.use_cache else {}

        # Filter chunks that need processing
        chunks_to_process = [c for c in chunks if c["id"] not in cached]

        if not chunks_to_process:
            logger.info(f"  All {len(chunks)} chunks cached for {config.name} ({prompt_type})")
            return cached

        logger.info(f"  Processing {len(chunks_to_process)}/{len(chunks)} chunks for {config.name} ({prompt_type})")

        # Create LLM
        llm = self._create_llm(config)

        results = dict(cached)

        for chunk in chunks_to_process:
            chunk_id = chunk["id"]
            text = chunk["text"]

            start_time = time.time()

            # Route to appropriate extraction method based on prompt type
            if prompt_type == "baseline":
                output_raw, nodes, edges, success, error = self._extract_baseline(llm, text)
            elif prompt_type == "advanced":
                output_raw, nodes, edges, success, error = self._extract_advanced(llm, text)
            elif prompt_type in ("minimal", "compact", "standard"):
                output_raw, nodes, edges, success, error = self._extract_skb(
                    llm, text, ALL_PROMPTS[prompt_type]
                )
            else:
                output_raw, nodes, edges, success, error = "", [], [], False, f"Unknown prompt type: {prompt_type}"

            latency = time.time() - start_time

            result = ExtractionResult(
                model_name=config.name,
                prompt_type=prompt_type,
                chunk_id=chunk_id,
                input_text=text,
                output_raw=output_raw,
                nodes=nodes,
                edges=edges,
                parse_success=success,
                error_message=error,
                latency_seconds=latency,
                timestamp=datetime.utcnow().isoformat(),
            )

            results[chunk_id] = result

            status = "OK" if success else f"FAIL: {error[:50]}"
            logger.info(f"    {chunk_id}: {len(nodes)} nodes, {len(edges)} edges ({latency:.2f}s) [{status}]")

        # Save cache
        if self.use_cache:
            save_model_cache(config.name, prompt_type, results)

        return results

    def run_benchmark(
        self,
        subject: str = "economics",
        num_docs: int = 5,
        start_doc: int = 0,
        run_local: bool = True,
        run_api: bool = True,
        local_models: List[str] = None,
        prompts: List[str] = None,
    ):
        """Run the full benchmark."""
        # Load dataset
        logger.info(f"Loading {subject} dataset...")

        if subject == "economics":
            dataset = load_dataset(
                "cais/wmdp-mmlu-auxiliary-corpora", "economics-corpus", split="train"
            )
        elif subject == "law":
            dataset = load_dataset(
                "cais/wmdp-mmlu-auxiliary-corpora", "law-corpus", split="train"
            )
        elif subject == "physics":
            dataset = load_dataset(
                "cais/wmdp-mmlu-auxiliary-corpora", "physics-corpus", split="train"
            )
        else:
            raise ValueError(f"Unknown subject: {subject}")

        # Select documents
        end_doc = min(start_doc + num_docs, len(dataset))
        dataset = dataset.select(range(start_doc, end_doc))

        # Prepare chunks
        all_chunks = []
        for doc_idx, entry in enumerate(dataset):
            actual_doc_idx = start_doc + doc_idx
            text = entry["text"]
            if len(text.strip()) < 50:
                continue
            chunks = self.text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks[:3]):  # Limit chunks per doc
                all_chunks.append({
                    "id": f"doc{actual_doc_idx}_chunk{chunk_idx}",
                    "text": chunk,
                    "doc_idx": actual_doc_idx,
                })

        logger.info(f"Processing {len(all_chunks)} chunks from {num_docs} documents")

        # Filter local models if specified
        models_to_run = LOCAL_MODELS
        if local_models:
            models_to_run = [m for m in LOCAL_MODELS if m.model_id in local_models]

        # Determine which prompts to run
        prompt_types = prompts if prompts else ["baseline", "advanced"]
        logger.info(f"Testing prompts: {prompt_types}")

        # Run API models first (they're faster and don't need VRAM management)
        if run_api:
            logger.info("\n" + "=" * 60)
            logger.info("RUNNING API MODELS")
            logger.info("=" * 60)

            for config in API_MODELS:
                logger.info(f"\n[{config.name}]")

                for prompt_type in prompt_types:
                    results = self.run_model(config, all_chunks, prompt_type)
                    self.results[config.name][prompt_type] = results

        # Run local models one at a time
        if run_local:
            logger.info("\n" + "=" * 60)
            logger.info("RUNNING LOCAL MODELS (one at a time)")
            logger.info("=" * 60)

            for config in models_to_run:
                logger.info(f"\n[{config.name}]")

                # Stop any running models first
                stop_all_ollama_models()
                time.sleep(2)  # Wait for VRAM to clear

                for prompt_type in prompt_types:
                    try:
                        results = self.run_model(config, all_chunks, prompt_type)
                        self.results[config.name][prompt_type] = results
                    except Exception as e:
                        logger.error(f"Failed to run {config.name} ({prompt_type}): {e}")

                # Stop model after use
                stop_ollama_model(config.model_id)
                time.sleep(2)

        # Save all I/O pairs
        save_all_io_pairs(dict(self.results))

        return self.results

    def run_llm_judge(self, judge_model: str = "gpt-5.2", max_chunks: int = 5):
        """Run LLM as a Judge evaluation."""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING LLM AS A JUDGE EVALUATION")
        logger.info("=" * 60)

        judge_llm = ChatOpenAI(model=judge_model, temperature=0.0)

        judge_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator of knowledge graph extractions.
Given the original text and two extractions (one using a baseline prompt, one using an advanced prompt),
evaluate and compare the quality of both extractions.

Score each dimension from 0 to 1:
- entity_completeness: Are all important entities from the text captured?
- entity_accuracy: Are extracted entities correct and well-named?
- relationship_completeness: Are important relationships captured?
- relationship_accuracy: Are relationships factually correct?
- schema_compliance: Does output follow the required schema?
- overall_quality: Overall extraction quality

Be strict but fair. Consider both precision (no spurious entities) and recall (all entities captured)."""),
            ("human", """Original Text:
{text}

Model: {model_name}
Prompt Type: {prompt_type}

Extracted Nodes:
{nodes}

Extracted Relationships:
{edges}

Evaluate this extraction's quality.""")
        ])

        structured_judge = judge_llm.with_structured_output(JudgeScores)
        chain = judge_prompt | structured_judge

        # Get common chunks across all models
        all_chunk_ids = set()
        for model_name, prompt_results in self.results.items():
            for prompt_type, chunk_results in prompt_results.items():
                for chunk_id in chunk_results.keys():
                    all_chunk_ids.add(chunk_id)

        chunks_to_evaluate = list(all_chunk_ids)[:max_chunks]
        logger.info(f"Evaluating {len(chunks_to_evaluate)} chunks")

        for model_name, prompt_results in self.results.items():
            for prompt_type, chunk_results in prompt_results.items():
                for chunk_id in chunks_to_evaluate:
                    if chunk_id not in chunk_results:
                        continue

                    cache_key = f"{model_name}_{prompt_type}_{chunk_id}"

                    if self.use_cache and cache_key in self.judge_results:
                        logger.info(f"  [cached] {model_name} ({prompt_type}) - {chunk_id}")
                        continue

                    result = chunk_results[chunk_id]

                    if not result.parse_success:
                        logger.info(f"  [skip] {model_name} ({prompt_type}) - {chunk_id} (parse failed)")
                        continue

                    logger.info(f"  Evaluating {model_name} ({prompt_type}) - {chunk_id}")

                    try:
                        nodes_str = json.dumps(result.nodes, indent=2)
                        edges_str = json.dumps(result.edges, indent=2)

                        scores = chain.invoke({
                            "text": result.input_text,
                            "model_name": model_name,
                            "prompt_type": prompt_type,
                            "nodes": nodes_str,
                            "edges": edges_str,
                        })

                        judge_result = JudgeResult(
                            model_name=model_name,
                            prompt_type=prompt_type,
                            chunk_id=chunk_id,
                            scores=scores.model_dump(),  # Pydantic v2 method
                            timestamp=datetime.utcnow().isoformat(),
                        )

                        self.judge_results[cache_key] = judge_result

                        if self.use_cache:
                            save_judge_cache(self.judge_results)

                        logger.info(f"    Overall: {scores.overall_quality:.2f}")

                    except Exception as e:
                        logger.error(f"    Judge failed: {e}")

        return self.judge_results

    def generate_report(self) -> str:
        """Generate comprehensive comparison report."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("PROMPT COMPARISON BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")

        # Extraction Summary
        lines.append("\n" + "-" * 80)
        lines.append("EXTRACTION SUMMARY")
        lines.append("-" * 80)

        summary_table = []
        for model_name, prompt_results in self.results.items():
            for prompt_type, chunk_results in prompt_results.items():
                total = len(chunk_results)
                success = sum(1 for r in chunk_results.values() if r.parse_success)
                total_nodes = sum(len(r.nodes) for r in chunk_results.values() if r.parse_success)
                total_edges = sum(len(r.edges) for r in chunk_results.values() if r.parse_success)
                avg_latency = sum(r.latency_seconds for r in chunk_results.values()) / max(total, 1)

                summary_table.append([
                    model_name,
                    prompt_type,
                    f"{success}/{total}",
                    f"{total_nodes/max(success,1):.1f}",
                    f"{total_edges/max(success,1):.1f}",
                    f"{avg_latency:.2f}s",
                ])

        headers = ["Model", "Prompt", "Success", "Avg Nodes", "Avg Edges", "Avg Latency"]
        lines.append(tabulate(summary_table, headers=headers, tablefmt="grid"))

        # Judge Scores Summary
        if self.judge_results:
            lines.append("\n" + "-" * 80)
            lines.append("LLM JUDGE SCORES (GPT-5.2)")
            lines.append("-" * 80)

            # Aggregate scores by model/prompt
            aggregated = defaultdict(lambda: defaultdict(list))
            for cache_key, result in self.judge_results.items():
                key = (result.model_name, result.prompt_type)
                for score_name, score_value in result.scores.items():
                    if isinstance(score_value, (int, float)):
                        aggregated[key][score_name].append(score_value)

            judge_table = []
            for (model_name, prompt_type), scores in aggregated.items():
                avg_entity_comp = sum(scores.get("entity_completeness", [0])) / max(len(scores.get("entity_completeness", [1])), 1)
                avg_entity_acc = sum(scores.get("entity_accuracy", [0])) / max(len(scores.get("entity_accuracy", [1])), 1)
                avg_rel_comp = sum(scores.get("relationship_completeness", [0])) / max(len(scores.get("relationship_completeness", [1])), 1)
                avg_rel_acc = sum(scores.get("relationship_accuracy", [0])) / max(len(scores.get("relationship_accuracy", [1])), 1)
                avg_overall = sum(scores.get("overall_quality", [0])) / max(len(scores.get("overall_quality", [1])), 1)

                judge_table.append([
                    model_name,
                    prompt_type,
                    f"{avg_entity_comp:.2f}",
                    f"{avg_entity_acc:.2f}",
                    f"{avg_rel_comp:.2f}",
                    f"{avg_rel_acc:.2f}",
                    f"{avg_overall:.2f}",
                ])

            headers = ["Model", "Prompt", "Ent Comp", "Ent Acc", "Rel Comp", "Rel Acc", "Overall"]
            lines.append(tabulate(judge_table, headers=headers, tablefmt="grid"))

            # Prompt Comparison by Model
            lines.append("\n" + "-" * 80)
            lines.append("BASELINE vs ADVANCED PROMPT COMPARISON")
            lines.append("-" * 80)

            model_names = set(r.model_name for r in self.judge_results.values())
            for model_name in model_names:
                baseline_scores = [r.scores for r in self.judge_results.values()
                                  if r.model_name == model_name and r.prompt_type == "baseline"]
                advanced_scores = [r.scores for r in self.judge_results.values()
                                  if r.model_name == model_name and r.prompt_type == "advanced"]

                if baseline_scores and advanced_scores:
                    baseline_avg = sum(s.get("overall_quality", 0) for s in baseline_scores) / len(baseline_scores)
                    advanced_avg = sum(s.get("overall_quality", 0) for s in advanced_scores) / len(advanced_scores)

                    diff = advanced_avg - baseline_avg
                    winner = "ADVANCED" if diff > 0.05 else "BASELINE" if diff < -0.05 else "TIE"

                    lines.append(f"\n{model_name}:")
                    lines.append(f"  Baseline Overall: {baseline_avg:.2f}")
                    lines.append(f"  Advanced Overall: {advanced_avg:.2f}")
                    lines.append(f"  Difference: {diff:+.2f}")
                    lines.append(f"  Winner: {winner}")

        # Sample Outputs
        lines.append("\n" + "-" * 80)
        lines.append("SAMPLE EXTRACTIONS")
        lines.append("-" * 80)

        # Get first chunk
        first_chunk_id = None
        for model_name, prompt_results in self.results.items():
            for prompt_type, chunk_results in prompt_results.items():
                if chunk_results:
                    first_chunk_id = list(chunk_results.keys())[0]
                    break
            if first_chunk_id:
                break

        if first_chunk_id:
            lines.append(f"\n[Chunk: {first_chunk_id}]")

            # Get input text
            for model_name, prompt_results in self.results.items():
                for prompt_type, chunk_results in prompt_results.items():
                    if first_chunk_id in chunk_results:
                        lines.append(f"\nInput Text (truncated):\n{chunk_results[first_chunk_id].input_text[:500]}...")
                        break
                break

            for model_name, prompt_results in self.results.items():
                for prompt_type, chunk_results in prompt_results.items():
                    if first_chunk_id in chunk_results:
                        result = chunk_results[first_chunk_id]
                        lines.append(f"\n{model_name} ({prompt_type}):")
                        if result.parse_success:
                            lines.append(f"  Nodes: {[n['id'] for n in result.nodes[:5]]}")
                            lines.append(f"  Edges: {[(e['source'], e['type'], e['target']) for e in result.edges[:5]]}")
                        else:
                            lines.append(f"  FAILED: {result.error_message[:100]}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Prompt Comparison Benchmark")
    parser.add_argument("--subject", type=str, default="economics",
                       choices=["economics", "law", "physics"])
    parser.add_argument("--num_docs", type=int, default=5)
    parser.add_argument("--start_doc", type=int, default=0)
    parser.add_argument("--skip_local", action="store_true", help="Skip local models")
    parser.add_argument("--skip_api", action="store_true", help="Skip API models")
    parser.add_argument("--local_models", type=str, nargs="+",
                       help="Specific local model IDs to run (e.g., ministral-3:14b)")
    parser.add_argument("--prompts", type=str, nargs="+",
                       choices=["minimal", "compact", "standard", "baseline", "advanced"],
                       default=None,
                       help="Which prompts to test (default: baseline, advanced). Options: minimal, compact, standard, baseline, advanced")
    parser.add_argument("--skip_judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--judge_chunks", type=int, default=5, help="Number of chunks for judge")
    parser.add_argument("--no_cache", action="store_true", help="Don't use cache")
    parser.add_argument("--output", type=str, default=None, help="Output file for report")

    args = parser.parse_args()

    benchmark = PromptComparisonBenchmark(use_cache=not args.no_cache)

    # Run benchmark
    benchmark.run_benchmark(
        subject=args.subject,
        num_docs=args.num_docs,
        start_doc=args.start_doc,
        run_local=not args.skip_local,
        run_api=not args.skip_api,
        local_models=args.local_models,
        prompts=args.prompts,
    )

    # Run LLM judge
    if not args.skip_judge:
        benchmark.run_llm_judge(max_chunks=args.judge_chunks)

    # Generate report
    report = benchmark.generate_report()
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
