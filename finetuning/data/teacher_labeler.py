"""
Teacher label generation using Gemini 2.5 Flash / GPT-5.2.

Generates gold-standard KG extraction labels for fine-tuning student models.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset

from ..config import DataConfig, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Reuse KG schema from benchmarks (keep consistent)
def normalize_entity(name: str) -> str:
    """Standardize entity naming: 'NEOCLASSICAL_ECONOMICS' -> 'Neoclassical Economics'"""
    return ' '.join(name.replace('_', ' ').split()).title()


class Node(BaseModel):
    """Entity node in the knowledge graph."""
    id: str = Field(description="Unique identifier, e.g., 'Albert Einstein'")
    type: str = Field(description="Category, e.g., 'Person', 'Location'")
    description: str = Field(description="A brief summary of this entity based on the text.")


class Edge(BaseModel):
    """Relationship edge in the knowledge graph."""
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    relation: str = Field(description="Relationship, e.g., 'born_in'")
    description: str = Field(description="Context explaining why this relationship exists.")


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph with nodes and edges."""
    nodes: List[Node]
    edges: List[Edge]


@dataclass
class LabeledSample:
    """A single labeled sample for training."""
    id: str
    domain: str
    doc_idx: int
    chunk_idx: int
    input_text: str
    knowledge_graph: Dict[str, Any]
    teacher_model: str
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabeledSample":
        return cls(**data)


class TeacherLabeler:
    """
    Generates gold-standard KG labels using teacher models.

    Uses Gemini 2.5 Flash as primary teacher, falling back to GPT-5.2
    when rate limits are hit.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self._setup_llms()
        self._setup_splitter()

        # Track rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 60.0 / config.requests_per_minute

    def _setup_llms(self):
        """Initialize teacher LLMs."""
        self.primary_llm = ChatGoogleGenerativeAI(
            model=self.config.primary_teacher,
            temperature=self.config.teacher_temperature,
        )
        self.fallback_llm = ChatOpenAI(
            model=self.config.fallback_teacher,
            temperature=self.config.teacher_temperature,
        )

        # Create chains with structured output
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Text: {text}"),
        ])

        self.primary_chain = self.prompt | self.primary_llm.with_structured_output(KnowledgeGraph)
        self.fallback_chain = self.prompt | self.fallback_llm.with_structured_output(KnowledgeGraph)

        self._using_fallback = False

    def _setup_splitter(self):
        """Initialize text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _extract_with_retry(self, text: str) -> tuple[KnowledgeGraph, str, float]:
        """
        Extract KG with retry logic and teacher fallback.

        Returns:
            tuple: (knowledge_graph, teacher_model_name, latency_seconds)
        """
        chain = self.fallback_chain if self._using_fallback else self.primary_chain
        teacher_name = self.config.fallback_teacher if self._using_fallback else self.config.primary_teacher

        for attempt in range(self.config.retry_attempts):
            try:
                self._rate_limit()
                start_time = time.time()
                result = chain.invoke({"text": text})
                latency = time.time() - start_time
                return result, teacher_name, latency

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limit errors
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if not self._using_fallback:
                        logger.warning(f"Rate limit hit on {self.config.primary_teacher}, switching to {self.config.fallback_teacher}")
                        self._using_fallback = True
                        chain = self.fallback_chain
                        teacher_name = self.config.fallback_teacher
                        continue

                # Exponential backoff for other errors
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)

        raise RuntimeError(f"Failed to extract KG after {self.config.retry_attempts} attempts")

    def _normalize_kg(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """Normalize entity names in the knowledge graph."""
        nodes = [
            {
                "id": normalize_entity(n.id),
                "type": n.type,
                "description": n.description
            }
            for n in kg.nodes
        ]
        edges = [
            {
                "source": normalize_entity(e.source),
                "target": normalize_entity(e.target),
                "relation": e.relation,
                "description": e.description
            }
            for e in kg.edges
        ]
        return {"nodes": nodes, "edges": edges}

    def _load_dataset(self, domain: str) -> Any:
        """Load dataset for a specific domain."""
        corpus_name = f"{domain}-corpus"
        return load_dataset(
            self.config.dataset_name,
            corpus_name,
            split="train"
        )

    def _get_cache_path(self, domain: str, split: str) -> str:
        """Get path for cached labels."""
        return os.path.join(
            self.config.teacher_labels_dir,
            f"{domain}_{split}_labels.jsonl"
        )

    def _load_cached_labels(self, domain: str, split: str) -> Dict[str, LabeledSample]:
        """Load previously generated labels from cache."""
        cache_path = self._get_cache_path(domain, split)
        cached = {}

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    sample = LabeledSample.from_dict(json.loads(line))
                    cached[sample.id] = sample
            logger.info(f"Loaded {len(cached)} cached labels from {cache_path}")

        return cached

    def _save_label(self, sample: LabeledSample, domain: str, split: str):
        """Append a single label to cache file."""
        cache_path = self._get_cache_path(domain, split)
        with open(cache_path, "a") as f:
            f.write(json.dumps(sample.to_dict()) + "\n")

    def generate_labels(
        self,
        domain: str,
        split: str = "train",
        num_samples: Optional[int] = None
    ) -> List[LabeledSample]:
        """
        Generate teacher labels for a domain.

        Args:
            domain: Dataset domain (economics, law, physics)
            split: Data split (train or eval)
            num_samples: Number of documents to process (default from config)

        Returns:
            List of LabeledSample objects
        """
        if num_samples is None:
            num_samples = (
                self.config.train_samples_per_domain
                if split == "train"
                else self.config.eval_samples_per_domain
            )

        logger.info(f"Generating {split} labels for {domain} ({num_samples} documents)")

        # Load dataset
        dataset = self._load_dataset(domain)
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Load cached labels
        cached = self._load_cached_labels(domain, split)
        results = []

        # Process documents
        for doc_idx, entry in enumerate(tqdm(dataset, desc=f"{domain}/{split}")):
            text = entry["text"]
            if len(text.strip()) < 50:
                continue

            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            chunks = chunks[:self.config.max_chunks_per_doc]

            for chunk_idx, chunk in enumerate(chunks):
                sample_id = f"{domain}_doc{doc_idx}_chunk{chunk_idx}"

                # Check cache
                if sample_id in cached:
                    results.append(cached[sample_id])
                    continue

                # Generate label
                try:
                    kg, teacher_name, latency = self._extract_with_retry(chunk)
                    normalized_kg = self._normalize_kg(kg)

                    sample = LabeledSample(
                        id=sample_id,
                        domain=domain,
                        doc_idx=doc_idx,
                        chunk_idx=chunk_idx,
                        input_text=chunk,
                        knowledge_graph=normalized_kg,
                        teacher_model=teacher_name,
                        latency_seconds=latency,
                    )

                    # Cache immediately
                    self._save_label(sample, domain, split)
                    results.append(sample)

                    logger.debug(
                        f"  {sample_id}: {len(normalized_kg['nodes'])} nodes, "
                        f"{len(normalized_kg['edges'])} edges ({latency:.2f}s)"
                    )

                except Exception as e:
                    logger.error(f"Failed to generate label for {sample_id}: {e}")
                    continue

        logger.info(f"Generated {len(results)} labels for {domain}/{split}")
        return results

    def generate_all_labels(self) -> Dict[str, Dict[str, List[LabeledSample]]]:
        """
        Generate labels for all domains and splits.

        Returns:
            Nested dict: {domain: {split: [samples]}}
        """
        all_labels = {}

        for domain in self.config.domains:
            all_labels[domain] = {
                "train": self.generate_labels(domain, "train"),
                "eval": self.generate_labels(domain, "eval"),
            }

        # Summary
        total_train = sum(len(all_labels[d]["train"]) for d in all_labels)
        total_eval = sum(len(all_labels[d]["eval"]) for d in all_labels)
        logger.info(f"Total labels generated: {total_train} train, {total_eval} eval")

        return all_labels

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about generated labels."""
        stats = {"domains": {}}

        for domain in self.config.domains:
            domain_stats = {}
            for split in ["train", "eval"]:
                cached = self._load_cached_labels(domain, split)
                if cached:
                    samples = list(cached.values())
                    avg_nodes = sum(len(s.knowledge_graph["nodes"]) for s in samples) / len(samples)
                    avg_edges = sum(len(s.knowledge_graph["edges"]) for s in samples) / len(samples)
                    domain_stats[split] = {
                        "count": len(samples),
                        "avg_nodes": round(avg_nodes, 2),
                        "avg_edges": round(avg_edges, 2),
                    }
            stats["domains"][domain] = domain_stats

        return stats
