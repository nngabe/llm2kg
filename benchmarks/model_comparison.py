"""
Model Comparison Benchmark for Knowledge Graph Extraction

This script reproduces the build_skb.py pipeline but runs multiple models
(including gold-standard models) to compare extraction quality without
writing to Neo4j.
"""

import os
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tabulate import tabulate
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE = "/app/benchmarks/results_cache.pkl"


def load_cache() -> Dict[str, List[Any]]:
    """Load cached results from disk."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(results: Dict[str, List[Any]]):
    """Save results to disk cache."""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)


# --- ONTOLOGY (Same as build_skb.py) ---
def normalize_entity(name: str) -> str:
    """Standardize entity naming: 'NEOCLASSICAL_ECONOMICS' -> 'Neoclassical Economics'"""
    return ' '.join(name.replace('_', ' ').split()).title()


class Node(BaseModel):
    id: str = Field(description="Unique identifier, e.g., 'Albert Einstein'")
    type: str = Field(description="Category, e.g., 'Person', 'Location'")
    description: str = Field(description="A brief summary of this entity based on the text.")


class Edge(BaseModel):
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    relation: str = Field(description="Relationship, e.g., 'born_in'")
    description: str = Field(description="Context explaining why this relationship exists.")


class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


# --- MODEL CONFIGURATIONS ---
@dataclass
class ModelConfig:
    name: str
    provider: str  # 'openai', 'anthropic', 'ollama'
    model_id: str
    is_gold_standard: bool = False
    temperature: float = 0.0
    base_url: Optional[str] = None


# Define all models to benchmark
BENCHMARK_MODELS = [
    # Gold-standard models (highest quality, use for ground truth)
    ModelConfig(
        name="o1 (Gold Standard)",
        provider="openai",
        model_id="o1",
        is_gold_standard=True,
    ),
    ModelConfig(
        name="GPT-5.2 (Gold Standard)",
        provider="openai",
        model_id="gpt-5.2",
        is_gold_standard=True,
    ),
    ModelConfig(
        name="Claude Opus 4.5 (Gold Standard)",
        provider="anthropic",
        model_id="claude-opus-4-5-20251101",
        is_gold_standard=True,
    ),
    # High-quality production models
    ModelConfig(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        is_gold_standard=False,
    ),
    ModelConfig(
        name="Claude Sonnet 4",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        is_gold_standard=False,
    ),
    # Cost-optimized models
    ModelConfig(
        name="GPT-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        is_gold_standard=False,
    ),
    ModelConfig(
        name="Claude Haiku 3.5",
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        is_gold_standard=False,
    ),
    # Local Ollama models (from original script)
    ModelConfig(
        name="Qwen3 30B (Local)",
        provider="ollama",
        model_id="qwen3:30b-a3b",
        is_gold_standard=False,
        base_url="http://host.docker.internal:11434",
    ),
    ModelConfig(
        name="Qwen3 8B (Local)",
        provider="ollama",
        model_id="qwen3:8b",
        is_gold_standard=False,
        base_url="http://host.docker.internal:11434",
    ),
    ModelConfig(
        name="Llama3.1 8B (Local)",
        provider="ollama",
        model_id="llama3.1:8b",
        is_gold_standard=False,
        base_url="http://host.docker.internal:11434",
    ),
]


@dataclass
class ExtractionResult:
    model_name: str
    chunk_id: str
    nodes: List[Dict]
    edges: List[Dict]
    parse_success: bool
    error_message: Optional[str] = None
    latency_seconds: float = 0.0


@dataclass
class ComparisonMetrics:
    model_name: str
    total_chunks: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    avg_nodes_per_chunk: float = 0.0
    avg_edges_per_chunk: float = 0.0
    avg_latency: float = 0.0
    # Comparison to gold standard
    node_overlap_ratio: float = 0.0  # How many nodes match gold standard
    edge_overlap_ratio: float = 0.0  # How many edges match gold standard
    missing_entities: List[str] = field(default_factory=list)
    extra_entities: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


class ModelBenchmark:
    """Runs the extraction pipeline on multiple models and compares outputs."""

    def __init__(self, models: List[ModelConfig] = None, use_cache: bool = True):
        self.models = models or BENCHMARK_MODELS
        self.results: Dict[str, List[ExtractionResult]] = defaultdict(list)
        self.use_cache = use_cache
        if use_cache:
            cached = load_cache()
            for k, v in cached.items():
                self.results[k] = v
            if cached:
                logger.info(f"Loaded {len(cached)} models from cache: {list(cached.keys())}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        self.system_prompt = """You are a Knowledge Graph expert. Extract a semi-structured graph from the text.

1. Identify Entities (Nodes): Include a 'description' summarizing who/what the entity is.
2. Identify Relationships (Edges): Include a 'description' explaining the context of the link.
3. Use consistent IDs."""

    def _create_llm(self, config: ModelConfig):
        """Create the appropriate LLM based on configuration."""
        if config.provider == "openai":
            # o1/o3 models don't support temperature parameter
            if config.model_id.startswith("o1") or config.model_id.startswith("o3"):
                return ChatOpenAI(model=config.model_id)
            return ChatOpenAI(model=config.model_id, temperature=config.temperature)
        elif config.provider == "anthropic":
            return ChatAnthropic(model=config.model_id, temperature=config.temperature)
        elif config.provider == "ollama":
            return ChatOllama(
                model=config.model_id,
                temperature=config.temperature,
                base_url=config.base_url,
            )
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    def _create_chain(self, config: ModelConfig):
        """Create the extraction chain for a model."""
        llm = self._create_llm(config)
        structured_llm = llm.with_structured_output(KnowledgeGraph)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Text: {text}"),
        ])
        return prompt | structured_llm

    def extract_from_chunk(
        self, config: ModelConfig, chunk: str, chunk_id: str
    ) -> ExtractionResult:
        """Extract knowledge graph from a single chunk using the specified model."""
        try:
            chain = self._create_chain(config)
            start_time = time.time()
            data = chain.invoke({"text": chunk})
            latency = time.time() - start_time

            # Normalize entity names for consistency across models
            nodes = [{"id": normalize_entity(n.id), "type": n.type, "description": n.description} for n in data.nodes]
            edges = [{"source": normalize_entity(e.source), "target": normalize_entity(e.target), "relation": e.relation, "description": e.description} for e in data.edges]

            return ExtractionResult(
                model_name=config.name,
                chunk_id=chunk_id,
                nodes=nodes,
                edges=edges,
                parse_success=True,
                latency_seconds=latency,
            )
        except Exception as e:
            return ExtractionResult(
                model_name=config.name,
                chunk_id=chunk_id,
                nodes=[],
                edges=[],
                parse_success=False,
                error_message=str(e),
            )

    def run_benchmark(
        self,
        subject: str = "economics",
        num_docs: int = 3,
        models_to_run: List[str] = None,
    ) -> Dict[str, List[ExtractionResult]]:
        """
        Run the benchmark on specified documents.

        Args:
            subject: Dataset subject ('economics', 'law', 'physics')
            num_docs: Number of documents to process
            models_to_run: List of model names to run (None = all)
        """
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
        dataset = dataset.select(range(min(num_docs, len(dataset))))

        # Prepare chunks from documents
        all_chunks = []
        for doc_idx, entry in enumerate(dataset):
            text = entry["text"]
            if len(text.strip()) < 50:
                continue
            chunks = self.text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks[:3]):  # Limit chunks per doc
                all_chunks.append({
                    "id": f"doc{doc_idx}_chunk{chunk_idx}",
                    "text": chunk,
                    "doc_idx": doc_idx,
                })

        logger.info(f"Processing {len(all_chunks)} chunks across {num_docs} documents")

        # Filter models if specified
        models_to_test = self.models
        if models_to_run:
            models_to_test = [m for m in self.models if m.name in models_to_run]

        # Run each model on all chunks
        for config in models_to_test:
            # Skip if already cached
            if config.name in self.results and len(self.results[config.name]) >= len(all_chunks):
                logger.info(f"Skipping {config.name} (cached with {len(self.results[config.name])} results)")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model: {config.name}")
            logger.info(f"{'='*60}")

            for chunk_data in all_chunks:
                logger.info(f"  Processing {chunk_data['id']}...")
                result = self.extract_from_chunk(
                    config, chunk_data["text"], chunk_data["id"]
                )
                self.results[config.name].append(result)

                if result.parse_success:
                    logger.info(
                        f"    Success: {len(result.nodes)} nodes, "
                        f"{len(result.edges)} edges ({result.latency_seconds:.2f}s)"
                    )
                else:
                    logger.warning(f"    Failed: {result.error_message[:100]}")

            # Save after each model completes
            if self.use_cache:
                save_cache(dict(self.results))
                logger.info(f"Cached results for {config.name}")

        return self.results

    def compute_metrics(self) -> Dict[str, ComparisonMetrics]:
        """Compute comparison metrics for each model."""
        metrics = {}

        # First, identify gold standard results
        gold_standard_results = {}
        for model_name, results in self.results.items():
            config = next((m for m in self.models if m.name == model_name), None)
            if config and config.is_gold_standard:
                for result in results:
                    if result.chunk_id not in gold_standard_results:
                        gold_standard_results[result.chunk_id] = result

        # Compute metrics for each model
        for model_name, results in self.results.items():
            m = ComparisonMetrics(model_name=model_name)
            m.total_chunks = len(results)
            m.successful_parses = sum(1 for r in results if r.parse_success)
            m.failed_parses = m.total_chunks - m.successful_parses

            successful_results = [r for r in results if r.parse_success]
            if successful_results:
                m.total_nodes = sum(len(r.nodes) for r in successful_results)
                m.total_edges = sum(len(r.edges) for r in successful_results)
                m.avg_nodes_per_chunk = m.total_nodes / len(successful_results)
                m.avg_edges_per_chunk = m.total_edges / len(successful_results)
                m.avg_latency = sum(r.latency_seconds for r in successful_results) / len(
                    successful_results
                )

            # Compare to gold standard
            config = next((c for c in self.models if c.name == model_name), None)
            if config and not config.is_gold_standard and gold_standard_results:
                node_overlaps = []
                edge_overlaps = []

                for result in successful_results:
                    gold = gold_standard_results.get(result.chunk_id)
                    if gold and gold.parse_success:
                        # Compare node IDs (normalized)
                        result_node_ids = {n["id"].lower().strip() for n in result.nodes}
                        gold_node_ids = {n["id"].lower().strip() for n in gold.nodes}

                        if gold_node_ids:
                            overlap = len(result_node_ids & gold_node_ids) / len(
                                gold_node_ids
                            )
                            node_overlaps.append(overlap)

                            # Track missing and extra entities
                            missing = gold_node_ids - result_node_ids
                            extra = result_node_ids - gold_node_ids
                            m.missing_entities.extend(list(missing)[:5])
                            m.extra_entities.extend(list(extra)[:5])

                        # Compare edges
                        result_edges = {
                            (e["source"].lower(), e["relation"].lower(), e["target"].lower())
                            for e in result.edges
                        }
                        gold_edges = {
                            (e["source"].lower(), e["relation"].lower(), e["target"].lower())
                            for e in gold.edges
                        }

                        if gold_edges:
                            edge_overlap = len(result_edges & gold_edges) / len(gold_edges)
                            edge_overlaps.append(edge_overlap)

                if node_overlaps:
                    m.node_overlap_ratio = sum(node_overlaps) / len(node_overlaps)
                if edge_overlaps:
                    m.edge_overlap_ratio = sum(edge_overlaps) / len(edge_overlaps)

            # Identify issues
            if m.failed_parses > 0:
                m.issues.append(
                    f"Parse failures: {m.failed_parses}/{m.total_chunks} "
                    f"({100*m.failed_parses/m.total_chunks:.1f}%)"
                )
            if m.avg_nodes_per_chunk < 2:
                m.issues.append("Low entity extraction (avg < 2 nodes/chunk)")
            if m.avg_edges_per_chunk < 1:
                m.issues.append("Low relationship extraction (avg < 1 edge/chunk)")
            if m.node_overlap_ratio > 0 and m.node_overlap_ratio < 0.5:
                m.issues.append(
                    f"Low entity overlap with gold standard ({100*m.node_overlap_ratio:.1f}%)"
                )
            if m.edge_overlap_ratio > 0 and m.edge_overlap_ratio < 0.3:
                m.issues.append(
                    f"Low relationship overlap with gold standard ({100*m.edge_overlap_ratio:.1f}%)"
                )

            metrics[model_name] = m

        return metrics

    def analyze_degradation(self, metrics: Dict[str, ComparisonMetrics]) -> str:
        """Analyze and explain quality degradation issues."""
        analysis = []
        analysis.append("\n" + "=" * 80)
        analysis.append("DEGRADATION ANALYSIS")
        analysis.append("=" * 80)

        # Sort models by whether they're gold standard
        gold_models = [m for m in metrics.values() if any(
            c.is_gold_standard and c.name == m.model_name for c in self.models
        )]
        other_models = [m for m in metrics.values() if m not in gold_models]

        if gold_models:
            analysis.append("\n[GOLD STANDARD BASELINE]")
            for m in gold_models:
                analysis.append(f"\n  {m.model_name}:")
                analysis.append(f"    - Success rate: {100*m.successful_parses/m.total_chunks:.1f}%")
                analysis.append(f"    - Avg nodes/chunk: {m.avg_nodes_per_chunk:.1f}")
                analysis.append(f"    - Avg edges/chunk: {m.avg_edges_per_chunk:.1f}")
                analysis.append(f"    - Avg latency: {m.avg_latency:.2f}s")

        if other_models:
            analysis.append("\n[MODEL COMPARISON TO GOLD STANDARD]")
            for m in other_models:
                analysis.append(f"\n  {m.model_name}:")
                analysis.append(f"    - Success rate: {100*m.successful_parses/m.total_chunks:.1f}%")
                analysis.append(f"    - Entity overlap: {100*m.node_overlap_ratio:.1f}%")
                analysis.append(f"    - Relationship overlap: {100*m.edge_overlap_ratio:.1f}%")
                analysis.append(f"    - Avg latency: {m.avg_latency:.2f}s")

                if m.issues:
                    analysis.append("    - Issues identified:")
                    for issue in m.issues:
                        analysis.append(f"      * {issue}")

                if m.missing_entities:
                    analysis.append(f"    - Sample missing entities: {m.missing_entities[:3]}")
                if m.extra_entities:
                    analysis.append(f"    - Sample extra entities: {m.extra_entities[:3]}")

        # Overall degradation summary
        analysis.append("\n[DEGRADATION PATTERNS]")

        # Check for common issues
        parse_failures = [m for m in other_models if m.failed_parses > 0]
        if parse_failures:
            analysis.append("\n  1. JSON/Structured Output Parsing Issues:")
            for m in parse_failures:
                analysis.append(f"     - {m.model_name}: {m.failed_parses} failures")
            analysis.append("     CAUSE: Smaller models may not reliably produce valid JSON")
            analysis.append("     FIX: Add retry logic or fallback parsing")

        low_extraction = [m for m in other_models if m.avg_nodes_per_chunk < 3]
        if low_extraction:
            analysis.append("\n  2. Under-extraction (missing entities):")
            for m in low_extraction:
                analysis.append(f"     - {m.model_name}: avg {m.avg_nodes_per_chunk:.1f} nodes/chunk")
            analysis.append("     CAUSE: Model lacks domain knowledge or context window limits")
            analysis.append("     FIX: Use larger context windows or domain-specific fine-tuning")

        low_overlap = [m for m in other_models if m.node_overlap_ratio < 0.5 and m.node_overlap_ratio > 0]
        if low_overlap:
            analysis.append("\n  3. Entity Inconsistency (different entity naming):")
            for m in low_overlap:
                analysis.append(f"     - {m.model_name}: {100*m.node_overlap_ratio:.1f}% overlap")
            analysis.append("     CAUSE: Models use different naming conventions for same entities")
            analysis.append("     FIX: Add entity normalization/resolution post-processing")

        return "\n".join(analysis)

    def explain_neo4j_writes(self) -> str:
        """Explain what would be written to Neo4j for each model."""
        explanation = []
        explanation.append("\n" + "=" * 80)
        explanation.append("NEO4J WRITE EXPLANATION (DRY RUN)")
        explanation.append("=" * 80)

        for model_name, results in self.results.items():
            explanation.append(f"\n[{model_name}]")
            successful = [r for r in results if r.parse_success]

            if not successful:
                explanation.append("  No successful extractions - nothing would be written.")
                continue

            # Aggregate all nodes and edges
            all_nodes = []
            all_edges = []
            for r in successful:
                all_nodes.extend(r.nodes)
                all_edges.extend(r.edges)

            # Deduplicate nodes by ID
            unique_nodes = {}
            for node in all_nodes:
                node_id = node["id"]
                if node_id not in unique_nodes:
                    unique_nodes[node_id] = node

            # Count edge types
            edge_types = defaultdict(int)
            for edge in all_edges:
                edge_types[edge["relation"]] += 1

            explanation.append(f"\n  NODES TO CREATE ({len(unique_nodes)} unique entities):")
            explanation.append("  " + "-" * 50)

            for i, (node_id, node) in enumerate(list(unique_nodes.items())[:5]):
                explanation.append(f"\n  Cypher: MERGE (n:Entity {{name: '{node_id}'}})")
                explanation.append(f"          ON CREATE SET n.type = '{node['type']}'")
                explanation.append(f"                        n.description = '{node['description'][:50]}...'")
                explanation.append(f"                        n.embedding = [vector of 1536 floats]")

            if len(unique_nodes) > 5:
                explanation.append(f"\n  ... and {len(unique_nodes) - 5} more nodes")

            explanation.append(f"\n  EDGES TO CREATE ({len(all_edges)} relationships):")
            explanation.append("  " + "-" * 50)

            for i, edge in enumerate(all_edges[:5]):
                rel_type = edge["relation"].upper().replace(" ", "_")
                explanation.append(
                    f"\n  Cypher: MATCH (s:Entity {{name: '{edge['source']}'}})"
                )
                explanation.append(f"          MATCH (t:Entity {{name: '{edge['target']}'}})")
                explanation.append(f"          MERGE (s)-[r:{rel_type}]->(t)")
                explanation.append(f"          ON CREATE SET r.description = '{edge['description'][:40]}...'")

            if len(all_edges) > 5:
                explanation.append(f"\n  ... and {len(all_edges) - 5} more edges")

            explanation.append(f"\n  RELATIONSHIP TYPE DISTRIBUTION:")
            for rel_type, count in sorted(edge_types.items(), key=lambda x: -x[1])[:10]:
                explanation.append(f"    - {rel_type}: {count}")

        return "\n".join(explanation)

    def generate_report(self) -> str:
        """Generate a full comparison report."""
        metrics = self.compute_metrics()

        # Create summary table
        table_data = []
        for model_name, m in metrics.items():
            config = next((c for c in self.models if c.name == model_name), None)
            table_data.append([
                model_name,
                "Yes" if config and config.is_gold_standard else "No",
                f"{m.successful_parses}/{m.total_chunks}",
                f"{m.avg_nodes_per_chunk:.1f}",
                f"{m.avg_edges_per_chunk:.1f}",
                f"{100*m.node_overlap_ratio:.1f}%" if m.node_overlap_ratio > 0 else "N/A",
                f"{m.avg_latency:.2f}s",
                len(m.issues),
            ])

        report = []
        report.append("\n" + "=" * 80)
        report.append("MODEL COMPARISON BENCHMARK REPORT")
        report.append("=" * 80)

        report.append("\n[SUMMARY TABLE]")
        headers = [
            "Model",
            "Gold Std",
            "Success",
            "Nodes/Chunk",
            "Edges/Chunk",
            "Entity Overlap",
            "Latency",
            "Issues",
        ]
        report.append(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Add degradation analysis
        report.append(self.analyze_degradation(metrics))

        # Add Neo4j write explanation
        report.append(self.explain_neo4j_writes())

        # Sample extraction comparison
        report.append("\n" + "=" * 80)
        report.append("SAMPLE EXTRACTION COMPARISON")
        report.append("=" * 80)

        # Get first chunk from each model
        first_chunk_id = None
        for results in self.results.values():
            if results:
                first_chunk_id = results[0].chunk_id
                break

        if first_chunk_id:
            report.append(f"\n[Chunk: {first_chunk_id}]")
            for model_name, results in self.results.items():
                matching = [r for r in results if r.chunk_id == first_chunk_id]
                if matching:
                    r = matching[0]
                    report.append(f"\n  {model_name}:")
                    if r.parse_success:
                        report.append(f"    Nodes: {[n['id'] for n in r.nodes]}")
                        report.append(f"    Edges: {[(e['source'], e['relation'], e['target']) for e in r.edges]}")
                    else:
                        report.append(f"    FAILED: {r.error_message[:80]}")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple models for KG extraction")
    parser.add_argument("--subject", type=str, default="economics", choices=["economics", "law", "physics"])
    parser.add_argument("--num_docs", type=int, default=3)
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to test")
    parser.add_argument("--skip_local", action="store_true", help="Skip Ollama local models")
    parser.add_argument("--skip_anthropic", action="store_true", help="Skip Anthropic models")
    parser.add_argument("--skip_openai", action="store_true", help="Skip OpenAI models")
    parser.add_argument("--output", type=str, default=None, help="Output file for report")
    parser.add_argument("--no_cache", action="store_true", help="Don't use cached results")
    parser.add_argument("--clear_cache", action="store_true", help="Clear cache before running")

    args = parser.parse_args()

    if args.clear_cache and os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("Cache cleared.")

    # Filter models
    models = BENCHMARK_MODELS
    if args.skip_local:
        models = [m for m in models if m.provider != "ollama"]
    if args.skip_anthropic:
        models = [m for m in models if m.provider != "anthropic"]
    if args.skip_openai:
        models = [m for m in models if m.provider != "openai"]
    if args.models:
        models = [m for m in models if m.name in args.models]

    print(f"Running benchmark with {len(models)} models on {args.num_docs} {args.subject} documents")
    print(f"Models: {[m.name for m in models]}")

    benchmark = ModelBenchmark(models=models, use_cache=not args.no_cache)

    try:
        benchmark.run_benchmark(subject=args.subject, num_docs=args.num_docs)
        report = benchmark.generate_report()
        print(report)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
