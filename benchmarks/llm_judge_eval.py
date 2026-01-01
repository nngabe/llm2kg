"""
LLM Judge Evaluation for Knowledge Graph Extraction Quality

Uses GPT-5.2 to evaluate the quality and meaningfulness of entity and relationship
extractions from local models, comparing them against gold standard results.
"""

import os
import sys
import pickle
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Ensure the parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.model_comparison import ExtractionResult

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tabulate import tabulate


CACHE_FILE = os.path.join(os.path.dirname(__file__), "results_cache.pkl")
JUDGE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "judge_cache.pkl")


class ExtractionQuality(BaseModel):
    """Structured output for LLM judge evaluation."""
    entity_precision: float = Field(description="Score 0-1: Are extracted entities meaningful and correct?")
    entity_recall: float = Field(description="Score 0-1: Are important entities from the text captured?")
    relationship_accuracy: float = Field(description="Score 0-1: Are relationships factually correct?")
    relationship_meaningfulness: float = Field(description="Score 0-1: Are relationships semantically meaningful?")
    naming_consistency: float = Field(description="Score 0-1: Are entity names consistent and normalized?")
    overall_quality: float = Field(description="Score 0-1: Overall extraction quality")
    strengths: List[str] = Field(description="Key strengths of this extraction")
    weaknesses: List[str] = Field(description="Key weaknesses or issues")
    missing_entities: List[str] = Field(description="Important entities missing from extraction")
    spurious_entities: List[str] = Field(description="Extracted entities that seem incorrect or irrelevant")


class OverlapAnalysis(BaseModel):
    """Structured output for overlap comparison."""
    entity_overlap_score: float = Field(description="Score 0-1: How much do entities overlap with gold standard?")
    relationship_overlap_score: float = Field(description="Score 0-1: How much do relationships overlap?")
    semantic_similarity: float = Field(description="Score 0-1: How semantically similar are the extractions?")
    matched_entities: List[str] = Field(description="Entities that match between models")
    unmatched_entities: List[str] = Field(description="Entities only in candidate model")
    missing_from_gold: List[str] = Field(description="Entities candidate has that gold standard missed")
    analysis: str = Field(description="Brief analysis of the overlap")


def load_cache() -> Dict[str, List[Any]]:
    """Load cached results from disk."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def load_judge_cache() -> Dict[str, Any]:
    """Load cached judge results."""
    if os.path.exists(JUDGE_CACHE_FILE):
        with open(JUDGE_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_judge_cache(cache: Dict[str, Any]):
    """Save judge cache to disk."""
    with open(JUDGE_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


class LLMJudge:
    """Uses GPT-5.2 as an LLM judge to evaluate extraction quality."""

    def __init__(self, use_cache: bool = True):
        self.llm = ChatOpenAI(model="gpt-5.2", temperature=0.0)
        self.results = load_cache()
        self.use_cache = use_cache
        self.judge_cache = load_judge_cache() if use_cache else {}

        # Quality evaluation prompt
        self.quality_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator of knowledge graph extractions.
Given the original text chunk and an extraction (nodes and edges), evaluate the quality of the extraction.

Focus on:
1. Entity Precision: Are the extracted entities real, meaningful concepts from the text?
2. Entity Recall: Are important entities from the text captured?
3. Relationship Accuracy: Are the relationships factually correct based on the text?
4. Relationship Meaningfulness: Do the relationships capture meaningful semantic connections?
5. Naming Consistency: Are entity names properly normalized (e.g., "U.S. Poverty Rate" not "U.S.Povertyrate")?
6. Overall Quality: Holistic assessment of the extraction.

Be strict but fair. Score 0-1 for each dimension."""),
            ("human", """Original Text:
{text}

Extracted Nodes:
{nodes}

Extracted Edges:
{edges}

Evaluate this extraction's quality.""")
        ])

        # Overlap comparison prompt
        self.overlap_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at comparing knowledge graph extractions.
Compare a candidate model's extraction against a gold standard extraction.

Consider:
1. Entity Overlap: Do they extract the same or semantically equivalent entities?
2. Relationship Overlap: Do they capture the same relationships (accounting for different naming)?
3. Semantic Similarity: Even if naming differs, do they convey the same information?

Be nuanced - "U.S. Poverty Rate" and "U.S.Povertyrate" are the same entity with different formatting.
"declined_in" and "decreased_during" are semantically equivalent relationships."""),
            ("human", """Gold Standard Extraction:
Nodes: {gold_nodes}
Edges: {gold_edges}

Candidate Model Extraction:
Nodes: {candidate_nodes}
Edges: {candidate_edges}

Compare and analyze the overlap.""")
        ])

    def _get_text_for_chunk(self, chunk_id: str) -> str:
        """Get the original text for a chunk ID by loading the dataset."""
        from datasets import load_dataset
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Parse chunk_id: "doc{N}_chunk{M}"
        parts = chunk_id.split("_")
        doc_idx = int(parts[0].replace("doc", ""))
        chunk_idx = int(parts[1].replace("chunk", ""))

        dataset = load_dataset(
            "cais/wmdp-mmlu-auxiliary-corpora", "economics-corpus", split="train"
        )
        text = dataset[doc_idx]["text"]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        if chunk_idx < len(chunks):
            return chunks[chunk_idx]
        return ""

    def evaluate_quality(self, model_name: str, chunk_id: str,
                        nodes: List[Dict], edges: List[Dict]) -> ExtractionQuality:
        """Evaluate the quality of an extraction using GPT-5.2."""
        cache_key = f"quality_{model_name}_{chunk_id}"

        if self.use_cache and cache_key in self.judge_cache:
            return self.judge_cache[cache_key]

        text = self._get_text_for_chunk(chunk_id)

        structured_llm = self.llm.with_structured_output(ExtractionQuality)
        chain = self.quality_prompt | structured_llm

        nodes_str = json.dumps([{"id": n["id"], "type": n["type"]} for n in nodes], indent=2)
        edges_str = json.dumps([{"source": e["source"], "relation": e["relation"], "target": e["target"]}
                                for e in edges], indent=2)

        result = chain.invoke({
            "text": text,
            "nodes": nodes_str,
            "edges": edges_str
        })

        if self.use_cache:
            self.judge_cache[cache_key] = result
            save_judge_cache(self.judge_cache)

        return result

    def compare_overlap(self, gold_nodes: List[Dict], gold_edges: List[Dict],
                       candidate_nodes: List[Dict], candidate_edges: List[Dict],
                       gold_model: str, candidate_model: str, chunk_id: str) -> OverlapAnalysis:
        """Compare overlap between gold standard and candidate extraction."""
        cache_key = f"overlap_{gold_model}_{candidate_model}_{chunk_id}"

        if self.use_cache and cache_key in self.judge_cache:
            return self.judge_cache[cache_key]

        structured_llm = self.llm.with_structured_output(OverlapAnalysis)
        chain = self.overlap_prompt | structured_llm

        gold_nodes_str = json.dumps([n["id"] for n in gold_nodes])
        gold_edges_str = json.dumps([f"({e['source']}) -[{e['relation']}]-> ({e['target']})"
                                     for e in gold_edges])
        candidate_nodes_str = json.dumps([n["id"] for n in candidate_nodes])
        candidate_edges_str = json.dumps([f"({e['source']}) -[{e['relation']}]-> ({e['target']})"
                                          for e in candidate_edges])

        result = chain.invoke({
            "gold_nodes": gold_nodes_str,
            "gold_edges": gold_edges_str,
            "candidate_nodes": candidate_nodes_str,
            "candidate_edges": candidate_edges_str
        })

        if self.use_cache:
            self.judge_cache[cache_key] = result
            save_judge_cache(self.judge_cache)

        return result

    def run_evaluation(self,
                      local_models: List[str] = None,
                      gold_models: List[str] = None,
                      max_chunks: int = 3) -> Tuple[Dict, Dict]:
        """
        Run full evaluation comparing local models against gold standards.

        Args:
            local_models: List of local model names to evaluate
            gold_models: List of gold standard model names for comparison
            max_chunks: Maximum chunks to evaluate per model

        Returns:
            Tuple of (quality_results, overlap_results)
        """
        if local_models is None:
            local_models = [
                "Ministral-3 8B (Local)",
                "Ministral-3 14B (Local)",
                "GPT-OSS 20B (Local)",
                "Gemma3 12B KG Extraction Q4 (Ollama)",
            ]

        if gold_models is None:
            gold_models = [
                "GPT-5.2 (Gold Standard)",
                "Gemini 2.5 Flash",
            ]

        quality_results = defaultdict(list)
        overlap_results = defaultdict(lambda: defaultdict(list))

        # Get common chunk IDs across all models
        all_chunk_ids = set()
        for model_name in local_models + gold_models:
            if model_name in self.results:
                for r in self.results[model_name]:
                    if r.parse_success:
                        all_chunk_ids.add(r.chunk_id)

        # Filter to chunks present in gold standards
        gold_chunk_ids = set()
        for gold_model in gold_models:
            if gold_model in self.results:
                for r in self.results[gold_model]:
                    if r.parse_success:
                        gold_chunk_ids.add(r.chunk_id)

        common_chunks = list(all_chunk_ids & gold_chunk_ids)[:max_chunks]
        print(f"Evaluating {len(common_chunks)} chunks: {common_chunks}")

        # Evaluate quality for each local model
        for model_name in local_models:
            if model_name not in self.results:
                print(f"Skipping {model_name} (no results)")
                continue

            print(f"\nEvaluating quality for: {model_name}")

            for result in self.results[model_name]:
                if not result.parse_success or result.chunk_id not in common_chunks:
                    continue

                print(f"  Evaluating chunk: {result.chunk_id}")
                quality = self.evaluate_quality(
                    model_name, result.chunk_id,
                    result.nodes, result.edges
                )
                quality_results[model_name].append({
                    "chunk_id": result.chunk_id,
                    "quality": quality
                })

        # Compare overlap with gold standards
        for local_model in local_models:
            if local_model not in self.results:
                continue

            for gold_model in gold_models:
                if gold_model not in self.results:
                    continue

                print(f"\nComparing {local_model} vs {gold_model}")

                # Build lookup for gold results
                gold_lookup = {r.chunk_id: r for r in self.results[gold_model] if r.parse_success}

                for result in self.results[local_model]:
                    if not result.parse_success or result.chunk_id not in common_chunks:
                        continue

                    if result.chunk_id not in gold_lookup:
                        continue

                    gold_result = gold_lookup[result.chunk_id]

                    print(f"  Comparing chunk: {result.chunk_id}")
                    overlap = self.compare_overlap(
                        gold_result.nodes, gold_result.edges,
                        result.nodes, result.edges,
                        gold_model, local_model, result.chunk_id
                    )
                    overlap_results[local_model][gold_model].append({
                        "chunk_id": result.chunk_id,
                        "overlap": overlap
                    })

        return dict(quality_results), dict(overlap_results)

    def generate_report(self, quality_results: Dict, overlap_results: Dict) -> str:
        """Generate a comprehensive evaluation report."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("GPT-5.2 LLM JUDGE EVALUATION REPORT")
        lines.append("=" * 80)

        # Quality Summary Table
        lines.append("\n[EXTRACTION QUALITY SCORES (Judged by GPT-5.2)]")

        quality_table = []
        for model_name, results in quality_results.items():
            if not results:
                continue

            avg_precision = sum(r["quality"].entity_precision for r in results) / len(results)
            avg_recall = sum(r["quality"].entity_recall for r in results) / len(results)
            avg_rel_acc = sum(r["quality"].relationship_accuracy for r in results) / len(results)
            avg_rel_mean = sum(r["quality"].relationship_meaningfulness for r in results) / len(results)
            avg_naming = sum(r["quality"].naming_consistency for r in results) / len(results)
            avg_overall = sum(r["quality"].overall_quality for r in results) / len(results)

            quality_table.append([
                model_name,
                f"{avg_precision:.2f}",
                f"{avg_recall:.2f}",
                f"{avg_rel_acc:.2f}",
                f"{avg_rel_mean:.2f}",
                f"{avg_naming:.2f}",
                f"{avg_overall:.2f}",
            ])

        headers = ["Model", "Ent Prec", "Ent Rec", "Rel Acc", "Rel Mean", "Naming", "Overall"]
        lines.append(tabulate(quality_table, headers=headers, tablefmt="grid"))

        # Overlap Summary Table
        lines.append("\n[OVERLAP WITH GOLD STANDARDS (Semantic Comparison)]")

        overlap_table = []
        for local_model, gold_comparisons in overlap_results.items():
            for gold_model, comparisons in gold_comparisons.items():
                if not comparisons:
                    continue

                avg_entity = sum(c["overlap"].entity_overlap_score for c in comparisons) / len(comparisons)
                avg_rel = sum(c["overlap"].relationship_overlap_score for c in comparisons) / len(comparisons)
                avg_semantic = sum(c["overlap"].semantic_similarity for c in comparisons) / len(comparisons)

                overlap_table.append([
                    local_model,
                    gold_model,
                    f"{avg_entity:.2f}",
                    f"{avg_rel:.2f}",
                    f"{avg_semantic:.2f}",
                ])

        headers = ["Local Model", "vs Gold Standard", "Entity Overlap", "Rel Overlap", "Semantic Sim"]
        lines.append(tabulate(overlap_table, headers=headers, tablefmt="grid"))

        # Detailed Findings
        lines.append("\n[DETAILED QUALITY ANALYSIS]")

        for model_name, results in quality_results.items():
            lines.append(f"\n{model_name}:")

            all_strengths = []
            all_weaknesses = []
            all_missing = []
            all_spurious = []

            for r in results:
                all_strengths.extend(r["quality"].strengths)
                all_weaknesses.extend(r["quality"].weaknesses)
                all_missing.extend(r["quality"].missing_entities)
                all_spurious.extend(r["quality"].spurious_entities)

            if all_strengths:
                lines.append(f"  Strengths: {list(set(all_strengths))[:5]}")
            if all_weaknesses:
                lines.append(f"  Weaknesses: {list(set(all_weaknesses))[:5]}")
            if all_missing:
                lines.append(f"  Commonly Missing: {list(set(all_missing))[:5]}")
            if all_spurious:
                lines.append(f"  Spurious Entities: {list(set(all_spurious))[:5]}")

        # Overlap Insights
        lines.append("\n[OVERLAP INSIGHTS]")

        for local_model, gold_comparisons in overlap_results.items():
            for gold_model, comparisons in gold_comparisons.items():
                if not comparisons:
                    continue

                lines.append(f"\n{local_model} vs {gold_model}:")

                all_matched = []
                all_unmatched = []
                all_missing_from_gold = []

                for c in comparisons:
                    all_matched.extend(c["overlap"].matched_entities)
                    all_unmatched.extend(c["overlap"].unmatched_entities)
                    all_missing_from_gold.extend(c["overlap"].missing_from_gold)

                if all_matched:
                    lines.append(f"  Matched Entities: {list(set(all_matched))[:5]}")
                if all_unmatched:
                    lines.append(f"  Only in Local Model: {list(set(all_unmatched))[:5]}")
                if all_missing_from_gold:
                    lines.append(f"  Local Found (Gold Missed): {list(set(all_missing_from_gold))[:5]}")

                # Sample analysis
                if comparisons:
                    lines.append(f"  Analysis: {comparisons[0]['overlap'].analysis[:200]}...")

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Judge Evaluation")
    parser.add_argument("--max_chunks", type=int, default=3, help="Max chunks to evaluate")
    parser.add_argument("--no_cache", action="store_true", help="Don't use cached results")
    parser.add_argument("--clear_cache", action="store_true", help="Clear judge cache")

    args = parser.parse_args()

    if args.clear_cache and os.path.exists(JUDGE_CACHE_FILE):
        os.remove(JUDGE_CACHE_FILE)
        print("Judge cache cleared.")

    judge = LLMJudge(use_cache=not args.no_cache)

    quality_results, overlap_results = judge.run_evaluation(max_chunks=args.max_chunks)
    report = judge.generate_report(quality_results, overlap_results)

    print(report)


if __name__ == "__main__":
    main()
