#!/usr/bin/env python3
"""
Ollama Model Benchmark for agent_qa.py

Tests different Ollama model pairs for:
1. JSON structured output (ReAct loop)
2. Complex multi-hop reasoning
3. Context compression quality
4. Cypher query generation
5. Answer synthesis with citations
6. Performance metrics (latency)

Usage:
    python benchmarks/ollama_model_benchmark.py --pairs all
    python benchmarks/ollama_model_benchmark.py --pair A
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# Model pairs to benchmark
MODEL_PAIRS = {
    "A": {
        "name": "qwen3:30b-a3b + ministral-3:14b",
        "main": "qwen3:30b-a3b",
        "utility": "ministral-3:14b",
        "ram": "27GB",
    },
    "B": {
        "name": "qwen3:32b + qwen3:14b",
        "main": "qwen3:32b",
        "utility": "qwen3:14b",
        "ram": "29GB",
    },
    "C": {
        "name": "ministral-3:14b (single)",
        "main": "ministral-3:14b",
        "utility": "ministral-3:14b",
        "ram": "9GB",
    },
    "D": {
        "name": "nemotron-3-nano:30b + ministral-3:14b",
        "main": "nemotron-3-nano:30b",
        "utility": "ministral-3:14b",
        "ram": "33GB",
    },
}

# ============================================================================
# TEST DATA
# ============================================================================

# Test 1: JSON Structured Output (ReAct style)
JSON_TEST_PROMPT = """Based on the question and context, decide what to do next.

Question: {question}

Context: The knowledge graph contains information about economics concepts including inflation,
monetary policy, fiscal policy, aggregate demand/supply, and their relationships.

Respond in this exact JSON format:
{{
    "thought": "Your reasoning about what to do next",
    "ready_to_answer": true or false,
    "action": {{
        "tool_name": "graph_lookup" or "web_search" or "none",
        "arguments": {{"key": "value"}}
    }}
}}

Only output valid JSON, no other text."""

JSON_QUESTIONS = [
    "What causes inflation?",
    "What is the relationship between aggregate supply and potential GDP?",
    "How does the Federal Reserve control money supply?",
    "What are the effects of expansionary fiscal policy?",
]

# Test 2: Complex Multi-Hop Reasoning
REASONING_PROMPT = """Answer the following economics question with detailed reasoning.
Show your reasoning chain step by step.

Question: {question}

Provide a thorough answer that demonstrates understanding of the causal relationships."""

REASONING_QUESTIONS = [
    "How does expansionary monetary policy affect unemployment through its impact on aggregate demand?",
    "What is the causal chain from Federal Reserve interest rate decisions to inflation expectations?",
    "Compare and contrast neoclassical and Keynesian views on government intervention in markets.",
]

# Test 3: Context Compression
COMPRESSION_PROMPT = """Compress the following context to only the facts relevant to the question.
Preserve entity names exactly. Remove redundant information.

Question: {question}

Context:
{context}

Output only bullet points of relevant facts (max 5 bullets)."""

COMPRESSION_TESTS = [
    {
        "question": "What is aggregate demand?",
        "context": """Aggregate demand (AD) is the total demand for final goods and services in an economy at a given time.
It is often called effective demand, though this term also has a distinct meaning. The aggregate demand curve
is plotted with real output on the horizontal axis and the price level on the vertical axis. It slopes
downward from left to right because of three effects: the Pigou effect (wealth effect), the Keynes effect
(interest rate effect), and the Mundell-Fleming effect (exchange rate effect). The equation for aggregate
demand is AD = C + I + G + (X − M), where C is consumption, I is investment, G is government spending,
X is exports and M is imports. Aggregate demand was introduced by John Maynard Keynes in his work
The General Theory of Employment, Interest and Money, published in 1936. Aggregate demand is different
from aggregate supply, which represents the total supply of goods and services in an economy. The
intersection of aggregate demand and aggregate supply determines the equilibrium price level and real GDP.
Various factors can shift the aggregate demand curve, including changes in consumer confidence, interest
rates, government policy, and international economic conditions. During recessions, aggregate demand
typically falls, leading to lower output and higher unemployment.""",
        "expected_facts": ["AD = C + I + G + (X - M)", "consumption", "investment", "government spending", "exports", "imports"],
    },
]

# Test 4: Cypher Query Generation
CYPHER_PROMPT = """Convert the following natural language pattern to a Cypher query.
The pattern uses this notation:
- Entity names are plain text
- Relationship types are in brackets like [CAUSES]
- ? means any node
- Direction is shown with arrows

Pattern: {pattern}

Output only the Cypher query, no explanation."""

CYPHER_PATTERNS = [
    {
        "pattern": "Inflation -[CAUSES]-> ?",
        "expected_contains": ["MATCH", "Inflation", "CAUSES", "->"],
    },
    {
        "pattern": "? -[PREVENTS]-> Recession",
        "expected_contains": ["MATCH", "PREVENTS", "Recession"],
    },
    {
        "pattern": "Federal Reserve -[CONTROLS]-> Interest Rates",
        "expected_contains": ["MATCH", "Federal Reserve", "CONTROLS", "Interest Rates"],
    },
    {
        "pattern": "Monetary Policy -[AFFECTS]-> ?",
        "expected_contains": ["MATCH", "Monetary Policy", "AFFECTS"],
    },
]

# Test 5: Answer Synthesis with Citations
SYNTHESIS_PROMPT = """Based on the context below, answer the question with citations.
Use [Source: X] notation for citations.

Question: {question}

Context:
{context}

Respond in this JSON format:
{{
    "answer": "Your answer with [Source: X] citations inline",
    "confidence": 0.0 to 1.0,
    "citations": [
        {{"source_type": "graph", "source_id": "entity name", "excerpt": "relevant text"}}
    ]
}}"""

SYNTHESIS_TESTS = [
    {
        "question": "What is fiscal policy and how does it differ from monetary policy?",
        "context": """[Entity: Fiscal Policy]
Description: Government policy regarding taxation and spending to influence the economy.
Relationships:
-> [INVOLVES] Taxation
-> [INVOLVES] Government Spending
-> [AFFECTS] Aggregate Demand

[Entity: Monetary Policy]
Description: Central bank policy regarding money supply and interest rates.
Relationships:
-> [CONTROLS] Money Supply
-> [CONTROLS] Interest Rates
-> [IMPLEMENTED_BY] Federal Reserve""",
    },
]


# ============================================================================
# BENCHMARK RESULTS
# ============================================================================

@dataclass
class TestResult:
    """Result from a single test."""
    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    latency_ms: int
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from benchmarking a model pair."""
    pair_id: str
    pair_name: str
    main_model: str
    utility_model: str
    ram_usage: str

    # Scores by category
    json_parse_rate: float = 0.0
    reasoning_score: float = 0.0
    compression_score: float = 0.0
    cypher_validity: float = 0.0
    citation_score: float = 0.0

    # Latencies (ms)
    avg_main_latency: int = 0
    avg_utility_latency: int = 0

    # Total score
    total_score: float = 0.0

    # Individual test results
    test_results: List[TestResult] = field(default_factory=list)

    def calculate_total(self):
        """Calculate weighted total score."""
        weights = {
            "json": 0.30,      # Most critical for ReAct loop
            "reasoning": 0.25,
            "compression": 0.15,
            "cypher": 0.15,
            "citation": 0.15,
        }
        self.total_score = (
            self.json_parse_rate * weights["json"] +
            self.reasoning_score * weights["reasoning"] +
            self.compression_score * weights["compression"] +
            self.cypher_validity * weights["cypher"] +
            self.citation_score * weights["citation"]
        ) * 100


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class OllamaModelBenchmark:
    """Benchmark runner for Ollama model pairs."""

    def __init__(self, pair_id: str):
        """Initialize benchmark for a specific model pair."""
        self.pair_id = pair_id
        self.pair_config = MODEL_PAIRS[pair_id]

        logger.info(f"\n{'='*60}")
        logger.info(f"Initializing: {self.pair_config['name']}")
        logger.info(f"Main model: {self.pair_config['main']}")
        logger.info(f"Utility model: {self.pair_config['utility']}")
        logger.info(f"RAM usage: {self.pair_config['ram']}")
        logger.info(f"{'='*60}")

        # Initialize models
        self.main_llm = ChatOllama(
            model=self.pair_config["main"],
            base_url=OLLAMA_HOST,
            temperature=0,
        )

        self.utility_llm = ChatOllama(
            model=self.pair_config["utility"],
            base_url=OLLAMA_HOST,
            temperature=0,
        )

        self.result = BenchmarkResult(
            pair_id=pair_id,
            pair_name=self.pair_config["name"],
            main_model=self.pair_config["main"],
            utility_model=self.pair_config["utility"],
            ram_usage=self.pair_config["ram"],
        )

        self.main_latencies = []
        self.utility_latencies = []

    def _invoke_main(self, prompt: str) -> Tuple[str, int]:
        """Invoke main model and return response + latency."""
        start = time.time()
        response = self.main_llm.invoke([HumanMessage(content=prompt)])
        latency = int((time.time() - start) * 1000)
        self.main_latencies.append(latency)
        return response.content, latency

    def _invoke_utility(self, prompt: str) -> Tuple[str, int]:
        """Invoke utility model and return response + latency."""
        start = time.time()
        response = self.utility_llm.invoke([HumanMessage(content=prompt)])
        latency = int((time.time() - start) * 1000)
        self.utility_latencies.append(latency)
        return response.content, latency

    def _parse_json(self, text: str) -> Tuple[bool, Optional[Dict]]:
        """Attempt to parse JSON from response."""
        # Try direct parse
        try:
            return True, json.loads(text)
        except:
            pass

        # Try extracting from markdown
        if "```json" in text:
            try:
                json_str = text.split("```json")[1].split("```")[0]
                return True, json.loads(json_str)
            except:
                pass

        if "```" in text:
            try:
                json_str = text.split("```")[1].split("```")[0]
                return True, json.loads(json_str)
            except:
                pass

        # Try finding JSON object
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return True, json.loads(text[start:end])
        except:
            pass

        return False, None

    def test_json_output(self) -> List[TestResult]:
        """Test JSON structured output capability."""
        logger.info("\n--- Test 1: JSON Structured Output ---")
        results = []

        for i, question in enumerate(JSON_QUESTIONS):
            prompt = JSON_TEST_PROMPT.format(question=question)

            try:
                response, latency = self._invoke_main(prompt)
                parsed_ok, parsed = self._parse_json(response)

                # Check required fields
                fields_ok = False
                if parsed:
                    required = ["thought", "ready_to_answer", "action"]
                    fields_ok = all(k in parsed for k in required)

                score = 1.0 if (parsed_ok and fields_ok) else (0.5 if parsed_ok else 0.0)

                result = TestResult(
                    test_name=f"json_{i+1}",
                    passed=parsed_ok and fields_ok,
                    score=score,
                    latency_ms=latency,
                    details={
                        "question": question[:50],
                        "parsed": parsed_ok,
                        "fields_ok": fields_ok,
                        "response_preview": response[:100] if response else "",
                    }
                )

                status = "PASS" if result.passed else "FAIL"
                logger.info(f"  [{status}] Q{i+1}: {question[:40]}... ({latency}ms)")

            except Exception as e:
                result = TestResult(
                    test_name=f"json_{i+1}",
                    passed=False,
                    score=0.0,
                    latency_ms=0,
                    error=str(e),
                )
                logger.info(f"  [ERROR] Q{i+1}: {str(e)[:50]}")

            results.append(result)

        return results

    def test_reasoning(self) -> List[TestResult]:
        """Test complex reasoning capability."""
        logger.info("\n--- Test 2: Complex Reasoning ---")
        results = []

        for i, question in enumerate(REASONING_QUESTIONS):
            prompt = REASONING_PROMPT.format(question=question)

            try:
                response, latency = self._invoke_main(prompt)

                # Simple heuristic scoring based on response quality
                score = 0.0

                # Check for reasoning indicators
                reasoning_words = ["because", "therefore", "leads to", "causes", "results in",
                                   "first", "second", "finally", "however", "thus"]
                reasoning_count = sum(1 for w in reasoning_words if w.lower() in response.lower())

                # Check response length (longer = more detailed)
                length_score = min(len(response) / 1000, 1.0)

                # Check for structure (paragraphs or bullet points)
                structure_score = 0.5 if ("\n\n" in response or "- " in response or "1." in response) else 0.0

                score = (
                    min(reasoning_count / 5, 1.0) * 0.4 +
                    length_score * 0.3 +
                    structure_score * 0.3
                )

                result = TestResult(
                    test_name=f"reasoning_{i+1}",
                    passed=score > 0.5,
                    score=score,
                    latency_ms=latency,
                    details={
                        "question": question[:50],
                        "response_length": len(response),
                        "reasoning_indicators": reasoning_count,
                    }
                )

                status = "PASS" if result.passed else "FAIR"
                logger.info(f"  [{status}] Q{i+1}: score={score:.2f} ({latency}ms)")

            except Exception as e:
                result = TestResult(
                    test_name=f"reasoning_{i+1}",
                    passed=False,
                    score=0.0,
                    latency_ms=0,
                    error=str(e),
                )
                logger.info(f"  [ERROR] Q{i+1}: {str(e)[:50]}")

            results.append(result)

        return results

    def test_compression(self) -> List[TestResult]:
        """Test context compression capability."""
        logger.info("\n--- Test 3: Context Compression ---")
        results = []

        for i, test in enumerate(COMPRESSION_TESTS):
            prompt = COMPRESSION_PROMPT.format(
                question=test["question"],
                context=test["context"],
            )

            try:
                response, latency = self._invoke_utility(prompt)

                # Check compression ratio
                original_len = len(test["context"])
                compressed_len = len(response)
                compression_ratio = compressed_len / original_len

                # Check for expected facts
                facts_found = sum(1 for fact in test["expected_facts"]
                                  if fact.lower() in response.lower())
                fact_score = facts_found / len(test["expected_facts"])

                # Good compression is 10-30% of original
                ratio_score = 1.0 if 0.1 <= compression_ratio <= 0.4 else 0.5

                score = fact_score * 0.7 + ratio_score * 0.3

                result = TestResult(
                    test_name=f"compression_{i+1}",
                    passed=score > 0.5,
                    score=score,
                    latency_ms=latency,
                    details={
                        "compression_ratio": f"{compression_ratio:.1%}",
                        "facts_found": f"{facts_found}/{len(test['expected_facts'])}",
                    }
                )

                status = "PASS" if result.passed else "FAIR"
                logger.info(f"  [{status}] Compression: {compression_ratio:.1%}, facts={facts_found}/{len(test['expected_facts'])} ({latency}ms)")

            except Exception as e:
                result = TestResult(
                    test_name=f"compression_{i+1}",
                    passed=False,
                    score=0.0,
                    latency_ms=0,
                    error=str(e),
                )
                logger.info(f"  [ERROR] Compression: {str(e)[:50]}")

            results.append(result)

        return results

    def test_cypher(self) -> List[TestResult]:
        """Test Cypher query generation."""
        logger.info("\n--- Test 4: Cypher Generation ---")
        results = []

        for i, test in enumerate(CYPHER_PATTERNS):
            prompt = CYPHER_PROMPT.format(pattern=test["pattern"])

            try:
                response, latency = self._invoke_utility(prompt)

                # Check for expected components
                components_found = sum(1 for c in test["expected_contains"]
                                       if c.upper() in response.upper())
                score = components_found / len(test["expected_contains"])

                # Basic Cypher syntax check
                has_match = "MATCH" in response.upper()
                has_return = "RETURN" in response.upper()
                syntax_score = 1.0 if (has_match and has_return) else 0.5 if has_match else 0.0

                final_score = score * 0.7 + syntax_score * 0.3

                result = TestResult(
                    test_name=f"cypher_{i+1}",
                    passed=final_score > 0.6,
                    score=final_score,
                    latency_ms=latency,
                    details={
                        "pattern": test["pattern"],
                        "components": f"{components_found}/{len(test['expected_contains'])}",
                        "response_preview": response[:80],
                    }
                )

                status = "PASS" if result.passed else "FAIL"
                logger.info(f"  [{status}] {test['pattern'][:30]}... ({latency}ms)")

            except Exception as e:
                result = TestResult(
                    test_name=f"cypher_{i+1}",
                    passed=False,
                    score=0.0,
                    latency_ms=0,
                    error=str(e),
                )
                logger.info(f"  [ERROR] {test['pattern'][:30]}: {str(e)[:30]}")

            results.append(result)

        return results

    def test_synthesis(self) -> List[TestResult]:
        """Test answer synthesis with citations."""
        logger.info("\n--- Test 5: Answer Synthesis ---")
        results = []

        for i, test in enumerate(SYNTHESIS_TESTS):
            prompt = SYNTHESIS_PROMPT.format(
                question=test["question"],
                context=test["context"],
            )

            try:
                response, latency = self._invoke_main(prompt)

                parsed_ok, parsed = self._parse_json(response)

                score = 0.0
                if parsed_ok and parsed:
                    # Check for answer
                    has_answer = bool(parsed.get("answer"))

                    # Check for citations
                    has_citations = bool(parsed.get("citations"))
                    citation_count = len(parsed.get("citations", []))

                    # Check for confidence
                    confidence = parsed.get("confidence", 0)
                    valid_confidence = 0 <= confidence <= 1

                    # Check for inline citations in answer
                    answer = parsed.get("answer", "")
                    has_inline_cite = "[Source:" in answer or "[source:" in answer.lower()

                    score = (
                        (0.3 if has_answer else 0.0) +
                        (0.3 if has_citations and citation_count > 0 else 0.0) +
                        (0.2 if valid_confidence else 0.0) +
                        (0.2 if has_inline_cite else 0.0)
                    )

                result = TestResult(
                    test_name=f"synthesis_{i+1}",
                    passed=parsed_ok and score > 0.5,
                    score=score,
                    latency_ms=latency,
                    details={
                        "parsed": parsed_ok,
                        "has_citations": has_citations if parsed_ok else False,
                    }
                )

                status = "PASS" if result.passed else "FAIL"
                logger.info(f"  [{status}] Synthesis: score={score:.2f} ({latency}ms)")

            except Exception as e:
                result = TestResult(
                    test_name=f"synthesis_{i+1}",
                    passed=False,
                    score=0.0,
                    latency_ms=0,
                    error=str(e),
                )
                logger.info(f"  [ERROR] Synthesis: {str(e)[:50]}")

            results.append(result)

        return results

    def run_all_tests(self) -> BenchmarkResult:
        """Run all benchmark tests."""
        logger.info(f"\n{'#'*60}")
        logger.info(f"# BENCHMARK: {self.pair_config['name']}")
        logger.info(f"{'#'*60}")

        # Run tests
        json_results = self.test_json_output()
        reasoning_results = self.test_reasoning()
        compression_results = self.test_compression()
        cypher_results = self.test_cypher()
        synthesis_results = self.test_synthesis()

        # Calculate scores
        self.result.json_parse_rate = (
            sum(r.score for r in json_results) / len(json_results)
            if json_results else 0.0
        )
        self.result.reasoning_score = (
            sum(r.score for r in reasoning_results) / len(reasoning_results)
            if reasoning_results else 0.0
        )
        self.result.compression_score = (
            sum(r.score for r in compression_results) / len(compression_results)
            if compression_results else 0.0
        )
        self.result.cypher_validity = (
            sum(r.score for r in cypher_results) / len(cypher_results)
            if cypher_results else 0.0
        )
        self.result.citation_score = (
            sum(r.score for r in synthesis_results) / len(synthesis_results)
            if synthesis_results else 0.0
        )

        # Calculate latencies
        self.result.avg_main_latency = (
            sum(self.main_latencies) // len(self.main_latencies)
            if self.main_latencies else 0
        )
        self.result.avg_utility_latency = (
            sum(self.utility_latencies) // len(self.utility_latencies)
            if self.utility_latencies else 0
        )

        # Store all results
        self.result.test_results = (
            json_results + reasoning_results + compression_results +
            cypher_results + synthesis_results
        )

        # Calculate total
        self.result.calculate_total()

        return self.result


def print_results_table(results: List[BenchmarkResult]):
    """Print comparison table of all results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Pair':<6} {'Model Pair':<35} {'JSON':<8} {'Reason':<8} {'Compress':<8} {'Cypher':<8} {'Cite':<8} {'TOTAL':<8}")
    print("-" * 80)

    # Sort by total score
    sorted_results = sorted(results, key=lambda r: r.total_score, reverse=True)

    for r in sorted_results:
        print(f"{r.pair_id:<6} {r.pair_name:<35} "
              f"{r.json_parse_rate*100:>5.1f}%  "
              f"{r.reasoning_score*100:>5.1f}%  "
              f"{r.compression_score*100:>5.1f}%  "
              f"{r.cypher_validity*100:>5.1f}%  "
              f"{r.citation_score*100:>5.1f}%  "
              f"{r.total_score:>5.1f}")

    print("-" * 80)

    # Latency comparison
    print(f"\n{'Pair':<6} {'Main Latency':<15} {'Utility Latency':<15} {'RAM':<10}")
    print("-" * 50)
    for r in sorted_results:
        print(f"{r.pair_id:<6} {r.avg_main_latency:>10}ms     {r.avg_utility_latency:>10}ms     {r.ram_usage:<10}")

    # Winner
    winner = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"WINNER: Pair {winner.pair_id} - {winner.pair_name}")
    print(f"Total Score: {winner.total_score:.1f}/100")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama model pairs for agent_qa.py")
    parser.add_argument("--pairs", type=str, default="all",
                        help="Pairs to test: 'all' or comma-separated like 'A,B,C'")
    parser.add_argument("--pair", type=str, help="Single pair to test: A, B, C, or D")
    args = parser.parse_args()

    # Determine which pairs to test
    if args.pair:
        pairs_to_test = [args.pair.upper()]
    elif args.pairs.lower() == "all":
        pairs_to_test = list(MODEL_PAIRS.keys())
    else:
        pairs_to_test = [p.strip().upper() for p in args.pairs.split(",")]

    # Validate pairs
    for p in pairs_to_test:
        if p not in MODEL_PAIRS:
            print(f"Error: Unknown pair '{p}'. Valid pairs: {list(MODEL_PAIRS.keys())}")
            sys.exit(1)

    print(f"\n{'#'*80}")
    print(f"# OLLAMA MODEL BENCHMARK FOR AGENT_QA.PY")
    print(f"# Testing pairs: {', '.join(pairs_to_test)}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#'*80}")

    results = []

    for pair_id in pairs_to_test:
        try:
            benchmark = OllamaModelBenchmark(pair_id)
            result = benchmark.run_all_tests()
            results.append(result)
        except Exception as e:
            logger.error(f"\nError testing pair {pair_id}: {e}")
            continue

    if results:
        print_results_table(results)
    else:
        print("\nNo results to display.")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
