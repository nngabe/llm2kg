#!/usr/bin/env python3
"""
Ablation Study for agent_qa.py

Tests the impact of different agent stages and retrieval algorithms:

1. Stage Ablations:
   - With/without retrieval planning
   - With/without context compression
   - With/without multi-hop traversal

2. Retrieval Algorithm Comparisons:
   - Vector search only
   - Graph traversal only
   - Hybrid (planning + vector + graph + patterns)

3. LLM-as-Judge Evaluation:
   - Uses GPT-5.2 to evaluate answer accuracy against ground truth
   - Scores: correctness, completeness, relevance

Usage:
    python benchmarks/ablation_study.py
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

MAIN_MODEL = "nemotron-3-nano:30b"
UTILITY_MODEL = "ministral-3:14b"
EMBEDDING_MODEL = "qwen3-embedding:8b"

# LLM-as-Judge configuration
JUDGE_MODEL = "gpt-5.2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# BENCHMARK QUESTIONS
# ============================================================================

BENCHMARK_QUESTIONS = [
    {
        "id": "q1",
        "question": "What is inflation and what causes it?",
        "expected_entities": ["inflation"],
        "difficulty": "easy",
        "expected_answer": """Inflation is a sustained increase in the general price level of goods and services in an economy over time. Key causes include:
1. Demand-pull inflation: When aggregate demand exceeds aggregate supply, often from increased consumer spending, government expenditure, or money supply expansion
2. Cost-push inflation: Rising production costs (wages, raw materials, energy) push prices up
3. Monetary factors: Excessive money supply growth by central banks
4. Built-in inflation: Wage-price spirals where workers demand higher wages expecting prices to rise
5. Supply shocks: Sudden decreases in supply (e.g., oil crises) raising prices""",
        "key_facts": ["price level increase", "demand-pull", "cost-push", "money supply", "aggregate demand"],
    },
    {
        "id": "q2",
        "question": "How does expansionary monetary policy affect unemployment?",
        "expected_entities": ["expansionary monetary policy", "unemployment"],
        "difficulty": "medium",
        "expected_answer": """Expansionary monetary policy reduces unemployment through the following mechanism:
1. Central bank lowers interest rates or increases money supply
2. Lower interest rates make borrowing cheaper for businesses and consumers
3. Increased borrowing leads to higher investment and consumer spending
4. Higher spending increases aggregate demand
5. Businesses respond to increased demand by hiring more workers
6. Unemployment decreases (at least in the short run)
This relationship is captured by the Phillips Curve trade-off between inflation and unemployment. However, long-run effects may differ according to different economic schools (Keynesian vs Neoclassical).""",
        "key_facts": ["lower interest rates", "increased aggregate demand", "more hiring", "Phillips Curve", "short-run reduction"],
    },
    {
        "id": "q3",
        "question": "What is the relationship between aggregate supply and potential GDP in neoclassical economics?",
        "expected_entities": ["aggregate supply", "potential gdp", "neoclassical"],
        "difficulty": "hard",
        "expected_answer": """In neoclassical economics, the long-run aggregate supply (LRAS) curve is vertical at the level of potential GDP. Key relationships:
1. Potential GDP represents the economy's maximum sustainable output when all resources are fully employed
2. The LRAS is vertical because in the long run, output is determined by real factors (technology, capital, labor) not price level
3. Short-run aggregate supply (SRAS) can deviate from potential GDP due to sticky wages/prices
4. The economy naturally tends toward potential GDP through price adjustments
5. Only supply-side factors (productivity, capital accumulation, labor force growth) can shift potential GDP
This contrasts with Keynesian views where demand can affect output even in longer periods.""",
        "key_facts": ["vertical LRAS", "full employment output", "real factors determine output", "price flexibility", "supply-side factors"],
    },
    {
        "id": "q4",
        "question": "Compare fiscal policy and monetary policy approaches to managing recession.",
        "expected_entities": ["fiscal policy", "monetary policy", "recession"],
        "difficulty": "hard",
        "expected_answer": """Fiscal Policy (Government):
- Tools: Government spending increases, tax cuts
- Mechanism: Direct injection into economy, increases aggregate demand
- Advantages: Can target specific sectors, effective when interest rates near zero
- Disadvantages: Implementation lags, political constraints, increases government debt

Monetary Policy (Central Bank):
- Tools: Lower interest rates, quantitative easing, reserve requirements
- Mechanism: Makes borrowing cheaper, increases money supply, stimulates investment
- Advantages: Faster implementation, independent of politics
- Disadvantages: Less effective at zero lower bound (liquidity trap), indirect effects

Key Differences:
- Fiscal is direct spending; monetary works through financial system
- Fiscal has longer lags; monetary is more flexible
- Both aim to increase aggregate demand to combat recession""",
        "key_facts": ["government spending", "tax cuts", "interest rates", "aggregate demand", "implementation lags", "liquidity trap"],
    },
    {
        "id": "q5",
        "question": "What is aggregate demand and what are its components?",
        "expected_entities": ["aggregate demand"],
        "difficulty": "easy",
        "expected_answer": """Aggregate Demand (AD) is the total demand for all goods and services in an economy at a given price level and time period. Components (using expenditure approach):
1. Consumption (C): Household spending on goods and services - typically the largest component
2. Investment (I): Business spending on capital goods, residential construction, inventory changes
3. Government Spending (G): Government purchases of goods and services (excludes transfers)
4. Net Exports (NX): Exports minus Imports (X - M)

Formula: AD = C + I + G + (X - M)

The AD curve slopes downward due to wealth effect, interest rate effect, and exchange rate effect.""",
        "key_facts": ["consumption", "investment", "government spending", "net exports", "C + I + G + (X-M)"],
    },
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetrievalResult:
    """Result from a retrieval algorithm."""
    algorithm: str
    context: str
    entities_found: int
    relationships_found: int
    latency_ms: int
    raw_results: List[Dict] = field(default_factory=list)


@dataclass
class JudgeScore:
    """Scores from LLM-as-judge evaluation."""
    correctness: float  # 0-1: Are the facts correct?
    completeness: float  # 0-1: Are all key facts covered?
    relevance: float  # 0-1: Is the answer relevant to the question?
    overall: float  # 0-1: Weighted average
    explanation: str  # Judge's reasoning
    key_facts_found: List[str] = field(default_factory=list)
    key_facts_missing: List[str] = field(default_factory=list)


@dataclass
class AblationResult:
    """Result from an ablation test."""
    config_name: str
    question_id: str
    answer: str
    confidence: float  # Self-reported by the model
    citations: int
    latency_ms: int
    context_length: int
    entities_retrieved: int
    judge_score: Optional[JudgeScore] = None  # GPT-5.2 evaluation
    error: Optional[str] = None


@dataclass
class AblationSummary:
    """Summary statistics for an ablation configuration."""
    config_name: str
    avg_confidence: float  # Self-reported
    avg_accuracy: float  # Judge's overall score
    avg_correctness: float  # Judge's correctness score
    avg_completeness: float  # Judge's completeness score
    avg_latency_ms: int
    avg_citations: float
    avg_context_length: int
    success_rate: float
    total_questions: int


# ============================================================================
# RETRIEVAL ALGORITHMS
# ============================================================================

class RetrievalAlgorithms:
    """Different retrieval algorithms for comparison."""

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embedding_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_HOST,
        )

    def close(self):
        self.driver.close()

    # -------------------------------------------------------------------------
    # Algorithm 1: Vector Search Only
    # -------------------------------------------------------------------------
    def vector_search_only(self, question: str, limit: int = 10) -> RetrievalResult:
        """Pure vector similarity search on entity embeddings."""
        start = time.time()
        results = []

        try:
            embedding = self.embedding_model.embed_query(question)

            with self.driver.session() as session:
                # Vector search on entity embeddings
                cypher = """
                CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
                YIELD node, score
                RETURN node, labels(node) as labels, score
                ORDER BY score DESC
                """
                result = session.run(cypher, embedding=embedding, limit=limit)

                for record in result:
                    node = dict(record["node"])
                    node["labels"] = record["labels"]
                    node["score"] = record["score"]
                    results.append(node)
        except Exception as e:
            logger.debug(f"Vector search error: {e}")

        # Format context
        context_parts = []
        for entity in results:
            name = entity.get("name", entity.get("id", "Unknown"))
            desc = entity.get("description", "")
            labels = entity.get("labels", [])
            score = entity.get("score", 0)
            context_parts.append(f"[Entity: {name} ({', '.join(labels)})] score={score:.3f}\n{desc}")

        context = "\n\n".join(context_parts)
        latency = int((time.time() - start) * 1000)

        return RetrievalResult(
            algorithm="vector_only",
            context=context,
            entities_found=len(results),
            relationships_found=0,
            latency_ms=latency,
            raw_results=results,
        )

    # -------------------------------------------------------------------------
    # Algorithm 2: Graph Traversal Only
    # -------------------------------------------------------------------------
    def graph_traversal_only(self, question: str, max_hops: int = 2) -> RetrievalResult:
        """Graph traversal based on keyword entity matching + N-hop expansion."""
        start = time.time()
        entities = []
        relationships = []

        # Extract key terms from question (simple keyword extraction)
        keywords = self._extract_keywords(question)

        with self.driver.session() as session:
            for keyword in keywords[:5]:  # Limit to top 5 keywords
                # Find matching entity
                entity_query = """
                MATCH (e)
                WHERE toLower(e.name) CONTAINS toLower($keyword)
                RETURN e, labels(e) as labels
                LIMIT 3
                """
                result = session.run(entity_query, keyword=keyword)

                for record in result:
                    entity = dict(record["e"])
                    entity["labels"] = record["labels"]
                    entities.append(entity)

                    # Get N-hop neighborhood
                    entity_name = entity.get("name", "")
                    if entity_name:
                        rel_query = f"""
                        MATCH (e {{name: $name}})-[r*1..{max_hops}]-(other)
                        WITH e, r as rels, other
                        UNWIND rels as rel
                        RETURN DISTINCT
                            type(rel) as rel_type,
                            startNode(rel).name as from_node,
                            endNode(rel).name as to_node
                        LIMIT 20
                        """
                        try:
                            rel_result = session.run(rel_query, name=entity_name)
                            for rec in rel_result:
                                relationships.append({
                                    "type": rec["rel_type"],
                                    "from": rec["from_node"],
                                    "to": rec["to_node"],
                                })
                        except:
                            pass

        # Format context
        context_parts = []

        # Entities
        seen_entities = set()
        for entity in entities:
            name = entity.get("name", "")
            if name and name not in seen_entities:
                seen_entities.add(name)
                desc = entity.get("description", "")
                labels = entity.get("labels", [])
                context_parts.append(f"[Entity: {name} ({', '.join(labels)})]\n{desc}")

        # Relationships
        if relationships:
            context_parts.append("\n[Relationships]")
            seen_rels = set()
            for rel in relationships:
                rel_str = f"  {rel['from']} -[{rel['type']}]-> {rel['to']}"
                if rel_str not in seen_rels:
                    seen_rels.add(rel_str)
                    context_parts.append(rel_str)

        context = "\n".join(context_parts)
        latency = int((time.time() - start) * 1000)

        return RetrievalResult(
            algorithm="graph_traversal",
            context=context,
            entities_found=len(seen_entities),
            relationships_found=len(relationships),
            latency_ms=latency,
            raw_results=entities,
        )

    # -------------------------------------------------------------------------
    # Algorithm 3: Hybrid (Vector + Graph + Pattern Matching)
    # -------------------------------------------------------------------------
    def hybrid_retrieval(self, question: str, max_hops: int = 2) -> RetrievalResult:
        """Hybrid retrieval combining vector search, graph traversal, and patterns."""
        start = time.time()

        all_entities = []
        all_relationships = []

        # Step 1: Vector search for semantic similarity
        vector_result = self.vector_search_only(question, limit=5)
        for entity in vector_result.raw_results:
            entity["source"] = "vector"
            all_entities.append(entity)

        # Step 2: Graph traversal from vector results
        with self.driver.session() as session:
            for entity in vector_result.raw_results[:3]:
                entity_name = entity.get("name", "")
                if entity_name:
                    # Get relationships
                    rel_query = f"""
                    MATCH (e {{name: $name}})-[r*1..{max_hops}]-(other)
                    WITH e, r as rels, other
                    UNWIND rels as rel
                    RETURN DISTINCT
                        type(rel) as rel_type,
                        startNode(rel).name as from_node,
                        endNode(rel).name as to_node,
                        other.name as other_name,
                        other.description as other_desc,
                        labels(other) as other_labels
                    LIMIT 15
                    """
                    try:
                        rel_result = session.run(rel_query, name=entity_name)
                        for rec in rel_result:
                            all_relationships.append({
                                "type": rec["rel_type"],
                                "from": rec["from_node"],
                                "to": rec["to_node"],
                            })
                            # Add connected entities
                            if rec["other_name"]:
                                all_entities.append({
                                    "name": rec["other_name"],
                                    "description": rec["other_desc"] or "",
                                    "labels": rec["other_labels"] or [],
                                    "source": "traversal",
                                })
                    except:
                        pass

            # Step 3: Pattern matching for common relationship types
            patterns = [
                ("CAUSES", question),
                ("AFFECTS", question),
                ("INFLUENCES", question),
                ("PART_OF", question),
            ]

            for rel_type, q in patterns:
                pattern_query = f"""
                MATCH (a)-[r:{rel_type}]->(b)
                WHERE toLower(a.name) CONTAINS toLower($keyword)
                   OR toLower(b.name) CONTAINS toLower($keyword)
                RETURN a.name as from_name, a.description as from_desc,
                       type(r) as rel_type,
                       b.name as to_name, b.description as to_desc
                LIMIT 5
                """
                keywords = self._extract_keywords(q)
                for keyword in keywords[:2]:
                    try:
                        result = session.run(pattern_query, keyword=keyword)
                        for rec in result:
                            all_relationships.append({
                                "type": rec["rel_type"],
                                "from": rec["from_name"],
                                "to": rec["to_name"],
                            })
                    except:
                        pass

        # Deduplicate and format
        seen_entities = {}
        for entity in all_entities:
            name = entity.get("name", "")
            if name and name not in seen_entities:
                seen_entities[name] = entity

        seen_rels = set()
        unique_rels = []
        for rel in all_relationships:
            rel_key = f"{rel['from']}-{rel['type']}-{rel['to']}"
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)

        # Format context
        context_parts = []

        for name, entity in list(seen_entities.items())[:10]:
            desc = entity.get("description", "")
            labels = entity.get("labels", [])
            source = entity.get("source", "")
            label_str = ', '.join(labels) if labels else "Entity"
            context_parts.append(f"[{name} ({label_str})] ({source})\n{desc}")

        if unique_rels:
            context_parts.append("\n[Relationships]")
            for rel in unique_rels[:15]:
                context_parts.append(f"  {rel['from']} -[{rel['type']}]-> {rel['to']}")

        context = "\n".join(context_parts)
        latency = int((time.time() - start) * 1000)

        return RetrievalResult(
            algorithm="hybrid",
            context=context,
            entities_found=len(seen_entities),
            relationships_found=len(unique_rels),
            latency_ms=latency,
            raw_results=list(seen_entities.values()),
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction."""
        # Remove common words
        stopwords = {"what", "is", "the", "a", "an", "and", "or", "how", "does",
                     "do", "are", "in", "to", "of", "for", "with", "on", "at",
                     "between", "from", "that", "which", "this", "its", "their"}

        words = text.lower().replace("?", "").replace(".", "").split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Also extract multi-word phrases
        phrases = []
        if "monetary policy" in text.lower():
            phrases.append("monetary policy")
        if "fiscal policy" in text.lower():
            phrases.append("fiscal policy")
        if "aggregate demand" in text.lower():
            phrases.append("aggregate demand")
        if "aggregate supply" in text.lower():
            phrases.append("aggregate supply")

        return phrases + keywords


# ============================================================================
# LLM-AS-JUDGE EVALUATOR
# ============================================================================

class LLMJudge:
    """GPT-5.2 based evaluation of answer quality against ground truth."""

    def __init__(self):
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set - judge evaluation will be skipped")
            self.judge_llm = None
        else:
            self.judge_llm = ChatOpenAI(
                model=JUDGE_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
            )

    def evaluate(self, question: Dict, generated_answer: str) -> Optional[JudgeScore]:
        """Evaluate a generated answer against the expected answer."""
        if not self.judge_llm:
            return None

        q_text = question.get("question", "")
        expected = question.get("expected_answer", "")
        key_facts = question.get("key_facts", [])

        prompt = f"""You are an expert evaluator assessing the quality of answers to economics questions.

QUESTION: {q_text}

EXPECTED ANSWER (Ground Truth):
{expected}

KEY FACTS that should be mentioned:
{json.dumps(key_facts)}

GENERATED ANSWER (to evaluate):
{generated_answer}

Evaluate the generated answer on these criteria:

1. CORRECTNESS (0.0-1.0): Are the stated facts accurate? Penalize factual errors heavily.
2. COMPLETENESS (0.0-1.0): Does it cover the key facts listed above?
3. RELEVANCE (0.0-1.0): Does it directly answer the question asked?

Respond in JSON format:
{{
    "correctness": 0.0-1.0,
    "completeness": 0.0-1.0,
    "relevance": 0.0-1.0,
    "key_facts_found": ["fact1", "fact2"],
    "key_facts_missing": ["fact3"],
    "explanation": "Brief explanation of scores"
}}"""

        try:
            response = self.judge_llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                parsed = json.loads(content[start_idx:end_idx])

                correctness = float(parsed.get("correctness", 0.5))
                completeness = float(parsed.get("completeness", 0.5))
                relevance = float(parsed.get("relevance", 0.5))

                # Weighted average: correctness most important
                overall = 0.5 * correctness + 0.3 * completeness + 0.2 * relevance

                return JudgeScore(
                    correctness=correctness,
                    completeness=completeness,
                    relevance=relevance,
                    overall=overall,
                    explanation=parsed.get("explanation", ""),
                    key_facts_found=parsed.get("key_facts_found", []),
                    key_facts_missing=parsed.get("key_facts_missing", []),
                )
        except Exception as e:
            logger.debug(f"Judge evaluation error: {e}")

        return None


# ============================================================================
# ABLATION RUNNER
# ============================================================================

class AblationStudy:
    """Run ablation studies on agent configurations."""

    def __init__(self):
        self.retrieval = RetrievalAlgorithms()
        self.judge = LLMJudge()
        self.main_llm = ChatOllama(
            model=MAIN_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
        )
        self.utility_llm = ChatOllama(
            model=UTILITY_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
        )

    def close(self):
        self.retrieval.close()

    def _evaluate_answer(self, question: Dict, answer: str) -> Optional[JudgeScore]:
        """Evaluate answer using GPT-5.2 judge."""
        return self.judge.evaluate(question, answer)

    def _generate_answer(self, question: str, context: str) -> Tuple[str, float, int, int]:
        """Generate answer using main LLM."""
        start = time.time()

        prompt = f"""Based on the context below, answer the question.
Provide a comprehensive answer with citations where possible.

Question: {question}

Context:
{context}

Respond in JSON format:
{{
    "answer": "Your comprehensive answer",
    "confidence": 0.0 to 1.0,
    "citations": ["source1", "source2"]
}}"""

        try:
            response = self.main_llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            try:
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    parsed = json.loads(content[start_idx:end_idx])
                    answer = parsed.get("answer", content)
                    confidence = float(parsed.get("confidence", 0.5))
                    citations = len(parsed.get("citations", []))
                else:
                    answer = content
                    confidence = 0.5
                    citations = 0
            except:
                answer = content
                confidence = 0.5
                citations = 0

            latency = int((time.time() - start) * 1000)
            return answer, confidence, citations, latency

        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return f"Error: {e}", 0.0, 0, latency

    def _compress_context(self, context: str, question: str) -> str:
        """Compress context using utility LLM."""
        if len(context) < 500:
            return context

        prompt = f"""Compress the following context to only the facts relevant to the question.
Preserve entity names exactly. Output bullet points (max 5).

Question: {question}

Context:
{context[:3000]}

Compressed facts:"""

        try:
            response = self.utility_llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except:
            return context[:1000]

    # -------------------------------------------------------------------------
    # Ablation Configurations
    # -------------------------------------------------------------------------

    def run_config_vector_only(self, question: Dict) -> AblationResult:
        """Vector search only, no compression."""
        q_id = question.get("id", "?")
        q_text = question.get("question", "")

        # Retrieve
        retrieval = self.retrieval.vector_search_only(q_text)

        # Generate answer
        answer, confidence, citations, llm_latency = self._generate_answer(q_text, retrieval.context)

        # Judge evaluation
        judge_score = self._evaluate_answer(question, answer)

        return AblationResult(
            config_name="vector_only",
            question_id=q_id,
            answer=answer,
            confidence=confidence,
            citations=citations,
            latency_ms=retrieval.latency_ms + llm_latency,
            context_length=len(retrieval.context),
            entities_retrieved=retrieval.entities_found,
            judge_score=judge_score,
        )

    def run_config_graph_only(self, question: Dict) -> AblationResult:
        """Graph traversal only, no compression."""
        q_id = question.get("id", "?")
        q_text = question.get("question", "")

        # Retrieve
        retrieval = self.retrieval.graph_traversal_only(q_text)

        # Generate answer
        answer, confidence, citations, llm_latency = self._generate_answer(q_text, retrieval.context)

        # Judge evaluation
        judge_score = self._evaluate_answer(question, answer)

        return AblationResult(
            config_name="graph_only",
            question_id=q_id,
            answer=answer,
            confidence=confidence,
            citations=citations,
            latency_ms=retrieval.latency_ms + llm_latency,
            context_length=len(retrieval.context),
            entities_retrieved=retrieval.entities_found,
            judge_score=judge_score,
        )

    def run_config_hybrid(self, question: Dict) -> AblationResult:
        """Hybrid retrieval, no compression."""
        q_id = question.get("id", "?")
        q_text = question.get("question", "")

        # Retrieve
        retrieval = self.retrieval.hybrid_retrieval(q_text)

        # Generate answer
        answer, confidence, citations, llm_latency = self._generate_answer(q_text, retrieval.context)

        # Judge evaluation
        judge_score = self._evaluate_answer(question, answer)

        return AblationResult(
            config_name="hybrid",
            question_id=q_id,
            answer=answer,
            confidence=confidence,
            citations=citations,
            latency_ms=retrieval.latency_ms + llm_latency,
            context_length=len(retrieval.context),
            entities_retrieved=retrieval.entities_found,
            judge_score=judge_score,
        )

    def run_config_hybrid_compressed(self, question: Dict) -> AblationResult:
        """Hybrid retrieval WITH compression."""
        q_id = question.get("id", "?")
        q_text = question.get("question", "")

        # Retrieve
        retrieval = self.retrieval.hybrid_retrieval(q_text)

        # Compress
        compress_start = time.time()
        compressed_context = self._compress_context(retrieval.context, q_text)
        compress_latency = int((time.time() - compress_start) * 1000)

        # Generate answer
        answer, confidence, citations, llm_latency = self._generate_answer(q_text, compressed_context)

        # Judge evaluation
        judge_score = self._evaluate_answer(question, answer)

        return AblationResult(
            config_name="hybrid_compressed",
            question_id=q_id,
            answer=answer,
            confidence=confidence,
            citations=citations,
            latency_ms=retrieval.latency_ms + compress_latency + llm_latency,
            context_length=len(compressed_context),
            entities_retrieved=retrieval.entities_found,
            judge_score=judge_score,
        )

    def run_config_full_agent(self, question: Dict) -> AblationResult:
        """Full agent with planning, compression, and all features."""
        from agent_qa import ReActQAAgent

        q_id = question.get("id", "?")
        q_text = question.get("question", "")

        start = time.time()

        try:
            agent = ReActQAAgent(
                use_retrieval_planning=True,
                compression_enabled=True,
                web_search_enabled=False,  # Disable web for fair comparison
                auto_add_documents=False,
            )

            response = agent.answer_question(q_text)

            latency = int((time.time() - start) * 1000)

            agent.close()

            # Judge evaluation
            judge_score = self._evaluate_answer(question, response.answer)

            return AblationResult(
                config_name="full_agent",
                question_id=q_id,
                answer=response.answer,
                confidence=response.confidence,
                citations=len(response.citations),
                latency_ms=latency,
                context_length=0,  # Not directly accessible
                entities_retrieved=0,
                judge_score=judge_score,
            )
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return AblationResult(
                config_name="full_agent",
                question_id=q_id,
                answer="",
                confidence=0.0,
                citations=0,
                latency_ms=latency,
                context_length=0,
                entities_retrieved=0,
                error=str(e),
            )

    def run_config_no_planning(self, question: Dict) -> AblationResult:
        """Full agent WITHOUT retrieval planning."""
        from agent_qa import ReActQAAgent

        q_id = question.get("id", "?")
        q_text = question.get("question", "")

        start = time.time()

        try:
            agent = ReActQAAgent(
                use_retrieval_planning=False,  # Disable planning
                compression_enabled=True,
                web_search_enabled=False,
                auto_add_documents=False,
            )

            response = agent.answer_question(q_text)

            latency = int((time.time() - start) * 1000)

            agent.close()

            # Judge evaluation
            judge_score = self._evaluate_answer(question, response.answer)

            return AblationResult(
                config_name="no_planning",
                question_id=q_id,
                answer=response.answer,
                confidence=response.confidence,
                citations=len(response.citations),
                latency_ms=latency,
                context_length=0,
                entities_retrieved=0,
                judge_score=judge_score,
            )
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return AblationResult(
                config_name="no_planning",
                question_id=q_id,
                answer="",
                confidence=0.0,
                citations=0,
                latency_ms=latency,
                context_length=0,
                entities_retrieved=0,
                error=str(e),
            )


def run_ablation_study():
    """Run the complete ablation study."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY: agent_qa.py")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Main Model: {MAIN_MODEL}")
    print(f"Utility Model: {UTILITY_MODEL}")
    print(f"Judge Model: {JUDGE_MODEL} (GPT-5.2)")
    print(f"Judge Enabled: {'Yes' if OPENAI_API_KEY else 'No (set OPENAI_API_KEY)'}")
    print("=" * 80)

    study = AblationStudy()

    # Configurations to test
    configs = [
        ("vector_only", study.run_config_vector_only),
        ("graph_only", study.run_config_graph_only),
        ("hybrid", study.run_config_hybrid),
        ("hybrid_compressed", study.run_config_hybrid_compressed),
        ("full_agent", study.run_config_full_agent),
        ("no_planning", study.run_config_no_planning),
    ]

    all_results: Dict[str, List[AblationResult]] = {name: [] for name, _ in configs}

    # Run each configuration on each question
    for q in BENCHMARK_QUESTIONS:
        print(f"\n--- Question: {q['id']} ({q['difficulty']}) ---")
        print(f"    {q['question'][:60]}...")

        for config_name, config_fn in configs:
            print(f"    Testing {config_name}...", end=" ", flush=True)
            try:
                result = config_fn(q)
                all_results[config_name].append(result)
                # Show both self-reported confidence and judge accuracy
                judge_str = ""
                if result.judge_score:
                    judge_str = f", accuracy={result.judge_score.overall:.2f}"
                print(f"conf={result.confidence:.2f}{judge_str}, latency={result.latency_ms}ms")
            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")
                all_results[config_name].append(AblationResult(
                    config_name=config_name,
                    question_id=q["id"],
                    answer="",
                    confidence=0.0,
                    citations=0,
                    latency_ms=0,
                    context_length=0,
                    entities_retrieved=0,
                    error=str(e),
                ))

    study.close()

    # Calculate summaries with judge scores
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)

    summaries = []
    for config_name, results in all_results.items():
        valid_results = [r for r in results if not r.error]
        if valid_results:
            # Calculate judge metrics
            judged_results = [r for r in valid_results if r.judge_score]
            avg_accuracy = sum(r.judge_score.overall for r in judged_results) / len(judged_results) if judged_results else 0
            avg_correctness = sum(r.judge_score.correctness for r in judged_results) / len(judged_results) if judged_results else 0
            avg_completeness = sum(r.judge_score.completeness for r in judged_results) / len(judged_results) if judged_results else 0

            summary = AblationSummary(
                config_name=config_name,
                avg_confidence=sum(r.confidence for r in valid_results) / len(valid_results),
                avg_accuracy=avg_accuracy,
                avg_correctness=avg_correctness,
                avg_completeness=avg_completeness,
                avg_latency_ms=sum(r.latency_ms for r in valid_results) // len(valid_results),
                avg_citations=sum(r.citations for r in valid_results) / len(valid_results),
                avg_context_length=sum(r.context_length for r in valid_results) // len(valid_results),
                success_rate=len(valid_results) / len(results),
                total_questions=len(results),
            )
            summaries.append(summary)

    # Sort by judge accuracy (true metric), fallback to confidence
    summaries.sort(key=lambda s: (s.avg_accuracy, s.avg_confidence), reverse=True)

    # Print main results table
    print(f"\n{'Config':<20} {'Accuracy':<10} {'Correct':<10} {'Complete':<10} {'Conf':<10} {'Latency':<12}")
    print("-" * 90)

    for s in summaries:
        print(f"{s.config_name:<20} {s.avg_accuracy:>7.1%}    {s.avg_correctness:>7.1%}    "
              f"{s.avg_completeness:>7.1%}    {s.avg_confidence:>7.1%}    {s.avg_latency_ms:>8}ms")

    # Self-reported vs Judge comparison
    print("\n" + "-" * 80)
    print("CONFIDENCE vs ACCURACY (Self-Reported vs Judge)")
    print("-" * 80)
    print("\nThis shows whether the model's self-reported confidence matches actual accuracy:")
    print(f"\n{'Config':<20} {'Self-Conf':<12} {'Judge Acc':<12} {'Delta':<12} {'Calibration':<15}")
    print("-" * 80)

    for s in summaries:
        delta = s.avg_confidence - s.avg_accuracy
        if abs(delta) < 0.1:
            calibration = "Well-calibrated"
        elif delta > 0:
            calibration = "Over-confident"
        else:
            calibration = "Under-confident"
        print(f"{s.config_name:<20} {s.avg_confidence:>8.1%}     {s.avg_accuracy:>8.1%}     "
              f"{delta:>+7.1%}      {calibration:<15}")

    # Retrieval algorithm comparison with judge scores
    print("\n" + "-" * 80)
    print("RETRIEVAL ALGORITHM COMPARISON (Judge Accuracy)")
    print("-" * 80)

    retrieval_configs = ["vector_only", "graph_only", "hybrid"]
    print(f"\n{'Algorithm':<20} {'Accuracy':<12} {'Correctness':<12} {'Completeness':<12} {'Latency':<12}")
    print("-" * 80)

    for config in retrieval_configs:
        results = all_results.get(config, [])
        valid = [r for r in results if not r.error]
        judged = [r for r in valid if r.judge_score]
        if judged:
            avg_acc = sum(r.judge_score.overall for r in judged) / len(judged)
            avg_cor = sum(r.judge_score.correctness for r in judged) / len(judged)
            avg_com = sum(r.judge_score.completeness for r in judged) / len(judged)
            avg_lat = sum(r.latency_ms for r in valid) // len(valid)
            print(f"{config:<20} {avg_acc:>8.1%}     {avg_cor:>8.1%}       {avg_com:>8.1%}       {avg_lat:>8}ms")

    # Compression impact with judge scores
    print("\n" + "-" * 80)
    print("COMPRESSION IMPACT (Judge Accuracy)")
    print("-" * 80)

    hybrid_results = [r for r in all_results.get("hybrid", []) if not r.error and r.judge_score]
    compressed_results = [r for r in all_results.get("hybrid_compressed", []) if not r.error and r.judge_score]

    if hybrid_results and compressed_results:
        h_acc = sum(r.judge_score.overall for r in hybrid_results) / len(hybrid_results)
        c_acc = sum(r.judge_score.overall for r in compressed_results) / len(compressed_results)
        h_ctx = sum(r.context_length for r in hybrid_results) // len(hybrid_results)
        c_ctx = sum(r.context_length for r in compressed_results) // len(compressed_results)

        print(f"\n{'Metric':<25} {'Without Compression':<20} {'With Compression':<20}")
        print("-" * 65)
        print(f"{'Judge Accuracy':<25} {h_acc:>15.1%}      {c_acc:>15.1%}")
        print(f"{'Avg Context Length':<25} {h_ctx:>15}      {c_ctx:>15}")
        print(f"{'Compression Ratio':<25} {'N/A':>15}      {c_ctx/h_ctx if h_ctx > 0 else 0:>14.1%}")

    # Per-question breakdown
    print("\n" + "-" * 80)
    print("PER-QUESTION ACCURACY BREAKDOWN")
    print("-" * 80)

    for q in BENCHMARK_QUESTIONS:
        print(f"\n{q['id']} ({q['difficulty']}): {q['question'][:50]}...")
        for config_name in [c[0] for c in configs]:
            results = all_results.get(config_name, [])
            for r in results:
                if r.question_id == q['id'] and r.judge_score:
                    print(f"    {config_name:<20} acc={r.judge_score.overall:.2f} "
                          f"(correct={r.judge_score.correctness:.2f}, complete={r.judge_score.completeness:.2f})")
                    if r.judge_score.key_facts_missing:
                        print(f"        Missing: {', '.join(r.judge_score.key_facts_missing[:3])}")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation_study()
