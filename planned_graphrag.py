"""
Plan-based GraphRAG implementation for agentic retrieval.

This module implements CLaRa-style retrieval with explicit retrieval planning:
1. RetrievalPlan - structured plan for what to retrieve
2. PlannedGraphRAG - executes retrieval plans against Neo4j
3. Context compression for reducing retrieved context to relevant facts
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from json_repair import repair_json

from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_UTILITY_MODEL = os.getenv("OLLAMA_UTILITY_MODEL", "ministral-3:14b")

# FalkorDB configuration
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "wikidata")

from prompts.retrieval_prompts import (
    format_compression_prompt,
    format_pattern_to_cypher_prompt,
    format_observation_compression_prompt,
    format_follow_up_plan_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# FALKORDB DRIVER WRAPPER
# =============================================================================

class FalkorDBDriverWrapper:
    """Wrapper to provide Neo4j-like interface for FalkorDB."""

    def __init__(self, host: str = FALKORDB_HOST, port: int = FALKORDB_PORT, graph_name: str = FALKORDB_GRAPH):
        from falkordb import FalkorDB
        self.client = FalkorDB(host=host, port=port)
        self.graph = self.client.select_graph(graph_name)
        self.graph_name = graph_name

    def close(self):
        pass  # FalkorDB client doesn't require explicit close

    def session(self):
        """Return a context manager that provides session-like interface."""
        return FalkorDBSessionWrapper(self.graph)


class FalkorDBSessionWrapper:
    """Session wrapper for FalkorDB to provide Neo4j-like interface."""

    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, query: str, **params):
        """Run a Cypher query and return results."""
        result = self.graph.query(query, params)
        return FalkorDBResultWrapper(result)


class FalkorDBResultWrapper:
    """Wrapper to provide Neo4j-like result interface for FalkorDB."""

    def __init__(self, result):
        self.result = result
        self.result_set = result.result_set
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self.result_set):
            raise StopIteration
        record = self.result_set[self._index]
        self._index += 1
        return FalkorDBRecordWrapper(record, self.result.header if hasattr(self.result, 'header') else [])

    def single(self):
        """Return single result or None."""
        if self.result_set:
            return FalkorDBRecordWrapper(self.result_set[0], self.result.header if hasattr(self.result, 'header') else [])
        return None


class FalkorDBRecordWrapper:
    """Wrapper to provide Neo4j-like record interface for FalkorDB."""

    def __init__(self, record, header):
        self._record = record
        # FalkorDB header format is [[type, name], ...] - extract just names
        self._header = []
        if header:
            for h in header:
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    self._header.append(h[1])  # Get the name from [type, name]
                else:
                    self._header.append(str(h))

        # Build a lookup dict for named access
        self._data = {}
        for i, h in enumerate(self._header):
            if i < len(self._record):
                val = self._record[i]
                # Convert FalkorDB node to dict
                if hasattr(val, 'properties'):
                    self._data[h] = dict(val.properties)
                elif hasattr(val, 'items'):  # Already dict-like
                    self._data[h] = dict(val)
                else:
                    self._data[h] = val

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._data:
                return self._data[key]
            # Try case-insensitive lookup
            for k, v in self._data.items():
                if str(k).lower() == key.lower():
                    return v
            raise KeyError(key)
        else:
            # Integer index access
            if key < len(self._record):
                val = self._record[key]
                if hasattr(val, 'properties'):
                    return dict(val.properties)
                return val
            raise IndexError(key)

    def get(self, key, default=None):
        """Get with default value support."""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self):
        """Return available keys."""
        return self._data.keys()

    def values(self):
        """Return values."""
        return self._data.values()

    def items(self):
        """Return items."""
        return self._data.items()


def create_driver(backend: str = "neo4j", **kwargs):
    """Factory function to create the appropriate driver.

    Args:
        backend: "neo4j" or "falkordb"
        **kwargs: Connection parameters

    Returns:
        Neo4j driver or FalkorDBDriverWrapper
    """
    if backend.lower() == "falkordb":
        host = kwargs.get('host', FALKORDB_HOST)
        port = kwargs.get('port', FALKORDB_PORT)
        graph_name = kwargs.get('graph_name', FALKORDB_GRAPH)
        return FalkorDBDriverWrapper(host=host, port=port, graph_name=graph_name)
    else:
        neo4j_uri = kwargs.get('uri') or os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
        neo4j_user = kwargs.get('user') or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = kwargs.get('password') or os.getenv("NEO4J_PASSWORD", "password")
        return GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


# =============================================================================
# DATA MODELS
# =============================================================================

class RetrievalPlan(BaseModel):
    """Explicit plan for what information to retrieve."""
    reasoning: str = Field(default="", description="Why this plan will help answer the question")
    information_needs: List[str] = Field(default_factory=list, description="What info is needed")
    entity_targets: List[str] = Field(default_factory=list, description="Specific entities to look up")
    relationship_queries: List[str] = Field(default_factory=list, description="Relationship patterns to find")
    fallback_searches: List[str] = Field(default_factory=list, description="Web searches if graph fails")


class FollowUpQuestion(BaseModel):
    """A follow-up question with reasoning."""
    question: str = Field(description="The follow-up question to search")
    reasoning: str = Field(default="", description="Why this question helps answer the original")
    priority: int = Field(default=1, ge=1, le=3, description="Priority 1=high, 2=medium, 3=low")


class FollowUpPlan(BaseModel):
    """Plan using follow-up questions instead of entity/relationship targets."""
    original_analysis: str = Field(default="", description="Analysis of what the original question is asking")
    reasoning_steps: List[str] = Field(default_factory=list, description="Step-by-step reasoning about what info is needed")
    follow_up_questions: List[FollowUpQuestion] = Field(default_factory=list, description="Follow-up questions to search")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts extracted from reasoning")


@dataclass
class EntityContext:
    """Context retrieved for a single entity."""
    entity_name: str
    entity_data: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    found: bool = False


@dataclass
class RelationshipContext:
    """Context retrieved from a relationship pattern query."""
    pattern: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    cypher_used: str = ""


@dataclass
class GraphContext:
    """Combined context from graph retrieval."""
    entities: List[EntityContext] = field(default_factory=list)
    relationships: List[RelationshipContext] = field(default_factory=list)
    raw_text: str = ""
    compressed_text: str = ""


# =============================================================================
# PLANNED GRAPHRAG
# =============================================================================

class PlannedGraphRAG:
    """
    GraphRAG that executes retrieval plans.

    Implements:
    - Entity-centric retrieval with N-hop traversal
    - Relationship pattern matching via Cypher
    - CLaRa-style context compression
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        backend: str = "neo4j",
        graph_name: str = FALKORDB_GRAPH,
        llm: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        compression_enabled: bool = True,
    ):
        """
        Initialize PlannedGraphRAG.

        Args:
            neo4j_uri: Neo4j connection URI (only used for neo4j backend)
            neo4j_user: Neo4j username (only used for neo4j backend)
            neo4j_password: Neo4j password (only used for neo4j backend)
            backend: Database backend ("neo4j" or "falkordb")
            graph_name: FalkorDB graph name (only used for falkordb backend)
            llm: LLM for compression and pattern conversion
            embedding_model: Embedding model for vector search
            compression_enabled: Whether to compress retrieved context
        """
        self.backend = backend.lower()
        if self.backend == "falkordb":
            self.driver = create_driver(backend="falkordb", graph_name=graph_name)
        else:
            neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
            neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.llm = llm or ChatOllama(
            model=OLLAMA_UTILITY_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
        )
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url=OLLAMA_HOST,
        )
        self.compression_enabled = compression_enabled

    def close(self):
        """Close database connection."""
        self.driver.close()

    def retrieve_with_plan(self, plan: RetrievalPlan, question: str) -> GraphContext:
        """
        Execute a retrieval plan and return combined context.

        Args:
            plan: The retrieval plan to execute
            question: The original question (for compression)

        Returns:
            GraphContext with all retrieved information
        """
        context = GraphContext()

        # 1. Entity-centric retrieval
        for entity_name in plan.entity_targets:
            entity_ctx = self._retrieve_entity(entity_name, max_hops=2)
            context.entities.append(entity_ctx)

        # 2. Relationship pattern queries
        for pattern in plan.relationship_queries:
            rel_ctx = self._execute_pattern(pattern)
            context.relationships.append(rel_ctx)

        # 3. Format raw context
        context.raw_text = self._format_raw_context(context)

        # 4. Compress context if enabled
        if self.compression_enabled and len(context.raw_text) > 1000:
            context.compressed_text = self._compress_context(context.raw_text, question)
        else:
            context.compressed_text = context.raw_text

        return context

    def _retrieve_entity(self, entity_name: str, max_hops: int = 2) -> EntityContext:
        """
        Retrieve entity with N-hop neighborhood.

        Args:
            entity_name: Name of entity to find
            max_hops: Maximum relationship hops to traverse

        Returns:
            EntityContext with entity data and relationships
        """
        ctx = EntityContext(entity_name=entity_name)

        with self.driver.session() as session:
            # Find entity (case-insensitive)
            entity_query = """
            MATCH (e)
            WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
            RETURN e, labels(e) as labels
            LIMIT 1
            """
            result = session.run(entity_query, name=entity_name)
            record = result.single()

            if not record:
                # Try vector search as fallback
                return self._vector_search_entity(entity_name)

            ctx.found = True
            ctx.entity_data = dict(record["e"])
            ctx.entity_data["labels"] = record["labels"]

            # Get relationships - use simple query for FalkorDB compatibility
            if max_hops >= 1:
                # Use simpler 1-hop query that works with both Neo4j and FalkorDB
                rel_query = """
                MATCH (e)-[r]-(other)
                WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
                RETURN DISTINCT
                    type(r) as rel_type,
                    other.name as other_name,
                    other.description as other_desc,
                    labels(other) as other_labels
                LIMIT 30
                """
                rel_result = session.run(rel_query, name=entity_name)

                for rec in rel_result:
                    other = {
                        "name": rec.get("other_name", "Unknown"),
                        "description": rec.get("other_desc", ""),
                        "labels": rec.get("other_labels", []),
                    }
                    ctx.relationships.append({
                        "type": rec["rel_type"],
                        "properties": {},
                        "other_entity": other,
                        "direction": "outgoing",  # Simplified
                    })

        return ctx

    def _vector_search_entity(self, query: str, limit: int = 3) -> EntityContext:
        """Fallback to vector search for entity retrieval."""
        ctx = EntityContext(entity_name=query)

        try:
            embedding = self.embedding_model.embed_query(query)

            with self.driver.session() as session:
                cypher = """
                CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
                YIELD node, score
                WHERE score > 0.7
                RETURN node, labels(node) as labels, score
                ORDER BY score DESC
                """
                result = session.run(cypher, embedding=embedding, limit=limit)

                records = list(result)
                if records:
                    best = records[0]
                    ctx.found = True
                    ctx.entity_data = dict(best["node"])
                    ctx.entity_data["labels"] = best["labels"]
                    ctx.entity_data["vector_score"] = best["score"]

        except Exception as e:
            logger.debug(f"Vector search failed: {e}")

        return ctx

    def _execute_pattern(self, pattern: str) -> RelationshipContext:
        """
        Execute a relationship pattern query.

        Args:
            pattern: Natural language pattern like "Entity -[CAUSES]-> ?"

        Returns:
            RelationshipContext with results
        """
        ctx = RelationshipContext(pattern=pattern)

        # Convert pattern to Cypher
        cypher = self._pattern_to_cypher(pattern)
        ctx.cypher_used = cypher

        if not cypher:
            return ctx

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                for record in result:
                    ctx.results.append(dict(record))
        except Exception as e:
            logger.warning(f"Pattern query failed: {e}")

        return ctx

    def _escape_entity(self, entity: str) -> str:
        """Escape entity names for safe use in Cypher queries."""
        if not entity:
            return ""
        # Escape single quotes and backslashes
        return entity.replace("\\", "\\\\").replace("'", "\\'")

    def _clean_cypher(self, query: str) -> str:
        """Clean Cypher query by removing markdown code blocks."""
        if not query:
            return ""
        query = query.strip()
        # Remove markdown code block markers
        if query.startswith("```cypher"):
            query = query[9:]  # Remove ```cypher
        elif query.startswith("```"):
            query = query[3:]  # Remove ```
        if query.endswith("```"):
            query = query[:-3]  # Remove trailing ```
        return query.strip()

    def _validate_cypher(self, query: str) -> bool:
        """Basic Cypher syntax validation."""
        if not query or not query.strip():
            return False
        # Reject if still has markdown markers after cleaning
        if "```" in query:
            return False
        query_upper = query.upper()
        # Must have MATCH and RETURN
        if "MATCH" not in query_upper:
            return False
        if "RETURN" not in query_upper:
            return False
        # Check balanced parentheses and brackets
        if query.count("(") != query.count(")"):
            return False
        if query.count("[") != query.count("]"):
            return False
        # Check for common syntax issues
        if "''" in query:  # Empty string literals often indicate issues
            return False
        return True

    def _pattern_to_cypher(self, pattern: str) -> str:
        """
        Convert natural language pattern to Cypher query.

        Supports patterns like:
        - "Entity -[REL_TYPE]-> ?"
        - "? -[REL_TYPE]-> Entity"
        - "Entity1 -[?]-> Entity2"
        """
        # Try rule-based conversion first
        cypher = self._rule_based_pattern_to_cypher(pattern)
        if cypher and self._validate_cypher(cypher):
            return cypher

        # Fallback to LLM conversion
        try:
            prompt = format_pattern_to_cypher_prompt(pattern)
            response = self.llm.invoke(prompt)
            cypher = response.content.strip()

            # Clean markdown code blocks from LLM response
            cypher = self._clean_cypher(cypher)

            # Validate the generated Cypher
            if self._validate_cypher(cypher):
                return cypher
            else:
                logger.warning(f"LLM generated invalid Cypher: {cypher[:100]}...")
        except Exception as e:
            logger.warning(f"LLM pattern conversion failed: {e}")

        return ""

    def _rule_based_pattern_to_cypher(self, pattern: str) -> str:
        """Rule-based pattern to Cypher conversion with proper escaping."""
        pattern = pattern.strip()

        # Pattern: "Entity -[REL_TYPE]-> ?"
        match = re.match(r"(.+?)\s*-\[([A-Z_]+)\]->\s*\?", pattern)
        if match:
            entity, rel_type = match.groups()
            entity = self._escape_entity(entity.strip())
            return f"""
            MATCH (a)-[r:{rel_type}]->(b)
            WHERE toLower(a.name) = toLower('{entity}') OR toLower(a.id) = toLower('{entity}')
            RETURN a.name as source, type(r) as rel_type, b.name as target, b.description as target_desc
            LIMIT 20
            """

        # Pattern: "? -[REL_TYPE]-> Entity"
        match = re.match(r"\?\s*-\[([A-Z_]+)\]->\s*(.+)", pattern)
        if match:
            rel_type, entity = match.groups()
            entity = self._escape_entity(entity.strip())
            return f"""
            MATCH (a)-[r:{rel_type}]->(b)
            WHERE toLower(b.name) = toLower('{entity}') OR toLower(b.id) = toLower('{entity}')
            RETURN a.name as source, a.description as source_desc, type(r) as rel_type, b.name as target
            LIMIT 20
            """

        # Pattern: "Entity1 -[?]-> Entity2"
        match = re.match(r"(.+?)\s*-\[\?\]->\s*(.+)", pattern)
        if match:
            entity1, entity2 = match.groups()
            entity1 = self._escape_entity(entity1.strip())
            entity2 = self._escape_entity(entity2.strip())
            return f"""
            MATCH (a)-[r]->(b)
            WHERE (toLower(a.name) = toLower('{entity1}') OR toLower(a.id) = toLower('{entity1}'))
              AND (toLower(b.name) = toLower('{entity2}') OR toLower(b.id) = toLower('{entity2}'))
            RETURN a.name as source, type(r) as rel_type, properties(r) as rel_props, b.name as target
            LIMIT 10
            """

        # Pattern: "Entity -[*N]-> ?" (N-hop)
        match = re.match(r"(.+?)\s*-\[\*(\d+)\]->\s*\?", pattern)
        if match:
            entity, hops = match.groups()
            entity = self._escape_entity(entity.strip())
            return f"""
            MATCH path = (a)-[*1..{hops}]->(b)
            WHERE toLower(a.name) = toLower('{entity}') OR toLower(a.id) = toLower('{entity}')
            WITH a, b, relationships(path) as rels
            RETURN a.name as source,
                   [r in rels | type(r)] as relationship_chain,
                   b.name as target, b.description as target_desc
            LIMIT 30
            """

        return ""

    def _format_raw_context(self, context: GraphContext) -> str:
        """Format raw context from entities and relationships."""
        parts = []

        # Format entities
        for entity_ctx in context.entities:
            if entity_ctx.found:
                entity = entity_ctx.entity_data
                name = entity.get("name", entity.get("id", "Unknown"))
                parts.append(f"=== Entity: {name} ===")

                if entity.get("description"):
                    parts.append(f"Description: {entity['description']}")
                if entity.get("ontology_type"):
                    parts.append(f"Type: {entity['ontology_type']}")

                if entity_ctx.relationships:
                    parts.append(f"Relationships ({len(entity_ctx.relationships)}):")
                    for rel in entity_ctx.relationships[:15]:
                        other = rel["other_entity"]
                        other_name = other.get("name", other.get("id", "?"))
                        direction = "->" if rel["direction"] == "outgoing" else "<-"
                        parts.append(f"  {direction} [{rel['type']}] {other_name}")
                parts.append("")
            else:
                parts.append(f"Entity '{entity_ctx.entity_name}' not found in knowledge graph.")
                parts.append("")

        # Format relationship query results
        for rel_ctx in context.relationships:
            if rel_ctx.results:
                parts.append(f"=== Pattern: {rel_ctx.pattern} ===")
                for result in rel_ctx.results[:10]:
                    result_str = ", ".join(f"{k}: {v}" for k, v in result.items() if v)
                    parts.append(f"  {result_str}")
                parts.append("")

        return "\n".join(parts)

    def _compress_context(self, context: str, question: str) -> str:
        """
        Compress context to relevant facts (CLaRa-style).

        Args:
            context: Raw retrieved context
            question: The original question

        Returns:
            Compressed context with only relevant facts
        """
        try:
            prompt = format_compression_prompt(question, context)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Context compression failed: {e}")
            # Return truncated raw context as fallback
            return context[:2000]

    def compress_observation(self, observation: str, question: str) -> str:
        """
        Compress a single tool observation.

        Args:
            observation: Tool output to compress
            question: The original question

        Returns:
            Compressed observation
        """
        if len(observation) < 500:
            return observation

        try:
            prompt = format_observation_compression_prompt(question, observation)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Observation compression failed: {e}")
            return observation[:500]


# =============================================================================
# RETRIEVAL PLAN PARSER
# =============================================================================

def parse_retrieval_plan(response_text: str) -> RetrievalPlan:
    """
    Parse LLM response into a RetrievalPlan.

    Args:
        response_text: JSON response from the planning LLM

    Returns:
        RetrievalPlan object
    """
    try:
        # Extract JSON from response
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Try to repair malformed JSON before parsing
        try:
            repaired_text = repair_json(text)
            data = json.loads(repaired_text)
        except Exception:
            data = json.loads(text)

        return RetrievalPlan(
            reasoning=data.get("reasoning", ""),
            information_needs=data.get("information_needs", []),
            entity_targets=data.get("entity_targets", []),
            relationship_queries=data.get("relationship_queries", []),
            fallback_searches=data.get("fallback_searches", []),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse retrieval plan: {e}")
        return RetrievalPlan()


# =============================================================================
# IMPROVED GRAPHRAG (Keywords + Vector → Fixed Hop Sampling)
# =============================================================================

class ImprovedGraphRAG:
    """
    Improved GraphRAG that avoids LLM-generated Cypher queries.

    Uses a reliable flow:
    1. Extract keywords from question (via LLM planning)
    2. Vector search for each keyword → top-K entities
    3. Fixed N-hop traversal from each entity (NO dynamic Cypher)
    4. Compress combined context

    This eliminates the failure modes of LLM-generated Cypher:
    - Deprecated syntax (EXISTS())
    - UNION column mismatches
    - Invalid property references
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        backend: str = "neo4j",
        graph_name: str = FALKORDB_GRAPH,
        llm: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        max_hops: int = 2,
        vector_limit: int = 5,
        compression_enabled: bool = True,
    ):
        """
        Initialize ImprovedGraphRAG.

        Args:
            neo4j_uri: Neo4j connection URI (only used for neo4j backend)
            neo4j_user: Neo4j username (only used for neo4j backend)
            neo4j_password: Neo4j password (only used for neo4j backend)
            backend: Database backend ("neo4j" or "falkordb")
            graph_name: FalkorDB graph name (only used for falkordb backend)
            llm: LLM for keyword extraction and compression
            embedding_model: Embedding model for vector search
            max_hops: Maximum hops for graph traversal (1, 2, or 3)
            vector_limit: Maximum entities per keyword from vector search
            compression_enabled: Whether to compress retrieved context
        """
        self.backend = backend.lower()
        if self.backend == "falkordb":
            self.driver = create_driver(backend="falkordb", graph_name=graph_name)
        else:
            neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
            neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.llm = llm or ChatOllama(
            model=OLLAMA_UTILITY_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
        )
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url=OLLAMA_HOST,
        )
        self.max_hops = max_hops
        self.vector_limit = vector_limit
        self.compression_enabled = compression_enabled

    def close(self):
        """Close database connection."""
        self.driver.close()

    def retrieve_with_keywords(self, keywords: List[str], question: str) -> GraphContext:
        """
        Keywords + Vector → Fixed Hop Sampling retrieval.

        Args:
            keywords: Keywords/topics extracted from question
            question: The original question (for compression)

        Returns:
            GraphContext with retrieved entities and relationships
        """
        all_entities = []
        seen_names = set()

        # Step 1: Vector search for each keyword
        for keyword in keywords:
            entities = self._vector_search_entities(keyword, limit=self.vector_limit)
            for entity in entities:
                if entity.entity_name.lower() not in seen_names:
                    all_entities.append(entity)
                    seen_names.add(entity.entity_name.lower())

        logger.info(f"Vector search found {len(all_entities)} unique entities from {len(keywords)} keywords")

        # Step 2: Fixed N-hop traversal from each entity (NO LLM Cypher)
        context = GraphContext()
        for entity in all_entities[:10]:  # Limit seed entities to avoid explosion
            neighbors = self._fixed_hop_traversal(entity.entity_name, hops=self.max_hops)
            entity.relationships = neighbors
            context.entities.append(entity)

        # Step 3: Format raw context
        context.raw_text = self._format_context(context)
        logger.info(f"Raw context size: {len(context.raw_text)} chars")

        # Step 4: Compress if enabled
        if self.compression_enabled and len(context.raw_text) > 1000:
            context.compressed_text = self._compress_context(context.raw_text, question)
            logger.info(f"Compressed to: {len(context.compressed_text)} chars")
        else:
            context.compressed_text = context.raw_text

        return context

    def retrieve_direct(self, question: str) -> GraphContext:
        """
        Direct vector search on question (no keyword extraction).

        Args:
            question: The question to retrieve context for

        Returns:
            GraphContext with retrieved entities
        """
        # Embed the question directly
        entities = self._vector_search_entities(question, limit=self.vector_limit * 2)

        context = GraphContext()
        for entity in entities[:10]:
            neighbors = self._fixed_hop_traversal(entity.entity_name, hops=self.max_hops)
            entity.relationships = neighbors
            context.entities.append(entity)

        context.raw_text = self._format_context(context)

        if self.compression_enabled and len(context.raw_text) > 1000:
            context.compressed_text = self._compress_context(context.raw_text, question)
        else:
            context.compressed_text = context.raw_text

        return context

    def _vector_search_entities(self, query: str, limit: int = 5) -> List[EntityContext]:
        """
        Vector search for entities matching query.

        Args:
            query: Search query (keyword or question)
            limit: Maximum entities to return

        Returns:
            List of EntityContext objects
        """
        try:
            # Embed the query
            query_embedding = self.embedding_model.embed_query(query)

            # Vector search query
            vector_query = """
            CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
            YIELD node, score
            RETURN node.name as name,
                   node.description as description,
                   labels(node) as labels,
                   score
            ORDER BY score DESC
            """

            with self.driver.session() as session:
                result = session.run(
                    vector_query,
                    embedding=query_embedding,
                    limit=limit
                )

                entities = []
                for record in result:
                    entities.append(EntityContext(
                        entity_name=record["name"],
                        entity_data={
                            "description": record["description"],
                            "labels": record["labels"],
                            "score": record["score"],
                        },
                        found=True,
                    ))
                return entities

        except Exception as e:
            logger.warning(f"Vector search failed for '{query}': {e}")
            # Fallback to name-based search
            return self._name_search_entities(query, limit)

    def _name_search_entities(self, query: str, limit: int = 5) -> List[EntityContext]:
        """Fallback name-based entity search."""
        try:
            name_query = """
            MATCH (e)
            WHERE toLower(e.name) CONTAINS toLower($query)
            RETURN e.name as name,
                   e.description as description,
                   labels(e) as labels
            LIMIT $limit
            """

            with self.driver.session() as session:
                result = session.run(name_query, query=query, limit=limit)

                entities = []
                for record in result:
                    entities.append(EntityContext(
                        entity_name=record["name"],
                        entity_data={
                            "description": record["description"],
                            "labels": record["labels"],
                        },
                        found=True,
                    ))
                return entities

        except Exception as e:
            logger.warning(f"Name search failed for '{query}': {e}")
            return []

    def _fixed_hop_traversal(self, entity_name: str, hops: int = 2) -> List[Dict[str, Any]]:
        """
        Fixed template traversal - uses simple 1-hop query for FalkorDB compatibility.

        Args:
            entity_name: Starting entity name
            hops: Number of hops (ignored for FalkorDB compatibility, uses 1-hop)

        Returns:
            List of relationship dictionaries
        """
        # Use simple 1-hop query for FalkorDB compatibility
        query = """
        MATCH (e)-[r]-(neighbor)
        WHERE toLower(e.name) = toLower($name)
        RETURN DISTINCT
            neighbor.name as neighbor_name,
            neighbor.description as neighbor_desc,
            labels(neighbor) as neighbor_labels,
            type(r) as rel_type,
            e.name as source
        LIMIT 30
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, name=entity_name)

                relationships = []
                for record in result:
                    relationships.append({
                        "neighbor_name": record.get("neighbor_name", "Unknown"),
                        "neighbor_desc": record.get("neighbor_desc", ""),
                        "neighbor_labels": record.get("neighbor_labels", []),
                        "rel_type": record.get("rel_type", "RELATED_TO"),
                        "source": record.get("source", entity_name),
                        "target": record.get("neighbor_name", "Unknown"),
                    })
                return relationships

        except Exception as e:
            logger.warning(f"Hop traversal failed for '{entity_name}': {e}")
            return []

    def _format_context(self, context: GraphContext) -> str:
        """Format retrieved context as readable text."""
        parts = []

        for i, entity in enumerate(context.entities, 1):
            entity_part = f"[Entity {i}] {entity.entity_name}"
            if entity.entity_data.get("description"):
                entity_part += f"\n  Description: {entity.entity_data['description']}"
            if entity.entity_data.get("labels"):
                entity_part += f"\n  Type: {', '.join(entity.entity_data['labels'])}"

            # Add relationships
            if entity.relationships:
                entity_part += "\n  Relationships:"
                seen_rels = set()
                for rel in entity.relationships[:10]:  # Limit per entity
                    rel_key = f"{rel['source']}-{rel['rel_type']}-{rel['target']}"
                    if rel_key not in seen_rels:
                        entity_part += f"\n    - {rel['source']} --[{rel['rel_type']}]--> {rel['target']}"
                        seen_rels.add(rel_key)

            parts.append(entity_part)

        return "\n\n".join(parts)

    def _compress_context(self, raw_context: str, question: str) -> str:
        """Compress context to relevant facts."""
        try:
            prompt = format_compression_prompt(question, raw_context)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            # Truncate as fallback
            return raw_context[:2000]

    def compress_observation(self, observation: str, question: str) -> str:
        """Compress a single tool observation."""
        if len(observation) < 500:
            return observation

        try:
            prompt = format_observation_compression_prompt(question, observation)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Observation compression failed: {e}")
            return observation[:500]


# =============================================================================
# FOLLOW-UP PLAN PARSER
# =============================================================================

def parse_follow_up_plan(response_text: str) -> FollowUpPlan:
    """
    Parse LLM response into a FollowUpPlan.

    Args:
        response_text: JSON response from the planning LLM

    Returns:
        FollowUpPlan object
    """
    try:
        # Extract JSON from response
        text = response_text.strip()

        # Handle thinking blocks - extract JSON after </detailed_thinking>
        if "</detailed_thinking>" in text:
            text = text.split("</detailed_thinking>")[-1].strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Try to repair malformed JSON before parsing
        try:
            repaired_text = repair_json(text)
            data = json.loads(repaired_text)
        except Exception:
            data = json.loads(text)

        # Parse follow-up questions
        follow_ups = []
        for fq in data.get("follow_up_questions", []):
            follow_ups.append(FollowUpQuestion(
                question=fq.get("question", ""),
                reasoning=fq.get("reasoning", ""),
                priority=fq.get("priority", 2),
            ))

        return FollowUpPlan(
            original_analysis=data.get("original_analysis", ""),
            reasoning_steps=data.get("reasoning_steps", []),
            follow_up_questions=follow_ups,
            key_concepts=data.get("key_concepts", []),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse follow-up plan: {e}")
        return FollowUpPlan()


# =============================================================================
# FOLLOW-UP GRAPHRAG (Dual Vector Search)
# =============================================================================

class FollowUpGraphRAG:
    """
    GraphRAG using follow-up question planning with dual vector search.

    Strategy:
    1. Generate follow-up questions from original query (with reasoning)
    2. Primary search: Original query with v5_h4 (5 vectors, 4 hops)
    3. Secondary search: Follow-up questions with v3_h2 (3 vectors, 2 hops)
    4. Merge and compress results

    This approach:
    - Uses LLM reasoning to decompose complex questions
    - Avoids brittle entity name extraction
    - Avoids LLM-generated Cypher (uses fixed templates only)
    - Provides broader context through multiple search angles
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        backend: str = "neo4j",
        graph_name: str = FALKORDB_GRAPH,
        planning_llm: Optional[Any] = None,
        utility_llm: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        primary_vector_limit: int = 5,
        primary_max_hops: int = 4,
        secondary_vector_limit: int = 3,
        secondary_max_hops: int = 2,
        compression_enabled: bool = True,
        planning_reasoning: bool = False,
    ):
        """
        Initialize FollowUpGraphRAG.

        Args:
            neo4j_uri: Neo4j connection URI (only used for neo4j backend)
            neo4j_user: Neo4j username (only used for neo4j backend)
            neo4j_password: Neo4j password (only used for neo4j backend)
            backend: Database backend ("neo4j" or "falkordb")
            graph_name: FalkorDB graph name (only used for falkordb backend)
            planning_llm: LLM for follow-up question generation (can have thinking enabled)
            utility_llm: LLM for compression (faster model)
            embedding_model: Embedding model for vector search
            primary_vector_limit: Max entities for original query search (default: 5)
            primary_max_hops: Max hops for original query traversal (default: 4)
            secondary_vector_limit: Max entities per follow-up question (default: 3)
            secondary_max_hops: Max hops for follow-up traversal (default: 2)
            compression_enabled: Whether to compress retrieved context
            planning_reasoning: Whether to use detailed thinking prompt
        """
        self.backend = backend.lower()
        if self.backend == "falkordb":
            self.driver = create_driver(backend="falkordb", graph_name=graph_name)
        else:
            neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
            neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Planning LLM (can be nemotron with thinking enabled)
        self.planning_llm = planning_llm or ChatOllama(
            model=os.getenv("OLLAMA_MAIN_MODEL", "nemotron-3-nano:30b"),
            base_url=OLLAMA_HOST,
            temperature=0,
            num_ctx=8192,
        )

        # Utility LLM for compression (smaller, faster)
        self.utility_llm = utility_llm or ChatOllama(
            model=OLLAMA_UTILITY_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
        )

        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url=OLLAMA_HOST,
        )

        # Search parameters
        self.primary_vector_limit = primary_vector_limit
        self.primary_max_hops = primary_max_hops
        self.secondary_vector_limit = secondary_vector_limit
        self.secondary_max_hops = secondary_max_hops
        self.compression_enabled = compression_enabled
        self.planning_reasoning = planning_reasoning

    def close(self):
        """Close database connection."""
        self.driver.close()

    def generate_follow_up_plan(self, question: str) -> FollowUpPlan:
        """
        Generate follow-up questions for the query.

        Args:
            question: Original question

        Returns:
            FollowUpPlan with reasoning and follow-up questions
        """
        try:
            prompt = format_follow_up_plan_prompt(question, use_thinking=self.planning_reasoning)
            response = self.planning_llm.invoke(prompt)
            plan = parse_follow_up_plan(response.content)

            logger.info(f"Generated follow-up plan: {len(plan.follow_up_questions)} questions, "
                       f"{len(plan.key_concepts)} concepts")
            return plan

        except Exception as e:
            logger.error(f"Follow-up plan generation failed: {e}")
            return FollowUpPlan()

    def retrieve_with_follow_ups(self, question: str, plan: Optional[FollowUpPlan] = None) -> GraphContext:
        """
        Dual vector search: original query (v5_h4) + follow-ups (v3_h2).

        Args:
            question: Original question
            plan: Optional pre-generated plan (if None, generates one)

        Returns:
            GraphContext with merged results
        """
        # Generate plan if not provided
        if plan is None:
            plan = self.generate_follow_up_plan(question)

        context = GraphContext()
        all_entities = []
        seen_names = set()

        # === PRIMARY SEARCH: Original query with v5_h4 ===
        logger.info(f"Primary search: '{question[:50]}...' (v{self.primary_vector_limit}_h{self.primary_max_hops})")
        primary_entities = self._vector_search_entities(question, limit=self.primary_vector_limit)

        for entity in primary_entities:
            if entity.entity_name.lower() not in seen_names:
                # Full traversal for primary entities
                neighbors = self._fixed_hop_traversal(entity.entity_name, hops=self.primary_max_hops)
                entity.relationships = neighbors
                entity.entity_data["search_source"] = "primary"
                all_entities.append(entity)
                seen_names.add(entity.entity_name.lower())

        logger.info(f"Primary search found {len(primary_entities)} entities")

        # === SECONDARY SEARCH: Follow-up questions with v3_h2 ===
        for i, fq in enumerate(plan.follow_up_questions[:3]):  # Max 3 follow-ups
            logger.info(f"Secondary search {i+1}: '{fq.question[:40]}...' (v{self.secondary_vector_limit}_h{self.secondary_max_hops})")
            secondary_entities = self._vector_search_entities(fq.question, limit=self.secondary_vector_limit)

            for entity in secondary_entities:
                if entity.entity_name.lower() not in seen_names:
                    # Shallower traversal for secondary entities
                    neighbors = self._fixed_hop_traversal(entity.entity_name, hops=self.secondary_max_hops)
                    entity.relationships = neighbors
                    entity.entity_data["search_source"] = f"followup_{i+1}"
                    entity.entity_data["followup_question"] = fq.question
                    all_entities.append(entity)
                    seen_names.add(entity.entity_name.lower())

        logger.info(f"Total unique entities: {len(all_entities)}")

        # === ALSO SEARCH KEY CONCEPTS ===
        for concept in plan.key_concepts[:5]:
            if concept.lower() not in seen_names:
                concept_entities = self._vector_search_entities(concept, limit=2)
                for entity in concept_entities:
                    if entity.entity_name.lower() not in seen_names:
                        neighbors = self._fixed_hop_traversal(entity.entity_name, hops=self.secondary_max_hops)
                        entity.relationships = neighbors
                        entity.entity_data["search_source"] = "key_concept"
                        all_entities.append(entity)
                        seen_names.add(entity.entity_name.lower())

        context.entities = all_entities

        # === FORMAT AND COMPRESS ===
        context.raw_text = self._format_context_with_plan(context, plan)
        logger.info(f"Raw context size: {len(context.raw_text)} chars")

        if self.compression_enabled and len(context.raw_text) > 1000:
            context.compressed_text = self._compress_context(context.raw_text, question)
            logger.info(f"Compressed to: {len(context.compressed_text)} chars")
        else:
            context.compressed_text = context.raw_text

        return context

    def _vector_search_entities(self, query: str, limit: int = 5) -> List[EntityContext]:
        """Vector search for entities matching query."""
        try:
            query_embedding = self.embedding_model.embed_query(query)

            vector_query = """
            CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
            YIELD node, score
            RETURN node.name as name,
                   node.description as description,
                   labels(node) as labels,
                   score
            ORDER BY score DESC
            """

            with self.driver.session() as session:
                result = session.run(
                    vector_query,
                    embedding=query_embedding,
                    limit=limit
                )

                entities = []
                for record in result:
                    entities.append(EntityContext(
                        entity_name=record["name"],
                        entity_data={
                            "description": record["description"],
                            "labels": record["labels"],
                            "score": record["score"],
                        },
                        found=True,
                    ))
                return entities

        except Exception as e:
            logger.warning(f"Vector search failed for '{query}': {e}")
            return []

    def _fixed_hop_traversal(self, entity_name: str, hops: int = 2) -> List[Dict[str, Any]]:
        """Fixed template traversal - uses simple 1-hop query for FalkorDB compatibility."""
        # Use simple 1-hop query for FalkorDB compatibility
        query = """
        MATCH (e)-[r]-(neighbor)
        WHERE toLower(e.name) = toLower($name)
        RETURN DISTINCT
            neighbor.name as neighbor_name,
            neighbor.description as neighbor_desc,
            labels(neighbor) as neighbor_labels,
            type(r) as rel_type,
            e.name as source
        LIMIT 30
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, name=entity_name)

                relationships = []
                for record in result:
                    relationships.append({
                        "neighbor_name": record.get("neighbor_name", "Unknown"),
                        "neighbor_desc": record.get("neighbor_desc", ""),
                        "neighbor_labels": record.get("neighbor_labels", []),
                        "rel_type": record.get("rel_type", "RELATED_TO"),
                        "source": record.get("source", entity_name),
                        "target": record.get("neighbor_name", "Unknown"),
                    })
                return relationships

        except Exception as e:
            logger.warning(f"Hop traversal failed for '{entity_name}': {e}")
            return []

    def _format_context_with_plan(self, context: GraphContext, plan: FollowUpPlan) -> str:
        """Format retrieved context with plan reasoning."""
        parts = []

        # Include reasoning if available
        if plan.reasoning_steps:
            parts.append("=== REASONING ===")
            for i, step in enumerate(plan.reasoning_steps, 1):
                parts.append(f"{i}. {step}")
            parts.append("")

        # Group entities by source
        primary = [e for e in context.entities if e.entity_data.get("search_source") == "primary"]
        secondary = [e for e in context.entities if "followup" in str(e.entity_data.get("search_source", ""))]
        concepts = [e for e in context.entities if e.entity_data.get("search_source") == "key_concept"]

        if primary:
            parts.append("=== PRIMARY RESULTS ===")
            parts.append(self._format_entity_list(primary))

        if secondary:
            parts.append("=== FOLLOW-UP RESULTS ===")
            parts.append(self._format_entity_list(secondary))

        if concepts:
            parts.append("=== KEY CONCEPT RESULTS ===")
            parts.append(self._format_entity_list(concepts))

        return "\n".join(parts)

    def _format_entity_list(self, entities: List[EntityContext]) -> str:
        """Format a list of entities."""
        parts = []
        for entity in entities:
            entity_part = f"[{entity.entity_name}]"
            if entity.entity_data.get("description"):
                entity_part += f"\n  {entity.entity_data['description']}"

            if entity.relationships:
                entity_part += "\n  Relationships:"
                seen_rels = set()
                for rel in entity.relationships[:8]:
                    rel_key = f"{rel['source']}-{rel['rel_type']}-{rel['target']}"
                    if rel_key not in seen_rels:
                        entity_part += f"\n    - {rel['source']} --[{rel['rel_type']}]--> {rel['target']}"
                        seen_rels.add(rel_key)

            parts.append(entity_part)
        return "\n\n".join(parts)

    def _compress_context(self, raw_context: str, question: str) -> str:
        """Compress context to relevant facts."""
        try:
            prompt = format_compression_prompt(question, raw_context)
            response = self.utility_llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return raw_context[:2000]

    def compress_observation(self, observation: str, question: str) -> str:
        """Compress a single tool observation."""
        if len(observation) < 500:
            return observation

        try:
            prompt = format_observation_compression_prompt(question, observation)
            response = self.utility_llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Observation compression failed: {e}")
            return observation[:500]


# =============================================================================
# CLI TESTING
# =============================================================================

def main():
    """Test the PlannedGraphRAG."""
    import argparse

    parser = argparse.ArgumentParser(description="Test PlannedGraphRAG")
    parser.add_argument("--question", "-q", required=True, help="Question to test")
    args = parser.parse_args()

    from langchain_openai import ChatOpenAI
    from prompts.retrieval_prompts import format_retrieval_plan_prompt

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graphrag = PlannedGraphRAG(llm=llm)

    try:
        # Generate retrieval plan
        plan_prompt = format_retrieval_plan_prompt(args.question)
        plan_response = llm.invoke(plan_prompt)
        plan = parse_retrieval_plan(plan_response.content)

        print("=== RETRIEVAL PLAN ===")
        print(f"Reasoning: {plan.reasoning}")
        print(f"Entity targets: {plan.entity_targets}")
        print(f"Relationship queries: {plan.relationship_queries}")
        print(f"Fallback searches: {plan.fallback_searches}")
        print()

        # Execute plan
        context = graphrag.retrieve_with_plan(plan, args.question)

        print("=== RAW CONTEXT ===")
        print(context.raw_text[:2000])
        print()

        print("=== COMPRESSED CONTEXT ===")
        print(context.compressed_text)

    finally:
        graphrag.close()


if __name__ == "__main__":
    main()
