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

from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_UTILITY_MODEL = os.getenv("OLLAMA_UTILITY_MODEL", "ministral-3:14b")

from prompts.retrieval_prompts import (
    format_compression_prompt,
    format_pattern_to_cypher_prompt,
    format_observation_compression_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        llm: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        compression_enabled: bool = True,
    ):
        """
        Initialize PlannedGraphRAG.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            llm: LLM for compression and pattern conversion
            embedding_model: Embedding model for vector search
            compression_enabled: Whether to compress retrieved context
        """
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

            # Get relationships up to max_hops
            if max_hops >= 1:
                rel_query = f"""
                MATCH (e)-[r*1..{max_hops}]-(other)
                WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
                WITH e, r as rels, other
                UNWIND rels as rel
                RETURN DISTINCT
                    type(rel) as rel_type,
                    properties(rel) as rel_props,
                    other,
                    labels(other) as other_labels,
                    startNode(rel) = e as is_outgoing
                LIMIT 30
                """
                rel_result = session.run(rel_query, name=entity_name)

                for rec in rel_result:
                    other = dict(rec["other"])
                    other["labels"] = rec["other_labels"]
                    ctx.relationships.append({
                        "type": rec["rel_type"],
                        "properties": dict(rec["rel_props"]) if rec["rel_props"] else {},
                        "other_entity": other,
                        "direction": "outgoing" if rec["is_outgoing"] else "incoming",
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
        if cypher:
            return cypher

        # Fallback to LLM conversion
        try:
            prompt = format_pattern_to_cypher_prompt(pattern)
            response = self.llm.invoke(prompt)
            cypher = response.content.strip()

            # Basic validation
            if cypher.upper().startswith("MATCH"):
                return cypher
        except Exception as e:
            logger.warning(f"LLM pattern conversion failed: {e}")

        return ""

    def _rule_based_pattern_to_cypher(self, pattern: str) -> str:
        """Rule-based pattern to Cypher conversion."""
        pattern = pattern.strip()

        # Pattern: "Entity -[REL_TYPE]-> ?"
        match = re.match(r"(.+?)\s*-\[([A-Z_]+)\]->\s*\?", pattern)
        if match:
            entity, rel_type = match.groups()
            entity = entity.strip()
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
            entity = entity.strip()
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
            entity1, entity2 = entity1.strip(), entity2.strip()
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
            entity = entity.strip()
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
