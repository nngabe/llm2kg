#!/usr/bin/env python3
"""
LangGraph ReAct Agent for Question Answering.

This agent performs question answering using:
- Hybrid GraphRAG + document retrieval from Neo4j
- Web search for external information when needed
- ReAct-style reasoning with Thought-Action-Observation loops
- Citation tracking with source attribution
- Automatic document ingestion from web searches
"""

import os
import re
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Tuple
from enum import Enum

from json_repair import repair_json

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from neo4j import GraphDatabase

from langgraph.graph import StateGraph, START, END

from planned_graphrag import (
    PlannedGraphRAG,
    ImprovedGraphRAG,
    FollowUpGraphRAG,
    RetrievalPlan,
    FollowUpPlan,
    parse_retrieval_plan,
    parse_follow_up_plan,
)
from prompts.retrieval_prompts import format_retrieval_plan_prompt, format_follow_up_plan_prompt

# KG extraction imports for auto document ingestion
from agent_skb import KnowledgeGraphAgent, AgentState as SKBAgentState

# Uncertainty metrics
from uncertainty_metrics import UncertaintyCalculator, UncertaintyScores

# Zep temporal query utilities
from knowledge_graph.temporal_queries import TemporalQueryBuilder, TemporalEntityManager

# Wikipedia search
from langchain_community.document_loaders import WikipediaLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# FalkorDB configuration
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "wikidata")

# Reranker configuration (zerank-2 CrossEncoder server)
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_URL = os.getenv("RERANKER_URL", "http://host.docker.internal:8081/rerank")
RERANKER_TIMEOUT = int(os.getenv("RERANKER_TIMEOUT", "120"))  # seconds for batch rerank
RERANKER_CANDIDATE_MULTIPLIER = 4  # Retrieve N*multiplier candidates, rerank to top N

# Ollama model configuration (benchmark winner: Pair D)
# Main model: Complex reasoning, ReAct loop, answer synthesis, compression, planning
OLLAMA_MAIN_MODEL = os.getenv("OLLAMA_MAIN_MODEL", "nemotron-3-nano:30b")

MAX_ITERATIONS = 5
CONTEXT_WINDOW_SIZE = 8000  # Max tokens for context


# --- DATA MODELS ---

class ToolCall(BaseModel):
    """A tool call made by the agent."""
    tool_name: str = Field(description="Name of the tool called")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool_name: str
    status: str = Field(description="'success', 'error', or 'not_found'")
    result: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Citation(BaseModel):
    """Citation for a piece of information."""
    source_type: Literal["graph", "document", "web_search"] = Field(
        description="Type of source"
    )
    source_id: str = Field(description="ID or URL of the source")
    source_title: Optional[str] = Field(default=None)
    trust_level: str = Field(default="unknown")
    excerpt: str = Field(default="", description="Relevant excerpt from source")


class ThoughtStep(BaseModel):
    """A single thought step in ReAct reasoning."""
    thought: str = Field(description="Agent's reasoning")
    action: Optional[ToolCall] = Field(default=None)
    observation: Optional[str] = Field(default=None)


class QAResponse(BaseModel):
    """Final response from the Q&A agent."""
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    external_info_used: bool = Field(
        default=False,
        description="Whether web search was used for this answer"
    )
    reasoning_steps: List[ThoughtStep] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: Optional[UncertaintyScores] = Field(
        default=None,
        description="Detailed uncertainty metrics (perplexity, semantic entropy, embedding consistency)"
    )


class ContextItem(BaseModel):
    """An item in the retrieved context."""
    source_type: Literal["entity", "relationship", "document", "web"]
    content: str
    source_id: str
    relevance_score: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- AGENT STATE ---

class QAAgentState(BaseModel):
    """State for the ReAct Q&A agent."""
    # Input
    question: str = ""

    # Retrieval planning (CLaRa-style)
    retrieval_plan: Optional[RetrievalPlan] = None
    compressed_context: str = ""

    # Retrieved context
    context: List[ContextItem] = Field(default_factory=list)
    context_formatted: str = ""

    # ReAct loop state
    thought_history: List[ThoughtStep] = Field(default_factory=list)
    current_thought: str = ""
    pending_action: Optional[ToolCall] = None
    last_observation: str = ""

    # Control flow
    iteration_count: int = 0
    max_iterations: int = MAX_ITERATIONS
    should_continue: bool = True
    ready_to_answer: bool = False

    # Output
    final_answer: str = ""
    citations: List[Citation] = Field(default_factory=list)
    external_info_used: bool = False
    confidence: float = 0.0
    uncertainty_scores: Optional[UncertaintyScores] = Field(
        default=None,
        description="Detailed uncertainty metrics"
    )

    # Error handling
    error: Optional[str] = None


# --- PROMPTS ---

SYSTEM_PROMPT = """You are a knowledgeable research assistant with access to a knowledge graph, Wikipedia, and web search.
Your goal is to answer questions accurately using available information sources.

Available tools:
1. graph_lookup(entity_name) - Look up an entity in the knowledge graph (includes temporal provenance)
2. wiki_search(query) - Search Wikipedia for encyclopedic information
3. web_search(query) - Search the web for information (use only when graph and wiki don't have the answer)
4. cypher_query(query) - Run a Cypher query on the knowledge graph
5. entity_resolve(entity_name, context) - Disambiguate an entity name
6. entity_timeline(entity_name) - Get temporal history of when facts about an entity were observed

Guidelines:
- ALWAYS try the knowledge graph first before using external sources
- Use wiki_search for general knowledge, definitions, and encyclopedic facts
- Use web_search for current events or when wiki doesn't have the answer
- Use entity_timeline to understand when facts about an entity were discovered over time
- When using external sources, clearly note that external information was used
- Provide citations for all claims
- If information conflicts between sources, note the discrepancy
- Be honest about uncertainty"""


THINK_PROMPT = """Based on the question and any previous observations, decide what to do next.

Question: {question}

Previous steps:
{thought_history}

Current context from retrievals:
{context}

Think step by step:
1. What information do I need to answer this question?
2. What do I already know from the context?
3. What is still missing?
4. What tool should I use next, or am I ready to answer?

Respond in this JSON format:
{{
    "thought": "Your reasoning about what to do next",
    "ready_to_answer": true/false,
    "action": {{
        "tool_name": "graph_lookup|wiki_search|web_search|cypher_query|entity_resolve|entity_timeline|none",
        "arguments": {{"key": "value"}}
    }}
}}

If ready_to_answer is true, action should have tool_name "none"."""


ANSWER_PROMPT = """Based on the retrieved context, provide a comprehensive answer to the question.

Question: {question}

Retrieved Context:
{context}

Reasoning Steps:
{thought_history}

Guidelines:
1. Answer the question directly and thoroughly
2. Cite sources using [Source: X] notation
3. If web search was used, explicitly state: "According to external sources..."
4. Note any conflicting information or uncertainties

Respond in this JSON format:
{{
    "answer": "Your comprehensive answer with citations",
    "citations": [
        {{
            "source_type": "graph|document|web_search",
            "source_id": "entity name or URL",
            "source_title": "optional title",
            "excerpt": "relevant excerpt"
        }}
    ]
}}"""


# --- NEO4J LOADER ---

class Neo4jQALoader:
    """Neo4j loader for Q&A operations."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        embedding_model: Optional[Any] = None,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url=OLLAMA_HOST,
            num_ctx=4096,
        )

    def close(self):
        self.driver.close()

    def vector_search(
        self,
        query: str,
        limit: int = 5,
        include_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search entities by vector similarity."""
        try:
            embedding = self.embedding_model.embed_query(query)

            with self.driver.session() as session:
                cypher = """
                CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
                YIELD node, score
                RETURN node, labels(node) as labels, score
                ORDER BY score DESC
                """
                result = session.run(cypher, embedding=embedding, limit=limit)

                entities = []
                for record in result:
                    node = dict(record["node"])
                    node["labels"] = record["labels"]
                    node["score"] = record["score"]
                    entities.append(node)

                return entities
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def get_entity_with_relationships(
        self,
        entity_name: str,
        max_relationships: int = 20,
    ) -> Dict[str, Any]:
        """Get entity and its relationships."""
        with self.driver.session() as session:
            # Find entity
            entity_query = """
            MATCH (e)
            WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
            RETURN e, labels(e) as labels
            LIMIT 1
            """
            result = session.run(entity_query, name=entity_name)
            record = result.single()

            if not record:
                return {"found": False, "entity_name": entity_name}

            entity = dict(record["e"])
            entity["labels"] = record["labels"]

            # Get relationships
            rel_query = """
            MATCH (e)-[r]-(other)
            WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
            RETURN type(r) as rel_type,
                   properties(r) as rel_props,
                   other,
                   labels(other) as other_labels,
                   startNode(r) = e as is_outgoing
            LIMIT $limit
            """
            rel_result = session.run(rel_query, name=entity_name, limit=max_relationships)

            relationships = []
            for rec in rel_result:
                other = dict(rec["other"])
                other["labels"] = rec["other_labels"]
                relationships.append({
                    "type": rec["rel_type"],
                    "properties": dict(rec["rel_props"]) if rec["rel_props"] else {},
                    "other_entity": other,
                    "direction": "outgoing" if rec["is_outgoing"] else "incoming",
                })

            return {
                "found": True,
                "entity": entity,
                "relationships": relationships,
            }

    def search_documents(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search document nodes by vector similarity."""
        try:
            embedding = self.embedding_model.embed_query(query)

            with self.driver.session() as session:
                # Check if document index exists
                cypher = """
                CALL db.index.vector.queryNodes('document_embeddings', $limit, $embedding)
                YIELD node, score
                WHERE 'Document' IN labels(node)
                RETURN node, score
                ORDER BY score DESC
                """
                result = session.run(cypher, embedding=embedding, limit=limit)

                documents = []
                for record in result:
                    doc = dict(record["node"])
                    doc["score"] = record["score"]
                    documents.append(doc)

                return documents
        except Exception as e:
            logger.debug(f"Document search not available: {e}")
            return []

    def add_document(
        self,
        url: str,
        title: str,
        content: str,
        domain: str,
        trust_level: str = "medium",
        source_type: str = "web_search",
    ) -> str:
        """Add a document to the database."""
        import uuid
        doc_id = str(uuid.uuid4())

        try:
            embedding = self.embedding_model.embed_query(f"{title}: {content[:500]}")

            with self.driver.session() as session:
                cypher = """
                CREATE (d:Document {
                    id: $id,
                    url: $url,
                    title: $title,
                    content: $content,
                    domain: $domain,
                    trust_level: $trust_level,
                    source_type: $source_type,
                    embedding: $embedding,
                    added_at: datetime()
                })
                RETURN d.id as id
                """
                result = session.run(
                    cypher,
                    id=doc_id,
                    url=url,
                    title=title,
                    content=content[:10000],  # Limit content size
                    domain=domain,
                    trust_level=trust_level,
                    source_type=source_type,
                    embedding=embedding,
                )
                record = result.single()
                return record["id"] if record else doc_id
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return doc_id

    def run_cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a Cypher query."""
        params = params or {}
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]


# --- FALKORDB LOADER ---

class FalkorDBQALoader:
    """FalkorDB loader for Q&A operations (alternative to Neo4jQALoader)."""

    def __init__(
        self,
        host: str = FALKORDB_HOST,
        port: int = FALKORDB_PORT,
        graph_name: str = FALKORDB_GRAPH,
        embedding_model: Optional[Any] = None,
        reranker_enabled: bool = RERANKER_ENABLED,
    ):
        from falkordb import FalkorDB
        self.client = FalkorDB(host=host, port=port)
        self.graph = self.client.select_graph(graph_name)
        self.graph_name = graph_name
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url=OLLAMA_HOST,
            num_ctx=4096,
        )
        self.reranker_enabled = reranker_enabled
        self._reranker_available = None  # Lazy check on first use

    def close(self):
        """Close the FalkorDB connection."""
        pass  # FalkorDB client doesn't require explicit close

    def vector_search(
        self,
        query: str,
        limit: int = 5,
        include_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search entities by vector similarity using FalkorDB."""
        try:
            embedding = self.embedding_model.embed_query(query)

            # Try FalkorDB vector search
            # FalkorDB uses a different vector search syntax
            cypher = """
            CALL db.idx.vector.queryNodes('entity_embeddings', $limit, vecf32($embedding))
            YIELD node, score
            RETURN node, labels(node) as labels, score
            ORDER BY score DESC
            """
            result = self.graph.query(cypher, {'embedding': embedding, 'limit': limit})

            entities = []
            for record in result.result_set:
                node = record[0].properties if hasattr(record[0], 'properties') else dict(record[0])
                entities.append({
                    **node,
                    'labels': record[1] if len(record) > 1 else [],
                    'score': record[2] if len(record) > 2 else 1.0,
                })
            return entities

        except Exception as e:
            logger.debug(f"FalkorDB vector search failed: {e}")
            # Fallback to text-based search
            return self._text_search_entities(query, limit)

    def _text_search_entities(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback text-based entity search for FalkorDB."""
        try:
            cypher = """
            MATCH (e)
            WHERE toLower(e.name) CONTAINS toLower($query)
            RETURN e, labels(e) as labels
            LIMIT $limit
            """
            result = self.graph.query(cypher, {'query': query, 'limit': limit})

            entities = []
            for record in result.result_set:
                node = record[0].properties if hasattr(record[0], 'properties') else dict(record[0])
                entities.append({
                    **node,
                    'labels': record[1] if len(record) > 1 else [],
                    'score': 0.5,  # Default score for text search
                })
            return entities

        except Exception as e:
            logger.debug(f"Text search failed: {e}")
            return []

    def get_entity_with_relationships(
        self,
        entity_name: str,
        max_relationships: int = 20,
        include_temporal: bool = True,
    ) -> Dict[str, Any]:
        """Get entity and its relationships from FalkorDB.

        Args:
            entity_name: Name of the entity to look up.
            max_relationships: Maximum number of relationships to return.
            include_temporal: Whether to include temporal info (Episode count, Community).

        Returns:
            Dictionary with entity, relationships, and optionally temporal metadata.
        """
        # Find entity with optional temporal filtering
        if include_temporal:
            entity_query = """
            MATCH (e:Entity)
            WHERE (toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name))
              AND (e.fact_status IS NULL OR e.fact_status = 'active')
            RETURN e, labels(e) as labels
            LIMIT 1
            """
        else:
            entity_query = """
            MATCH (e)
            WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
            RETURN e, labels(e) as labels
            LIMIT 1
            """
        result = self.graph.query(entity_query, {'name': entity_name})

        if not result.result_set:
            return {"found": False, "entity_name": entity_name}

        record = result.result_set[0]
        entity = record[0].properties if hasattr(record[0], 'properties') else dict(record[0])
        entity["labels"] = record[1] if len(record) > 1 else []

        # Get relationships (filter related entities by fact_status if temporal)
        if include_temporal:
            rel_query = """
            MATCH (e:Entity)-[r]-(other)
            WHERE (toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name))
              AND (e.fact_status IS NULL OR e.fact_status = 'active')
              AND (other.fact_status IS NULL OR other.fact_status = 'active')
            RETURN type(r) as rel_type,
                   properties(r) as rel_props,
                   other,
                   labels(other) as other_labels,
                   startNode(r) = e as is_outgoing
            LIMIT $limit
            """
        else:
            rel_query = """
            MATCH (e)-[r]-(other)
            WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
            RETURN type(r) as rel_type,
                   properties(r) as rel_props,
                   other,
                   labels(other) as other_labels,
                   startNode(r) = e as is_outgoing
            LIMIT $limit
            """
        rel_result = self.graph.query(rel_query, {'name': entity_name, 'limit': max_relationships})

        relationships = []
        for rec in rel_result.result_set:
            other_node = rec[2].properties if hasattr(rec[2], 'properties') else dict(rec[2])
            other_node["labels"] = rec[3] if len(rec) > 3 else []
            relationships.append({
                "type": rec[0],
                "properties": dict(rec[1]) if rec[1] else {},
                "other_entity": other_node,
                "direction": "outgoing" if rec[4] else "incoming",
            })

        result_dict = {
            "found": True,
            "entity": entity,
            "relationships": relationships,
        }

        # Add temporal metadata if requested
        if include_temporal:
            # Get Episode count
            try:
                name_escaped = entity_name.replace("'", "\\'")
                ep_count_result = self.graph.query(f"""
                    MATCH (ep:Episode)-[:CONTAINS]->(e:Entity)
                    WHERE toLower(e.name) = toLower('{name_escaped}')
                    RETURN count(ep) as episode_count
                """)
                if ep_count_result.result_set:
                    result_dict["episode_count"] = ep_count_result.result_set[0][0]
            except Exception:
                result_dict["episode_count"] = 0

            # Get Community info
            try:
                community_result = self.graph.query(f"""
                    MATCH (e:Entity)-[:BELONGS_TO]->(c:Community)
                    WHERE toLower(e.name) = toLower('{name_escaped}')
                    RETURN c.name as community_name, c.community_id as community_id
                    LIMIT 1
                """)
                if community_result.result_set:
                    result_dict["community"] = {
                        "name": community_result.result_set[0][0],
                        "id": community_result.result_set[0][1],
                    }
            except Exception:
                pass

        return result_dict

    def search_documents(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search DocumentChunk nodes by text matching in FalkorDB.

        Two-stage retrieval pipeline:
        1. CANDIDATE RETRIEVAL: Broad keyword search to gather candidates
           - Uses multiple keywords from query
           - Filters for technical content (MW, %, efficiency, etc.)
           - Retrieves 4x the final limit to give reranker good options

        2. RERANKING: Cross-encoder model (zerank-2) scores each candidate
           - Understands semantic relevance beyond keyword matching
           - Prioritizes exact product matches and specific specs
           - Gracefully falls back to keyword scoring if unavailable
        """
        try:
            # Extract keywords from query (words > 2 chars, excluding common words)
            stop_words = {'the', 'is', 'at', 'which', 'on', 'what', 'how', 'does', 'are', 'was', 'for', 'and', 'gas', 'turbine'}
            keywords = [w.strip("'\",.?!") for w in query.lower().split()
                       if len(w) > 2 and w.lower() not in stop_words]

            if not keywords:
                keywords = [query.lower()]

            all_docs = {}  # chunk_id -> doc with score

            # Candidate pool size: retrieve more for reranking
            candidate_limit = limit * RERANKER_CANDIDATE_MULTIPLIER if self.reranker_enabled else limit

            # First pass: search for main keywords in chunks with technical specs
            for kw in keywords[:4]:  # Use up to 4 keywords for broader coverage
                cypher = """
                MATCH (d:DocumentChunk)
                WHERE d.source_type = 'webpage'
                AND toLower(d.content) CONTAINS toLower($keyword)
                AND (
                    d.content CONTAINS 'MW' OR
                    d.content CONTAINS 'efficiency' OR
                    d.content CONTAINS 'min.' OR
                    d.content CONTAINS 'min ' OR
                    d.content CONTAINS 'output' OR
                    d.content CONTAINS 'hydrogen' OR
                    d.content CONTAINS 'kV' OR
                    d.content CONTAINS 'Net ' OR
                    d.content CONTAINS 'capability' OR
                    d.content CONTAINS 'start time' OR
                    d.chunk_type IN ['cluster', 'root']
                )
                RETURN d, d.content as content, d.source_url as url
                LIMIT $limit
                """
                result = self.graph.query(cypher, {'keyword': kw, 'limit': candidate_limit})

                for record in result.result_set:
                    doc = record[0].properties if hasattr(record[0], 'properties') else dict(record[0])
                    doc["content"] = record[1] if len(record) > 1 else doc.get("content", "")
                    doc["source_url"] = record[2] if len(record) > 2 else doc.get("source_url", "")
                    chunk_id = doc.get("chunk_id", id(doc))

                    if chunk_id in all_docs:
                        all_docs[chunk_id]["score"] += 1
                    else:
                        # Higher score for cluster/root summaries
                        base_score = 1.5 if doc.get("chunk_type") in ["cluster", "root"] else 1.0
                        doc["score"] = base_score
                        all_docs[chunk_id] = doc

            # Second pass: broader search for product-specific chunks that may not match all keywords
            # This catches chunks like "5 min. start time" that don't contain "fast"
            product_terms = [kw for kw in keywords if len(kw) > 4]  # Likely product names
            for term in product_terms[:2]:
                cypher = """
                MATCH (d:DocumentChunk)
                WHERE d.source_type = 'webpage'
                AND toLower(d.content) CONTAINS toLower($term)
                AND (d.content CONTAINS 'MW' OR d.content CONTAINS '%' OR d.content CONTAINS 'min.')
                RETURN d, d.content as content, d.source_url as url
                LIMIT $limit
                """
                result = self.graph.query(cypher, {'term': term, 'limit': candidate_limit // 2})

                for record in result.result_set:
                    doc = record[0].properties if hasattr(record[0], 'properties') else dict(record[0])
                    doc["content"] = record[1] if len(record) > 1 else doc.get("content", "")
                    doc["source_url"] = record[2] if len(record) > 2 else doc.get("source_url", "")
                    chunk_id = doc.get("chunk_id", id(doc))

                    if chunk_id not in all_docs:
                        doc["score"] = 0.8  # Product match without keyword match
                        all_docs[chunk_id] = doc

            # Third pass: search Wikipedia for relevant content
            if len(all_docs) < candidate_limit:
                for kw in keywords[:2]:
                    cypher = """
                    MATCH (d:DocumentChunk)
                    WHERE d.source_type = 'wikipedia' AND toLower(d.content) CONTAINS toLower($keyword)
                    RETURN d, d.content as content, d.source_url as url
                    LIMIT $limit
                    """
                    result = self.graph.query(cypher, {'keyword': kw, 'limit': limit})

                    for record in result.result_set:
                        doc = record[0].properties if hasattr(record[0], 'properties') else dict(record[0])
                        doc["content"] = record[1] if len(record) > 1 else doc.get("content", "")
                        doc["source_url"] = record[2] if len(record) > 2 else doc.get("source_url", "")
                        chunk_id = doc.get("chunk_id", id(doc))

                        if chunk_id not in all_docs:
                            doc["score"] = 0.3
                            all_docs[chunk_id] = doc

            # Convert to list and sort by keyword score
            documents = list(all_docs.values())
            documents.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Apply reranking if enabled (takes top candidates, returns top `limit`)
            if self.reranker_enabled and len(documents) > limit:
                # Send top candidates to reranker
                candidates = documents[:candidate_limit]
                return self._rerank_documents(query, candidates, limit)
            else:
                return documents[:limit]

        except Exception as e:
            logger.debug(f"Document search not available: {e}")
            return []

    def _check_reranker_available(self) -> bool:
        """Check if the reranker service is available (cached)."""
        if self._reranker_available is not None:
            return self._reranker_available

        try:
            # Health check endpoint on the CrossEncoder server
            health_url = RERANKER_URL.replace("/rerank", "/health")
            response = requests.get(health_url, timeout=15)
            data = response.json()
            self._reranker_available = response.status_code == 200 and data.get("model_loaded", False)
        except Exception as e:
            logger.debug(f"Reranker health check failed: {e}")
            self._reranker_available = False

        if self._reranker_available:
            logger.info(f"Reranker available at {RERANKER_URL}")
        else:
            logger.warning(f"Reranker not available at {RERANKER_URL}, using keyword-only scoring")

        return self._reranker_available

    def _rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using zerank-2 cross-encoder model.

        Design Justification:
        ---------------------
        1. TWO-STAGE RETRIEVAL: Initial keyword retrieval is fast but imprecise
           (e.g., "fast" matches navigation menus). Cross-encoder reranking is slower
           but understands semantic relevance (e.g., "5 min. start time" answers
           "fast start capability" even without keyword "fast").

        2. CROSS-ENCODER SCORING: zerank-2 is a CrossEncoder model that directly
           scores query-document relevance. We send all documents in a single batch
           request for efficiency.

        3. DOCUMENT TRUNCATION: Sends first 2000 chars to reranker. Cross-encoders
           assess relevance from key signals, don't need full document.

        4. GRACEFUL FALLBACK: If reranker fails, documents keep original keyword
           scores. System degrades gracefully rather than failing entirely.

        Args:
            query: The user's question
            documents: List of candidate documents from keyword retrieval
            limit: Number of documents to return after reranking

        Returns:
            Top `limit` documents sorted by rerank score
        """
        if not documents:
            return []

        # Check reranker availability
        if not self.reranker_enabled or not self._check_reranker_available():
            return documents[:limit]

        try:
            # Build request for CrossEncoder server (batch all documents)
            rerank_docs = []
            for i, doc in enumerate(documents):
                rerank_docs.append({
                    "id": str(i),
                    "content": doc.get("content", "")[:2000],
                })

            response = requests.post(
                RERANKER_URL,
                json={
                    "query": query,
                    "documents": rerank_docs,
                },
                timeout=RERANKER_TIMEOUT,
            )

            result = response.json()

            if "error" in result:
                logger.warning(f"Reranker error: {result['error']}")
                return documents[:limit]

            # Map scores back to documents
            score_map = {item["id"]: item["score"] for item in result.get("scores", [])}

            for i, doc in enumerate(documents):
                doc["rerank_score"] = score_map.get(str(i), 0.0)

            # Sort by rerank score (descending), use keyword score as tiebreaker
            documents.sort(key=lambda x: (x.get("rerank_score", 0), x.get("score", 0)), reverse=True)

            # Log reranking results for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Reranking results for: {query[:50]}...")
                for i, doc in enumerate(documents[:limit]):
                    logger.debug(f"  {i+1}. score={doc.get('rerank_score', '?'):.3f} url={doc.get('source_url', '?')[-30:]}")

            return documents[:limit]

        except Exception as e:
            logger.warning(f"Reranker failed, using keyword scores: {e}")
            return documents[:limit]

    def add_document(
        self,
        url: str,
        title: str,
        content: str,
        domain: str,
        trust_level: str = "medium",
        source_type: str = "web_search",
    ) -> str:
        """Add a document to FalkorDB."""
        import uuid
        from datetime import datetime
        doc_id = str(uuid.uuid4())

        try:
            cypher = """
            CREATE (d:Document {
                id: $id,
                url: $url,
                title: $title,
                content: $content,
                domain: $domain,
                trust_level: $trust_level,
                source_type: $source_type,
                added_at: $timestamp
            })
            RETURN d.id as id
            """
            result = self.graph.query(
                cypher,
                {
                    'id': doc_id,
                    'url': url,
                    'title': title,
                    'content': content[:10000],  # Limit content size
                    'domain': domain,
                    'trust_level': trust_level,
                    'source_type': source_type,
                    'timestamp': datetime.now().isoformat(),
                }
            )
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document to FalkorDB: {e}")
            return doc_id

    def run_cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a Cypher query on FalkorDB."""
        params = params or {}
        result = self.graph.query(query, params)
        results = []
        for record in result.result_set:
            # Convert FalkorDB result to dict
            if len(record) == 1:
                if hasattr(record[0], 'properties'):
                    results.append(record[0].properties)
                else:
                    results.append({'value': record[0]})
            else:
                row = {}
                for i, val in enumerate(record):
                    if hasattr(val, 'properties'):
                        row[f'col_{i}'] = val.properties
                    else:
                        row[f'col_{i}'] = val
                results.append(row)
        return results

    def get_entity_episodes(
        self,
        entity_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get Episodes that mention this entity via CONTAINS relationship.

        This method retrieves temporal provenance information showing when
        and where an entity was mentioned in the knowledge graph.

        Args:
            entity_name: Name of the entity to look up.
            limit: Maximum number of episodes to return.

        Returns:
            List of episode dictionaries with episode_id, name, content,
            reference_time, source_url, source_type, and confidence.
        """
        try:
            # Escape single quotes in entity name for Cypher
            name_escaped = entity_name.replace("'", "\\'")
            result = self.graph.query(f"""
                MATCH (ep:Episode)-[r:CONTAINS]->(e:Entity)
                WHERE toLower(e.name) = toLower('{name_escaped}')
                  AND (e.fact_status IS NULL OR e.fact_status = 'active')
                RETURN ep.episode_id as episode_id,
                       ep.name as name,
                       ep.content as content,
                       ep.reference_time as reference_time,
                       ep.source_url as source_url,
                       ep.source_type as source_type,
                       r.confidence as confidence
                ORDER BY ep.reference_time DESC
                LIMIT {limit}
            """)
            return [dict(zip(['episode_id', 'name', 'content', 'reference_time',
                              'source_url', 'source_type', 'confidence'], row))
                    for row in result.result_set]
        except Exception as e:
            logger.debug(f"Failed to get entity episodes: {e}")
            return []

    def get_community_members(
        self,
        entity_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get entities in the same community as this entity.

        This method finds related entities through community membership,
        enabling community-aware search expansion.

        Args:
            entity_name: Name of the entity to look up.
            limit: Maximum number of community members to return.

        Returns:
            List of entity dictionaries with entity_id, name, type,
            description, community_name, and community_id.
        """
        try:
            # Escape single quotes in entity name for Cypher
            name_escaped = entity_name.replace("'", "\\'")
            result = self.graph.query(f"""
                MATCH (e1:Entity)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(e2:Entity)
                WHERE toLower(e1.name) = toLower('{name_escaped}')
                  AND e1 <> e2
                  AND (e2.fact_status IS NULL OR e2.fact_status = 'active')
                RETURN DISTINCT e2.entity_id as entity_id,
                       e2.name as name,
                       e2.ontology_type as type,
                       e2.description as description,
                       c.name as community_name,
                       c.community_id as community_id
                LIMIT {limit}
            """)
            return [dict(zip(['entity_id', 'name', 'type', 'description',
                              'community_name', 'community_id'], row))
                    for row in result.result_set]
        except Exception as e:
            logger.debug(f"Failed to get community members: {e}")
            return []

    def get_entity_timeline(
        self,
        entity_name: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get temporal timeline of when this entity was mentioned.

        This method returns a chronological view of when facts about
        an entity were observed, providing temporal context for the
        knowledge graph.

        Args:
            entity_name: Name of the entity to look up.
            limit: Maximum number of timeline entries to return.

        Returns:
            List of timeline entries with observed_at, source, url, and excerpt.
        """
        try:
            # Escape single quotes in entity name for Cypher
            name_escaped = entity_name.replace("'", "\\'")
            result = self.graph.query(f"""
                MATCH (ep:Episode)-[:CONTAINS]->(e:Entity)
                WHERE toLower(e.name) = toLower('{name_escaped}')
                RETURN ep.reference_time as observed_at,
                       ep.source_type as source,
                       ep.source_url as url,
                       ep.content as excerpt
                ORDER BY ep.reference_time
                LIMIT {limit}
            """)
            return [dict(zip(['observed_at', 'source', 'url', 'excerpt'], row))
                    for row in result.result_set]
        except Exception as e:
            logger.debug(f"Failed to get entity timeline: {e}")
            return []


def create_qa_loader(
    backend: str = "neo4j",
    graph_name: str = FALKORDB_GRAPH,
    **kwargs
):
    """Factory function to create the appropriate QA loader.

    Args:
        backend: "neo4j" or "falkordb"
        graph_name: Name of the FalkorDB graph (only used for falkordb backend)
        **kwargs: Additional arguments passed to the loader

    Returns:
        Neo4jQALoader or FalkorDBQALoader instance
    """
    if backend.lower() == "falkordb":
        return FalkorDBQALoader(graph_name=graph_name, **kwargs)
    else:
        return Neo4jQALoader(**kwargs)


# --- REACT AGENT ---

class ReActQAAgent:
    """
    ReAct-style Q&A agent using LangGraph.

    Implements Thought-Action-Observation loops for question answering
    with hybrid GraphRAG and web search capabilities.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        neo4j_loader: Optional[Neo4jQALoader] = None,
        db_loader: Optional[Any] = None,
        backend: str = "neo4j",
        graph_name: str = FALKORDB_GRAPH,
        web_search_enabled: bool = True,
        auto_add_documents: bool = True,
        use_retrieval_planning: bool = True,
        compression_enabled: bool = True,
        use_improved_retrieval: bool = False,
        use_followup_planning: bool = False,
        planning_reasoning: bool = False,
        max_hops: int = 2,
        vector_limit: int = 5,
        primary_vector_limit: int = 5,
        primary_max_hops: int = 4,
        secondary_vector_limit: int = 3,
        secondary_max_hops: int = 2,
        n_generations: int = 3,
        skip_uncertainty: bool = False,
        wiki_search_enabled: bool = True,
        wiki_max_results: int = 2,
        parse_response_max_retries: int = 2,
        tool_call_max_retries: int = 1,
        pass_tool_errors_to_agent: bool = True,
    ):
        """
        Initialize the ReAct Q&A agent.

        Args:
            llm: Language model to use. Defaults to Nemotron-3-Nano.
            neo4j_loader: Neo4j loader for graph operations (deprecated, use db_loader).
            db_loader: Database loader for graph operations (Neo4j or FalkorDB).
            backend: Database backend to use ("neo4j" or "falkordb").
            graph_name: Name of the FalkorDB graph (only used for falkordb backend).
            web_search_enabled: Whether to allow web searches.
            auto_add_documents: Whether to automatically add web results to DB.
            use_retrieval_planning: Whether to use CLaRa-style retrieval planning.
            compression_enabled: Whether to compress context after retrieval.
            use_improved_retrieval: Use ImprovedGraphRAG (vector + fixed hop sampling).
            use_followup_planning: Use FollowUpGraphRAG (follow-up questions + dual vector search).
            planning_reasoning: Enable detailed thinking for follow-up planning (nemotron).
            max_hops: Maximum hops for graph traversal (1, 2, or 3).
            vector_limit: Maximum entities per keyword from vector search.
            primary_vector_limit: Max entities for original query (FollowUpGraphRAG).
            primary_max_hops: Max hops for original query (FollowUpGraphRAG).
            secondary_vector_limit: Max entities per follow-up question.
            secondary_max_hops: Max hops for follow-up traversal.
            n_generations: Number of generations for uncertainty metrics (default: 3).
            skip_uncertainty: Skip uncertainty computation for faster inference.
            wiki_search_enabled: Whether to allow Wikipedia searches.
            wiki_max_results: Maximum Wikipedia articles to return per search (default: 2).
            parse_response_max_retries: Max retries for JSON parse failures (default: 2).
            tool_call_max_retries: Max retries for failed tool calls (default: 1).
            pass_tool_errors_to_agent: Pass tool errors to agent for recovery (default: True).
        """
        # Nemotron-3-Nano: A/B testing showed temp=0 performs equally well
        # and is more deterministic. Keeping original settings.
        self.llm = llm or ChatOllama(
            model=OLLAMA_MAIN_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
            num_ctx=8192,  # Increased context for complex reasoning
        )

        # Initialize database loader (supports Neo4j and FalkorDB)
        self.backend = backend.lower()
        if db_loader is not None:
            self.neo4j_loader = db_loader
        elif neo4j_loader is not None:
            self.neo4j_loader = neo4j_loader
        else:
            self.neo4j_loader = create_qa_loader(backend=backend, graph_name=graph_name)
        self.web_search_enabled = web_search_enabled
        self.auto_add_documents = auto_add_documents
        self.use_retrieval_planning = use_retrieval_planning
        self.compression_enabled = compression_enabled
        self.use_improved_retrieval = use_improved_retrieval
        self.use_followup_planning = use_followup_planning
        self.planning_reasoning = planning_reasoning
        self.max_hops = max_hops
        self.vector_limit = vector_limit
        self.n_generations = n_generations
        self.skip_uncertainty = skip_uncertainty
        self.wiki_search_enabled = wiki_search_enabled
        self.wiki_max_results = wiki_max_results
        self.parse_response_max_retries = parse_response_max_retries
        self.tool_call_max_retries = tool_call_max_retries
        self.pass_tool_errors_to_agent = pass_tool_errors_to_agent

        # Initialize uncertainty calculator (uses main LLM with diversity settings for sampling)
        self.uncertainty_calculator = None
        if not skip_uncertainty:
            self.uncertainty_calculator = UncertaintyCalculator(
                llm=self.llm,
                embeddings=self.neo4j_loader.embedding_model,
                ollama_host=OLLAMA_HOST,
                model=OLLAMA_MAIN_MODEL,
            )
            logger.info(f"Uncertainty calculator initialized (n_generations={n_generations})")

        # Initialize GraphRAG - choose between original, improved, or follow-up
        # NOTE: All GraphRAG instances now use main LLM (utility_llm removed)
        self.planned_graphrag = None
        self.improved_graphrag = None
        self.followup_graphrag = None

        if use_followup_planning:
            # Follow-up question planning with dual vector search
            self.followup_graphrag = FollowUpGraphRAG(
                backend=self.backend,
                graph_name=graph_name,
                planning_llm=self.llm,
                utility_llm=self.llm,  # Use main LLM for all tasks
                primary_vector_limit=primary_vector_limit,
                primary_max_hops=primary_max_hops,
                secondary_vector_limit=secondary_vector_limit,
                secondary_max_hops=secondary_max_hops,
                compression_enabled=compression_enabled,
                planning_reasoning=planning_reasoning,
            )
            logger.info(f"Using FollowUpGraphRAG (backend={self.backend}, v{primary_vector_limit}_h{primary_max_hops} + "
                       f"v{secondary_vector_limit}_h{secondary_max_hops}, reasoning={planning_reasoning})")
        elif use_improved_retrieval:
            self.improved_graphrag = ImprovedGraphRAG(
                backend=self.backend,
                graph_name=graph_name,
                llm=self.llm,  # Use main LLM
                max_hops=max_hops,
                vector_limit=vector_limit,
                compression_enabled=compression_enabled,
            )
            logger.info(f"Using ImprovedGraphRAG (backend={self.backend}, max_hops={max_hops}, vector_limit={vector_limit})")
        else:
            self.planned_graphrag = PlannedGraphRAG(
                backend=self.backend,
                graph_name=graph_name,
                llm=self.llm,  # Use main LLM
                compression_enabled=compression_enabled,
            )

        # Initialize KG extraction agent for auto document ingestion
        self._kg_extraction_agent = None
        if auto_add_documents:
            try:
                self._kg_extraction_agent = KnowledgeGraphAgent(
                    provider='openai',  # Use OpenAI for quality extraction
                    ontology='medium',  # Use medium ontology for balance
                )
                logger.info("KG extraction agent initialized for auto document ingestion")
            except Exception as e:
                logger.warning(f"KG extraction agent not available: {e}")

        # Initialize web search tool
        self._web_search_tool = None
        if web_search_enabled:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "agent_tools",
                    os.path.join(os.path.dirname(__file__), "finetuning/agent/tools.py")
                )
                tools_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tools_module)
                self._web_search_tool = tools_module.WebSearchTool()
            except Exception as e:
                logger.warning(f"Web search tool not available: {e}")

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(QAAgentState)

        if self.use_retrieval_planning:
            # CLaRa-style flow with retrieval planning
            # START → plan_retrieval → retrieve_planned → think → [conditional]
            #         ├→ execute_action → compress_observation → think (loop)
            #         └→ synthesize → END
            graph.add_node("plan_retrieval", self._plan_retrieval_node)
            graph.add_node("retrieve_planned", self._retrieve_planned_node)
            graph.add_node("think", self._think_node)
            graph.add_node("execute_action", self._execute_action_node)
            graph.add_node("compress_observation", self._compress_observation_node)
            graph.add_node("synthesize", self._synthesize_node)

            graph.add_edge(START, "plan_retrieval")
            graph.add_edge("plan_retrieval", "retrieve_planned")
            graph.add_edge("retrieve_planned", "think")
            graph.add_conditional_edges(
                "think",
                self._should_continue,
                {
                    "execute": "execute_action",
                    "answer": "synthesize",
                    "end": END,
                }
            )
            graph.add_edge("execute_action", "compress_observation")
            graph.add_edge("compress_observation", "think")
            graph.add_edge("synthesize", END)
        else:
            # Original flow without retrieval planning
            graph.add_node("retrieve_initial", self._retrieve_initial_node)
            graph.add_node("think", self._think_node)
            graph.add_node("execute_action", self._execute_action_node)
            graph.add_node("observe", self._observe_node)
            graph.add_node("synthesize", self._synthesize_node)

            graph.add_edge(START, "retrieve_initial")
            graph.add_edge("retrieve_initial", "think")
            graph.add_conditional_edges(
                "think",
                self._should_continue,
                {
                    "execute": "execute_action",
                    "answer": "synthesize",
                    "end": END,
                }
            )
            graph.add_edge("execute_action", "observe")
            graph.add_edge("observe", "think")
            graph.add_edge("synthesize", END)

        return graph.compile()

    def _should_continue(self, state: QAAgentState) -> str:
        """Determine next step based on state."""
        if state.error:
            return "end"
        if state.ready_to_answer:
            return "answer"
        if state.iteration_count >= state.max_iterations:
            return "answer"  # Force answer after max iterations
        if state.pending_action:
            return "execute"
        return "answer"

    # =========================================================================
    # CLaRa-style Retrieval Planning Nodes
    # =========================================================================

    def _plan_retrieval_node(self, state: QAAgentState) -> Dict[str, Any]:
        """
        Create a retrieval plan for the question (CLaRa-style).

        This node generates an explicit plan specifying:
        - Entity targets to look up
        - Relationship patterns to query
        - Fallback web searches if needed
        """
        question = state.question
        logger.info(f"Creating retrieval plan for: {question[:100]}...")

        try:
            # Generate retrieval plan
            plan_prompt = format_retrieval_plan_prompt(question)
            response = self.llm.invoke(plan_prompt)
            plan = parse_retrieval_plan(response.content)

            logger.info(f"Retrieval plan: {len(plan.entity_targets)} entities, "
                       f"{len(plan.relationship_queries)} patterns")

            return {"retrieval_plan": plan}

        except Exception as e:
            logger.error(f"Retrieval planning failed: {e}")
            # Return empty plan - will fall back to basic retrieval
            return {"retrieval_plan": RetrievalPlan()}

    def _retrieve_planned_node(self, state: QAAgentState) -> Dict[str, Any]:
        """
        Execute the retrieval plan and return compressed context.

        Uses FollowUpGraphRAG, ImprovedGraphRAG, or PlannedGraphRAG to:
        1. Look up targeted entities with N-hop traversal
        2. Execute relationship pattern queries (PlannedGraphRAG only)
        3. Compress retrieved context to relevant facts
        """
        plan = state.retrieval_plan
        question = state.question

        # NEW: Use FollowUpGraphRAG if enabled (generates its own plan with follow-up questions)
        if self.use_followup_planning and self.followup_graphrag:
            return self._retrieve_with_followup_planning(question)

        if not plan or (not plan.entity_targets and not plan.relationship_queries):
            # Fallback to basic retrieval
            logger.info("No retrieval plan, falling back to vector search")
            return self._retrieve_initial_node(state)

        try:
            # Use ImprovedGraphRAG if enabled (Keywords + Vector → Fixed Hop Sampling)
            if self.use_improved_retrieval and self.improved_graphrag:
                # Use entity_targets as keywords for vector search
                keywords = plan.entity_targets + plan.information_needs[:3]
                graph_context = self.improved_graphrag.retrieve_with_keywords(keywords, question)

                # Convert to ContextItems (simplified format from ImprovedGraphRAG)
                context = []
                for entity_ctx in graph_context.entities:
                    content = f"[{entity_ctx.entity_name}]"
                    if entity_ctx.entity_data.get("description"):
                        content += f"\n  {entity_ctx.entity_data['description']}"
                    if entity_ctx.relationships:
                        content += "\n  Relationships:"
                        seen = set()
                        for rel in entity_ctx.relationships[:10]:
                            rel_key = f"{rel.get('source', '?')}-{rel.get('rel_type', '?')}-{rel.get('target', '?')}"
                            if rel_key not in seen:
                                content += f"\n    - {rel.get('source', '?')} --[{rel.get('rel_type', '?')}]--> {rel.get('target', '?')}"
                                seen.add(rel_key)

                    context.append(ContextItem(
                        source_type="entity",
                        content=content,
                        source_id=entity_ctx.entity_name,
                        relevance_score=entity_ctx.entity_data.get("score", 1.0),
                        metadata={"from_improved": True, "hops": self.max_hops},
                    ))

                context_formatted = self._format_context(context)
                compressed = graph_context.compressed_text

                logger.info(f"ImprovedGraphRAG: {len(context)} entities, "
                           f"{len(graph_context.raw_text)} → {len(compressed)} chars")

                return {
                    "context": context,
                    "context_formatted": context_formatted,
                    "compressed_context": compressed,
                }

            # Use original PlannedGraphRAG
            graph_context = self.planned_graphrag.retrieve_with_plan(plan, question)

            # Convert to ContextItems
            context = []

            # Add entity contexts
            for entity_ctx in graph_context.entities:
                if entity_ctx.found:
                    content = self._format_entity(entity_ctx.entity_data)
                    if entity_ctx.relationships:
                        rel_lines = []
                        for rel in entity_ctx.relationships[:10]:
                            other = rel["other_entity"]
                            other_name = other.get("name", other.get("id", "?"))
                            direction = "->" if rel["direction"] == "outgoing" else "<-"
                            rel_lines.append(f"  {direction} [{rel['type']}] {other_name}")
                        content += f"\nRelationships:\n" + "\n".join(rel_lines)

                    context.append(ContextItem(
                        source_type="entity",
                        content=content,
                        source_id=entity_ctx.entity_name,
                        relevance_score=1.0,
                        metadata={"from_plan": True},
                    ))

            # Add relationship query results
            for rel_ctx in graph_context.relationships:
                if rel_ctx.results:
                    content = f"[Pattern: {rel_ctx.pattern}]\n"
                    for result in rel_ctx.results[:5]:
                        result_str = ", ".join(f"{k}: {v}" for k, v in result.items() if v)
                        content += f"  {result_str}\n"

                    context.append(ContextItem(
                        source_type="relationship",
                        content=content,
                        source_id=rel_ctx.pattern,
                        relevance_score=0.9,
                        metadata={"cypher": rel_ctx.cypher_used},
                    ))

            # Format and store both raw and compressed context
            context_formatted = self._format_context(context)
            compressed = graph_context.compressed_text

            logger.info(f"Retrieved {len(context)} context items, "
                       f"compressed from {len(graph_context.raw_text)} to {len(compressed)} chars")

            return {
                "context": context,
                "context_formatted": context_formatted,
                "compressed_context": compressed,
            }

        except Exception as e:
            logger.error(f"Planned retrieval failed: {e}")
            # Fallback to basic retrieval
            return self._retrieve_initial_node(state)

    def _retrieve_with_followup_planning(self, question: str) -> Dict[str, Any]:
        """
        Retrieve context using FollowUpGraphRAG (follow-up questions + dual vector search).

        This combines planning and retrieval:
        1. Generate follow-up questions from the original query
        2. Primary search: Original query (v5_h4 by default)
        3. Secondary search: Follow-up questions (v3_h2 by default)
        4. Merge and compress results
        """
        try:
            # FollowUpGraphRAG handles planning internally
            graph_context = self.followup_graphrag.retrieve_with_follow_ups(question)

            # Convert to ContextItems
            context = []
            for entity_ctx in graph_context.entities:
                content = f"[{entity_ctx.entity_name}]"
                if entity_ctx.entity_data.get("description"):
                    content += f"\n  {entity_ctx.entity_data['description']}"

                # Include search source info
                search_source = entity_ctx.entity_data.get("search_source", "unknown")
                if "followup" in search_source:
                    followup_q = entity_ctx.entity_data.get("followup_question", "")
                    content += f"\n  (from follow-up: {followup_q[:50]}...)" if followup_q else ""

                if entity_ctx.relationships:
                    content += "\n  Relationships:"
                    seen = set()
                    for rel in entity_ctx.relationships[:10]:
                        rel_key = f"{rel.get('source', '?')}-{rel.get('rel_type', '?')}-{rel.get('target', '?')}"
                        if rel_key not in seen:
                            content += f"\n    - {rel.get('source', '?')} --[{rel.get('rel_type', '?')}]--> {rel.get('target', '?')}"
                            seen.add(rel_key)

                context.append(ContextItem(
                    source_type="entity",
                    content=content,
                    source_id=entity_ctx.entity_name,
                    relevance_score=entity_ctx.entity_data.get("score", 1.0),
                    metadata={
                        "from_followup": True,
                        "search_source": search_source,
                    },
                ))

            context_formatted = self._format_context(context)
            compressed = graph_context.compressed_text

            logger.info(f"FollowUpGraphRAG: {len(context)} entities, "
                       f"{len(graph_context.raw_text)} → {len(compressed)} chars")

            return {
                "context": context,
                "context_formatted": context_formatted,
                "compressed_context": compressed,
            }

        except Exception as e:
            logger.error(f"Follow-up retrieval failed: {e}")
            # Fallback to direct vector search
            if self.improved_graphrag:
                return self._retrieve_initial_node_improved(question)
            return {"context": [], "context_formatted": "", "compressed_context": ""}

    def _retrieve_initial_node_improved(self, question: str) -> Dict[str, Any]:
        """Fallback retrieval using ImprovedGraphRAG."""
        try:
            graph_context = self.improved_graphrag.retrieve_direct(question)

            context = []
            for entity_ctx in graph_context.entities:
                content = f"[{entity_ctx.entity_name}]"
                if entity_ctx.entity_data.get("description"):
                    content += f"\n  {entity_ctx.entity_data['description']}"

                context.append(ContextItem(
                    source_type="entity",
                    content=content,
                    source_id=entity_ctx.entity_name,
                    relevance_score=entity_ctx.entity_data.get("score", 1.0),
                    metadata={"fallback": True},
                ))

            return {
                "context": context,
                "context_formatted": self._format_context(context),
                "compressed_context": graph_context.compressed_text,
            }
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return {"context": [], "context_formatted": "", "compressed_context": ""}

    def _compress_observation_node(self, state: QAAgentState) -> Dict[str, Any]:
        """
        Compress the tool observation before recording it (CLaRa-style).

        This reduces noise in the observation and focuses on facts
        relevant to the question.
        """
        observation = state.last_observation
        question = state.question

        if not observation:
            return self._observe_node(state)

        try:
            # Compress observation if it's long
            if len(observation) > 500 and self.compression_enabled:
                # Use whichever GraphRAG is available for compression
                if self.followup_graphrag:
                    compressed = self.followup_graphrag.compress_observation(observation, question)
                elif self.improved_graphrag:
                    compressed = self.improved_graphrag.compress_observation(observation, question)
                elif self.planned_graphrag:
                    compressed = self.planned_graphrag.compress_observation(observation, question)
                else:
                    compressed = observation[:500]  # Fallback: truncate
                logger.debug(f"Compressed observation from {len(observation)} to {len(compressed)} chars")
            else:
                compressed = observation

            # Add to thought history
            new_history = list(state.thought_history)
            new_history.append(ThoughtStep(
                thought=state.current_thought,
                action=state.pending_action,
                observation=compressed,
            ))

            return {
                "thought_history": new_history,
                "pending_action": None,
                "last_observation": compressed,
            }

        except Exception as e:
            logger.warning(f"Observation compression failed: {e}")
            # Fallback to normal observe
            return self._observe_node(state)

    # =========================================================================
    # Original Retrieval Nodes (non-planning fallback)
    # =========================================================================

    def _retrieve_initial_node(self, state: QAAgentState) -> Dict[str, Any]:
        """Initial retrieval based on the question."""
        question = state.question
        context = []

        # Use ImprovedGraphRAG if enabled (with fixed hop traversal)
        if self.use_improved_retrieval and self.improved_graphrag:
            graph_context = self.improved_graphrag.retrieve_direct(question)

            # Convert to ContextItems
            for entity_ctx in graph_context.entities:
                content = f"[{entity_ctx.entity_name}]"
                if entity_ctx.entity_data.get("description"):
                    content += f"\n  {entity_ctx.entity_data['description']}"
                if entity_ctx.relationships:
                    content += "\n  Relationships:"
                    seen = set()
                    for rel in entity_ctx.relationships[:10]:
                        rel_key = f"{rel.get('source', '?')}-{rel.get('rel_type', '?')}-{rel.get('target', '?')}"
                        if rel_key not in seen:
                            content += f"\n    - {rel.get('source', '?')} --[{rel.get('rel_type', '?')}]--> {rel.get('target', '?')}"
                            seen.add(rel_key)

                context.append(ContextItem(
                    source_type="entity",
                    content=content,
                    source_id=entity_ctx.entity_name,
                    relevance_score=entity_ctx.entity_data.get("score", 1.0),
                    metadata={"from_improved": True, "hops": self.max_hops},
                ))

            context_formatted = self._format_context(context)
            compressed = graph_context.compressed_text

            return {
                "context": context,
                "context_formatted": context_formatted,
                "compressed_context": compressed,
            }

        # Fallback to basic neo4j_loader vector search
        entities = self.neo4j_loader.vector_search(question, limit=5)
        for entity in entities:
            context.append(ContextItem(
                source_type="entity",
                content=self._format_entity(entity),
                source_id=entity.get("name", entity.get("id", "unknown")),
                relevance_score=entity.get("score", 0.0),
                metadata={"labels": entity.get("labels", [])},
            ))

        # Search documents if available
        documents = self.neo4j_loader.search_documents(question, limit=3)
        for doc in documents:
            context.append(ContextItem(
                source_type="document",
                content=f"[Document: {doc.get('title', 'Untitled')}]\n{doc.get('content', '')[:1000]}",
                source_id=doc.get("url", doc.get("id", "unknown")),
                relevance_score=doc.get("score", 0.0),
                metadata={
                    "domain": doc.get("domain"),
                    "trust_level": doc.get("trust_level"),
                },
            ))

        # Format context
        context_formatted = self._format_context(context)

        return {
            "context": context,
            "context_formatted": context_formatted,
        }

    def _format_entity(self, entity: Dict[str, Any]) -> str:
        """Format an entity for context."""
        name = entity.get("name", entity.get("id", "Unknown"))
        entity_type = entity.get("ontology_type", entity.get("open_type", "Entity"))
        description = entity.get("description", "")

        parts = [f"[Entity: {name} ({entity_type})]"]
        if description:
            parts.append(f"Description: {description}")

        return "\n".join(parts)

    def _format_context(self, context: List[ContextItem]) -> str:
        """Format all context items into a string."""
        if not context:
            return "No relevant context found."

        parts = []
        for i, item in enumerate(context, 1):
            parts.append(f"--- Context {i} ({item.source_type}) ---")
            parts.append(item.content)
            parts.append("")

        return "\n".join(parts)

    def _think_node(self, state: QAAgentState) -> Dict[str, Any]:
        """Generate next thought and action."""
        # Format thought history
        history_parts = []

        # Include retrieval plan reasoning if available
        if state.retrieval_plan and state.retrieval_plan.reasoning:
            history_parts.append(f"Retrieval Plan: {state.retrieval_plan.reasoning}")
            history_parts.append("")

        for step in state.thought_history:
            history_parts.append(f"Thought: {step.thought}")
            if step.action:
                history_parts.append(f"Action: {step.action.tool_name}({step.action.arguments})")
            if step.observation:
                history_parts.append(f"Observation: {step.observation[:500]}")
            history_parts.append("")

        thought_history = "\n".join(history_parts) if history_parts else "No previous steps."

        # Use compressed context if available, otherwise use raw formatted context
        context_to_use = state.compressed_context if state.compressed_context else state.context_formatted
        context_to_use = context_to_use[:CONTEXT_WINDOW_SIZE]

        # Generate next thought
        prompt = THINK_PROMPT.format(
            question=state.question,
            thought_history=thought_history,
            context=context_to_use,
        )

        # Retry loop for parse failures (NeMo-style)
        last_parse_error = None
        for attempt in range(self.parse_response_max_retries + 1):
            try:
                response = self.llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])

                # Parse response
                content = response.content
                if not content or not content.strip():
                    logger.warning("Empty LLM response in think node, using fallback")
                    return {
                        "ready_to_answer": True,
                        "current_thought": "Unable to generate thought due to empty response.",
                        "error": "Empty LLM response",
                    }

                # Try to extract JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                # Check for empty content after extraction
                if not content or not content.strip():
                    logger.warning("Empty JSON content after extraction in think node")
                    return {
                        "ready_to_answer": True,
                        "current_thought": "Unable to parse response.",
                        "error": "Empty JSON content",
                    }

                # Try to repair malformed JSON before parsing
                try:
                    repaired_content = repair_json(content)
                    parsed = json.loads(repaired_content)
                except Exception as repair_err:
                    logger.warning(f"JSON repair failed, trying raw parse: {repair_err}")
                    parsed = json.loads(content)

                # Successfully parsed - break out of retry loop
                break

            except json.JSONDecodeError as e:
                last_parse_error = e
                if attempt < self.parse_response_max_retries:
                    logger.warning(f"JSON parse retry {attempt + 1}/{self.parse_response_max_retries}: {e}")
                    continue
                else:
                    # Final attempt failed
                    logger.error(f"JSON parse failed after {self.parse_response_max_retries + 1} attempts: {e}")
                    return {
                        "ready_to_answer": True,
                        "current_thought": f"Unable to parse response after {self.parse_response_max_retries + 1} attempts.",
                        "error": f"JSON parse error: {str(e)}",
                    }
            except Exception as e:
                logger.error(f"Think node error: {e}")
                return {
                    "ready_to_answer": True,
                    "error": str(e),
                }

        # Process parsed result (reached via break)
        thought = parsed.get("thought", "")
        ready_to_answer = parsed.get("ready_to_answer", False)
        action_data = parsed.get("action", {})

        pending_action = None
        if not ready_to_answer and action_data.get("tool_name", "none") != "none":
            pending_action = ToolCall(
                tool_name=action_data["tool_name"],
                arguments=action_data.get("arguments", {}),
            )

        return {
            "current_thought": thought,
            "ready_to_answer": ready_to_answer,
            "pending_action": pending_action,
            "iteration_count": state.iteration_count + 1,
        }

    def _execute_action_node(self, state: QAAgentState) -> Dict[str, Any]:
        """Execute the pending action with retry logic."""
        if not state.pending_action:
            return {"last_observation": "No action to execute"}

        action = state.pending_action
        tool_name = action.tool_name
        args = action.arguments

        # Tool call retry loop (NeMo-style)
        last_error = None
        for attempt in range(self.tool_call_max_retries + 1):
            try:
                if tool_name == "graph_lookup":
                    entity_name = args.get("entity_name", "")

                    # Get entity with relationships (includes temporal filtering)
                    result = self.neo4j_loader.get_entity_with_relationships(
                        entity_name=entity_name,
                        include_temporal=True,
                    )

                    # Zep temporal features: Get Episode context (temporal provenance)
                    if result.get("found") and hasattr(self.neo4j_loader, 'get_entity_episodes'):
                        try:
                            episodes = self.neo4j_loader.get_entity_episodes(entity_name, limit=5)
                            result["episodes"] = episodes
                        except Exception as e:
                            logger.debug(f"Failed to get episodes for {entity_name}: {e}")

                    # Zep temporal features: Get Community context (related entities)
                    if result.get("found") and hasattr(self.neo4j_loader, 'get_community_members'):
                        try:
                            community_members = self.neo4j_loader.get_community_members(entity_name, limit=5)
                            result["community_members"] = community_members
                        except Exception as e:
                            logger.debug(f"Failed to get community members for {entity_name}: {e}")

                    observation = self._format_graph_lookup_result(result)

                    # Add to context if found
                    new_context = list(state.context)
                    if result.get("found"):
                        new_context.append(ContextItem(
                            source_type="entity",
                            content=observation,
                            source_id=entity_name or "unknown",
                            relevance_score=1.0,
                        ))

                    return {
                        "last_observation": observation,
                        "context": new_context,
                        "context_formatted": self._format_context(new_context),
                    }

                elif tool_name == "wiki_search":
                    if not self.wiki_search_enabled:
                        return {"last_observation": "Wikipedia search is not enabled"}

                    query = args.get("query", "")
                    observation, new_context = self._execute_wiki_search(query, state.context)

                    return {
                        "last_observation": observation,
                        "context": new_context,
                        "context_formatted": self._format_context(new_context),
                        "external_info_used": True,
                    }

                elif tool_name == "web_search":
                    if not self.web_search_enabled or not self._web_search_tool:
                        return {"last_observation": "Web search is not enabled"}

                    result = self._web_search_tool.execute(
                        query=args.get("query", ""),
                        num_results=args.get("num_results", 5),
                    )
                    observation = self._format_web_search_result(result)

                    # Add to context and optionally extract KG to database
                    new_context = list(state.context)
                    for item in result.get("results", []):
                        new_context.append(ContextItem(
                            source_type="web",
                            content=f"[Web: {item.get('title', 'Untitled')}]\n{item.get('snippet', '')}",
                            source_id=item.get("url", "unknown"),
                            relevance_score=item.get("score", 0.5),
                            metadata={
                                "domain": item.get("domain"),
                                "trust_level": item.get("trust_level"),
                            },
                        ))

                        # Extract KG from web content using same pipeline as agent_skb.py
                        if self.auto_add_documents and self._kg_extraction_agent:
                            try:
                                self._extract_kg_from_web_result(item)
                            except Exception as e:
                                logger.warning(f"Failed to extract KG from web result: {e}")

                    return {
                        "last_observation": observation,
                        "context": new_context,
                        "context_formatted": self._format_context(new_context),
                        "external_info_used": True,
                    }

                elif tool_name == "cypher_query":
                    query = args.get("query", "")
                    params = args.get("params", {})
                    result = self.neo4j_loader.run_cypher(query, params)
                    observation = f"Query returned {len(result)} results:\n{json.dumps(result[:10], indent=2, default=str)}"

                    return {"last_observation": observation}

                elif tool_name == "entity_resolve":
                    entity_name = args.get("entity_name", "")
                    context = args.get("context", "")

                    # Vector search for disambiguation
                    entities = self.neo4j_loader.vector_search(
                        f"{entity_name} {context}",
                        limit=5,
                    )

                    candidates = []
                    for entity in entities:
                        candidates.append({
                            "name": entity.get("name", entity.get("id")),
                            "type": entity.get("ontology_type", entity.get("open_type")),
                            "description": entity.get("description", "")[:200],
                            "score": entity.get("score", 0),
                        })

                    observation = f"Found {len(candidates)} candidates:\n{json.dumps(candidates, indent=2)}"
                    return {"last_observation": observation}

                elif tool_name == "entity_timeline":
                    entity_name = args.get("entity_name", "")

                    # Get temporal timeline using Zep-style method
                    if hasattr(self.neo4j_loader, 'get_entity_timeline'):
                        timeline = self.neo4j_loader.get_entity_timeline(entity_name, limit=20)
                    else:
                        timeline = []

                    if not timeline:
                        observation = f"No temporal history found for entity '{entity_name}'."
                    else:
                        parts = [f"Temporal history for '{entity_name}' ({len(timeline)} observations):"]
                        for entry in timeline:
                            observed_at = entry.get('observed_at', 'unknown time')
                            source = entry.get('source', 'unknown')
                            url = entry.get('url', '')
                            excerpt = entry.get('excerpt', '')
                            if excerpt:
                                excerpt = excerpt[:100] + "..." if len(excerpt) > 100 else excerpt
                            if url:
                                parts.append(f"  - {observed_at} ({source}): {excerpt} [{url}]")
                            else:
                                parts.append(f"  - {observed_at} ({source}): {excerpt}")
                        observation = "\n".join(parts)

                    return {"last_observation": observation}

                else:
                    return {"last_observation": f"Unknown tool: {tool_name}"}

            except Exception as e:
                last_error = e
                if attempt < self.tool_call_max_retries:
                    logger.warning(f"Tool call retry {attempt + 1}/{self.tool_call_max_retries} for {tool_name}: {e}")
                    continue
                else:
                    # Final attempt failed
                    error_msg = f"Error executing {tool_name} after {self.tool_call_max_retries + 1} attempts: {str(e)}"
                    logger.error(error_msg)

                    # Pass error to agent if configured (allows agent to recover)
                    if self.pass_tool_errors_to_agent:
                        return {"last_observation": error_msg}
                    else:
                        raise

        # Shouldn't reach here, but safety return
        return {"last_observation": f"Tool {tool_name} failed: {last_error}"}

    def _format_graph_lookup_result(self, result: Dict[str, Any]) -> str:
        """Format graph lookup result with Zep temporal features."""
        if not result.get("found"):
            return f"Entity '{result.get('entity_name')}' not found in knowledge graph."

        entity = result["entity"]
        parts = [f"Entity: {entity.get('name', entity.get('id'))}"]

        if entity.get("description"):
            parts.append(f"Description: {entity['description']}")

        if entity.get("ontology_type"):
            parts.append(f"Type: {entity['ontology_type']}")

        # Add temporal metadata (fact_status, valid_from/valid_to)
        if entity.get("fact_status"):
            parts.append(f"Status: {entity['fact_status']}")
        if entity.get("valid_from") or entity.get("valid_to"):
            validity = []
            if entity.get("valid_from"):
                validity.append(f"from {entity['valid_from']}")
            if entity.get("valid_to"):
                validity.append(f"to {entity['valid_to']}")
            parts.append(f"Valid: {' '.join(validity)}")

        # Add Community info if available
        if result.get("community"):
            community = result["community"]
            parts.append(f"Community: {community.get('name', community.get('id', 'Unknown'))}")

        # Add Episode count if available
        if result.get("episode_count"):
            parts.append(f"Mentioned in {result['episode_count']} episode(s)")

        relationships = result.get("relationships", [])
        if relationships:
            parts.append(f"\nRelationships ({len(relationships)}):")
            for rel in relationships[:10]:
                other = rel["other_entity"]
                other_name = other.get("name", other.get("id", "?"))
                direction = "->" if rel["direction"] == "outgoing" else "<-"
                parts.append(f"  {direction} [{rel['type']}] {other_name}")

        # Add Episode provenance (temporal context from Zep)
        episodes = result.get("episodes", [])
        if episodes:
            parts.append("\nObserved in Episodes:")
            for ep in episodes[:3]:
                source_type = ep.get('source_type', 'unknown')
                ref_time = ep.get('reference_time', 'unknown time')
                source_url = ep.get('source_url', '')
                if source_url:
                    parts.append(f"  - {source_type}: {ref_time} ({source_url})")
                else:
                    parts.append(f"  - {source_type}: {ref_time}")

        # Add Community members (related entities from Zep)
        community_members = result.get("community_members", [])
        if community_members:
            parts.append("\nRelated entities (same community):")
            for member in community_members[:5]:
                member_name = member.get('name', 'Unknown')
                member_type = member.get('type', '')
                if member_type:
                    parts.append(f"  - {member_name} ({member_type})")
                else:
                    parts.append(f"  - {member_name}")

        return "\n".join(parts)

    def _format_web_search_result(self, result: Dict[str, Any]) -> str:
        """Format web search result."""
        if result.get("status") != "success":
            return f"Web search failed: {result.get('error', 'Unknown error')}"

        results = result.get("results", [])
        if not results:
            return "No web search results found."

        parts = [f"Found {len(results)} web results:"]
        for i, item in enumerate(results, 1):
            parts.append(f"\n{i}. {item.get('title', 'Untitled')}")
            parts.append(f"   URL: {item.get('url')}")
            parts.append(f"   Trust: {item.get('trust_level', 'unknown')}")
            parts.append(f"   Snippet: {item.get('snippet', '')[:200]}...")

        return "\n".join(parts)

    def _search_wiki_pages(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search local WikiPage nodes by name.

        Uses the WikiPage nodes built from Wikidata to find relevant
        Wikipedia articles stored in Neo4j.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching WikiPage nodes with name, qid, and url.
        """
        try:
            with self.neo4j_loader.driver.session() as session:
                result = session.run("""
                    MATCH (w:WikiPage)
                    WHERE toLower(w.name) CONTAINS toLower($query)
                    RETURN w.name as name, w.wikidata_id as qid, w.wikipedia_url as url
                    ORDER BY size(w.name)
                    LIMIT $limit
                """, query=query, limit=limit)

                return [dict(record) for record in result]
        except Exception as e:
            logger.debug(f"WikiPage search not available: {e}")
            return []

    def _execute_wiki_search(
        self,
        query: str,
        current_context: List[ContextItem],
    ) -> Tuple[str, List[ContextItem]]:
        """Execute Wikipedia search and return formatted results.

        First checks local WikiPage nodes from Wikidata, then falls back
        to direct Wikipedia search if no local matches found.

        Args:
            query: Search query for Wikipedia
            current_context: Current context items

        Returns:
            Tuple of (observation string, updated context list)
        """
        try:
            # NEW: Check WikiPage nodes first
            wiki_pages = self._search_wiki_pages(query, limit=self.wiki_max_results)

            if wiki_pages:
                # Use stored URLs/names to fetch content from Wikipedia
                logger.info(f"Found {len(wiki_pages)} local WikiPage matches for '{query}'")
                parts = [f"Found {len(wiki_pages)} matching WikiPage(s) in knowledge graph:"]
                new_context = list(current_context)
                docs_loaded = 0

                for wp in wiki_pages:
                    try:
                        # Use the exact name for Wikipedia lookup
                        loader = WikipediaLoader(query=wp["name"], load_max_docs=1)
                        docs = loader.load()

                        if docs:
                            doc = docs[0]
                            source = wp.get("url") or doc.metadata.get("source", "Wikipedia")
                            title = doc.metadata.get("title", wp["name"])
                            content = doc.page_content[:1500]

                            parts.append(f"\n[WikiPage: {title}] (QID: {wp['qid']})")
                            parts.append(f"Source: {source}")
                            parts.append(f"Content: {content}...")

                            new_context.append(ContextItem(
                                source_type="document",
                                content=f"[WikiPage: {title}]\n{content}",
                                source_id=source,
                                relevance_score=0.95,  # Higher score for local matches
                                metadata={
                                    "source": "wikipage",
                                    "title": title,
                                    "wikidata_id": wp["qid"],
                                    "trust_level": "high",
                                },
                            ))
                            docs_loaded += 1

                    except Exception as e:
                        logger.warning(f"Failed to load Wikipedia content for {wp['name']}: {e}")
                        # Still include the WikiPage metadata even if content load fails
                        parts.append(f"\n[WikiPage: {wp['name']}] (QID: {wp['qid']})")
                        if wp.get("url"):
                            parts.append(f"URL: {wp['url']}")

                if docs_loaded > 0:
                    return "\n".join(parts), new_context

            # Fallback to direct Wikipedia search (existing logic)
            loader = WikipediaLoader(query=query, load_max_docs=self.wiki_max_results)
            docs = loader.load()

            if not docs:
                return "No Wikipedia articles found for this query.", list(current_context)

            # Format results
            parts = [f"Found {len(docs)} Wikipedia article(s):"]
            new_context = list(current_context)

            for doc in docs:
                source = doc.metadata.get("source", "Wikipedia")
                title = doc.metadata.get("title", "Untitled")
                content = doc.page_content[:1500]  # Limit content length

                parts.append(f"\n[Wikipedia: {title}]")
                parts.append(f"Source: {source}")
                parts.append(f"Content: {content}...")

                new_context.append(ContextItem(
                    source_type="document",
                    content=f"[Wikipedia: {title}]\n{content}",
                    source_id=source,
                    relevance_score=0.9,
                    metadata={
                        "source": "wikipedia",
                        "title": title,
                        "trust_level": "high",
                    },
                ))

            return "\n".join(parts), new_context

        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return f"Wikipedia search failed: {str(e)}", list(current_context)

    def _extract_kg_from_web_result(self, web_result: Dict[str, Any]) -> None:
        """
        Extract knowledge graph from web search result using agent_skb.py pipeline.

        This ensures web search results are ingested into the knowledge graph
        using the same extraction and ontology as the main SKB construction.

        Args:
            web_result: Web search result with title, url, snippet, etc.
        """
        if not self._kg_extraction_agent:
            return

        # Combine title and snippet for extraction
        title = web_result.get("title", "")
        snippet = web_result.get("snippet", "")
        url = web_result.get("url", "")
        domain = web_result.get("domain", "")

        if not snippet:
            logger.debug(f"Skipping KG extraction for {url}: no content")
            return

        # Create text chunk for extraction (include URL context)
        text_chunk = f"Source: {url}\nTitle: {title}\n\n{snippet}"

        # Create agent state for extraction (using actual AgentState fields)
        extraction_state = SKBAgentState(
            text_chunk=text_chunk,
            chunk_index=0,
            doc_index=0,
        )

        try:
            # Run the extraction pipeline
            result = self._kg_extraction_agent.graph.invoke(extraction_state)

            # Log extraction results
            if result.get("extracted_kg"):
                kg = result["extracted_kg"]
                node_count = len(kg.nodes) if hasattr(kg, 'nodes') else 0
                edge_count = len(kg.relationships) if hasattr(kg, 'relationships') else 0
                logger.info(f"Extracted KG from web result: {node_count} nodes, {edge_count} edges from {domain}")
            else:
                logger.debug(f"No KG extracted from {url}")

        except Exception as e:
            logger.warning(f"KG extraction failed for {url}: {e}")

    def _observe_node(self, state: QAAgentState) -> Dict[str, Any]:
        """Record observation from action."""
        # Add to thought history
        new_history = list(state.thought_history)
        new_history.append(ThoughtStep(
            thought=state.current_thought,
            action=state.pending_action,
            observation=state.last_observation,
        ))

        return {
            "thought_history": new_history,
            "pending_action": None,
        }

    def _synthesize_node(self, state: QAAgentState) -> Dict[str, Any]:
        """Generate final answer with citations."""
        # Format thought history
        history_parts = []
        for step in state.thought_history:
            history_parts.append(f"Thought: {step.thought}")
            if step.action:
                history_parts.append(f"Action: {step.action.tool_name}")
            if step.observation:
                history_parts.append(f"Observation: {step.observation[:300]}...")
            history_parts.append("")

        thought_history = "\n".join(history_parts) if history_parts else "Direct answer."

        prompt = ANSWER_PROMPT.format(
            question=state.question,
            context=state.context_formatted[:CONTEXT_WINDOW_SIZE],
            thought_history=thought_history,
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            # Parse response
            content = response.content
            if not content or not content.strip():
                logger.warning("Empty LLM response in synthesize, using fallback")
                return {
                    "final_answer": "I was unable to generate an answer due to an empty response.",
                    "confidence": 0.0,
                }

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # Check for empty content after extraction
            if not content or not content.strip():
                logger.warning("Empty JSON content after extraction in synthesize")
                return {
                    "final_answer": "I was unable to parse the response.",
                    "confidence": 0.0,
                }

            # Try to repair malformed JSON before parsing
            try:
                repaired_content = repair_json(content)
                parsed = json.loads(repaired_content)
            except Exception as repair_err:
                logger.warning(f"JSON repair failed, trying raw parse: {repair_err}")
                parsed = json.loads(content)

            # Handle case where LLM returns a list instead of dict
            if isinstance(parsed, list):
                # Try to find a dict in the list
                for item in parsed:
                    if isinstance(item, dict) and "answer" in item:
                        parsed = item
                        break
                else:
                    # No valid dict found, create fallback
                    parsed = {"answer": str(parsed[0]) if parsed else "I was unable to generate an answer."}

            answer = parsed.get("answer", "I was unable to generate an answer.")

            # Build citations
            citations = []
            for cit in parsed.get("citations", []):
                # Normalize source_type to valid enum values
                source_type = cit.get("source_type", "graph")
                if source_type in ("graph_query", "cypher_query", "graph_lookup", "knowledge_graph"):
                    source_type = "graph"
                elif source_type not in ("graph", "document", "web_search"):
                    source_type = "graph"  # Default fallback

                citations.append(Citation(
                    source_type=source_type,
                    source_id=cit.get("source_id", "unknown"),
                    source_title=cit.get("source_title"),
                    excerpt=cit.get("excerpt", ""),
                ))

            # Add external info notice if needed
            if state.external_info_used:
                if not answer.startswith("According to external"):
                    answer = f"[Note: This answer includes information from web search.]\n\n{answer}"

            # Compute uncertainty metrics (replaces LLM self-reported confidence)
            uncertainty_scores = None
            confidence = 0.5  # Default fallback

            if self.uncertainty_calculator is not None:
                try:
                    uncertainty_scores = self.uncertainty_calculator.compute_all(
                        question=state.question,
                        answer=answer,
                        context=state.context_formatted[:CONTEXT_WINDOW_SIZE],
                        n_generations=self.n_generations,
                    )
                    confidence = uncertainty_scores.combined_confidence
                    logger.info(f"Uncertainty metrics computed: confidence={confidence:.2f}")
                except Exception as ue:
                    logger.warning(f"Uncertainty calculation failed: {ue}")
                    confidence = 0.5
            else:
                logger.debug("Uncertainty calculation skipped (skip_uncertainty=True)")

            return {
                "final_answer": answer,
                "citations": citations,
                "confidence": confidence,
                "uncertainty_scores": uncertainty_scores,
            }

        except Exception as e:
            logger.error(f"Synthesize error: {e}")
            return {
                "final_answer": f"I encountered an error generating the answer: {str(e)}",
                "confidence": 0.0,
            }

    def answer_question(self, question: str) -> QAResponse:
        """
        Answer a question using the ReAct loop.

        Args:
            question: The question to answer.

        Returns:
            QAResponse with answer, citations, reasoning, and uncertainty metrics.
        """
        initial_state = QAAgentState(question=question)

        try:
            final_state = self.graph.invoke(initial_state)

            return QAResponse(
                question=question,
                answer=final_state.get("final_answer", ""),
                citations=final_state.get("citations", []),
                external_info_used=final_state.get("external_info_used", False),
                reasoning_steps=final_state.get("thought_history", []),
                confidence=final_state.get("confidence", 0.0),
                uncertainty=final_state.get("uncertainty_scores"),
            )

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return QAResponse(
                question=question,
                answer=f"I encountered an error: {str(e)}",
                confidence=0.0,
            )

    def close(self):
        """Close connections."""
        if self.neo4j_loader:
            self.neo4j_loader.close()
        if self.planned_graphrag:
            self.planned_graphrag.close()
        if self.improved_graphrag:
            self.improved_graphrag.close()


# --- ASYNC WRAPPER FOR CHAINLIT ---

class AsyncReActQAAgent:
    """Async wrapper for ReActQAAgent for use with Chainlit."""

    def __init__(self, agent: Optional[ReActQAAgent] = None, **kwargs):
        self._agent = agent
        self._kwargs = kwargs
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            if self._agent is None:
                self._agent = ReActQAAgent(**self._kwargs)
            self._initialized = True

    async def answer_question(self, question: str) -> QAResponse:
        """Async version of answer_question."""
        import asyncio
        self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._agent.answer_question,
            question,
        )

    async def close(self):
        """Close connections."""
        if self._agent:
            self._agent.close()


# --- CLI INTERFACE ---

def main():
    """CLI for testing the Q&A agent."""
    import argparse

    parser = argparse.ArgumentParser(description="ReAct Q&A Agent with CLaRa-style Retrieval")
    parser.add_argument("--question", "-q", type=str, help="Question to answer")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", type=str, default="neo4j", choices=["neo4j", "falkordb"],
                        help="Database backend (default: neo4j)")
    parser.add_argument("--graph", type=str, default=FALKORDB_GRAPH,
                        help=f"FalkorDB graph name (default: {FALKORDB_GRAPH})")
    parser.add_argument("--no-web", action="store_true", help="Disable web search")
    parser.add_argument("--no-wiki", action="store_true", help="Disable Wikipedia search")
    parser.add_argument("--wiki-results", type=int, default=2, help="Max Wikipedia results (default: 2)")
    parser.add_argument("--no-planning", action="store_true", help="Disable retrieval planning")
    parser.add_argument("--no-compression", action="store_true", help="Disable context compression")
    parser.add_argument("--followup", action="store_true", help="Use FollowUpGraphRAG (follow-up questions + dual vector search)")
    parser.add_argument("--reasoning", action="store_true", help="Enable detailed thinking for follow-up planning")
    parser.add_argument("--primary-vec", type=int, default=5, help="Primary vector limit (default: 5)")
    parser.add_argument("--primary-hops", type=int, default=4, help="Primary max hops (default: 4)")
    parser.add_argument("--secondary-vec", type=int, default=3, help="Secondary vector limit (default: 3)")
    parser.add_argument("--secondary-hops", type=int, default=2, help="Secondary max hops (default: 2)")
    parser.add_argument("--n-generations", type=int, default=3, help="Number of generations for uncertainty metrics (default: 3)")
    parser.add_argument("--skip-uncertainty", action="store_true", help="Skip uncertainty computation for faster inference")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed uncertainty metrics")

    # Retry and robustness flags (NeMo-style)
    parser.add_argument("--parse-retries", type=int, default=2, help="Max retries for JSON parse failures (default: 2)")
    parser.add_argument("--tool-retries", type=int, default=1, help="Max retries for failed tool calls (default: 1)")

    # Profiling flags
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")

    args = parser.parse_args()

    agent = ReActQAAgent(
        backend=args.backend,
        graph_name=args.graph,
        web_search_enabled=not args.no_web,
        use_retrieval_planning=not args.no_planning and not args.followup,
        compression_enabled=not args.no_compression,
        use_followup_planning=args.followup,
        planning_reasoning=args.reasoning,
        primary_vector_limit=args.primary_vec,
        primary_max_hops=args.primary_hops,
        secondary_vector_limit=args.secondary_vec,
        secondary_max_hops=args.secondary_hops,
        n_generations=args.n_generations,
        skip_uncertainty=args.skip_uncertainty,
        wiki_search_enabled=not args.no_wiki,
        wiki_max_results=args.wiki_results,
        parse_response_max_retries=args.parse_retries,
        tool_call_max_retries=args.tool_retries,
    )

    # Initialize profiler if requested
    profiler = None
    if args.profile:
        from agent_profiler import AgentProfiler, print_profile_report
        profiler = AgentProfiler()

    try:
        if args.interactive:
            print("ReAct Q&A Agent - Interactive Mode")
            print("Type 'quit' to exit\n")

            while True:
                question = input("Question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break
                if not question:
                    continue

                print("\nProcessing...\n")
                response = agent.answer_question(question)

                print(f"Answer: {response.answer}\n")
                print(f"Confidence: {response.confidence:.2f}")
                if args.verbose and response.uncertainty:
                    u = response.uncertainty
                    print(f"\nUncertainty Scores:")
                    if u.perplexity >= 0:
                        print(f"  Perplexity: {u.perplexity:.2f} (normalized: {u.perplexity_normalized:.2f})")
                    else:
                        print(f"  Perplexity: N/A (logprobs not supported by model)")
                    print(f"  Semantic Entropy: {u.semantic_entropy:.2f} (normalized: {u.semantic_entropy_normalized:.2f})")
                    print(f"  Embedding Consistency: {u.embedding_consistency:.2f}")
                    print(f"  Combined Confidence: {u.combined_confidence:.2%}")
                print(f"External sources used: {response.external_info_used}")
                if response.citations:
                    print(f"Citations: {len(response.citations)}")
                    for cit in response.citations:
                        print(f"  - [{cit.source_type}] {cit.source_id}")
                print()

        elif args.question:
            # Start profiler if enabled
            if profiler:
                profiler.start(args.question)

            response = agent.answer_question(args.question)

            # Finish profiler and print report
            if profiler:
                profile_report = profiler.finish(response.answer)
                print_profile_report(profile_report)
                print()

            print(f"Question: {args.question}")
            print(f"\nAnswer: {response.answer}")
            print(f"\nConfidence: {response.confidence:.2f}")
            if args.verbose and response.uncertainty:
                u = response.uncertainty
                print(f"\nUncertainty Scores:")
                if u.perplexity >= 0:
                    print(f"  Perplexity: {u.perplexity:.2f} (normalized: {u.perplexity_normalized:.2f})")
                else:
                    print(f"  Perplexity: N/A (logprobs not supported by model)")
                print(f"  Semantic Entropy: {u.semantic_entropy:.2f} (normalized: {u.semantic_entropy_normalized:.2f})")
                print(f"  Embedding Consistency: {u.embedding_consistency:.2f}")
                print(f"  Combined Confidence: {u.combined_confidence:.2%}")
            print(f"External sources used: {response.external_info_used}")
            if response.citations:
                print(f"\nCitations: {len(response.citations)}")
                for cit in response.citations:
                    print(f"  - [{cit.source_type}] {cit.source_id}")

        else:
            parser.print_help()

    finally:
        agent.close()


if __name__ == "__main__":
    main()
