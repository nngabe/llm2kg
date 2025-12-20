#!/usr/bin/env python3
"""
LangGraph Agent for Semantic Knowledge Base Construction.

This agent performs iterative knowledge graph extraction with:
- Open extraction ontology that builds into a stricter ontology
- Entity deduplication with description merging
- Vector embeddings based on node descriptions
- Temporal information tracking with uncertainty levels
"""

import os
import re
import time
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Annotated
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PERIODIC MAINTENANCE SETTINGS ---
ENTITY_RESOLUTION_INTERVAL = 100  # Run entity resolution every N documents
ONTOLOGY_CONSOLIDATION_INTERVAL = 100  # Run ontology consolidation every N documents

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# --- TEMPORAL ONTOLOGY ---
class TemporalCertainty(str, Enum):
    """Level of certainty for temporal information."""
    ABSOLUTELY_CERTAIN = "absolutely_certain"      # Exact date from source
    HIGHLY_CERTAIN = "highly_certain"              # Clear timeframe, minor ambiguity
    MODERATELY_CERTAIN = "moderately_certain"      # Approximate, contextual clues
    SLIGHTLY_CERTAIN = "slightly_certain"          # Inferred from indirect evidence
    COMPLETELY_UNCERTAIN = "completely_uncertain"  # No temporal information available


class TemporalInfo(BaseModel):
    """Temporal information with uncertainty quantification."""
    time_period: Optional[str] = Field(
        default=None,
        description="Time period or date (e.g., '1905', '1900-1910', 'early 20th century', 'during WWII')"
    )
    certainty: TemporalCertainty = Field(
        default=TemporalCertainty.COMPLETELY_UNCERTAIN,
        description="Level of certainty for the temporal estimate"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of how the time period was determined"
    )


# --- ONTOLOGY SCHEMAS ---
def normalize_entity(name: str) -> str:
    """Standardize entity naming: 'NEOCLASSICAL_ECONOMICS' -> 'Neoclassical Economics'"""
    return ' '.join(name.replace('_', ' ').split()).title()


class ExtractedNode(BaseModel):
    """Node extracted from text with open ontology."""
    id: str = Field(description="Unique identifier, e.g., 'Albert Einstein'")
    type: str = Field(description="Open category, e.g., 'Person', 'Concept', 'Organization'")
    description: str = Field(description="Summary of this entity based on the text context")
    temporal: Optional[TemporalInfo] = Field(
        default=None,
        description="Temporal information about when this entity or description is relevant"
    )


class ExtractedEdge(BaseModel):
    """Edge extracted from text with open ontology."""
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    relation: str = Field(description="Open relationship type, e.g., 'developed', 'influenced_by'")
    description: str = Field(description="Context explaining this relationship")
    temporal: Optional[TemporalInfo] = Field(
        default=None,
        description="Temporal information about when this relationship held"
    )


class ExtractedKnowledgeGraph(BaseModel):
    """Knowledge graph extracted from a single text chunk."""
    nodes: List[ExtractedNode]
    edges: List[ExtractedEdge]


# --- ONTOLOGY CONSOLIDATION ---
class OntologyLabel(BaseModel):
    """A canonical label in the strict ontology."""
    canonical_name: str = Field(description="The standardized label name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names that map to this label")
    description: str = Field(description="Definition of what this label represents")
    count: int = Field(default=1, description="Number of times this label has been used")


class StrictOntology(BaseModel):
    """Consolidated ontology built from open extractions."""
    node_types: Dict[str, OntologyLabel] = Field(default_factory=dict)
    edge_types: Dict[str, OntologyLabel] = Field(default_factory=dict)


# --- AGENT STATE ---
class AgentState(BaseModel):
    """State for the knowledge graph extraction agent."""
    # Input
    text_chunk: str = ""
    chunk_index: int = 0
    doc_index: int = 0

    # Extraction results
    extracted_kg: Optional[ExtractedKnowledgeGraph] = None

    # Entities to process
    nodes_to_merge: List[Dict[str, Any]] = Field(default_factory=list)
    edges_to_create: List[Dict[str, Any]] = Field(default_factory=list)

    # Ontology building
    new_node_types: List[str] = Field(default_factory=list)
    new_edge_types: List[str] = Field(default_factory=list)

    # Processing metadata
    processing_timestamp: str = ""
    error_message: Optional[str] = None

    # Counters
    nodes_created: int = 0
    nodes_merged: int = 0
    edges_created: int = 0


# --- NEO4J LOADER WITH MERGE SUPPORT ---
class Neo4jAgentLoader:
    """Neo4j loader with entity merging and ontology tracking."""

    def __init__(self, uri: str, user: str, password: str, provider: str = 'local'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.provider = provider

        if provider == 'local':
            self.embedding_model = OllamaEmbeddings(
                base_url="http://host.docker.internal:11434",
                model="qwen3-embedding:8b",
            )
            self.embedding_dim = 4096
        elif provider == 'openai':
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
            self.embedding_dim = 1536
        else:
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
            self.embedding_dim = 1536

    def close(self):
        self.driver.close()

    def sanitize(self, text: str) -> str:
        return re.sub(r'\W+', '_', text).upper()

    def init_indices(self):
        """Initialize Neo4j indices for entities and ontology."""
        queries = [
            # Entity vector index
            f"""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (n:Entity)
            ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            # Ontology label index
            """
            CREATE INDEX ontology_label_idx IF NOT EXISTS
            FOR (n:OntologyLabel)
            ON (n.name)
            """,
            # Text index for entity search
            """
            CREATE TEXT INDEX entity_name_idx IF NOT EXISTS
            FOR (n:Entity)
            ON (n.name)
            """,
        ]

        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug(f"Index creation note: {e}")

    def get_existing_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Check if an entity already exists and return its data."""
        query = """
        MATCH (n:Entity {name: $name})
        RETURN n.name AS name, n.type AS type, n.description AS description,
               n.description_history AS history, n.created_at AS created_at,
               n.updated_at AS updated_at
        """
        with self.driver.session() as session:
            result = session.run(query, name=normalize_entity(name))
            record = result.single()
            if record:
                return dict(record)
        return None

    def create_or_merge_node(
        self,
        node: ExtractedNode,
        timestamp: str,
    ) -> tuple[bool, bool]:
        """
        Create a new node or merge with existing.

        Returns:
            tuple: (was_created, was_merged)
        """
        normalized_name = normalize_entity(node.id)
        existing = self.get_existing_entity(node.id)

        # Prepare temporal info
        temporal_data = None
        if node.temporal:
            temporal_data = {
                "time_period": node.temporal.time_period,
                "certainty": node.temporal.certainty.value,
                "reasoning": node.temporal.reasoning,
            }

        if existing:
            # Merge: append new description
            return self._merge_node(
                name=normalized_name,
                new_description=node.description,
                new_type=node.type,
                temporal=temporal_data,
                timestamp=timestamp,
                existing=existing,
            )
        else:
            # Create new node
            return self._create_node(
                name=normalized_name,
                node_type=node.type,
                description=node.description,
                temporal=temporal_data,
                timestamp=timestamp,
            )

    def _create_node(
        self,
        name: str,
        node_type: str,
        description: str,
        temporal: Optional[Dict],
        timestamp: str,
    ) -> tuple[bool, bool]:
        """Create a new entity node."""
        # Generate embedding from description
        text_to_embed = f"{name} ({node_type}): {description}"
        vector = self.embedding_model.embed_query(text_to_embed)

        # Build temporal properties
        temporal_props = ""
        temporal_params = {}
        if temporal:
            temporal_props = """
                n.temporal_period = $temporal_period,
                n.temporal_certainty = $temporal_certainty,
                n.temporal_reasoning = $temporal_reasoning,
            """
            temporal_params = {
                "temporal_period": temporal.get("time_period"),
                "temporal_certainty": temporal.get("certainty"),
                "temporal_reasoning": temporal.get("reasoning"),
            }

        query = f"""
        CREATE (n:Entity {{
            name: $name,
            type: $type,
            description: $description,
            description_history: [$description_entry],
            embedding: $vector,
            created_at: $timestamp,
            updated_at: $timestamp,
            {temporal_props}
            extraction_count: 1
        }})
        """

        description_entry = json.dumps({
            "text": description,
            "timestamp": timestamp,
            "temporal": temporal,
        })

        with self.driver.session() as session:
            session.run(
                query,
                name=name,
                type=node_type,
                description=description,
                description_entry=description_entry,
                vector=vector,
                timestamp=timestamp,
                **temporal_params,
            )

        return (True, False)  # created, not merged

    def _merge_node(
        self,
        name: str,
        new_description: str,
        new_type: str,
        temporal: Optional[Dict],
        timestamp: str,
        existing: Dict[str, Any],
    ) -> tuple[bool, bool]:
        """Merge new description with existing entity."""
        # Append new description to existing
        existing_desc = existing.get("description", "")
        merged_description = f"{existing_desc}\n\n[{timestamp}] {new_description}"

        # Generate new embedding from merged description
        text_to_embed = f"{name} ({new_type}): {merged_description}"
        vector = self.embedding_model.embed_query(text_to_embed)

        # Build description history entry
        description_entry = json.dumps({
            "text": new_description,
            "timestamp": timestamp,
            "temporal": temporal,
        })

        # Build temporal update if provided
        temporal_update = ""
        temporal_params = {}
        if temporal:
            temporal_update = """
                n.temporal_period = $temporal_period,
                n.temporal_certainty = $temporal_certainty,
                n.temporal_reasoning = $temporal_reasoning,
            """
            temporal_params = {
                "temporal_period": temporal.get("time_period"),
                "temporal_certainty": temporal.get("certainty"),
                "temporal_reasoning": temporal.get("reasoning"),
            }

        query = f"""
        MATCH (n:Entity {{name: $name}})
        SET n.description = $merged_description,
            n.description_history = n.description_history + $description_entry,
            n.embedding = $vector,
            n.updated_at = $timestamp,
            n.extraction_count = n.extraction_count + 1,
            {temporal_update}
            n.type = CASE WHEN n.type = $new_type THEN n.type ELSE n.type + '/' + $new_type END
        """

        with self.driver.session() as session:
            session.run(
                query,
                name=name,
                merged_description=merged_description,
                description_entry=description_entry,
                vector=vector,
                timestamp=timestamp,
                new_type=new_type,
                **temporal_params,
            )

        return (False, True)  # not created, was merged

    def create_edge(
        self,
        edge: ExtractedEdge,
        timestamp: str,
    ) -> bool:
        """Create or update an edge between entities."""
        rel_type = self.sanitize(edge.relation)
        source = normalize_entity(edge.source)
        target = normalize_entity(edge.target)

        # Prepare temporal info
        temporal_props = ""
        temporal_params = {}
        if edge.temporal:
            temporal_props = """
                r.temporal_period = $temporal_period,
                r.temporal_certainty = $temporal_certainty,
                r.temporal_reasoning = $temporal_reasoning,
            """
            temporal_params = {
                "temporal_period": edge.temporal.time_period,
                "temporal_certainty": edge.temporal.certainty.value,
                "temporal_reasoning": edge.temporal.reasoning,
            }

        query = f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET
            r.description = $description,
            r.created_at = $timestamp,
            r.extraction_count = 1,
            {temporal_props}
            r.updated_at = $timestamp
        ON MATCH SET
            r.description = r.description + ' | ' + $description,
            r.extraction_count = r.extraction_count + 1,
            r.updated_at = $timestamp
        """

        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    source=source,
                    target=target,
                    description=edge.description,
                    timestamp=timestamp,
                    **temporal_params,
                )
            return True
        except Exception as e:
            logger.warning(f"Failed to create edge {source} -> {target}: {e}")
            return False

    def update_ontology(self, label_type: str, label: str, is_node: bool = True):
        """Track ontology labels for building strict ontology."""
        category = "NodeType" if is_node else "EdgeType"

        query = """
        MERGE (o:OntologyLabel {name: $label, category: $category})
        ON CREATE SET o.count = 1, o.first_seen = datetime()
        ON MATCH SET o.count = o.count + 1, o.last_seen = datetime()
        """

        with self.driver.session() as session:
            session.run(query, label=label, category=category)

    def get_ontology_stats(self) -> Dict[str, List[Dict]]:
        """Get current ontology statistics."""
        query = """
        MATCH (o:OntologyLabel)
        RETURN o.category AS category, o.name AS name, o.count AS count
        ORDER BY o.count DESC
        """

        stats = {"NodeType": [], "EdgeType": []}
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                stats[record["category"]].append({
                    "name": record["name"],
                    "count": record["count"],
                })

        return stats

    def find_potential_duplicates(self, similarity_threshold: float = 0.90) -> List[tuple]:
        """
        Find entities with similar vector embeddings that may be duplicates.
        Returns list of (entity1, entity2, score) tuples.
        """
        query = """
        MATCH (n:Entity)
        WHERE n.embedding IS NOT NULL
        CALL db.index.vector.queryNodes('entity_embeddings', 5, n.embedding)
        YIELD node AS candidate, score
        WHERE score > $threshold AND n.name <> candidate.name
        AND id(n) < id(candidate)
        RETURN n.name AS entity1, n.description AS desc1,
               candidate.name AS entity2, candidate.description AS desc2,
               score
        ORDER BY score DESC
        """

        duplicates = []
        with self.driver.session() as session:
            result = session.run(query, threshold=similarity_threshold)
            for record in result:
                duplicates.append((
                    record["entity1"],
                    record["desc1"],
                    record["entity2"],
                    record["desc2"],
                    record["score"],
                ))

        return duplicates

    def merge_duplicate_nodes(self, keep_name: str, discard_name: str, merged_description: str, timestamp: str):
        """
        Merge two duplicate entities using APOC.
        Combines descriptions and updates the vector embedding.
        """
        # Generate new embedding from merged description
        text_to_embed = f"{keep_name}: {merged_description}"
        vector = self.embedding_model.embed_query(text_to_embed)

        # Use APOC to merge nodes
        query = """
        MATCH (keep:Entity {name: $keep_name})
        MATCH (discard:Entity {name: $discard_name})

        // Update keep node with merged data before APOC merge
        SET keep.description = $merged_description,
            keep.embedding = $vector,
            keep.updated_at = $timestamp,
            keep.merged_from = COALESCE(keep.merged_from, []) + [$discard_name]

        WITH keep, discard
        CALL apoc.refactor.mergeNodes([keep, discard], {
            properties: {
                name: 'discard',
                embedding: 'discard',
                description: 'discard',
                type: 'combine'
            },
            mergeRels: true
        })
        YIELD node
        RETURN node.name AS name
        """

        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    keep_name=keep_name,
                    discard_name=discard_name,
                    merged_description=merged_description,
                    vector=vector,
                    timestamp=timestamp,
                )
            return True
        except Exception as e:
            logger.warning(f"Failed to merge {discard_name} into {keep_name}: {e}")
            return False

    def get_similar_ontology_labels(self, category: str = "NodeType") -> List[tuple]:
        """Find similar ontology labels that could be consolidated."""
        query = """
        MATCH (o1:OntologyLabel {category: $category})
        MATCH (o2:OntologyLabel {category: $category})
        WHERE id(o1) < id(o2)
        AND apoc.text.levenshteinSimilarity(toLower(o1.name), toLower(o2.name)) > 0.7
        RETURN o1.name AS label1, o1.count AS count1,
               o2.name AS label2, o2.count AS count2,
               apoc.text.levenshteinSimilarity(toLower(o1.name), toLower(o2.name)) AS similarity
        ORDER BY similarity DESC
        """

        similar = []
        try:
            with self.driver.session() as session:
                result = session.run(query, category=category)
                for record in result:
                    similar.append((
                        record["label1"],
                        record["count1"],
                        record["label2"],
                        record["count2"],
                        record["similarity"],
                    ))
        except Exception as e:
            logger.debug(f"Ontology similarity check failed (APOC may not be available): {e}")

        return similar

    def consolidate_ontology_label(self, canonical: str, alias: str, category: str = "NodeType"):
        """Map an alias label to a canonical label."""
        # Update all entities using the alias to use the canonical label
        if category == "NodeType":
            query = """
            MATCH (n:Entity)
            WHERE n.type CONTAINS $alias
            SET n.type = replace(n.type, $alias, $canonical)
            """
        else:
            # For edge types, we'd need to recreate relationships
            # This is more complex, so we just track the mapping
            query = """
            MATCH (o:OntologyLabel {name: $alias, category: $category})
            SET o.canonical = $canonical, o.is_alias = true
            """

        with self.driver.session() as session:
            session.run(query, alias=alias, canonical=canonical, category=category)


# --- LANGGRAPH AGENT ---
class KnowledgeGraphAgent:
    """LangGraph agent for knowledge graph extraction."""

    def __init__(self, provider: str = 'local'):
        self.provider = provider
        self.loader = Neo4jAgentLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, provider)

        # Initialize LLM
        if provider == 'local':
            self.llm = ChatOllama(
                model="qwen3:30b-a3b",
                temperature=0,
                base_url="http://host.docker.internal:11434",
            )
        elif provider == 'openai':
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        elif provider == 'google':
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        else:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        self.structured_llm = self.llm.with_structured_output(ExtractedKnowledgeGraph)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Define the graph
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("extract", self._extract_node)
        builder.add_node("process_entities", self._process_entities_node)
        builder.add_node("create_edges", self._create_edges_node)
        builder.add_node("update_ontology", self._update_ontology_node)

        # Add edges
        builder.add_edge(START, "extract")
        builder.add_conditional_edges(
            "extract",
            self._should_continue,
            {
                "continue": "process_entities",
                "end": END,
            }
        )
        builder.add_edge("process_entities", "create_edges")
        builder.add_edge("create_edges", "update_ontology")
        builder.add_edge("update_ontology", END)

        return builder.compile()

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue processing."""
        if state.error_message or state.extracted_kg is None:
            return "end"
        if not state.extracted_kg.nodes:
            return "end"
        return "continue"

    def _extract_node(self, state: AgentState) -> Dict[str, Any]:
        """Extract knowledge graph from text chunk."""
        timestamp = datetime.utcnow().isoformat()

        system_prompt = """You are a Knowledge Graph expert. Extract a semi-structured graph from the text.

For each entity (node):
1. Provide a unique 'id' (the entity name)
2. Assign a 'type' (e.g., Person, Concept, Organization, Event, Location, Theory, Law)
3. Write a 'description' summarizing who/what the entity is based on the text
4. If temporal information is available or can be inferred, include it with:
   - time_period: The date, year, or time range (e.g., "1905", "1900-1910", "early 20th century")
   - certainty: How certain this temporal info is:
     * absolutely_certain: Exact date explicitly stated
     * highly_certain: Clear timeframe with minor ambiguity
     * moderately_certain: Approximate, from contextual clues
     * slightly_certain: Inferred from indirect evidence
     * completely_uncertain: No temporal information available
   - reasoning: Brief explanation of how the time was determined

For each relationship (edge):
1. Identify source and target entities
2. Use descriptive relation types (e.g., 'developed', 'influenced_by', 'occurred_during')
3. Include a description explaining the relationship context
4. Include temporal information if the relationship has a specific time frame

Be thorough but precise. Extract all meaningful entities and relationships from the text."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Text: {text}"),
        ])

        chain = prompt | self.structured_llm

        try:
            extracted = chain.invoke({"text": state.text_chunk})

            return {
                "extracted_kg": extracted,
                "processing_timestamp": timestamp,
                "error_message": None,
            }
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            return {
                "extracted_kg": None,
                "processing_timestamp": timestamp,
                "error_message": str(e),
            }

    def _process_entities_node(self, state: AgentState) -> Dict[str, Any]:
        """Process extracted entities - create or merge."""
        if not state.extracted_kg:
            return {}

        nodes_created = 0
        nodes_merged = 0

        for node in state.extracted_kg.nodes:
            created, merged = self.loader.create_or_merge_node(
                node=node,
                timestamp=state.processing_timestamp,
            )
            if created:
                nodes_created += 1
            if merged:
                nodes_merged += 1

        return {
            "nodes_created": nodes_created,
            "nodes_merged": nodes_merged,
        }

    def _create_edges_node(self, state: AgentState) -> Dict[str, Any]:
        """Create edges between entities."""
        if not state.extracted_kg:
            return {}

        edges_created = 0

        for edge in state.extracted_kg.edges:
            success = self.loader.create_edge(
                edge=edge,
                timestamp=state.processing_timestamp,
            )
            if success:
                edges_created += 1

        return {
            "edges_created": edges_created,
        }

    def _update_ontology_node(self, state: AgentState) -> Dict[str, Any]:
        """Update ontology tracking with new labels."""
        if not state.extracted_kg:
            return {}

        new_node_types = []
        new_edge_types = []

        # Track node types
        for node in state.extracted_kg.nodes:
            self.loader.update_ontology(node.type, node.type, is_node=True)
            new_node_types.append(node.type)

        # Track edge types
        for edge in state.extracted_kg.edges:
            self.loader.update_ontology(edge.relation, edge.relation, is_node=False)
            new_edge_types.append(edge.relation)

        return {
            "new_node_types": list(set(new_node_types)),
            "new_edge_types": list(set(new_edge_types)),
        }

    def process_chunk(self, text: str, chunk_index: int = 0, doc_index: int = 0) -> AgentState:
        """Process a single text chunk through the agent."""
        initial_state = AgentState(
            text_chunk=text,
            chunk_index=chunk_index,
            doc_index=doc_index,
        )

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Convert back to AgentState if needed
        if isinstance(result, dict):
            return AgentState(**result)
        return result

    def get_ontology_report(self) -> str:
        """Generate a report of the current ontology."""
        stats = self.loader.get_ontology_stats()

        lines = [
            "=" * 60,
            "ONTOLOGY REPORT",
            "=" * 60,
            "",
            "NODE TYPES:",
            "-" * 40,
        ]

        for item in stats["NodeType"][:20]:
            lines.append(f"  {item['name']}: {item['count']} occurrences")

        lines.extend([
            "",
            "EDGE TYPES:",
            "-" * 40,
        ])

        for item in stats["EdgeType"][:20]:
            lines.append(f"  {item['name']}: {item['count']} occurrences")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _llm_check_same_entity(self, name1: str, desc1: str, name2: str, desc2: str) -> bool:
        """Use LLM to verify if two entities are the same."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data cleaning expert. Determine if two entities refer to the same real-world object.
Consider both the names and descriptions when making your decision."""),
            ("human", """Entity 1: {name1}
Description: {desc1}

Entity 2: {name2}
Description: {desc2}

Are these likely the same entity? Answer ONLY with 'YES' or 'NO'."""),
        ])

        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "name1": name1, "desc1": desc1 or "No description",
                "name2": name2, "desc2": desc2 or "No description",
            }).content.strip().upper()
            return "YES" in response
        except Exception as e:
            logger.warning(f"LLM entity check failed: {e}")
            return False

    def _llm_check_same_label(self, label1: str, label2: str, category: str) -> tuple[bool, str]:
        """Use LLM to verify if two ontology labels are the same and pick canonical."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an ontology expert. Determine if two {category} labels refer to the same concept.
If they are the same, pick the more standard/canonical label."""),
            ("human", """Label 1: {label1}
Label 2: {label2}

Are these the same concept? If YES, which label is more canonical?
Answer in format: YES|canonical_label or NO"""),
        ])

        chain = prompt | self.llm
        try:
            response = chain.invoke({"label1": label1, "label2": label2}).content.strip()
            if response.upper().startswith("YES"):
                parts = response.split("|")
                canonical = parts[1].strip() if len(parts) > 1 else label1
                return True, canonical
            return False, ""
        except Exception as e:
            logger.warning(f"LLM label check failed: {e}")
            return False, ""

    def run_entity_resolution(self, similarity_threshold: float = 0.92) -> int:
        """
        Run entity resolution to find and merge duplicate entities.
        Returns number of entities merged.
        """
        logger.info("Running entity resolution...")
        timestamp = datetime.utcnow().isoformat()

        # Find potential duplicates
        candidates = self.loader.find_potential_duplicates(similarity_threshold)
        logger.info(f"Found {len(candidates)} potential duplicate pairs")

        merged_count = 0

        for name1, desc1, name2, desc2, score in candidates:
            logger.info(f"Analyzing: '{name1}' vs '{name2}' (score: {score:.4f})")

            # LLM verification
            is_match = self._llm_check_same_entity(name1, desc1, name2, desc2)

            if is_match:
                logger.info(f"  LLM confirmed match")

                # Keep the longer/more descriptive name
                if len(name1) >= len(name2):
                    keep, discard = name1, name2
                    keep_desc, discard_desc = desc1, desc2
                else:
                    keep, discard = name2, name1
                    keep_desc, discard_desc = desc2, desc1

                # Merge descriptions
                merged_desc = f"{keep_desc}\n\n[Merged from {discard}] {discard_desc}"

                # Execute merge
                success = self.loader.merge_duplicate_nodes(
                    keep_name=keep,
                    discard_name=discard,
                    merged_description=merged_desc,
                    timestamp=timestamp,
                )

                if success:
                    merged_count += 1
                    logger.info(f"  Merged '{discard}' into '{keep}'")
            else:
                logger.info(f"  LLM rejected match (false positive)")

        logger.info(f"Entity resolution complete: merged {merged_count} entities")
        return merged_count

    def run_ontology_consolidation(self) -> Dict[str, int]:
        """
        Run ontology consolidation to map similar labels to canonical forms.
        Returns counts of consolidated labels per category.
        """
        logger.info("Running ontology consolidation...")

        consolidated = {"NodeType": 0, "EdgeType": 0}

        for category in ["NodeType", "EdgeType"]:
            similar_pairs = self.loader.get_similar_ontology_labels(category)
            logger.info(f"Found {len(similar_pairs)} similar {category} pairs")

            for label1, count1, label2, count2, similarity in similar_pairs:
                logger.info(f"Analyzing {category}: '{label1}' vs '{label2}' (sim: {similarity:.2f})")

                # LLM verification
                is_match, canonical = self._llm_check_same_label(label1, label2, category)

                if is_match:
                    # Determine which to keep based on LLM suggestion and usage count
                    if canonical == label2 or (canonical not in [label1, label2] and count2 > count1):
                        canonical, alias = label2, label1
                    else:
                        canonical, alias = label1, label2

                    logger.info(f"  Consolidating '{alias}' -> '{canonical}'")

                    self.loader.consolidate_ontology_label(
                        canonical=canonical,
                        alias=alias,
                        category=category,
                    )
                    consolidated[category] += 1

        logger.info(f"Ontology consolidation complete: {consolidated}")
        return consolidated

    def run_periodic_maintenance(self) -> Dict[str, Any]:
        """Run all periodic maintenance tasks."""
        results = {
            "entities_merged": 0,
            "ontology_consolidated": {"NodeType": 0, "EdgeType": 0},
        }

        # Entity resolution
        results["entities_merged"] = self.run_entity_resolution()

        # Ontology consolidation
        results["ontology_consolidated"] = self.run_ontology_consolidation()

        return results

    def close(self):
        """Clean up resources."""
        self.loader.close()


# --- PIPELINE EXECUTION ---
def run_agent_pipeline(
    provider: str = 'local',
    limit_docs: int = 5,
    restart_index: int = 0,
    subject: str = 'economics',
    entity_resolution_interval: int = ENTITY_RESOLUTION_INTERVAL,
    ontology_consolidation_interval: int = ONTOLOGY_CONSOLIDATION_INTERVAL,
):
    """Run the knowledge graph extraction agent pipeline."""

    agent = KnowledgeGraphAgent(provider=provider)

    try:
        # Initialize indices
        agent.loader.init_indices()

        print("Loading Data...")

        # Load dataset
        if subject == 'economics':
            dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "economics-corpus", split='train')
        elif subject == 'law':
            dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "law-corpus", split='train')
        elif subject == 'physics':
            dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "physics-corpus", split='train')
        else:
            raise ValueError(f"Unknown subject: {subject}")

        end_index = restart_index + limit_docs if limit_docs > 0 else len(dataset)
        dataset = dataset.select(range(restart_index, min(end_index, len(dataset))))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        print(f"\n[doc/total_docs][chunk] Processing {len(dataset)} documents\n")
        print(f"Periodic maintenance: entity resolution every {entity_resolution_interval} docs, "
              f"ontology consolidation every {ontology_consolidation_interval} docs\n")

        total_nodes_created = 0
        total_nodes_merged = 0
        total_edges_created = 0
        total_entities_resolved = 0
        total_labels_consolidated = {"NodeType": 0, "EdgeType": 0}

        # Track documents processed for periodic maintenance
        docs_since_entity_resolution = 0
        docs_since_ontology_consolidation = 0

        for doc_idx, entry in enumerate(dataset):
            text = entry['text']
            if len(text.strip()) < 50:
                continue

            chunks = text_splitter.split_text(text)

            tic = time.time()
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # Process through agent
                    result = agent.process_chunk(
                        text=chunk,
                        chunk_index=chunk_idx,
                        doc_index=doc_idx + restart_index,
                    )

                    if result.error_message:
                        logger.warning(
                            f"[{doc_idx + restart_index}/{end_index}][{chunk_idx}] "
                            f"Error: {result.error_message[:100]}"
                        )
                        continue

                    total_nodes_created += result.nodes_created
                    total_nodes_merged += result.nodes_merged
                    total_edges_created += result.edges_created

                    toc = time.time() - tic
                    tic = time.time()

                    print(
                        f"[{doc_idx + restart_index}/{end_index}][{chunk_idx}] "
                        f"Created: {result.nodes_created} nodes, "
                        f"Merged: {result.nodes_merged} nodes, "
                        f"Edges: {result.edges_created} "
                        f"({toc:.2f}s)"
                    )

                    # Show new ontology labels if any
                    if result.new_node_types:
                        logger.debug(f"  Node types: {result.new_node_types}")
                    if result.new_edge_types:
                        logger.debug(f"  Edge types: {result.new_edge_types}")

                except Exception as e:
                    logger.error(
                        f"[{doc_idx + restart_index}/{end_index}][{chunk_idx}] "
                        f"Unexpected error: {e}"
                    )

            # Increment document counters
            docs_since_entity_resolution += 1
            docs_since_ontology_consolidation += 1

            # --- PERIODIC ENTITY RESOLUTION ---
            if docs_since_entity_resolution >= entity_resolution_interval:
                print(f"\n{'=' * 60}")
                print(f"PERIODIC ENTITY RESOLUTION (after {docs_since_entity_resolution} documents)")
                print(f"{'=' * 60}")

                try:
                    merged = agent.run_entity_resolution()
                    total_entities_resolved += merged
                    docs_since_entity_resolution = 0
                    print(f"Resolved {merged} duplicate entities")
                except Exception as e:
                    logger.error(f"Entity resolution failed: {e}")

                print(f"{'=' * 60}\n")

            # --- PERIODIC ONTOLOGY CONSOLIDATION ---
            if docs_since_ontology_consolidation >= ontology_consolidation_interval:
                print(f"\n{'=' * 60}")
                print(f"PERIODIC ONTOLOGY CONSOLIDATION (after {docs_since_ontology_consolidation} documents)")
                print(f"{'=' * 60}")

                try:
                    consolidated = agent.run_ontology_consolidation()
                    total_labels_consolidated["NodeType"] += consolidated["NodeType"]
                    total_labels_consolidated["EdgeType"] += consolidated["EdgeType"]
                    docs_since_ontology_consolidation = 0
                    print(f"Consolidated {consolidated['NodeType']} node types, {consolidated['EdgeType']} edge types")
                except Exception as e:
                    logger.error(f"Ontology consolidation failed: {e}")

                print(f"{'=' * 60}\n")

        # --- FINAL MAINTENANCE RUN ---
        print(f"\n{'=' * 60}")
        print("FINAL MAINTENANCE RUN")
        print(f"{'=' * 60}")

        try:
            # Final entity resolution
            merged = agent.run_entity_resolution()
            total_entities_resolved += merged

            # Final ontology consolidation
            consolidated = agent.run_ontology_consolidation()
            total_labels_consolidated["NodeType"] += consolidated["NodeType"]
            total_labels_consolidated["EdgeType"] += consolidated["EdgeType"]
        except Exception as e:
            logger.error(f"Final maintenance failed: {e}")

        print(f"{'=' * 60}\n")

        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total nodes created: {total_nodes_created}")
        print(f"Total nodes merged (same name): {total_nodes_merged}")
        print(f"Total edges created: {total_edges_created}")
        print(f"Total entities resolved (duplicates): {total_entities_resolved}")
        print(f"Total node types consolidated: {total_labels_consolidated['NodeType']}")
        print(f"Total edge types consolidated: {total_labels_consolidated['EdgeType']}")
        print("=" * 60)

        # Print ontology report
        print("\n" + agent.get_ontology_report())

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    finally:
        agent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangGraph Agent for Knowledge Graph Extraction"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default='local',
        choices=['local', 'openai', 'google'],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--limit_docs",
        type=int,
        default=5,
        help="Number of documents to process (0 for all)",
    )
    parser.add_argument(
        "--restart_index",
        type=int,
        default=0,
        help="Document index to start from",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default='economics',
        choices=['economics', 'law', 'physics'],
        help="Subject corpus to use",
    )
    parser.add_argument(
        "--entity_resolution_interval",
        type=int,
        default=ENTITY_RESOLUTION_INTERVAL,
        help=f"Run entity resolution every N documents (default: {ENTITY_RESOLUTION_INTERVAL})",
    )
    parser.add_argument(
        "--ontology_consolidation_interval",
        type=int,
        default=ONTOLOGY_CONSOLIDATION_INTERVAL,
        help=f"Run ontology consolidation every N documents (default: {ONTOLOGY_CONSOLIDATION_INTERVAL})",
    )

    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  Provider: {args.provider}")
    print(f"  Subject: {args.subject}")
    print(f"  Documents: {args.limit_docs} starting from {args.restart_index}")
    print(f"  Entity resolution interval: {args.entity_resolution_interval} docs")
    print(f"  Ontology consolidation interval: {args.ontology_consolidation_interval} docs")
    print(f"\nBuilding {args.subject} KG with LangGraph agent...\n")

    run_agent_pipeline(
        provider=args.provider,
        limit_docs=args.limit_docs,
        restart_index=args.restart_index,
        subject=args.subject,
        entity_resolution_interval=args.entity_resolution_interval,
        ontology_consolidation_interval=args.ontology_consolidation_interval,
    )
