#!/usr/bin/env python3
"""
LangGraph Agent for Semantic Knowledge Base Construction.

This agent performs iterative knowledge graph extraction with:
- Open extraction ontology that converges to a strict ontology (5-15 entity types, 5-20 relationship types)
- Dual labels: open extraction labels kept alongside assigned ontology labels
- Entity deduplication with description merging
- Vector embeddings based on node descriptions
- Ontology stabilization detection with automatic freezing and node reassignment
"""

import os
import re
import time
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

from langgraph.graph import StateGraph, START, END

from prompts import (
    KG_EXTRACTION_PROMPT,
    get_extraction_prompt,
    ONTOLOGY_MAPPING_PROMPT_TEMPLATE,
    FROZEN_ONTOLOGY_MAPPING_PROMPT_TEMPLATE,
    LABEL_GENERALIZATION_PROMPT_TEMPLATE,
    ENTITY_VERIFICATION_PROMPT,
    LABEL_VERIFICATION_PROMPT_TEMPLATE,
    ONTOLOGY_MERGE_PROMPT_TEMPLATE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logging from httpx/langchain
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

# --- PERIODIC MAINTENANCE SETTINGS ---
ENTITY_RESOLUTION_INTERVAL = 600  # Run entity resolution every N documents
ONTOLOGY_CONSOLIDATION_INTERVAL = 9999999  # Run ontology consolidation every N documents

# --- MODEL DEFAULTS ---
DEFAULT_LOCAL_EXTRACTION_MODEL = "ministral-3:14b"
DEFAULT_LOCAL_UTILITY_MODEL = "nemotron-3-nano:30b"
DEFAULT_LOCAL_EMBEDDING_MODEL = "qwen3-embedding:8b"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash"
DEFAULT_MAINTENANCE_MODEL = "gpt-5.2"  # Used for entity resolution & ontology consolidation

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# --- ONTOLOGY CONFIGURATION ---
TARGET_ENTITY_TYPES = (5, 15)  # min, max entity types for convergence
TARGET_RELATIONSHIP_TYPES = (5, 20)  # min, max relationship types for convergence
STABILITY_THRESHOLD = 3  # versions without changes to consider stable

# --- ONTOLOGY IMPORTS ---
from ontologies import (
    get_ontology,
    OntologyConfig,
    ONTOLOGY_SMALL,
    ONTOLOGY_MEDIUM,
    ONTOLOGY_LARGE,
    DEFAULT_ONTOLOGY,
    ONTOLOGIES,
)


# --- ONTOLOGY SCHEMAS ---
def normalize_entity(name: str) -> str:
    """Standardize entity naming: 'NEOCLASSICAL_ECONOMICS' -> 'Neoclassical Economics'"""
    return ' '.join(name.replace('_', ' ').split()).title()


class NodeProperties(BaseModel):
    """Properties attached to a node."""
    description: str = Field(description="Summary of this entity based on the text context")


class RelationshipProperties(BaseModel):
    """Properties attached to a relationship."""
    description: str = Field(description="Detailed explanation of the relationship")


class ExtractedNode(BaseModel):
    """Node extracted from text with open ontology."""
    id: str = Field(description="Unique identifier using canonical name, e.g., 'Albert Einstein'")
    labels: List[str] = Field(description="PascalCase entity types, e.g., ['Person', 'Scientist']")
    properties: NodeProperties = Field(description="Node properties including description")

    @property
    def open_type(self) -> str:
        """Get first label for backward compatibility."""
        return self.labels[0] if self.labels else "Unknown"

    @property
    def description(self) -> str:
        """Get description for backward compatibility."""
        return self.properties.description


class ExtractedRelationship(BaseModel):
    """Relationship extracted from text with open ontology."""
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    type: str = Field(description="SCREAMING_SNAKE_CASE relationship type, e.g., 'DEVELOPED', 'INFLUENCED_BY'")
    properties: RelationshipProperties = Field(description="Relationship properties including description")

    @property
    def open_relation(self) -> str:
        """Get type for backward compatibility."""
        return self.type

    @property
    def description(self) -> str:
        """Get description for backward compatibility."""
        return self.properties.description


class ExtractedKnowledgeGraph(BaseModel):
    """Knowledge graph extracted from a single text chunk."""
    nodes: List[ExtractedNode] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

    @property
    def edges(self) -> List[ExtractedRelationship]:
        """Alias for backward compatibility."""
        return self.relationships


# --- ONTOLOGY CONSOLIDATION ---
class OntologyLabel(BaseModel):
    """A canonical label in the strict ontology."""
    canonical_name: str = Field(description="The standardized label name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names that map to this label")
    description: str = Field(default="", description="Definition of what this label represents")
    count: int = Field(default=1, description="Number of times this label has been used")


class OntologyState(BaseModel):
    """Tracks the evolving ontology state with stabilization detection."""
    entity_types: Dict[str, OntologyLabel] = Field(default_factory=dict)
    relationship_types: Dict[str, OntologyLabel] = Field(default_factory=dict)
    is_frozen: bool = Field(default=False, description="Whether the ontology has been frozen")
    version: int = Field(default=0, description="Current ontology version")
    last_change_version: int = Field(default=0, description="Version when last change occurred")

    def check_stabilization(self) -> bool:
        """Check if ontology has stabilized (no changes for STABILITY_THRESHOLD versions)."""
        if self.version - self.last_change_version >= STABILITY_THRESHOLD:
            entity_count = len(self.entity_types)
            rel_count = len(self.relationship_types)
            if (TARGET_ENTITY_TYPES[0] <= entity_count <= TARGET_ENTITY_TYPES[1] and
                TARGET_RELATIONSHIP_TYPES[0] <= rel_count <= TARGET_RELATIONSHIP_TYPES[1]):
                return True
        return False

    def get_entity_type_names(self) -> List[str]:
        """Get list of canonical entity type names."""
        return list(self.entity_types.keys())

    def get_relationship_type_names(self) -> List[str]:
        """Get list of canonical relationship type names."""
        return list(self.relationship_types.keys())


class OntologyAssignment(BaseModel):
    """Assignment of an open label to an ontology label."""
    open_label: str = Field(description="The original open extraction label")
    ontology_label: str = Field(description="The assigned canonical ontology label")
    is_new: bool = Field(default=False, description="Whether this is a new ontology label")


class OntologyAssignmentBatch(BaseModel):
    """Batch of ontology assignments for nodes and edges."""
    node_assignments: List[OntologyAssignment] = Field(default_factory=list)
    edge_assignments: List[OntologyAssignment] = Field(default_factory=list)


class OntologyMergeProposal(BaseModel):
    """Proposal to merge ontology labels."""
    merge_from: str = Field(description="Label to merge away")
    merge_to: str = Field(description="Label to merge into")
    reason: str = Field(description="Reason for the merge")


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

    # Ontology assignments (open_label -> ontology_label)
    node_ontology_assignments: Dict[str, str] = Field(default_factory=dict)
    edge_ontology_assignments: Dict[str, str] = Field(default_factory=dict)

    # Ontology building
    new_node_types: List[str] = Field(default_factory=list)
    new_edge_types: List[str] = Field(default_factory=list)
    new_ontology_labels: List[str] = Field(default_factory=list)

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
            self.embedding_model_name = DEFAULT_LOCAL_EMBEDDING_MODEL
            self.embedding_model = OllamaEmbeddings(
                base_url="http://host.docker.internal:11434",
                model=self.embedding_model_name,
            )
            self.embedding_dim = 4096
        elif provider == 'openai':
            self.embedding_model_name = DEFAULT_OPENAI_EMBEDDING_MODEL
            self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
            self.embedding_dim = 1536
        else:
            self.embedding_model_name = DEFAULT_OPENAI_EMBEDDING_MODEL
            self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
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
        RETURN n.name AS name, n.open_type AS open_type, n.ontology_type AS ontology_type,
               n.description AS description, n.open_type_history AS open_type_history,
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
        ontology_type: str,
        timestamp: str,
    ) -> tuple[bool, bool]:
        """
        Create a new node or merge with existing.

        Args:
            node: The extracted node with open_type
            ontology_type: The assigned ontology label
            timestamp: Processing timestamp

        Returns:
            tuple: (was_created, was_merged)
        """
        normalized_name = normalize_entity(node.id)
        existing = self.get_existing_entity(node.id)

        if existing:
            # Merge: append new description
            return self._merge_node(
                name=normalized_name,
                new_description=node.description,
                open_type=node.open_type,
                ontology_type=ontology_type,
                timestamp=timestamp,
                existing=existing,
            )
        else:
            # Create new node
            return self._create_node(
                name=normalized_name,
                open_type=node.open_type,
                ontology_type=ontology_type,
                description=node.description,
                timestamp=timestamp,
            )

    def _create_node(
        self,
        name: str,
        open_type: str,
        ontology_type: str,
        description: str,
        timestamp: str,
    ) -> tuple[bool, bool]:
        """Create a new entity node with both open and ontology types."""
        # Generate embedding from description
        text_to_embed = f"{name} ({ontology_type}): {description}"
        vector = self.embedding_model.embed_query(text_to_embed)

        query = """
        CREATE (n:Entity {
            name: $name,
            open_type: $open_type,
            ontology_type: $ontology_type,
            open_type_history: [$open_type],
            description: $description,
            description_history: [$description_entry],
            embedding: $vector,
            created_at: $timestamp,
            updated_at: $timestamp,
            extraction_count: 1
        })
        """

        description_entry = json.dumps({
            "text": description,
            "timestamp": timestamp,
            "open_type": open_type,
        })

        with self.driver.session() as session:
            session.run(
                query,
                name=name,
                open_type=open_type,
                ontology_type=ontology_type,
                description=description,
                description_entry=description_entry,
                vector=vector,
                timestamp=timestamp,
            )

        return (True, False)  # created, not merged

    def _merge_node(
        self,
        name: str,
        new_description: str,
        open_type: str,
        ontology_type: str,
        timestamp: str,
        existing: Dict[str, Any],
    ) -> tuple[bool, bool]:
        """Merge new description with existing entity."""
        # Append new description to existing
        existing_desc = existing.get("description", "")
        merged_description = f"{existing_desc}\n\n[{timestamp}] {new_description}"

        # Generate new embedding from merged description
        text_to_embed = f"{name} ({ontology_type}): {merged_description}"
        vector = self.embedding_model.embed_query(text_to_embed)

        # Build description history entry
        description_entry = json.dumps({
            "text": new_description,
            "timestamp": timestamp,
            "open_type": open_type,
        })

        query = """
        MATCH (n:Entity {name: $name})
        SET n.description = $merged_description,
            n.description_history = n.description_history + $description_entry,
            n.open_type_history = CASE
                WHEN $open_type IN n.open_type_history THEN n.open_type_history
                ELSE n.open_type_history + $open_type
            END,
            n.ontology_type = $ontology_type,
            n.embedding = $vector,
            n.updated_at = $timestamp,
            n.extraction_count = n.extraction_count + 1
        """

        with self.driver.session() as session:
            session.run(
                query,
                name=name,
                merged_description=merged_description,
                description_entry=description_entry,
                open_type=open_type,
                ontology_type=ontology_type,
                vector=vector,
                timestamp=timestamp,
            )

        return (False, True)  # not created, was merged

    def create_edge(
        self,
        edge: ExtractedRelationship,
        ontology_relation: str,
        timestamp: str,
    ) -> bool:
        """Create or update an edge between entities with both open and ontology relations."""
        # Use ontology_relation for the relationship type in Neo4j
        rel_type = self.sanitize(ontology_relation)
        source = normalize_entity(edge.source)
        target = normalize_entity(edge.target)

        query = f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET
            r.open_relation = $open_relation,
            r.ontology_relation = $ontology_relation,
            r.open_relation_history = [$open_relation],
            r.description = $description,
            r.created_at = $timestamp,
            r.extraction_count = 1,
            r.updated_at = $timestamp
        ON MATCH SET
            r.description = r.description + ' | ' + $description,
            r.open_relation_history = CASE
                WHEN $open_relation IN r.open_relation_history THEN r.open_relation_history
                ELSE r.open_relation_history + $open_relation
            END,
            r.extraction_count = r.extraction_count + 1,
            r.updated_at = $timestamp
        """

        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    source=source,
                    target=target,
                    open_relation=edge.open_relation,
                    ontology_relation=ontology_relation,
                    description=edge.description,
                    timestamp=timestamp,
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
    """LangGraph agent for knowledge graph extraction with ontology convergence."""

    def __init__(self, provider: str = 'local', model: str = None, utility_model: str = None,
                 ontology: str = None):
        self.provider = provider
        self.loader = Neo4jAgentLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, provider)

        # Load ontology configuration
        self.ontology_config = get_ontology(ontology)
        self.ontology_name = self.ontology_config.name

        # Initialize extraction LLM (fine-tuned for KG extraction)
        if provider == 'local':
            self.extraction_model = model or DEFAULT_LOCAL_EXTRACTION_MODEL
            self.llm = ChatOllama(
                model=self.extraction_model,
                temperature=0.0,  # Slight temperature to avoid empty responses
                base_url="http://host.docker.internal:11434",
                num_ctx=16384,  # Increase context window for long prompts + output
                num_predict=8192,  # Limit tokens for KG extraction
                format="json",  # Enable JSON mode for structured output
            )
            # Utility LLM for ontology assignment and entity checking
            # A/B testing showed temp=0 performs equally well and is more deterministic
            self.utility_model = utility_model or DEFAULT_LOCAL_UTILITY_MODEL
            self.utility_llm = ChatOllama(
                model=self.utility_model,
                temperature=0,
                base_url="http://host.docker.internal:11434",
                num_predict=256,  # Limit tokens for label assignment
            )
        elif provider == 'openai':
            self.extraction_model = model or DEFAULT_OPENAI_MODEL
            self.utility_model = utility_model or self.extraction_model
            self.llm = ChatOpenAI(model=self.extraction_model, temperature=0)
            self.utility_llm = self.llm  # Use same model for non-local
        elif provider == 'google':
            self.extraction_model = model or DEFAULT_GOOGLE_MODEL
            self.utility_model = utility_model or self.extraction_model
            self.llm = ChatGoogleGenerativeAI(model=self.extraction_model, temperature=0)
            self.utility_llm = self.llm  # Use same model for non-local
        else:
            self.extraction_model = model or DEFAULT_OPENAI_MODEL
            self.utility_model = utility_model or self.extraction_model
            self.llm = ChatOpenAI(model=self.extraction_model, temperature=0)
            self.utility_llm = self.llm

        self.structured_llm = self.llm.with_structured_output(ExtractedKnowledgeGraph)

        # Maintenance LLM for entity resolution & ontology consolidation (always GPT-5.2)
        self.maintenance_model = DEFAULT_MAINTENANCE_MODEL
        self.maintenance_llm = ChatOpenAI(model=self.maintenance_model, temperature=0)

        # Ontology state tracking - initialize with clean default labels
        self.ontology_state = self._create_default_ontology()
        self.previous_ontology_state: Optional[OntologyState] = None

        # Build the graph
        self.graph = self._build_graph()

    def _create_default_ontology(self) -> OntologyState:
        """Create ontology state initialized from the selected ontology config."""
        entity_types = {
            label: OntologyLabel(
                canonical_name=label,
                aliases=[],
                description=desc,
                count=0,
            )
            for label, desc in self.ontology_config.entity_types.items()
        }

        relationship_types = {
            label: OntologyLabel(
                canonical_name=label,
                aliases=[],
                description=desc,
                count=0,
            )
            for label, desc in self.ontology_config.relationship_types.items()
        }

        return OntologyState(
            entity_types=entity_types,
            relationship_types=relationship_types,
            is_frozen=True,  # Start frozen - use strict ontology
            version=0,
            last_change_version=0,
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Define the graph
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("extract", self._extract_node)
        builder.add_node("assign_ontology", self._assign_ontology_node)
        builder.add_node("process_entities", self._process_entities_node)
        builder.add_node("create_edges", self._create_edges_node)
        builder.add_node("update_ontology", self._update_ontology_node)

        # Add edges
        builder.add_edge(START, "extract")
        builder.add_conditional_edges(
            "extract",
            self._should_continue,
            {
                "continue": "assign_ontology",
                "end": END,
            }
        )
        builder.add_edge("assign_ontology", "process_entities")
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

        # Use adaptive prompt selection based on text length
        text_length = len(state.text_chunk)
        extraction_prompt = get_extraction_prompt(text_length)

        # Add strict ontology context to the prompt
        ontology_context = self.ontology_config.format_for_prompt()
        full_prompt = f"{extraction_prompt}\n\n{ontology_context}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", full_prompt),
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

    def _assign_ontology_node(self, state: AgentState) -> Dict[str, Any]:
        """Assign open extraction labels to ontology labels."""
        if not state.extracted_kg:
            return {}

        node_assignments = {}
        edge_assignments = {}
        new_ontology_labels = []

        # Get current ontology types
        current_entity_types = self.ontology_state.get_entity_type_names()
        current_rel_types = self.ontology_state.get_relationship_type_names()

        # Collect unique open types from this extraction
        unique_node_types = set(node.open_type for node in state.extracted_kg.nodes)
        unique_edge_types = set(edge.open_relation for edge in state.extracted_kg.edges)

        # Assign node types
        for open_type in unique_node_types:
            ontology_type = self._assign_single_label(
                open_label=open_type,
                current_ontology=current_entity_types,
                is_node=True,
            )
            node_assignments[open_type] = ontology_type
            if ontology_type not in current_entity_types:
                new_ontology_labels.append(f"Entity:{ontology_type}")
                # Add to ontology state
                if ontology_type not in self.ontology_state.entity_types:
                    self.ontology_state.entity_types[ontology_type] = OntologyLabel(
                        canonical_name=ontology_type,
                        aliases=[open_type] if open_type != ontology_type else [],
                        count=1,
                    )
                    self.ontology_state.last_change_version = self.ontology_state.version

        # Assign edge types
        for open_relation in unique_edge_types:
            ontology_relation = self._assign_single_label(
                open_label=open_relation,
                current_ontology=current_rel_types,
                is_node=False,
            )
            edge_assignments[open_relation] = ontology_relation
            if ontology_relation not in current_rel_types:
                new_ontology_labels.append(f"Relation:{ontology_relation}")
                # Add to ontology state
                if ontology_relation not in self.ontology_state.relationship_types:
                    self.ontology_state.relationship_types[ontology_relation] = OntologyLabel(
                        canonical_name=ontology_relation,
                        aliases=[open_relation] if open_relation != ontology_relation else [],
                        count=1,
                    )
                    self.ontology_state.last_change_version = self.ontology_state.version

        return {
            "node_ontology_assignments": node_assignments,
            "edge_ontology_assignments": edge_assignments,
            "new_ontology_labels": new_ontology_labels,
        }

    def _clean_label_response(self, response: str, is_relationship: bool = False) -> str:
        """Clean LLM response to extract just the label name.

        Removes markdown formatting, prefixes, and other noise from LLM responses.
        Ensures the result is a valid Neo4j label (no numbers at start, reasonable length).
        Returns empty string if no valid label can be extracted (caller should use fallback).
        """
        original = response
        cleaned = response.replace('\n', ' ').strip()

        # Remove markdown bold formatting (**text**)
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)

        # Remove markdown italic formatting (*text*)
        cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)

        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            r'^Canonical\s+(Entity\s+)?Type:\s*',
            r'^Canonical\s+Relationship\s+Type:\s*',
            r'^Generalized\s+Label:\s*',
            r'^Ontology\s+Label:\s*',
            r'^Label:\s*',
            r'^Type:\s*',
            r'^\d+\.\s*',  # Numbered list items like "1. "
            r'^-\s*',  # Bullet points
        ]
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)

        # Remove quotes and extra whitespace
        cleaned = cleaned.strip().strip('"').strip("'").strip()

        # If there's a colon, the label is usually after it
        if ':' in cleaned:
            parts = cleaned.split(':')
            # Take the shortest non-empty part (likely the label, not explanation)
            candidates = [p.strip() for p in parts if p.strip() and len(p.strip()) < 50]
            if candidates:
                cleaned = min(candidates, key=len)

        # If response is way too long, LLM gave explanation instead of label
        # Try to extract a valid label pattern
        if len(cleaned) > 40:
            if is_relationship:
                # First, look for common relationship types in the response
                common_rels = ['RELATES_TO', 'CONTAINS', 'PART_OF', 'IS_A', 'HAS', 'USES',
                              'INFLUENCES', 'CAUSES', 'CREATES', 'LEADS_TO', 'DEPENDS_ON',
                              'SUPPORTS', 'OPPOSES', 'REFLECTS', 'REPRESENTS', 'INVOLVES']
                for rel in common_rels:
                    if rel in cleaned.upper():
                        cleaned = rel
                        break
                else:
                    # Look for SCREAMING_SNAKE_CASE patterns (reasonable length)
                    matches = re.findall(r'\b([A-Z]{2,}(?:_[A-Z0-9]+){0,2})\b', cleaned)
                    if matches:
                        # Pick shortest reasonable one (3-25 chars)
                        valid = [m for m in matches if 3 <= len(m) <= 25]
                        if valid:
                            cleaned = min(valid, key=len)
                        else:
                            # Give up - return empty
                            return ""
                    else:
                        return ""
            else:
                # Look for PascalCase patterns
                matches = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b', cleaned)
                if matches:
                    valid = [m for m in matches if 3 <= len(m) <= 30]
                    if valid:
                        cleaned = valid[0]
                    else:
                        cleaned = matches[0][:30]
                else:
                    # Try to get first meaningful word
                    words = re.findall(r'\b([A-Za-z]{3,15})\b', cleaned)
                    if words:
                        cleaned = words[0].title()
                    else:
                        return ""

        # Neo4j labels/types cannot start with a number
        if cleaned and cleaned[0].isdigit():
            match = re.search(r'[A-Za-z]', cleaned)
            if match:
                cleaned = cleaned[match.start():]
            else:
                return ""  # No valid label

        # Ensure valid characters only
        if is_relationship:
            cleaned = re.sub(r'[^A-Za-z0-9_]', '_', cleaned).upper()
            cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        else:
            cleaned = re.sub(r'[^A-Za-z0-9]', '', cleaned)

        # Must be at least 2 chars to be meaningful
        if len(cleaned) < 2:
            return ""

        # Final length limit (25 for relationships, 30 for entities)
        max_len = 25 if is_relationship else 30
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len]

        return cleaned

    def _assign_single_label(
        self,
        open_label: str,
        current_ontology: List[str],
        is_node: bool,
    ) -> str:
        """Assign a single open label to an ontology label using LLM."""
        # If ontology is frozen, must map to existing label
        if self.ontology_state.is_frozen:
            if not current_ontology:
                return open_label
            return self._map_to_existing_label(open_label, current_ontology, is_node)

        # If ontology is not frozen, can propose new label or use existing
        if not current_ontology:
            # First label - use a generalized version
            return self._generalize_label(open_label, is_node)

        # Check if open_label matches or is similar to existing
        label_type = "entity type" if is_node else "relationship type"
        target_range = TARGET_ENTITY_TYPES if is_node else TARGET_RELATIONSHIP_TYPES

        system_msg = ONTOLOGY_MAPPING_PROMPT_TEMPLATE.format(
            label_type=label_type,
            current_ontology=current_ontology,
            target_min=target_range[0],
            target_max=target_range[1],
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", f"Extracted {label_type}: {open_label}"),
        ])

        try:
            response = (prompt | self.utility_llm).invoke({})
            ontology_label = self._clean_label_response(response.content, is_relationship=not is_node)
            return ontology_label if ontology_label else open_label
        except Exception as e:
            logger.warning(f"Label assignment failed for {open_label}: {e}")
            return open_label

    def _map_to_existing_label(self, open_label: str, ontology: List[str], is_node: bool) -> str:
        """Map an open label to the closest existing ontology label (when frozen)."""
        label_type = "entity type" if is_node else "relationship type"

        system_msg = FROZEN_ONTOLOGY_MAPPING_PROMPT_TEMPLATE.format(
            label_type=label_type,
            ontology=ontology,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", f"Extracted {label_type}: {open_label}"),
        ])

        try:
            response = (prompt | self.utility_llm).invoke({})
            ontology_label = self._clean_label_response(response.content, is_relationship=not is_node)
            # Validate it's in the ontology
            if ontology_label in ontology:
                return ontology_label
            # Find closest match
            for existing in ontology:
                if existing.lower() == ontology_label.lower():
                    return existing
            # Default to first
            return ontology[0]
        except Exception as e:
            logger.warning(f"Label mapping failed for {open_label}: {e}")
            return ontology[0] if ontology else open_label

    def _generalize_label(self, open_label: str, is_node: bool) -> str:
        """Generalize an open label to a broader ontology label."""
        label_type = "entity type" if is_node else "relationship type"
        target_range = TARGET_ENTITY_TYPES if is_node else TARGET_RELATIONSHIP_TYPES

        system_msg = LABEL_GENERALIZATION_PROMPT_TEMPLATE.format(
            label_type=label_type,
            target_min=target_range[0],
            target_max=target_range[1],
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", f"Specific {label_type}: {open_label}"),
        ])

        try:
            response = (prompt | self.utility_llm).invoke({})
            cleaned = self._clean_label_response(response.content, is_relationship=not is_node)
            return cleaned if cleaned else open_label
        except Exception as e:
            logger.warning(f"Label generalization failed for {open_label}: {e}")
            return open_label

    def _process_entities_node(self, state: AgentState) -> Dict[str, Any]:
        """Process extracted entities - create or merge with ontology assignments."""
        if not state.extracted_kg:
            return {}

        nodes_created = 0
        nodes_merged = 0

        for node in state.extracted_kg.nodes:
            # Get the ontology type from assignments
            ontology_type = state.node_ontology_assignments.get(node.open_type, node.open_type)

            created, merged = self.loader.create_or_merge_node(
                node=node,
                ontology_type=ontology_type,
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
        """Create edges between entities with ontology assignments."""
        if not state.extracted_kg:
            return {}

        edges_created = 0

        for edge in state.extracted_kg.edges:
            # Get the ontology relation from assignments
            ontology_relation = state.edge_ontology_assignments.get(edge.open_relation, edge.open_relation)

            success = self.loader.create_edge(
                edge=edge,
                ontology_relation=ontology_relation,
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

        # Track node types (both open and ontology)
        for node in state.extracted_kg.nodes:
            ontology_type = state.node_ontology_assignments.get(node.open_type, node.open_type)
            # Track the ontology type in Neo4j
            self.loader.update_ontology(ontology_type, ontology_type, is_node=True)
            new_node_types.append(ontology_type)

        # Track edge types (both open and ontology)
        for edge in state.extracted_kg.edges:
            ontology_relation = state.edge_ontology_assignments.get(edge.open_relation, edge.open_relation)
            self.loader.update_ontology(ontology_relation, ontology_relation, is_node=False)
            new_edge_types.append(ontology_relation)

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
        """Use LLM to verify if two entities are the same. Uses GPT-5.2 for accuracy."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", ENTITY_VERIFICATION_PROMPT),
            ("human", """Entity 1: {name1}
Description: {desc1}

Entity 2: {name2}
Description: {desc2}

Are these likely the same entity? Answer ONLY with 'YES' or 'NO'."""),
        ])

        chain = prompt | self.maintenance_llm  # Use GPT-5.2 for entity resolution
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
        """Use LLM to verify if two ontology labels are the same. Uses GPT-5.2 for accuracy."""
        system_msg = LABEL_VERIFICATION_PROMPT_TEMPLATE.format(category=category)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", """Label 1: {label1}
Label 2: {label2}

Are these the same concept? If YES, which label is more canonical?
Answer in format: YES|canonical_label or NO"""),
        ])

        chain = prompt | self.maintenance_llm  # Use GPT-5.2 for ontology consolidation
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

    def run_ontology_convergence(self) -> Dict[str, Any]:
        """
        Run ontology convergence to reduce labels toward target sizes.
        Uses LLM to propose merges when ontology is too large.
        Returns dict with changes made and stabilization status.
        """
        if self.ontology_state.is_frozen:
            logger.info("Ontology is frozen, skipping convergence")
            return {"is_frozen": True, "changes": 0}

        # Save previous state for comparison
        self.previous_ontology_state = self.ontology_state.model_copy(deep=True)

        # Increment version
        self.ontology_state.version += 1

        # Get current ontology stats from Neo4j
        stats = self.loader.get_ontology_stats()

        changes_made = 0

        # Process entity types
        entity_types = [item['name'] for item in stats.get("NodeType", [])]
        if len(entity_types) > TARGET_ENTITY_TYPES[1]:
            logger.info(f"Entity types ({len(entity_types)}) exceed target max ({TARGET_ENTITY_TYPES[1]}), running convergence...")
            merges = self._propose_ontology_merges(entity_types, is_node=True)
            for merge_from, merge_to in merges.items():
                self.loader.consolidate_ontology_label(merge_to, merge_from, "NodeType")
                # Update local state
                if merge_from in self.ontology_state.entity_types:
                    if merge_to in self.ontology_state.entity_types:
                        self.ontology_state.entity_types[merge_to].aliases.append(merge_from)
                    del self.ontology_state.entity_types[merge_from]
                changes_made += 1
                self.ontology_state.last_change_version = self.ontology_state.version

        # Process relationship types
        rel_types = [item['name'] for item in stats.get("EdgeType", [])]
        if len(rel_types) > TARGET_RELATIONSHIP_TYPES[1]:
            logger.info(f"Relationship types ({len(rel_types)}) exceed target max ({TARGET_RELATIONSHIP_TYPES[1]}), running convergence...")
            merges = self._propose_ontology_merges(rel_types, is_node=False)
            for merge_from, merge_to in merges.items():
                self.loader.consolidate_ontology_label(merge_to, merge_from, "EdgeType")
                # Update local state
                if merge_from in self.ontology_state.relationship_types:
                    if merge_to in self.ontology_state.relationship_types:
                        self.ontology_state.relationship_types[merge_to].aliases.append(merge_from)
                    del self.ontology_state.relationship_types[merge_from]
                changes_made += 1
                self.ontology_state.last_change_version = self.ontology_state.version

        # Check for stabilization
        is_stable = self.ontology_state.check_stabilization()

        return {
            "is_frozen": False,
            "changes": changes_made,
            "is_stable": is_stable,
            "version": self.ontology_state.version,
            "versions_since_change": self.ontology_state.version - self.ontology_state.last_change_version,
        }

    def _propose_ontology_merges(self, labels: List[str], is_node: bool) -> Dict[str, str]:
        """Use LLM to propose merges for ontology labels. Uses GPT-5.2 for accuracy."""
        label_type = "entity types" if is_node else "relationship types"
        target_range = TARGET_ENTITY_TYPES if is_node else TARGET_RELATIONSHIP_TYPES

        system_msg = ONTOLOGY_MERGE_PROMPT_TEMPLATE.format(
            label_type=label_type,
            target_min=target_range[0],
            target_max=target_range[1],
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", f"Current {label_type}: {labels}\n\nPropose merges:"),
        ])

        try:
            response = (prompt | self.maintenance_llm).invoke({})  # Use GPT-5.2
            content = response.content.strip()
            # Try to parse JSON
            import re
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except Exception as e:
            logger.warning(f"Ontology merge proposal failed: {e}")
            return {}

    def freeze_ontology(self) -> bool:
        """Freeze the ontology and reassign all nodes to current labels."""
        if self.ontology_state.is_frozen:
            logger.info("Ontology already frozen")
            return False

        logger.info("=" * 60)
        logger.info("FREEZING ONTOLOGY")
        logger.info("=" * 60)

        self.ontology_state.is_frozen = True

        # Get final ontology state
        stats = self.loader.get_ontology_stats()
        entity_types = [item['name'] for item in stats.get("NodeType", [])]
        rel_types = [item['name'] for item in stats.get("EdgeType", [])]

        logger.info(f"Final entity types ({len(entity_types)}): {entity_types}")
        logger.info(f"Final relationship types ({len(rel_types)}): {rel_types}")

        # Reassign all nodes to current ontology
        reassigned = self._reassign_all_nodes_to_ontology(entity_types, rel_types)

        logger.info(f"Reassigned {reassigned['nodes']} nodes and {reassigned['edges']} edges")
        logger.info("=" * 60)

        return True

    def _reassign_all_nodes_to_ontology(self, entity_types: List[str], rel_types: List[str]) -> Dict[str, int]:
        """Reassign all nodes and edges to current frozen ontology labels."""
        timestamp = datetime.utcnow().isoformat()
        reassigned = {"nodes": 0, "edges": 0}

        # Get all unique open_types that need reassignment
        query = """
        MATCH (n:Entity)
        WHERE n.ontology_type IS NULL OR NOT n.ontology_type IN $entity_types
        RETURN DISTINCT n.open_type AS open_type, count(*) AS count
        """

        with self.loader.driver.session() as session:
            result = session.run(query, entity_types=entity_types)
            open_types_to_reassign = [(r["open_type"], r["count"]) for r in result]

        # Map each open_type to the best ontology type
        for open_type, count in open_types_to_reassign:
            if not open_type:
                continue
            best_type = self._map_to_existing_label(open_type, entity_types, is_node=True)

            # Update all nodes with this open_type
            update_query = """
            MATCH (n:Entity)
            WHERE n.open_type = $open_type
            SET n.ontology_type = $ontology_type,
                n.updated_at = $timestamp
            """
            with self.loader.driver.session() as session:
                session.run(update_query, open_type=open_type, ontology_type=best_type, timestamp=timestamp)
                reassigned["nodes"] += count
                logger.info(f"  Reassigned {count} nodes: {open_type} -> {best_type}")

        return reassigned

    def get_ontology_change_report(self) -> str:
        """Generate a report of ontology changes since last convergence."""
        lines = [
            "=" * 60,
            "ONTOLOGY STATUS REPORT",
            "=" * 60,
            f"Version: {self.ontology_state.version}",
            f"Frozen: {self.ontology_state.is_frozen}",
            f"Versions since last change: {self.ontology_state.version - self.ontology_state.last_change_version}",
            "",
        ]

        # Get current stats from Neo4j
        stats = self.loader.get_ontology_stats()

        entity_count = len(stats.get("NodeType", []))
        rel_count = len(stats.get("EdgeType", []))

        lines.append(f"ENTITY TYPES ({entity_count}, target: {TARGET_ENTITY_TYPES[0]}-{TARGET_ENTITY_TYPES[1]}):")
        lines.append("-" * 40)
        for item in stats.get("NodeType", [])[:15]:
            lines.append(f"  {item['name']}: {item['count']} occurrences")
        if entity_count > 15:
            lines.append(f"  ... and {entity_count - 15} more")

        lines.append("")
        lines.append(f"RELATIONSHIP TYPES ({rel_count}, target: {TARGET_RELATIONSHIP_TYPES[0]}-{TARGET_RELATIONSHIP_TYPES[1]}):")
        lines.append("-" * 40)
        for item in stats.get("EdgeType", [])[:20]:
            lines.append(f"  {item['name']}: {item['count']} occurrences")
        if rel_count > 20:
            lines.append(f"  ... and {rel_count - 20} more")

        # Stabilization status
        lines.append("")
        if self.ontology_state.is_frozen:
            lines.append("STATUS: FROZEN (ontology is locked)")
        elif self.ontology_state.check_stabilization():
            lines.append("STATUS: STABLE (ready to freeze)")
        else:
            versions_to_stable = STABILITY_THRESHOLD - (self.ontology_state.version - self.ontology_state.last_change_version)
            lines.append(f"STATUS: EVOLVING ({versions_to_stable} more stable versions needed)")

        lines.append("=" * 60)

        return "\n".join(lines)

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
def load_subject_dataset(subject: str, restart_index: int = 0, limit_docs: int = 0):
    """Load a dataset for a specific subject."""
    if subject == 'economics':
        dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "economics-corpus", split='train')
    elif subject == 'law':
        dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "law-corpus", split='train')
    elif subject == 'physics':
        dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", "physics-corpus", split='train')
    else:
        raise ValueError(f"Unknown subject: {subject}")

    end_index = restart_index + limit_docs if limit_docs > 0 else len(dataset)
    return dataset.select(range(restart_index, min(end_index, len(dataset))))


def run_agent_pipeline(
    provider: str = 'local',
    model: str = None,
    utility_model: str = None,
    limit_docs: int = 5,
    restart_index: int = 0,
    subject: str = 'economics',
    entity_resolution_interval: int = ENTITY_RESOLUTION_INTERVAL,
    ontology_consolidation_interval: int = ONTOLOGY_CONSOLIDATION_INTERVAL,
    ontology: str = None,
):
    """Run the knowledge graph extraction agent pipeline."""

    agent = KnowledgeGraphAgent(
        provider=provider,
        model=model,
        utility_model=utility_model,
        ontology=ontology,
    )

    # Determine which subjects to process
    if subject == 'all':
        subjects = ['economics', 'law', 'physics']
    else:
        subjects = [subject]

    try:
        # Initialize indices
        agent.loader.init_indices()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        print(f"Periodic maintenance: entity resolution every {entity_resolution_interval} docs, "
              f"ontology consolidation every {ontology_consolidation_interval} docs\n", flush=True)

        total_nodes_created = 0
        total_nodes_merged = 0
        total_edges_created = 0
        total_entities_resolved = 0
        total_labels_consolidated = {"NodeType": 0, "EdgeType": 0}

        # Track documents processed for periodic maintenance (across all subjects)
        docs_since_entity_resolution = 0
        docs_since_ontology_consolidation = 0
        global_doc_idx = 0

        # Process each subject
        for subj_idx, current_subject in enumerate(subjects):
            print(f"\n{'=' * 60}", flush=True)
            print(f"PROCESSING SUBJECT: {current_subject.upper()} ({subj_idx + 1}/{len(subjects)})", flush=True)
            print(f"{'=' * 60}", flush=True)

            print(f"Loading {current_subject} dataset...", flush=True)
            dataset = load_subject_dataset(current_subject, restart_index, limit_docs)
            end_index = restart_index + len(dataset)

            print(f"Processing {len(dataset)} documents from {current_subject}\n", flush=True)

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
                            doc_index=global_doc_idx,
                        )

                        if result.error_message:
                            logger.warning(
                                f"[{current_subject}][{doc_idx + restart_index}/{end_index}][{chunk_idx}] "
                                f"Error: {result.error_message[:100]}"
                            )
                            continue

                        total_nodes_created += result.nodes_created
                        total_nodes_merged += result.nodes_merged
                        total_edges_created += result.edges_created

                        toc = time.time() - tic
                        tic = time.time()

                        # Build status line
                        status = (
                            f"[{current_subject}][{doc_idx + restart_index}/{end_index}][{chunk_idx}] "
                            f"Created: {result.nodes_created} nodes, "
                            f"Merged: {result.nodes_merged} nodes, "
                            f"Edges: {result.edges_created} "
                            f"({toc:.2f}s)"
                        )

                        # Add new ontology labels if any
                        if result.new_ontology_labels:
                            status += f" | New: {result.new_ontology_labels}"

                        print(status, flush=True)

                        # Show detailed type info in debug mode
                        if result.new_node_types:
                            logger.debug(f"  Node types: {result.new_node_types}")
                        if result.new_edge_types:
                            logger.debug(f"  Edge types: {result.new_edge_types}")

                    except Exception as e:
                        logger.error(
                            f"[{current_subject}][{doc_idx + restart_index}/{end_index}][{chunk_idx}] "
                            f"Unexpected error: {e}"
                        )

                # Increment document counters
                docs_since_entity_resolution += 1
                docs_since_ontology_consolidation += 1
                global_doc_idx += 1

                # --- PERIODIC ENTITY RESOLUTION ---
                if docs_since_entity_resolution >= entity_resolution_interval:
                    print(f"\n{'=' * 60}", flush=True)
                    print(f"PERIODIC ENTITY RESOLUTION (after {docs_since_entity_resolution} documents)", flush=True)
                    print(f"{'=' * 60}", flush=True)

                    try:
                        merged = agent.run_entity_resolution()
                        total_entities_resolved += merged
                        print(f"Resolved {merged} duplicate entities", flush=True)
                    except Exception as e:
                        logger.error(f"Entity resolution failed: {e}")

                    # Always reset counter, even on failure
                    docs_since_entity_resolution = 0
                    print(f"{'=' * 60}\n", flush=True)

                # --- PERIODIC ONTOLOGY CONVERGENCE ---
                if docs_since_ontology_consolidation >= ontology_consolidation_interval:
                    print(f"\n{'=' * 60}", flush=True)
                    print(f"PERIODIC ONTOLOGY CONVERGENCE (after {docs_since_ontology_consolidation} documents)", flush=True)
                    print(f"{'=' * 60}", flush=True)

                    try:
                        # Run ontology consolidation (merge similar labels)
                        consolidated = agent.run_ontology_consolidation()
                        total_labels_consolidated["NodeType"] += consolidated["NodeType"]
                        total_labels_consolidated["EdgeType"] += consolidated["EdgeType"]
                        print(f"Consolidated {consolidated['NodeType']} node types, {consolidated['EdgeType']} edge types", flush=True)

                        # Run ontology convergence (reduce toward target size)
                        convergence_result = agent.run_ontology_convergence()

                        # Print intermediate ontology report
                        print(agent.get_ontology_change_report(), flush=True)

                        # Check if ontology is stable and should be frozen
                        if convergence_result.get("is_stable") and not convergence_result.get("is_frozen"):
                            print("\n*** ONTOLOGY HAS STABILIZED ***", flush=True)
                            agent.freeze_ontology()
                            print("Ontology is now FROZEN. All future extractions will use fixed labels.", flush=True)
                    except Exception as e:
                        logger.error(f"Ontology convergence failed: {e}")

                    # Always reset counter, even on failure
                    docs_since_ontology_consolidation = 0
                    print(f"{'=' * 60}\n", flush=True)

            # Print subject completion summary
            print(f"\n[{current_subject.upper()}] Completed: {len(dataset)} documents processed", flush=True)

        # --- FINAL MAINTENANCE RUN ---
        print(f"\n{'=' * 60}", flush=True)
        print("FINAL MAINTENANCE RUN", flush=True)
        print(f"{'=' * 60}", flush=True)

        try:
            # Final entity resolution
            merged = agent.run_entity_resolution()
            total_entities_resolved += merged

            # Final ontology consolidation
            consolidated = agent.run_ontology_consolidation()
            total_labels_consolidated["NodeType"] += consolidated["NodeType"]
            total_labels_consolidated["EdgeType"] += consolidated["EdgeType"]

            # Final ontology convergence
            convergence_result = agent.run_ontology_convergence()

            # If not already frozen and stable, freeze now
            if not agent.ontology_state.is_frozen:
                print("\nFreezing ontology at end of pipeline...", flush=True)
                agent.freeze_ontology()

        except Exception as e:
            logger.error(f"Final maintenance failed: {e}")

        print(f"{'=' * 60}\n", flush=True)

        # Print summary
        print("\n" + "=" * 60, flush=True)
        print("PIPELINE COMPLETE", flush=True)
        print("=" * 60, flush=True)
        print(f"Total nodes created: {total_nodes_created}", flush=True)
        print(f"Total nodes merged (same name): {total_nodes_merged}", flush=True)
        print(f"Total edges created: {total_edges_created}", flush=True)
        print(f"Total entities resolved (duplicates): {total_entities_resolved}", flush=True)
        print(f"Total node types consolidated: {total_labels_consolidated['NodeType']}", flush=True)
        print(f"Total edge types consolidated: {total_labels_consolidated['EdgeType']}", flush=True)
        print(f"Ontology frozen: {agent.ontology_state.is_frozen}", flush=True)
        print(f"Ontology version: {agent.ontology_state.version}", flush=True)
        print("=" * 60, flush=True)

        # Print final ontology report
        print("\n" + agent.get_ontology_change_report(), flush=True)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

    finally:
        agent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Agent for Knowledge Graph Extraction")
    parser.add_argument("--provider", default='local', choices=['local', 'openai', 'google'], help="LLM provider")
    parser.add_argument("--model", default=None, help="Extraction model")
    parser.add_argument("--utility_model", default=None, help="Utility model for ontology tasks")
    parser.add_argument("--limit_docs", type=int, default=5, help="Docs to process (0=all)")
    parser.add_argument("--restart_index", type=int, default=0, help="Start index")
    parser.add_argument("--subject", default='economics', choices=['economics', 'law', 'physics', 'all'], help="Corpus")
    parser.add_argument("--entity_resolution_interval", type=int, default=ENTITY_RESOLUTION_INTERVAL, help="Entity resolution every N docs")
    parser.add_argument("--ontology_consolidation_interval", type=int, default=ONTOLOGY_CONSOLIDATION_INTERVAL, help="Ontology consolidation every N docs")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, choices=list(ONTOLOGIES.keys()), help="Ontology size (small/medium/large)")
    args = parser.parse_args()

    # Determine actual models based on provider and arguments
    if args.provider == 'local':
        extraction_model = args.model or DEFAULT_LOCAL_EXTRACTION_MODEL
        utility_model = args.utility_model or DEFAULT_LOCAL_UTILITY_MODEL
        embedding_model = DEFAULT_LOCAL_EMBEDDING_MODEL
    elif args.provider == 'openai':
        extraction_model = args.model or DEFAULT_OPENAI_MODEL
        utility_model = args.utility_model or extraction_model
        embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL
    elif args.provider == 'google':
        extraction_model = args.model or DEFAULT_GOOGLE_MODEL
        utility_model = args.utility_model or extraction_model
        embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL
    else:
        extraction_model = args.model or DEFAULT_OPENAI_MODEL
        utility_model = args.utility_model or extraction_model
        embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL

    # Get ontology info
    ontology_config = get_ontology(args.ontology)

    print(f"\nConfiguration:", flush=True)
    print(f"  Provider: {args.provider}", flush=True)
    print(f"  Extraction model: {extraction_model}", flush=True)
    print(f"  Utility model: {utility_model}", flush=True)
    print(f"  Embedding model: {embedding_model}", flush=True)
    print(f"  Maintenance model: {DEFAULT_MAINTENANCE_MODEL} (entity resolution & ontology consolidation)", flush=True)
    print(f"  Ontology: {args.ontology} ({len(ontology_config.entity_types)} entity types, {len(ontology_config.relationship_types)} relationship types)", flush=True)
    print(f"  Subject: {args.subject}", flush=True)
    if args.subject == 'all':
        print(f"  Documents: {args.limit_docs} per subject (economics, law, physics) = {args.limit_docs * 3} total", flush=True)
    else:
        print(f"  Documents: {args.limit_docs} starting from {args.restart_index}", flush=True)
    print(f"  Entity resolution interval: {args.entity_resolution_interval} docs", flush=True)
    print(f"  Ontology consolidation interval: {args.ontology_consolidation_interval} docs", flush=True)
    print(f"\nBuilding KG with LangGraph agent...\n", flush=True)

    run_agent_pipeline(
        provider=args.provider,
        model=args.model,
        utility_model=args.utility_model,
        limit_docs=args.limit_docs,
        restart_index=args.restart_index,
        subject=args.subject,
        entity_resolution_interval=args.entity_resolution_interval,
        ontology_consolidation_interval=args.ontology_consolidation_interval,
        ontology=args.ontology,
    )
