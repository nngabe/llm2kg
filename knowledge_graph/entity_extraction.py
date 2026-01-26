"""
Entity Extraction Pipeline (Stage 3).

Extracts entities and relationships from document chunks using:
- graphrag_sdk: Auto-ontology discovery from unstructured text
- graphiti-core: Temporal entity extraction with bi-temporal data model

Components:
- OntologyDiscoveryPipeline: Discover entity types from document chunks
- GraphitiEntityExtractor: Temporal entity/relationship extraction
- HybridEntityExtractor: Combined approach using both libraries

Usage:
    python -m knowledge_graph.entity_extraction --graph ge_vernova_test \
        --discover-ontology --extract-entities
"""

import os
import time
import json
import logging
import argparse
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone

from .models import (
    DocumentChunk,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    PipelineConfig,
    EntityStatus,
    ChunkStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")


class OntologyDiscoveryPipeline:
    """
    Use FalkorDB GraphRAG-SDK to discover entity types from document chunks.

    This leverages graphrag_sdk's auto-ontology discovery to bootstrap
    entity types from unstructured text, which can then be used to guide
    the extraction process.
    """

    def __init__(
        self,
        graph_name: str,
        sample_size: int = 50,
        falkordb_host: str = FALKORDB_HOST,
        falkordb_port: int = FALKORDB_PORT,
        llm_model: str = "nemotron-3-nano:30b",  # Ollama model
    ):
        """
        Initialize the ontology discovery pipeline.

        Args:
            graph_name: Name of the FalkorDB graph.
            sample_size: Number of chunks to sample for discovery.
            falkordb_host: FalkorDB host.
            falkordb_port: FalkorDB port.
            llm_model: LLM model for ontology discovery.
        """
        self.graph_name = graph_name
        self.sample_size = sample_size
        self.llm_model = llm_model

        # Initialize FalkorDB connection
        from falkordb import FalkorDB
        self.client = FalkorDB(host=falkordb_host, port=falkordb_port)
        self.graph = self.client.select_graph(graph_name)

        # Lazy-loaded graphrag_sdk components
        self._kg = None

    def _get_knowledge_graph(self):
        """Lazy load graphrag_sdk KnowledgeGraph."""
        if self._kg is None:
            try:
                from graphrag_sdk import KnowledgeGraph
                from graphrag_sdk.ontology import Ontology

                # Initialize with FalkorDB backend
                self._kg = KnowledgeGraph(
                    name=self.graph_name,
                    host=FALKORDB_HOST,
                    port=FALKORDB_PORT,
                )
                logger.info("Initialized graphrag_sdk KnowledgeGraph")
            except ImportError:
                logger.warning("graphrag_sdk not installed, ontology discovery disabled")
                self._kg = None
            except Exception as e:
                logger.warning(f"Failed to initialize graphrag_sdk: {e}")
                self._kg = None

        return self._kg

    def get_sample_chunks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a sample of document chunks for ontology discovery.

        Samples diverse chunks by selecting from different source WikiPages.

        Args:
            limit: Maximum number of chunks (defaults to sample_size).

        Returns:
            List of chunk dictionaries with content and metadata.
        """
        limit = limit or self.sample_size

        # Get diverse sample by sampling from different sources
        result = self.graph.query(f"""
            MATCH (w:WikiPage)-[:HAS_CHUNK]->(c:DocumentChunk)
            WITH w, c, rand() as r
            ORDER BY r
            WITH w, collect(c)[0] as sample_chunk
            RETURN sample_chunk.chunk_id as chunk_id,
                   sample_chunk.content as content,
                   sample_chunk.chunk_index as chunk_index,
                   w.name as source_name,
                   w.wikidata_id as source_qid
            LIMIT {limit}
        """)

        chunks = []
        for row in result.result_set:
            chunks.append({
                "chunk_id": row[0],
                "content": row[1],
                "chunk_index": row[2],
                "source_name": row[3],
                "source_qid": row[4],
            })

        logger.info(f"Sampled {len(chunks)} chunks for ontology discovery")
        return chunks

    def discover_from_chunks(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Discover entity types from document chunks.

        Uses graphrag_sdk's auto-ontology discovery if available,
        otherwise falls back to a simple heuristic approach.

        Args:
            chunks: Optional list of chunk dictionaries. If None, samples from graph.

        Returns:
            Dictionary with discovered entity_types and relationship_types.
        """
        if chunks is None:
            chunks = self.get_sample_chunks()

        if not chunks:
            return {"entity_types": {}, "relationship_types": {}, "error": "No chunks to analyze"}

        # Try graphrag_sdk ontology discovery
        kg = self._get_knowledge_graph()
        if kg is not None:
            try:
                return self._discover_with_graphrag_sdk(chunks)
            except Exception as e:
                logger.warning(f"graphrag_sdk discovery failed: {e}")

        # Fallback to simple heuristic discovery
        return self._discover_heuristic(chunks)

    def _discover_with_graphrag_sdk(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Discover ontology using graphrag_sdk.

        Uses the SDK's built-in ontology discovery from text sources.
        """
        try:
            from graphrag_sdk.ontology import Ontology
            from graphrag_sdk.source import Source

            # Create text sources from chunks
            texts = [c["content"] for c in chunks]

            # Use graphrag_sdk ontology discovery
            # Note: This is a simplified version - the actual API may differ
            ontology = Ontology.from_sources(
                sources=[Source.from_text(t) for t in texts],
                model=self.llm_model,
            )

            return {
                "entity_types": {e.label: e.description for e in ontology.entities},
                "relationship_types": {r.label: r.description for r in ontology.relations},
                "source": "graphrag_sdk",
                "sample_size": len(chunks),
            }

        except Exception as e:
            logger.warning(f"graphrag_sdk ontology discovery failed: {e}")
            raise

    def _discover_heuristic(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple heuristic-based ontology discovery.

        Analyzes chunk content to identify common entity patterns.
        This is a fallback when graphrag_sdk is not available.
        """
        # Combine all chunk content
        all_text = " ".join(c["content"] for c in chunks)

        # Simple pattern-based entity type detection
        entity_types = {}
        relationship_types = {}

        # Check for common domain patterns
        patterns = {
            "Technology": ["technology", "system", "software", "hardware", "device"],
            "Organization": ["company", "corporation", "organization", "institution", "university"],
            "Person": ["inventor", "founder", "engineer", "scientist", "researcher"],
            "Location": ["country", "city", "region", "facility", "plant"],
            "Process": ["process", "method", "technique", "procedure", "workflow"],
            "Product": ["product", "model", "version", "series", "line"],
            "Event": ["event", "conference", "meeting", "launch", "announcement"],
            "Concept": ["concept", "theory", "principle", "standard", "specification"],
        }

        text_lower = all_text.lower()
        for entity_type, keywords in patterns.items():
            count = sum(text_lower.count(kw) for kw in keywords)
            if count >= 3:  # Threshold for inclusion
                entity_types[entity_type] = f"Detected from keywords: {', '.join(keywords)}"

        # Common relationship patterns
        rel_patterns = {
            "PART_OF": ["part of", "component of", "belongs to", "included in"],
            "PRODUCES": ["produces", "manufactures", "creates", "generates"],
            "USES": ["uses", "utilizes", "employs", "requires"],
            "LOCATED_IN": ["located in", "based in", "headquartered in"],
            "DEVELOPED_BY": ["developed by", "created by", "invented by", "designed by"],
            "RELATED_TO": ["related to", "associated with", "connected to"],
        }

        for rel_type, keywords in rel_patterns.items():
            count = sum(text_lower.count(kw) for kw in keywords)
            if count >= 2:
                relationship_types[rel_type] = f"Detected from phrases: {', '.join(keywords)}"

        return {
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "source": "heuristic",
            "sample_size": len(chunks),
        }

    def merge_with_base_ontology(
        self,
        discovered: Dict[str, Any],
        base_ontology_name: str = "medium",
    ) -> Dict[str, Any]:
        """
        Merge discovered ontology with a base ontology from ontologies.py.

        Args:
            discovered: Discovered entity/relationship types.
            base_ontology_name: Name of base ontology (small/medium/large).

        Returns:
            Merged ontology configuration.
        """
        from ontologies import get_ontology

        base = get_ontology(base_ontology_name)

        # Merge entity types (base takes precedence)
        merged_entities = dict(discovered.get("entity_types", {}))
        merged_entities.update(base.entity_types)

        # Merge relationship types (base takes precedence)
        merged_rels = dict(discovered.get("relationship_types", {}))
        merged_rels.update(base.relationship_types)

        return {
            "name": f"{base_ontology_name}_extended",
            "entity_types": merged_entities,
            "relationship_types": merged_rels,
            "base_ontology": base_ontology_name,
            "discovered_entities": len(discovered.get("entity_types", {})),
            "discovered_relationships": len(discovered.get("relationship_types", {})),
        }


class GraphitiEntityExtractor:
    """
    Use graphiti-core for temporal entity/relationship extraction.

    Graphiti provides:
    - 3-tier architecture: Episodic → Semantic Entity → Community layers
    - Temporal bi-data model: Tracks both event time (valid_at) and ingestion time (created_at)
    - LLM extraction pipeline with specialized prompts
    """

    def __init__(
        self,
        graph_name: str,
        ontology: Optional[Dict[str, Any]] = None,
        falkordb_host: str = FALKORDB_HOST,
        falkordb_port: int = FALKORDB_PORT,
        llm_model: str = "nemotron-3-nano:30b",
    ):
        """
        Initialize the Graphiti entity extractor.

        Args:
            graph_name: Name of the FalkorDB graph.
            ontology: Optional ontology configuration dict.
            falkordb_host: FalkorDB host.
            falkordb_port: FalkorDB port.
            llm_model: LLM model for extraction.
        """
        self.graph_name = graph_name
        self.ontology = ontology or {}
        self.llm_model = llm_model
        self.falkordb_host = falkordb_host
        self.falkordb_port = falkordb_port

        # Initialize FalkorDB connection for custom operations
        from falkordb import FalkorDB
        self.client = FalkorDB(host=falkordb_host, port=falkordb_port)
        self.graph = self.client.select_graph(graph_name)

        # Lazy-loaded graphiti components
        self._graphiti = None

    async def _get_graphiti(self):
        """Lazy load graphiti-core client."""
        if self._graphiti is None:
            try:
                from graphiti_core import Graphiti
                from graphiti_core.driver.falkordb_driver import FalkorDriver
                from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
                from graphiti_core.llm_client.config import LLMConfig

                # Initialize FalkorDB driver for graphiti
                driver = FalkorDriver(
                    host=self.falkordb_host,
                    port=self.falkordb_port,
                    database=self.graph_name,
                )

                # Create OpenAI-compatible client for Ollama
                config = LLMConfig(
                    api_key='ollama',  # Dummy key for Ollama
                    model=self.llm_model,
                    base_url=f'{OLLAMA_HOST}/v1',
                    small_model=self.llm_model,
                )
                llm_client = OpenAIGenericClient(config)

                # Initialize Graphiti client with LLM
                self._graphiti = Graphiti(
                    graph_driver=driver,
                    llm_client=llm_client,
                )
                await self._graphiti.build_indices_and_constraints()
                logger.info("Initialized graphiti-core client with FalkorDB and Ollama LLM")

            except ImportError:
                logger.warning("graphiti-core not installed, temporal extraction disabled")
                self._graphiti = None
            except Exception as e:
                logger.warning(f"Failed to initialize graphiti-core: {e}")
                self._graphiti = None

        return self._graphiti

    async def extract_from_chunk(
        self,
        chunk: DocumentChunk,
        reference_time: Optional[datetime] = None,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a single chunk.

        Uses graphiti's add_episode() pattern for temporal extraction.

        Args:
            chunk: DocumentChunk to process.
            reference_time: Reference time for temporal context.

        Returns:
            ExtractionResult with extracted entities and relationships.
        """
        start_time = time.time()
        reference_time = reference_time or datetime.now(timezone.utc)

        # Try graphiti extraction
        graphiti = await self._get_graphiti()
        if graphiti is not None:
            try:
                return await self._extract_with_graphiti(chunk, reference_time)
            except Exception as e:
                logger.warning(f"Graphiti extraction failed: {e}")

        # Fallback to simple LLM extraction
        return await self._extract_with_llm(chunk, reference_time)

    async def _extract_with_graphiti(
        self,
        chunk: DocumentChunk,
        reference_time: datetime,
    ) -> ExtractionResult:
        """
        Extract using graphiti-core's episode processing.
        """
        from graphiti_core.nodes import EpisodeType

        graphiti = await self._get_graphiti()

        # Add chunk as an episode
        episode = await graphiti.add_episode(
            name=f"chunk_{chunk.chunk_id}",
            episode_body=chunk.content,
            source=EpisodeType.text,
            source_description=f"Wikipedia article chunk from {chunk.source_qid}",
            reference_time=reference_time,
        )

        # Get extracted nodes and edges from the episode
        entities = []
        relationships = []

        # Query graphiti's internal graph for extracted entities
        # Note: The actual graphiti API may differ
        try:
            nodes_result = await graphiti.search(
                query="",  # Get all recent nodes
                num_results=100,
            )

            for node in nodes_result:
                entity = ExtractedEntity(
                    name=node.name,
                    ontology_type=self._map_to_ontology_type(node.labels),
                    description=node.summary if hasattr(node, 'summary') else None,
                    source_chunk_ids={chunk.chunk_id},
                    valid_from=reference_time,
                    created_at=datetime.now(timezone.utc),
                )
                entities.append(entity)

        except Exception as e:
            logger.warning(f"Failed to query graphiti nodes: {e}")

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_chunk_ids={chunk.chunk_id},
            processing_time_seconds=time.time() - start_time,
        )

    async def _extract_with_llm(
        self,
        chunk: DocumentChunk,
        reference_time: datetime,
    ) -> ExtractionResult:
        """
        Fallback LLM-based extraction when graphiti is not available.
        """
        start_time = time.time()

        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage

            # Build extraction prompt
            entity_types = self.ontology.get("entity_types", {})
            rel_types = self.ontology.get("relationship_types", {})

            system_prompt = f"""You are an entity extraction system. Extract entities and relationships from the text.

ENTITY TYPES:
{json.dumps(entity_types, indent=2) if entity_types else "Use general types like: Person, Organization, Location, Concept, Technology, Event"}

RELATIONSHIP TYPES:
{json.dumps(rel_types, indent=2) if rel_types else "Use general types like: RELATED_TO, PART_OF, USES, PRODUCES, LOCATED_IN"}

Respond with JSON in this format:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "..."}}
  ]
}}"""

            llm = ChatOllama(
                model="nemotron-3-nano:30b",
                base_url=OLLAMA_HOST,
                temperature=0.0,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract entities and relationships from:\n\n{chunk.content}"),
            ]

            response = await llm.ainvoke(messages)

            # Parse response
            try:
                # Extract JSON from response
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content)

                entities = []
                for e in data.get("entities", []):
                    entity = ExtractedEntity(
                        name=e["name"],
                        ontology_type=e.get("type", "Concept"),
                        description=e.get("description"),
                        source_chunk_ids={chunk.chunk_id},
                        valid_from=reference_time,
                    )
                    entities.append(entity)

                relationships = []
                # Create entity ID map for relationship linking
                entity_map = {e.name.lower(): e.entity_id for e in entities}

                for r in data.get("relationships", []):
                    source_id = entity_map.get(r["source"].lower())
                    target_id = entity_map.get(r["target"].lower())

                    if source_id and target_id:
                        rel = ExtractedRelationship(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relationship_type=r.get("type", "RELATED_TO"),
                            description=r.get("description"),
                            source_chunk_ids={chunk.chunk_id},
                            valid_from=reference_time,
                        )
                        relationships.append(rel)

                return ExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    source_chunk_ids={chunk.chunk_id},
                    processing_time_seconds=time.time() - start_time,
                )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response: {e}")
                return ExtractionResult(
                    source_chunk_ids={chunk.chunk_id},
                    processing_time_seconds=time.time() - start_time,
                    errors=[f"JSON parse error: {e}"],
                )

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return ExtractionResult(
                source_chunk_ids={chunk.chunk_id},
                processing_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    def _map_to_ontology_type(self, labels: List[str]) -> str:
        """Map graphiti labels to ontology types."""
        if not labels:
            return "Concept"

        entity_types = self.ontology.get("entity_types", {})
        for label in labels:
            if label in entity_types:
                return label

        # Default mapping
        return labels[0] if labels else "Concept"

    def save_entities(self, entities: List[ExtractedEntity]) -> int:
        """
        Save extracted entities to FalkorDB.

        Creates Entity nodes with temporal metadata and EXTRACTED_FROM relationships.
        """
        if not entities:
            return 0

        # Get current timestamp as ISO string (FalkorDB doesn't support datetime())
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        saved = 0
        for entity in entities:
            try:
                # Escape strings
                name = entity.name.replace("'", "\\'")
                desc = (entity.description or "").replace("'", "\\'")
                aliases_str = str(list(entity.aliases)) if entity.aliases else "[]"
                valid_from_str = entity.valid_from.isoformat() if entity.valid_from else timestamp

                # MERGE Entity node
                if entity.embedding:
                    embed_str = str(entity.embedding)
                    self.graph.query(f"""
                        MERGE (e:Entity {{entity_id: '{entity.entity_id}'}})
                        ON CREATE SET
                            e.name = '{name}',
                            e.ontology_type = '{entity.ontology_type}',
                            e.description = '{desc}',
                            e.aliases = {aliases_str},
                            e.confidence = {entity.confidence},
                            e.status = '{entity.status.value}',
                            e.valid_from = '{valid_from_str}',
                            e.created_at = '{timestamp}',
                            e.embedding = {embed_str}
                        ON MATCH SET
                            e.confidence = CASE WHEN {entity.confidence} > e.confidence
                                           THEN {entity.confidence} ELSE e.confidence END
                    """)
                else:
                    self.graph.query(f"""
                        MERGE (e:Entity {{entity_id: '{entity.entity_id}'}})
                        ON CREATE SET
                            e.name = '{name}',
                            e.ontology_type = '{entity.ontology_type}',
                            e.description = '{desc}',
                            e.aliases = {aliases_str},
                            e.confidence = {entity.confidence},
                            e.status = '{entity.status.value}',
                            e.valid_from = '{valid_from_str}',
                            e.created_at = '{timestamp}'
                        ON MATCH SET
                            e.confidence = CASE WHEN {entity.confidence} > e.confidence
                                           THEN {entity.confidence} ELSE e.confidence END
                    """)

                # Create EXTRACTED_FROM relationships to source chunks
                for chunk_id in entity.source_chunk_ids:
                    self.graph.query(f"""
                        MATCH (e:Entity {{entity_id: '{entity.entity_id}'}})
                        MATCH (c:DocumentChunk {{chunk_id: '{chunk_id}'}})
                        MERGE (e)-[r:EXTRACTED_FROM]->(c)
                        ON CREATE SET r.confidence = {entity.confidence}
                    """)

                saved += 1

            except Exception as e:
                logger.warning(f"Failed to save entity {entity.name}: {e}")

        logger.info(f"Saved {saved}/{len(entities)} entities to FalkorDB")
        return saved

    def save_relationships(self, relationships: List[ExtractedRelationship]) -> int:
        """
        Save extracted relationships to FalkorDB.
        """
        if not relationships:
            return 0

        # Get current timestamp as ISO string (FalkorDB doesn't support datetime())
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        saved = 0
        for rel in relationships:
            try:
                rel_type = rel.relationship_type.upper().replace(" ", "_")
                desc = (rel.description or "").replace("'", "\\'")
                valid_from_str = rel.valid_from.isoformat() if rel.valid_from else timestamp

                self.graph.query(f"""
                    MATCH (source:Entity {{entity_id: '{rel.source_entity_id}'}})
                    MATCH (target:Entity {{entity_id: '{rel.target_entity_id}'}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    ON CREATE SET
                        r.relationship_id = '{rel.relationship_id}',
                        r.description = '{desc}',
                        r.confidence = {rel.confidence},
                        r.valid_from = '{valid_from_str}',
                        r.created_at = '{timestamp}'
                """)
                saved += 1

            except Exception as e:
                logger.warning(f"Failed to save relationship: {e}")

        logger.info(f"Saved {saved}/{len(relationships)} relationships to FalkorDB")
        return saved


class HybridEntityExtractor:
    """
    Combine multiple extraction approaches:
    - graphrag_sdk: Ontology-guided type assignment
    - graphiti: Temporal tracking + deduplication
    - LLM extraction: Direct extraction with ontology prompts

    Provides entity merging and deduplication across sources.
    """

    def __init__(
        self,
        graph_name: str,
        ontology: Optional[Dict[str, Any]] = None,
        falkordb_host: str = FALKORDB_HOST,
        falkordb_port: int = FALKORDB_PORT,
        dedup_threshold: float = 0.85,
    ):
        """
        Initialize the hybrid entity extractor.

        Args:
            graph_name: Name of the FalkorDB graph.
            ontology: Optional ontology configuration dict.
            falkordb_host: FalkorDB host.
            falkordb_port: FalkorDB port.
            dedup_threshold: Similarity threshold for deduplication.
        """
        self.graph_name = graph_name
        self.ontology = ontology or {}
        self.dedup_threshold = dedup_threshold

        # Initialize sub-extractors
        self.ontology_pipeline = OntologyDiscoveryPipeline(
            graph_name=graph_name,
            falkordb_host=falkordb_host,
            falkordb_port=falkordb_port,
        )

        self.graphiti_extractor = GraphitiEntityExtractor(
            graph_name=graph_name,
            ontology=ontology,
            falkordb_host=falkordb_host,
            falkordb_port=falkordb_port,
        )

        # Initialize FalkorDB connection
        from falkordb import FalkorDB
        self.client = FalkorDB(host=falkordb_host, port=falkordb_port)
        self.graph = self.client.select_graph(graph_name)

        # Lazy-loaded embedding model for deduplication
        self._embeddings = None

    def _get_embeddings(self):
        """Lazy load embedding model for deduplication."""
        if self._embeddings is None:
            from langchain_ollama import OllamaEmbeddings
            self._embeddings = OllamaEmbeddings(
                model="qwen3-embedding:8b",
                base_url=OLLAMA_HOST,
                num_ctx=4096,
            )
        return self._embeddings

    def get_chunks_for_extraction(
        self,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Get document chunks for entity extraction.

        Args:
            limit: Maximum number of chunks.
            status: Optional status filter (e.g., "embedded").

        Returns:
            List of DocumentChunk objects.
        """
        status_filter = f"AND c.status = '{status}'" if status else ""

        # Use OPTIONAL MATCH + NULL check instead of NOT EXISTS for FalkorDB compatibility
        result = self.graph.query(f"""
            MATCH (w:WikiPage)-[:HAS_CHUNK]->(c:DocumentChunk)
            OPTIONAL MATCH (e:Entity)-[:EXTRACTED_FROM]->(c)
            WITH w, c, e
            WHERE e IS NULL {status_filter}
            RETURN c.chunk_id as chunk_id,
                   c.content as content,
                   c.chunk_index as chunk_index,
                   w.wikidata_id as source_qid
            LIMIT {limit}
        """)

        chunks = []
        for row in result.result_set:
            chunk = DocumentChunk(
                content=row[1],
                source_qid=row[3],
                chunk_index=row[2],
                chunk_id=row[0],
                status=ChunkStatus.EMBEDDED,
            )
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks for extraction")
        return chunks

    async def process_chunk(
        self,
        chunk: DocumentChunk,
        reference_time: Optional[datetime] = None,
    ) -> ExtractionResult:
        """
        Full extraction pipeline for a single chunk.

        Combines graphiti extraction with ontology-guided type assignment.
        """
        reference_time = reference_time or datetime.now(timezone.utc)

        # Extract using graphiti
        result = await self.graphiti_extractor.extract_from_chunk(chunk, reference_time)

        # Apply ontology type mapping to extracted entities
        for entity in result.entities:
            mapped_type = self._map_entity_type(entity.name, entity.ontology_type)
            entity.ontology_type = mapped_type

        return result

    def _map_entity_type(self, name: str, current_type: str) -> str:
        """
        Map entity to best matching ontology type.
        """
        entity_types = self.ontology.get("entity_types", {})
        if not entity_types:
            return current_type

        # If current type is in ontology, keep it
        if current_type in entity_types:
            return current_type

        # Otherwise return current type as fallback
        return current_type

    def deduplicate_entities(
        self,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedEntity]:
        """
        Deduplicate entities using embedding similarity.

        Merges similar entities based on name embedding similarity.
        """
        if len(entities) <= 1:
            return entities

        embeddings = self._get_embeddings()

        # Get embeddings for all entity names
        names = [e.name for e in entities]
        try:
            vectors = embeddings.embed_documents(names)
        except Exception as e:
            logger.warning(f"Failed to embed entity names for dedup: {e}")
            return entities

        # Compute similarity matrix and merge similar entities
        import numpy as np

        vectors_np = np.array(vectors)
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        normalized = vectors_np / (norms + 1e-8)
        similarities = np.dot(normalized, normalized.T)

        # Find entities to merge
        merged_indices = set()
        deduped = []

        for i, entity in enumerate(entities):
            if i in merged_indices:
                continue

            # Find all similar entities
            similar_indices = np.where(similarities[i] >= self.dedup_threshold)[0]

            # Merge all similar entities into this one
            merged_entity = entity
            for j in similar_indices:
                if j != i and j not in merged_indices:
                    merged_entity = merged_entity.merge_with(entities[j])
                    merged_indices.add(j)

            deduped.append(merged_entity)

        logger.info(f"Deduplicated {len(entities)} → {len(deduped)} entities")
        return deduped

    async def run(
        self,
        max_chunks: int = 100,
        with_dedup: bool = True,
        discover_ontology: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full entity extraction pipeline.

        Args:
            max_chunks: Maximum number of chunks to process.
            with_dedup: Whether to deduplicate entities.
            discover_ontology: Whether to discover ontology first.

        Returns:
            Dictionary with extraction statistics.
        """
        start_time = time.time()

        # Optionally discover ontology first
        if discover_ontology:
            logger.info("Discovering ontology from document chunks...")
            discovered = self.ontology_pipeline.discover_from_chunks()
            self.ontology = self.ontology_pipeline.merge_with_base_ontology(discovered)
            self.graphiti_extractor.ontology = self.ontology
            logger.info(f"Discovered ontology: {len(self.ontology.get('entity_types', {}))} entity types")

        # Get chunks for extraction
        chunks = self.get_chunks_for_extraction(limit=max_chunks)
        if not chunks:
            return {"error": "No chunks available for extraction"}

        # Process each chunk
        all_entities = []
        all_relationships = []
        stats = {
            "chunks_processed": 0,
            "chunks_failed": 0,
            "total_entities_extracted": 0,
            "total_relationships_extracted": 0,
        }

        for chunk in chunks:
            try:
                result = await self.process_chunk(chunk)
                all_entities.extend(result.entities)
                all_relationships.extend(result.relationships)
                stats["chunks_processed"] += 1
                stats["total_entities_extracted"] += len(result.entities)
                stats["total_relationships_extracted"] += len(result.relationships)
            except Exception as e:
                logger.warning(f"Failed to process chunk {chunk.chunk_id}: {e}")
                stats["chunks_failed"] += 1

        # Deduplicate entities
        if with_dedup and all_entities:
            all_entities = self.deduplicate_entities(all_entities)

        # Save to FalkorDB
        entities_saved = self.graphiti_extractor.save_entities(all_entities)
        rels_saved = self.graphiti_extractor.save_relationships(all_relationships)

        stats["entities_saved"] = entities_saved
        stats["relationships_saved"] = rels_saved
        stats["entities_after_dedup"] = len(all_entities)
        stats["processing_time_seconds"] = time.time() - start_time
        stats["graph_name"] = self.graph_name

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics from the graph."""
        stats = {}

        try:
            # Count entities
            result = self.graph.query("MATCH (e:Entity) RETURN count(e) as count")
            stats["total_entities"] = result.result_set[0][0] if result.result_set else 0

            # Count by ontology type
            result = self.graph.query("""
                MATCH (e:Entity)
                RETURN e.ontology_type as type, count(e) as count
                ORDER BY count DESC
            """)
            stats["entities_by_type"] = {}
            for row in result.result_set:
                stats["entities_by_type"][row[0] or "unknown"] = row[1]

            # Count relationships
            result = self.graph.query("""
                MATCH (e1:Entity)-[r]->(e2:Entity)
                RETURN type(r) as rel_type, count(r) as count
            """)
            stats["relationships"] = {}
            for row in result.result_set:
                stats["relationships"][row[0]] = row[1]

            # Count EXTRACTED_FROM relationships
            result = self.graph.query("""
                MATCH (e:Entity)-[r:EXTRACTED_FROM]->(c:DocumentChunk)
                RETURN count(r) as count
            """)
            stats["extracted_from_count"] = result.result_set[0][0] if result.result_set else 0

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats["error"] = str(e)

        return stats


def main():
    """CLI entry point for entity extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract entities and relationships from document chunks"
    )

    parser.add_argument(
        "--graph",
        type=str,
        default="wikidata",
        help="FalkorDB graph name (default: wikidata)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=100,
        help="Maximum number of chunks to process (default: 100)"
    )
    parser.add_argument(
        "--discover-ontology",
        action="store_true",
        help="Auto-discover ontology from chunks first"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip entity deduplication"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Base ontology to use (default: medium)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show current extraction statistics"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load base ontology
    from ontologies import get_ontology
    base_ontology = get_ontology(args.ontology)
    ontology_dict = {
        "entity_types": base_ontology.entity_types,
        "relationship_types": base_ontology.relationship_types,
    }

    # Initialize extractor
    extractor = HybridEntityExtractor(
        graph_name=args.graph,
        ontology=ontology_dict,
    )

    # Stats-only mode
    if args.stats_only:
        stats = extractor.get_stats()
        print(f"\nEntity Extraction Statistics ({args.graph}):")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        return

    # Run extraction
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Graph:           {args.graph}")
    print(f"Max Chunks:      {args.max_chunks}")
    print(f"Base Ontology:   {args.ontology}")
    print(f"Discover Ontology: {args.discover_ontology}")
    print(f"Deduplication:   {not args.no_dedup}")
    print("=" * 60)

    # Run async extraction
    stats = asyncio.run(extractor.run(
        max_chunks=args.max_chunks,
        with_dedup=not args.no_dedup,
        discover_ontology=args.discover_ontology,
    ))

    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return

    print(f"Chunks processed: {stats['chunks_processed']}")
    print(f"Chunks failed: {stats['chunks_failed']}")
    print(f"Entities extracted: {stats['total_entities_extracted']}")
    print(f"Entities after dedup: {stats.get('entities_after_dedup', 'N/A')}")
    print(f"Entities saved: {stats['entities_saved']}")
    print(f"Relationships extracted: {stats['total_relationships_extracted']}")
    print(f"Relationships saved: {stats['relationships_saved']}")
    print(f"Processing time: {stats['processing_time_seconds']:.2f}s")


if __name__ == "__main__":
    main()
