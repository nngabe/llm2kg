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
    Episode,
    Community,
    PipelineConfig,
    EntityStatus,
    FactStatus,
    ChunkStatus,
    SourceType,
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

        # Initialize Episode/Community schema
        self.init_episode_schema()

    def init_episode_schema(self):
        """
        Create Episode and Community indices for Zep-style temporal layer.

        Creates indexes for efficient querying of:
        - Episodes by episode_id, source_chunk_id, reference_time
        - Communities by community_id
        - Entities by valid_from, fact_status for temporal queries
        """
        indices = [
            "CREATE INDEX FOR (ep:Episode) ON (ep.episode_id)",
            "CREATE INDEX FOR (ep:Episode) ON (ep.source_chunk_id)",
            "CREATE INDEX FOR (ep:Episode) ON (ep.reference_time)",
            "CREATE INDEX FOR (c:Community) ON (c.community_id)",
            "CREATE INDEX FOR (e:Entity) ON (e.valid_from)",
            "CREATE INDEX FOR (e:Entity) ON (e.fact_status)",
        ]
        for idx in indices:
            try:
                self.graph.query(idx)
                logger.debug(f"Created index: {idx}")
            except Exception:
                pass  # Index may already exist

        logger.info("Episode and Community schema initialized")

    def save_episode(self, episode: Episode) -> bool:
        """
        Persist Episode node and link to source DocumentChunk.

        Args:
            episode: Episode object to save.

        Returns:
            True if saved successfully, False otherwise.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        ref_time = episode.reference_time.isoformat() if episode.reference_time else timestamp

        try:
            # Escape content for Cypher
            content_escaped = episode.content[:2000].replace("'", "\\'").replace("\n", "\\n")
            name_escaped = episode.name.replace("'", "\\'")
            source_url = (episode.source_url or "").replace("'", "\\'")

            # Create Episode node
            self.graph.query(f"""
                MERGE (ep:Episode {{episode_id: '{episode.episode_id}'}})
                ON CREATE SET
                    ep.name = '{name_escaped}',
                    ep.content = '{content_escaped}',
                    ep.source_type = '{episode.source_type}',
                    ep.source_chunk_id = '{episode.source_chunk_id}',
                    ep.source_url = '{source_url}',
                    ep.reference_time = '{ref_time}',
                    ep.created_at = '{timestamp}',
                    ep.valid_from = '{ref_time}'
            """)

            # Link to DocumentChunk via DERIVED_FROM
            self.graph.query(f"""
                MATCH (ep:Episode {{episode_id: '{episode.episode_id}'}})
                MATCH (c:DocumentChunk {{chunk_id: '{episode.source_chunk_id}'}})
                MERGE (ep)-[:DERIVED_FROM]->(c)
            """)

            logger.debug(f"Saved Episode: {episode.episode_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save Episode {episode.episode_id}: {e}")
            return False

    def link_episode_to_entities(
        self,
        episode_id: str,
        entity_ids: List[str],
        confidence: float = 0.9,
    ) -> int:
        """
        Create CONTAINS relationships from Episode to extracted Entities.

        Args:
            episode_id: Episode ID to link from.
            entity_ids: List of Entity IDs to link to.
            confidence: Confidence score for the relationships.

        Returns:
            Number of relationships created.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        created = 0

        for entity_id in entity_ids:
            try:
                self.graph.query(f"""
                    MATCH (ep:Episode {{episode_id: '{episode_id}'}})
                    MATCH (e:Entity {{entity_id: '{entity_id}'}})
                    MERGE (ep)-[r:CONTAINS]->(e)
                    ON CREATE SET
                        r.confidence = {confidence},
                        r.created_at = '{timestamp}'
                """)
                created += 1
            except Exception as e:
                logger.warning(f"Failed to link Episode {episode_id} to Entity {entity_id}: {e}")

        if created > 0:
            logger.debug(f"Created {created} CONTAINS relationships for Episode {episode_id}")

        return created

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
        create_episode: bool = True,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a single chunk.

        Uses graphiti's add_episode() pattern for temporal extraction.
        Creates Episode node and CONTAINS relationships (Zep architecture).

        Args:
            chunk: DocumentChunk to process.
            reference_time: Reference time for temporal context.
            create_episode: Whether to create Episode node (default True).

        Returns:
            ExtractionResult with extracted entities and relationships.
        """
        start_time = time.time()
        reference_time = reference_time or datetime.now(timezone.utc)

        # Step 1: Create Episode from chunk (Zep episodic layer)
        episode = None
        if create_episode:
            source_type_val = chunk.source_type.value if chunk.source_type else "text"
            episode = Episode(
                name=f"chunk_{chunk.chunk_id}",
                content=chunk.content,
                source_chunk_id=chunk.chunk_id,
                source_type=source_type_val,
                source_url=chunk.source_url,
                reference_time=reference_time,
            )
            self.save_episode(episode)

        # Step 2: Try graphiti extraction
        graphiti = await self._get_graphiti()
        if graphiti is not None:
            try:
                result = await self._extract_with_graphiti(chunk, reference_time)
            except Exception as e:
                logger.warning(f"Graphiti extraction failed: {e}")
                result = await self._extract_with_llm(chunk, reference_time)
        else:
            # Fallback to simple LLM extraction
            result = await self._extract_with_llm(chunk, reference_time)

        # Step 3: Set fact_status to active for all entities
        for entity in result.entities:
            entity.fact_status = FactStatus.ACTIVE.value

        # Step 4: Link Episode to Entities via CONTAINS
        if episode and result.entities:
            entity_ids = [e.entity_id for e in result.entities]
            self.link_episode_to_entities(episode.episode_id, entity_ids)
            episode.entity_ids = set(entity_ids)

        # Add episode_id to result metadata
        if episode:
            result.metadata["episode_id"] = episode.episode_id

        result.processing_time_seconds = time.time() - start_time
        return result

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

        Creates Entity nodes with bi-temporal metadata (Zep-style),
        multi-source provenance, and EXTRACTED_FROM relationships.
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

                # Multi-source provenance
                source_urls_str = str(list(entity.source_urls)) if entity.source_urls else "[]"
                source_types_str = str(list(entity.source_types)) if entity.source_types else "[]"

                # Bi-temporal fields (Zep-style)
                fact_status = entity.fact_status or "active"

                # MERGE Entity node with bi-temporal and multi-source fields
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
                            e.source_urls = {source_urls_str},
                            e.source_types = {source_types_str},
                            e.valid_from = '{valid_from_str}',
                            e.created_at = '{timestamp}',
                            e.fact_status = '{fact_status}',
                            e.embedding = {embed_str}
                        ON MATCH SET
                            e.confidence = CASE WHEN {entity.confidence} > e.confidence
                                           THEN {entity.confidence} ELSE e.confidence END,
                            e.modified_at = '{timestamp}'
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
                            e.source_urls = {source_urls_str},
                            e.source_types = {source_types_str},
                            e.valid_from = '{valid_from_str}',
                            e.created_at = '{timestamp}',
                            e.fact_status = '{fact_status}'
                        ON MATCH SET
                            e.confidence = CASE WHEN {entity.confidence} > e.confidence
                                           THEN {entity.confidence} ELSE e.confidence END,
                            e.modified_at = '{timestamp}'
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
        source_type: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Get document chunks for entity extraction from both WikiPage and WebPage sources.

        Args:
            limit: Maximum number of chunks.
            status: Optional status filter (e.g., "embedded").
            source_type: Optional filter by source type ("wikipedia" or "webpage").

        Returns:
            List of DocumentChunk objects.
        """
        status_filter = f"AND c.status = '{status}'" if status else ""
        source_type_filter = f"AND c.source_type = '{source_type}'" if source_type else ""

        # Query chunks from both WikiPage and WebPage sources
        # Use OPTIONAL MATCH + NULL check instead of NOT EXISTS for FalkorDB compatibility
        result = self.graph.query(f"""
            MATCH (c:DocumentChunk)
            WHERE c.source_id IS NOT NULL
            OPTIONAL MATCH (e:Entity)-[:EXTRACTED_FROM]->(c)
            WITH c, e
            WHERE e IS NULL {status_filter} {source_type_filter}
            RETURN c.chunk_id as chunk_id,
                   c.content as content,
                   c.chunk_index as chunk_index,
                   c.source_id as source_id,
                   c.source_type as source_type,
                   c.source_url as source_url
            LIMIT {limit}
        """)

        chunks = []
        for row in result.result_set:
            source_type_val = row[4] or "wikipedia"
            chunk = DocumentChunk(
                content=row[1],
                chunk_index=row[2],
                chunk_id=row[0],
                source_id=row[3],
                source_type=SourceType(source_type_val) if source_type_val else SourceType.WIKIPEDIA,
                source_url=row[5],
                # Backwards compatibility
                source_qid=row[3] if source_type_val == "wikipedia" else None,
                status=ChunkStatus.EMBEDDED,
            )
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks for extraction (source_type={source_type or 'all'})")
        return chunks

    async def process_chunk(
        self,
        chunk: DocumentChunk,
        reference_time: Optional[datetime] = None,
    ) -> ExtractionResult:
        """
        Full extraction pipeline for a single chunk.

        Combines graphiti extraction with ontology-guided type assignment.
        Adds multi-source provenance (source_urls, source_types) to entities.
        """
        reference_time = reference_time or datetime.now(timezone.utc)

        # Extract using graphiti
        result = await self.graphiti_extractor.extract_from_chunk(chunk, reference_time)

        # Apply ontology type mapping and add multi-source provenance
        for entity in result.entities:
            mapped_type = self._map_entity_type(entity.name, entity.ontology_type)
            entity.ontology_type = mapped_type

            # Add source URL and type from chunk
            if chunk.source_url:
                entity.source_urls.add(chunk.source_url)
            if chunk.source_type:
                entity.source_types.add(chunk.source_type.value)

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
        """Get extraction statistics from the graph including Zep temporal layers."""
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

            # Count by fact_status (Zep temporal)
            result = self.graph.query("""
                MATCH (e:Entity)
                WHERE e.fact_status IS NOT NULL
                RETURN e.fact_status as status, count(e) as count
            """)
            stats["entities_by_fact_status"] = {}
            for row in result.result_set:
                stats["entities_by_fact_status"][row[0] or "unknown"] = row[1]

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

            # Zep Episodic Layer stats
            result = self.graph.query("MATCH (ep:Episode) RETURN count(ep) as count")
            stats["total_episodes"] = result.result_set[0][0] if result.result_set else 0

            # Count CONTAINS relationships (Episode -> Entity)
            result = self.graph.query("MATCH ()-[r:CONTAINS]->() RETURN count(r) as count")
            stats["contains_relationships"] = result.result_set[0][0] if result.result_set else 0

            # Count DERIVED_FROM relationships (Episode -> DocumentChunk)
            result = self.graph.query("MATCH ()-[r:DERIVED_FROM]->() RETURN count(r) as count")
            stats["derived_from_relationships"] = result.result_set[0][0] if result.result_set else 0

            # Zep Community Layer stats
            result = self.graph.query("MATCH (c:Community) RETURN count(c) as count")
            stats["total_communities"] = result.result_set[0][0] if result.result_set else 0

            # Count BELONGS_TO relationships (Entity -> Community)
            result = self.graph.query("MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count")
            stats["belongs_to_relationships"] = result.result_set[0][0] if result.result_set else 0

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
        "--detect-communities",
        action="store_true",
        help="Run Louvain community detection after extraction"
    )
    parser.add_argument(
        "--community-min-size",
        type=int,
        default=3,
        help="Minimum community size for detection (default: 3)"
    )
    parser.add_argument(
        "--temporal-stats",
        action="store_true",
        help="Show temporal layer statistics (Episodes, Communities)"
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

    # Temporal stats mode
    if args.temporal_stats:
        from .temporal_queries import get_temporal_statistics
        stats = get_temporal_statistics(extractor.graph)
        print(f"\nTemporal Layer Statistics ({args.graph}):")
        print("-" * 40)
        print(f"Episodes: {stats.get('episodes', 0)}")
        print(f"Communities: {stats.get('communities', 0)}")
        print(f"CONTAINS relationships: {stats.get('contains_relationships', 0)}")
        print(f"BELONGS_TO relationships: {stats.get('belongs_to_relationships', 0)}")
        print(f"DERIVED_FROM relationships: {stats.get('derived_from_relationships', 0)}")
        print(f"SUPERSEDES relationships: {stats.get('supersedes_relationships', 0)}")
        if stats.get("entities_by_fact_status"):
            print("\nEntities by fact_status:")
            for status, count in stats["entities_by_fact_status"].items():
                print(f"  {status}: {count}")
        if stats.get("episodes_by_source_type"):
            print("\nEpisodes by source_type:")
            for stype, count in stats["episodes_by_source_type"].items():
                print(f"  {stype}: {count}")
        return

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

    # Run community detection if requested
    if args.detect_communities:
        print("\n" + "=" * 60)
        print("COMMUNITY DETECTION")
        print("=" * 60)

        from .community_detection import EntityCommunityDetector

        detector = EntityCommunityDetector(
            graph=extractor.graph,
            min_community_size=args.community_min_size,
        )
        community_count = detector.run()
        print(f"Created {community_count} entity communities")

        # Show community stats
        comm_stats = detector.get_stats()
        if comm_stats.get("communities"):
            print("\nTop communities:")
            for c in comm_stats["communities"][:5]:
                print(f"  {c['name']}: {c['member_count']} members")


if __name__ == "__main__":
    main()
