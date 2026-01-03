#!/usr/bin/env python3
"""
Hybrid RAG implementation combining Graph and Document retrieval.

Provides unified retrieval from:
- Entity vector search (existing knowledge graph)
- Relationship traversal (graph structure)
- Document vector search (ingested documents)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class SourceType(str, Enum):
    """Types of retrieval sources."""
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    DOCUMENT = "document"


class RetrievedItem(BaseModel):
    """A single retrieved item from hybrid search."""

    source_type: SourceType
    source_id: str
    content: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_context_string(self) -> str:
        """Format as context string for LLM."""
        prefix = f"[{self.source_type.value.upper()}]"
        return f"{prefix} {self.content}"


class HybridContext(BaseModel):
    """Combined context from hybrid retrieval."""

    entities: List[RetrievedItem] = Field(default_factory=list)
    relationships: List[RetrievedItem] = Field(default_factory=list)
    documents: List[RetrievedItem] = Field(default_factory=list)
    query: str = ""

    @property
    def all_items(self) -> List[RetrievedItem]:
        """Get all items sorted by relevance."""
        all_items = self.entities + self.relationships + self.documents
        return sorted(all_items, key=lambda x: x.relevance_score, reverse=True)

    @property
    def total_count(self) -> int:
        """Total number of retrieved items."""
        return len(self.entities) + len(self.relationships) + len(self.documents)

    def format_for_llm(self, max_chars: int = 8000) -> str:
        """Format context for LLM consumption."""
        parts = []
        current_length = 0

        for item in self.all_items:
            context_str = item.to_context_string()
            if current_length + len(context_str) > max_chars:
                break
            parts.append(context_str)
            current_length += len(context_str) + 2  # +2 for newlines

        return "\n\n".join(parts) if parts else "No relevant context found."

    def get_source_breakdown(self) -> Dict[str, int]:
        """Get count breakdown by source type."""
        return {
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "documents": len(self.documents),
        }


class HybridRAG:
    """
    Hybrid RAG that combines graph and document retrieval.

    Retrieval strategy:
    1. Vector search on entity embeddings
    2. Traverse relationships from matched entities
    3. Vector search on document embeddings
    4. Combine and rank by relevance
    """

    def __init__(
        self,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_password: str = NEO4J_PASSWORD,
        embedding_model: Optional[Any] = None,
        llm: Optional[Any] = None,
    ):
        """
        Initialize HybridRAG.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            embedding_model: Embedding model for vector search.
            llm: LLM for answer generation.
        """
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )
        self.embedding_model = embedding_model or OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)

        # Check available indices
        self._entity_index_exists = self._check_index("entity_embeddings")
        self._document_index_exists = self._check_index("document_embeddings")

    def _check_index(self, index_name: str) -> bool:
        """Check if a vector index exists."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    "SHOW INDEXES WHERE name = $name",
                    name=index_name,
                )
                return result.single() is not None
        except Exception:
            return False

    def close(self):
        """Close database connection."""
        self.driver.close()

    def retrieve_hybrid(
        self,
        query: str,
        k_entities: int = 5,
        k_relationships: int = 10,
        k_documents: int = 5,
        include_relationships: bool = True,
    ) -> HybridContext:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query.
            k_entities: Number of entities to retrieve.
            k_relationships: Max relationships to include per entity.
            k_documents: Number of documents to retrieve.
            include_relationships: Whether to traverse relationships.

        Returns:
            HybridContext with all retrieved items.
        """
        context = HybridContext(query=query)

        # 1. Entity vector search
        entities = self._search_entities(query, k_entities)
        context.entities = entities

        # 2. Relationship traversal
        if include_relationships and entities:
            entity_ids = [e.source_id for e in entities]
            relationships = self._get_relationships(entity_ids, k_relationships)
            context.relationships = relationships

        # 3. Document vector search
        if self._document_index_exists:
            documents = self._search_documents(query, k_documents)
            context.documents = documents

        return context

    def _search_entities(
        self,
        query: str,
        limit: int = 5,
    ) -> List[RetrievedItem]:
        """Search entities by vector similarity."""
        if not self._entity_index_exists:
            logger.warning("Entity embedding index not found")
            return []

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

                items = []
                for record in result:
                    node = dict(record["node"])
                    name = node.get("name", node.get("id", "Unknown"))
                    entity_type = node.get("ontology_type", node.get("open_type", "Entity"))
                    description = node.get("description", "")

                    content = f"{name} ({entity_type})"
                    if description:
                        content += f": {description}"

                    items.append(RetrievedItem(
                        source_type=SourceType.ENTITY,
                        source_id=name,
                        content=content,
                        relevance_score=record["score"],
                        metadata={
                            "labels": record["labels"],
                            "entity_type": entity_type,
                        },
                    ))

                return items

        except Exception as e:
            logger.error(f"Entity search error: {e}")
            return []

    def _get_relationships(
        self,
        entity_ids: List[str],
        limit_per_entity: int = 10,
    ) -> List[RetrievedItem]:
        """Get relationships for entities."""
        items = []

        try:
            with self.driver.session() as session:
                for entity_id in entity_ids[:5]:  # Limit to avoid too many queries
                    cypher = """
                    MATCH (e)-[r]-(other)
                    WHERE e.name = $name OR e.id = $name
                    RETURN e.name as source_name,
                           type(r) as rel_type,
                           properties(r) as rel_props,
                           other.name as target_name,
                           other.description as target_desc,
                           labels(other) as target_labels,
                           startNode(r) = e as is_outgoing
                    LIMIT $limit
                    """
                    result = session.run(
                        cypher,
                        name=entity_id,
                        limit=limit_per_entity,
                    )

                    for record in result:
                        source = record["source_name"]
                        target = record["target_name"]
                        rel_type = record["rel_type"]
                        direction = "->" if record["is_outgoing"] else "<-"

                        # Build content
                        content = f"{source} {direction}[{rel_type}]{direction} {target}"
                        rel_props = record["rel_props"]
                        if rel_props and rel_props.get("description"):
                            content += f" ({rel_props['description']})"

                        items.append(RetrievedItem(
                            source_type=SourceType.RELATIONSHIP,
                            source_id=f"{source}-{rel_type}-{target}",
                            content=content,
                            relevance_score=0.7,  # Fixed score for relationships
                            metadata={
                                "source": source,
                                "target": target,
                                "rel_type": rel_type,
                                "direction": "outgoing" if record["is_outgoing"] else "incoming",
                            },
                        ))

        except Exception as e:
            logger.error(f"Relationship fetch error: {e}")

        return items

    def _search_documents(
        self,
        query: str,
        limit: int = 5,
    ) -> List[RetrievedItem]:
        """Search documents by vector similarity."""
        if not self._document_index_exists:
            return []

        try:
            embedding = self.embedding_model.embed_query(query)

            with self.driver.session() as session:
                cypher = """
                CALL db.index.vector.queryNodes('document_embeddings', $limit, $embedding)
                YIELD node, score
                WHERE 'Document' IN labels(node)
                RETURN node, score
                ORDER BY score DESC
                """
                result = session.run(cypher, embedding=embedding, limit=limit)

                items = []
                for record in result:
                    node = dict(record["node"])
                    title = node.get("title", "Untitled")
                    content = node.get("content", "")[:500]  # Limit content length
                    url = node.get("url", "")

                    full_content = f"{title}"
                    if url:
                        full_content += f" ({url})"
                    full_content += f"\n{content}"

                    items.append(RetrievedItem(
                        source_type=SourceType.DOCUMENT,
                        source_id=node.get("id", url),
                        content=full_content,
                        relevance_score=record["score"],
                        metadata={
                            "url": url,
                            "domain": node.get("domain", ""),
                            "trust_level": node.get("trust_level", "unknown"),
                            "source_type": node.get("source_type", "unknown"),
                        },
                    ))

                return items

        except Exception as e:
            logger.error(f"Document search error: {e}")
            return []

    def generate_answer(
        self,
        query: str,
        context: Optional[HybridContext] = None,
        k_entities: int = 5,
        k_documents: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate an answer using hybrid retrieval.

        Args:
            query: User question.
            context: Pre-retrieved context (optional).
            k_entities: Number of entities to retrieve.
            k_documents: Number of documents to retrieve.

        Returns:
            Dictionary with answer and metadata.
        """
        # Retrieve context if not provided
        if context is None:
            context = self.retrieve_hybrid(
                query=query,
                k_entities=k_entities,
                k_documents=k_documents,
            )

        # Format context for LLM
        formatted_context = context.format_for_llm()

        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant that answers questions using the provided context.

Guidelines:
- Answer based on the context provided
- Cite sources using [Source: X] notation
- If the context doesn't contain enough information, say so
- Be accurate and concise"""),
            ("human", """Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context above."""),
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "context": formatted_context,
                "question": query,
            })

            return {
                "answer": response.content,
                "context": context,
                "sources": {
                    "entities": [e.source_id for e in context.entities],
                    "documents": [d.source_id for d in context.documents],
                },
                "source_breakdown": context.get_source_breakdown(),
            }

        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context": context,
                "error": str(e),
            }

    def retrieve_for_entity(
        self,
        entity_name: str,
        include_neighbors: bool = True,
        max_neighbors: int = 20,
    ) -> HybridContext:
        """
        Retrieve context for a specific entity.

        Args:
            entity_name: Name of the entity.
            include_neighbors: Whether to include connected entities.
            max_neighbors: Maximum neighbors to include.

        Returns:
            HybridContext focused on the entity.
        """
        context = HybridContext(query=entity_name)

        try:
            with self.driver.session() as session:
                # Get entity
                entity_query = """
                MATCH (e)
                WHERE toLower(e.name) = toLower($name) OR toLower(e.id) = toLower($name)
                RETURN e, labels(e) as labels
                LIMIT 1
                """
                result = session.run(entity_query, name=entity_name)
                record = result.single()

                if record:
                    node = dict(record["e"])
                    name = node.get("name", node.get("id", entity_name))
                    entity_type = node.get("ontology_type", node.get("open_type", "Entity"))
                    description = node.get("description", "")

                    content = f"{name} ({entity_type})"
                    if description:
                        content += f": {description}"

                    context.entities.append(RetrievedItem(
                        source_type=SourceType.ENTITY,
                        source_id=name,
                        content=content,
                        relevance_score=1.0,
                        metadata={
                            "labels": record["labels"],
                            "entity_type": entity_type,
                        },
                    ))

                    # Get relationships
                    if include_neighbors:
                        relationships = self._get_relationships([name], max_neighbors)
                        context.relationships = relationships

        except Exception as e:
            logger.error(f"Entity retrieval error: {e}")

        return context


class AsyncHybridRAG:
    """Async wrapper for HybridRAG for use with Chainlit."""

    def __init__(self, rag: Optional[HybridRAG] = None, **kwargs):
        self._rag = rag
        self._kwargs = kwargs
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            if self._rag is None:
                self._rag = HybridRAG(**self._kwargs)
            self._initialized = True

    async def retrieve_hybrid(
        self,
        query: str,
        **kwargs,
    ) -> HybridContext:
        """Async hybrid retrieval."""
        import asyncio
        self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._rag.retrieve_hybrid(query, **kwargs),
        )

    async def generate_answer(
        self,
        query: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Async answer generation."""
        import asyncio
        self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._rag.generate_answer(query, **kwargs),
        )

    async def close(self):
        """Close connections."""
        if self._rag:
            self._rag.close()


# CLI for testing
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid RAG")
    parser.add_argument("--query", "-q", type=str, required=True, help="Query to search")
    parser.add_argument("--entities", "-e", type=int, default=5, help="Number of entities")
    parser.add_argument("--documents", "-d", type=int, default=5, help="Number of documents")
    args = parser.parse_args()

    rag = HybridRAG()

    try:
        print(f"Query: {args.query}\n")

        # Retrieve
        context = rag.retrieve_hybrid(
            query=args.query,
            k_entities=args.entities,
            k_documents=args.documents,
        )

        print(f"Retrieved: {context.get_source_breakdown()}\n")

        # Generate answer
        result = rag.generate_answer(args.query, context=context)

        print("Answer:")
        print(result["answer"])
        print(f"\nSources: {result.get('sources', {})}")

    finally:
        rag.close()


if __name__ == "__main__":
    main()
