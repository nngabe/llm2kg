"""
GraphRAG integration for Chainlit frontend.

Provides async wrappers around the GraphRAG class for use with Chainlit's
async event loop.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class ChainlitGraphRAG:
    """
    Async-compatible GraphRAG wrapper for Chainlit.

    Wraps sync Neo4j and LLM calls with asyncio executor for
    non-blocking operation in Chainlit's event loop.
    """

    def __init__(
        self,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_password: str = NEO4J_PASSWORD,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "qwen3-embedding:8b",
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://host.docker.internal:11434",
        )
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def close(self):
        """Close connections."""
        self.driver.close()
        self.executor.shutdown(wait=False)

    def _retrieve_graph_context_sync(
        self,
        query_text: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Synchronous graph context retrieval."""
        query_vector = self.embeddings.embed_query(query_text)

        cypher_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
        YIELD node AS head, score

        OPTIONAL MATCH (head)-[r]->(tail)

        RETURN
            head.name AS entity,
            head.type AS type,
            head.description AS description,
            score,
            COLLECT({
                relation: type(r),
                description: r.description,
                target: tail.name,
                target_type: tail.type
            }) AS relationships
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, limit=limit, embedding=query_vector)
            return [record.data() for record in result]

    async def retrieve_graph_context(
        self,
        query_text: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Async graph context retrieval."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._retrieve_graph_context_sync,
            query_text,
            limit,
        )

    def _extract_entities_sync(self, query_text: str) -> List[Dict[str, Any]]:
        """Extract entity mentions from query using LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract entity names mentioned in the query.
Return a JSON list of objects with 'name' and 'type' (if determinable).
Example: [{"name": "Albert Einstein", "type": "Person"}]
If no entities found, return empty list: []"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({"query": query_text})
            import json
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    async def extract_entities(self, query_text: str) -> List[Dict[str, Any]]:
        """Async entity extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_entities_sync,
            query_text,
        )

    def _resolve_entity_sync(
        self,
        entity_name: str,
        context: str = "",
        max_candidates: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find matching entities in the graph."""
        # Try exact match first
        exact_query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) = toLower($name)
        RETURN n.name AS name, n.type AS type, n.description AS description, 1.0 AS score
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(exact_query, name=entity_name)
            exact_matches = [record.data() for record in result]
            if exact_matches:
                return exact_matches

        # Fuzzy search using text index
        fuzzy_query = """
        CALL db.index.fulltext.queryNodes('entity_name_idx', $name + '~')
        YIELD node, score
        RETURN node.name AS name, node.type AS type, node.description AS description, score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(fuzzy_query, name=entity_name, limit=max_candidates)
                return [record.data() for record in result]
        except Exception:
            # Fall back to CONTAINS search if fulltext index doesn't exist
            contains_query = """
            MATCH (n:Entity)
            WHERE toLower(n.name) CONTAINS toLower($name)
            RETURN n.name AS name, n.type AS type, n.description AS description, 0.8 AS score
            LIMIT $limit
            """
            with self.driver.session() as session:
                result = session.run(contains_query, name=entity_name, limit=max_candidates)
                return [record.data() for record in result]

    async def resolve_entity(
        self,
        entity_name: str,
        context: str = "",
        max_candidates: int = 5,
    ) -> List[Dict[str, Any]]:
        """Async entity resolution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._resolve_entity_sync,
            entity_name,
            context,
            max_candidates,
        )

    def format_context(self, context_data: List[Dict]) -> str:
        """Format graph context for LLM."""
        if not context_data:
            return "No relevant data found in the knowledge graph."

        formatted_text = "Retrieved Knowledge Graph Data:\n"

        for entry in context_data:
            formatted_text += f"\n=== Entity: {entry.get('entity', 'Unknown')} ({entry.get('type', 'Unknown')}) ===\n"
            if entry.get('description'):
                formatted_text += f"Summary: {entry['description']}\n"

            formatted_text += "Connections:\n"
            relationships = entry.get('relationships', [])
            if relationships:
                for rel in relationships:
                    if rel.get('relation'):
                        formatted_text += f"  -> [{rel['relation']}] -> {rel.get('target', 'Unknown')} ({rel.get('target_type', '')})\n"
                        if rel.get('description'):
                            formatted_text += f"     Context: {rel['description']}\n"
            else:
                formatted_text += "  (No outgoing relationships found)\n"

        return formatted_text

    def _generate_answer_sync(
        self,
        question: str,
        context_data: List[Dict],
    ) -> str:
        """Synchronous answer generation."""
        context_str = self.format_context(context_data)

        system_prompt = """You are a helpful assistant answering questions based *only* on the provided Knowledge Graph data.

The graph data includes:
- Entities (Nodes) with descriptions
- Connections (Edges) with context explaining relationships

Guidelines:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain relevant information, say so
3. Cite the entities and relationships you used
4. Be precise and accurate - don't hallucinate information"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "context": context_str,
            "question": question,
        })

    async def generate_answer(
        self,
        question: str,
        context_data: Optional[List[Dict]] = None,
    ) -> str:
        """Async answer generation."""
        if context_data is None:
            context_data = await self.retrieve_graph_context(question)

        if not context_data:
            return "I couldn't find any relevant information in the Knowledge Graph for your question."

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_answer_sync,
            question,
            context_data,
        )

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Full RAG query with context and answer.

        Returns dict with:
        - context: List of graph context entries
        - context_formatted: Formatted context string
        - answer: Generated answer
        - entities: Extracted entities from question
        """
        # Extract entities from question
        entities = await self.extract_entities(question)

        # Retrieve graph context
        context_data = await self.retrieve_graph_context(question)

        # Format context
        context_formatted = self.format_context(context_data)

        # Generate answer
        answer = await self.generate_answer(question, context_data)

        return {
            "context": context_data,
            "context_formatted": context_formatted,
            "answer": answer,
            "entities": entities,
        }

    def _run_cypher_sync(self, query: str, params: Dict = None) -> List[Dict]:
        """Run arbitrary Cypher query."""
        with self.driver.session() as session:
            result = session.run(query, **(params or {}))
            return [record.data() for record in result]

    async def run_cypher(self, query: str, params: Dict = None) -> List[Dict]:
        """Async Cypher query execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_cypher_sync,
            query,
            params,
        )
