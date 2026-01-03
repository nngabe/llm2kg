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
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from neo4j import GraphDatabase

from langgraph.graph import StateGraph, START, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

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

    # Error handling
    error: Optional[str] = None


# --- PROMPTS ---

SYSTEM_PROMPT = """You are a knowledgeable research assistant with access to a knowledge graph and web search.
Your goal is to answer questions accurately using available information sources.

Available tools:
1. graph_lookup(entity_name) - Look up an entity in the knowledge graph
2. web_search(query) - Search the web for information (use only when graph doesn't have the answer)
3. cypher_query(query) - Run a Cypher query on the knowledge graph
4. entity_resolve(entity_name, context) - Disambiguate an entity name

Guidelines:
- ALWAYS try the knowledge graph first before using web search
- When using web search, clearly note that external information was used
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
        "tool_name": "graph_lookup|web_search|cypher_query|entity_resolve|none",
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
5. Rate your confidence (0-1) based on source quality and completeness

Respond in this JSON format:
{{
    "answer": "Your comprehensive answer with citations",
    "confidence": 0.0-1.0,
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
        self.embedding_model = embedding_model or OpenAIEmbeddings(
            model="text-embedding-3-small"
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
        web_search_enabled: bool = True,
        auto_add_documents: bool = True,
    ):
        """
        Initialize the ReAct Q&A agent.

        Args:
            llm: Language model to use. Defaults to GPT-4.
            neo4j_loader: Neo4j loader for graph operations.
            web_search_enabled: Whether to allow web searches.
            auto_add_documents: Whether to automatically add web results to DB.
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.neo4j_loader = neo4j_loader or Neo4jQALoader()
        self.web_search_enabled = web_search_enabled
        self.auto_add_documents = auto_add_documents

        # Initialize web search tool
        self._web_search_tool = None
        if web_search_enabled:
            try:
                from finetuning.agent.tools import WebSearchTool
                self._web_search_tool = WebSearchTool()
            except ImportError:
                logger.warning("Web search tool not available")

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(QAAgentState)

        # Add nodes
        graph.add_node("retrieve_initial", self._retrieve_initial_node)
        graph.add_node("think", self._think_node)
        graph.add_node("execute_action", self._execute_action_node)
        graph.add_node("observe", self._observe_node)
        graph.add_node("synthesize", self._synthesize_node)

        # Add edges
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

    def _retrieve_initial_node(self, state: QAAgentState) -> Dict[str, Any]:
        """Initial retrieval based on the question."""
        question = state.question
        context = []

        # Vector search on entities
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
        for step in state.thought_history:
            history_parts.append(f"Thought: {step.thought}")
            if step.action:
                history_parts.append(f"Action: {step.action.tool_name}({step.action.arguments})")
            if step.observation:
                history_parts.append(f"Observation: {step.observation[:500]}")
            history_parts.append("")

        thought_history = "\n".join(history_parts) if history_parts else "No previous steps."

        # Generate next thought
        prompt = THINK_PROMPT.format(
            question=state.question,
            thought_history=thought_history,
            context=state.context_formatted[:CONTEXT_WINDOW_SIZE],
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            # Parse response
            content = response.content
            # Try to extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content)

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

        except Exception as e:
            logger.error(f"Think node error: {e}")
            return {
                "ready_to_answer": True,
                "error": str(e),
            }

    def _execute_action_node(self, state: QAAgentState) -> Dict[str, Any]:
        """Execute the pending action."""
        if not state.pending_action:
            return {"last_observation": "No action to execute"}

        action = state.pending_action
        tool_name = action.tool_name
        args = action.arguments

        try:
            if tool_name == "graph_lookup":
                result = self.neo4j_loader.get_entity_with_relationships(
                    entity_name=args.get("entity_name", ""),
                )
                observation = self._format_graph_lookup_result(result)

                # Add to context if found
                new_context = list(state.context)
                if result.get("found"):
                    new_context.append(ContextItem(
                        source_type="entity",
                        content=observation,
                        source_id=args.get("entity_name", "unknown"),
                        relevance_score=1.0,
                    ))

                return {
                    "last_observation": observation,
                    "context": new_context,
                    "context_formatted": self._format_context(new_context),
                }

            elif tool_name == "web_search":
                if not self.web_search_enabled or not self._web_search_tool:
                    return {"last_observation": "Web search is not enabled"}

                result = self._web_search_tool.execute(
                    query=args.get("query", ""),
                    num_results=args.get("num_results", 5),
                )
                observation = self._format_web_search_result(result)

                # Add to context and optionally to database
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

                    # Optionally add to database
                    if self.auto_add_documents and item.get("url"):
                        try:
                            self.neo4j_loader.add_document(
                                url=item["url"],
                                title=item.get("title", ""),
                                content=item.get("snippet", ""),
                                domain=item.get("domain", ""),
                                trust_level=item.get("trust_level", "medium"),
                                source_type="web_search",
                            )
                        except Exception as e:
                            logger.warning(f"Failed to add document: {e}")

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

            else:
                return {"last_observation": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Action execution error: {e}")
            return {"last_observation": f"Error executing {tool_name}: {str(e)}"}

    def _format_graph_lookup_result(self, result: Dict[str, Any]) -> str:
        """Format graph lookup result."""
        if not result.get("found"):
            return f"Entity '{result.get('entity_name')}' not found in knowledge graph."

        entity = result["entity"]
        parts = [f"Entity: {entity.get('name', entity.get('id'))}"]

        if entity.get("description"):
            parts.append(f"Description: {entity['description']}")

        if entity.get("ontology_type"):
            parts.append(f"Type: {entity['ontology_type']}")

        relationships = result.get("relationships", [])
        if relationships:
            parts.append(f"\nRelationships ({len(relationships)}):")
            for rel in relationships[:10]:
                other = rel["other_entity"]
                other_name = other.get("name", other.get("id", "?"))
                direction = "->" if rel["direction"] == "outgoing" else "<-"
                parts.append(f"  {direction} [{rel['type']}] {other_name}")

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
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content)

            answer = parsed.get("answer", "I was unable to generate an answer.")
            confidence = parsed.get("confidence", 0.5)

            # Build citations
            citations = []
            for cit in parsed.get("citations", []):
                citations.append(Citation(
                    source_type=cit.get("source_type", "graph"),
                    source_id=cit.get("source_id", "unknown"),
                    source_title=cit.get("source_title"),
                    excerpt=cit.get("excerpt", ""),
                ))

            # Add external info notice if needed
            if state.external_info_used:
                if not answer.startswith("According to external"):
                    answer = f"[Note: This answer includes information from web search.]\n\n{answer}"

            return {
                "final_answer": answer,
                "citations": citations,
                "confidence": confidence,
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
            QAResponse with answer, citations, and reasoning.
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

    parser = argparse.ArgumentParser(description="ReAct Q&A Agent")
    parser.add_argument("--question", "-q", type=str, help="Question to answer")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-web", action="store_true", help="Disable web search")
    args = parser.parse_args()

    agent = ReActQAAgent(web_search_enabled=not args.no_web)

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
                print(f"External sources used: {response.external_info_used}")
                if response.citations:
                    print(f"Citations: {len(response.citations)}")
                    for cit in response.citations:
                        print(f"  - [{cit.source_type}] {cit.source_id}")
                print()

        elif args.question:
            response = agent.answer_question(args.question)
            print(f"Question: {args.question}")
            print(f"\nAnswer: {response.answer}")
            print(f"\nConfidence: {response.confidence:.2f}")
            print(f"External sources used: {response.external_info_used}")

        else:
            parser.print_help()

    finally:
        agent.close()


if __name__ == "__main__":
    main()
